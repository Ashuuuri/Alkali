#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/JSON.h"

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

// Analysis pass for converting the pointer value to LLVM type
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "ep2/passes/LiftUtils.h"
#include "ep2/Utilities.h"

#include <algorithm>
#include <functional>

#define DEBUG_TYPE "ep2-lift-llvm"

namespace mlir {
namespace ep2 {

namespace {

using namespace dataflow;

struct StackSlotValue : public AbstractDenseLattice {
  llvm::DenseMap<Value, Value> updateTable{};
  using AbstractDenseLattice::AbstractDenseLattice;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StackSlotValue);

  ChangeResult join(const AbstractDenseLattice &rhs) override {
    auto rvalue = static_cast<const StackSlotValue &>(rhs);
    auto changed = ChangeResult::NoChange;

    for (auto &pair : rvalue.updateTable) {
        auto [it, inserted] = updateTable.try_emplace(pair.first, pair.second);
        if (inserted)
            changed = ChangeResult::Change;
    }

    return changed;
  }

  // print and lookup

  Value lookup(Value value) const {
    llvm::DenseMap<Value, Value>::const_iterator it;
    llvm::SmallPtrSet<void *, 16> visited;
    while ((it = updateTable.find(value)) != updateTable.end()) {
      if (it->first == it->second)
        break;
      // Cycle detection: if we've seen this value before, break to avoid
      // infinite loop. Cycles should not occur given correct update() usage,
      // but this guard prevents hangs in unexpected edge cases.
      if (!visited.insert(value.getAsOpaquePointer()).second)
        break;
      value = it->second;
    }
    return value;
  }


  void print(llvm::raw_ostream &os) const override {
    os << "Dumping Dict...\n";
    for (auto &pair : updateTable) {
      os << "  ";
      pair.first.getLoc().print(os);
      os << " -> ";
      auto v = lookup(pair.second);
      if (v.getDefiningOp())
        v.getDefiningOp()->print(os);
      os << "\n";
    }
    os << "\n";
  }

  void printValue(llvm::raw_ostream &os, Value value) const {
    auto it = updateTable.find(value);
    if (it == updateTable.end())
      os << "Not a stack variable.";
    else {
      os << "definition = ";
      it->second.getDefiningOp()->print(os);
    }
  }
  
  // interfaces
  ChangeResult reset() {
    if (updateTable.empty())
      return ChangeResult::NoChange;
    updateTable.clear();
    return ChangeResult::Change;
  }
  ChangeResult init(Value value, Value source) {
    auto [it, inserted] = updateTable.try_emplace(value, source);
    return inserted ? ChangeResult::Change : ChangeResult::NoChange;
  }
  ChangeResult init(Value value) { return init(value, value); }

  ChangeResult update(Value value, Value newValue) {
    // Resolve newValue to its canonical representative first to prevent cycles.
    // If newValue already has a chain, we want to point to the terminal of that
    // chain, not to newValue itself (which might be mid-chain).
    newValue = lookup(newValue);

    llvm::DenseMap<Value, Value>::iterator it;
    auto changed = ChangeResult::NoChange;
    while ((it = updateTable.find(value)) != updateTable.end()) {
      if (it->first == it->second) {
        if (it->second != newValue) {
          it->second = newValue;
          changed = ChangeResult::Change;
        }
        break;
      }
      value = it->second;
    }
    if (changed == ChangeResult::Change)
      updateTable.try_emplace(newValue, newValue);
    return changed;
  }
};

class StackVariableAnalysis : public DenseForwardDataFlowAnalysis<StackSlotValue> {
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

  void visitOperation(Operation *op, const StackSlotValue &before,
                              StackSlotValue *after) override final {
    auto changed = after->join(before);

    auto opChanged =
        TypeSwitch<Operation *, ChangeResult>(op)
            // Only allocate creates the stack slot
            .Case<LLVM::AllocaOp, ep2::InitOp>([&](Operation *op) {
              return after->init(op->getResult(0));
            })
            .Case<LLVM::StoreOp>([&](LLVM::StoreOp &op) {
              return after->update(op.getAddr(), op.getValue());
            })
            .Case<ep2::AssignOp>([&](ep2::AssignOp &op) {
              // LHS is address
              return after->update(op.getLhs(), op.getRhs());
            })
            .Case<UnrealizedConversionCastOp, LLVM::LoadOp>([&](Operation *op) {
              return after->init(op->getResult(0), op->getOperand(0));
            })
            .Default([&](Operation *op) { return ChangeResult::NoChange; });

    propagateIfChanged(after, changed | opChanged);
  }

  // hook for external function
  void visitCallControlFlowTransfer(CallOpInterface call,
                                            CallControlFlowAction action,
                                            const StackSlotValue &before,
                                            StackSlotValue *after) override {
    llvm::errs() << "Call Control Flow Transfer\n";
    call->dump();
  }

  void setToEntryState(StackSlotValue *state) override {
    propagateIfChanged(state, state->reset());
  }
};

// XXX(zhiyuang): this is a hacky way to do this
static std::map<std::string, std::vector<int>> tableDesc;

Type liftLLVMType(OpBuilder &builder, Type type) {
  return TypeSwitch<Type, Type>(type)
      .Case<LLVM::LLVMPointerType>([&](LLVM::LLVMPointerType ptrType) {
        return ptrType;
      })
      .Case<LLVM::LLVMStructType>([&](LLVM::LLVMStructType structType) {
        // TODO(zhiyuang): support recursive struct
        // Drop the first "struct." prefix
        auto name = structType.getName().drop_front(7).str();
        auto it = tableDesc.find(name);

        if (name.empty() || it == tableDesc.end()) {
          auto liftedTypes = llvm::map_to_vector(
              structType.getBody(), [&](Type type) { return liftLLVMType(builder, type); });
          return builder.getType<ep2::StructType>(false, liftedTypes, "__anno__");
        } else {
          // we have a spec for the table
          // TODO(zhiyuang): now assume all tables contains integer. support better in the future!
          auto liftedTypes = llvm::map_to_vector(
              llvm::zip(structType.getBody(), it->second), [&](std::tuple<Type, int> tuple) -> Type {
                auto [type, size] = tuple;
                return builder.getIntegerType(size);
              });

          return builder.getType<ep2::StructType>(false, liftedTypes, name);
        }
      })
      .Default([&](Type type) { return type; });
}

// Dialect Conversion
void populateTypeConversion(TypeConverter &typeConverter, OpBuilder &builder) {

  // bind the funciion within the context
  typeConverter.addConversion([&](Type type) { return liftLLVMType(builder, type); });

  // wildcard conversion to make system work. remove this later!
  typeConverter.addSourceMaterialization(
      [&](OpBuilder &builder, Type type, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return builder
            .create<UnrealizedConversionCastOp>(loc, TypeRange{type}, inputs)
            .getOutputs()[0];
      });
  typeConverter.addArgumentMaterialization(
      [&](OpBuilder &builder, Type type, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        llvm::errs() << "wildcard conversion: argument\n";
        if (inputs.size() != 1)
          return std::nullopt;
        return builder
            .create<UnrealizedConversionCastOp>(loc, TypeRange{type}, inputs)
            .getOutputs()[0];
      });
  typeConverter.addTargetMaterialization(
      [&](OpBuilder &builder, Type type, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        llvm::errs() << "wildcard conversion: target\n";
        if (inputs.size() != 1)
          return std::nullopt;
        return builder
            .create<UnrealizedConversionCastOp>(loc, TypeRange{type}, inputs)
            .getOutputs()[0];
      });


}


Type convertLLVMPointer(OpBuilder &builder, Value pointer) {
  auto ptrType = pointer.getType().cast<LLVM::LLVMPointerType>();

  if (auto elementType = ptrType.getElementType())
    return liftLLVMType(builder, elementType);
  else if (auto allocaOp = dyn_cast_or_null<LLVM::AllocaOp>(pointer.getDefiningOp())) {
    auto type = allocaOp.getElemType();
    if (type)
      return liftLLVMType(builder, *type);
  }

  return builder.getType<ep2::AnyType>();
}

Value castEP2Value(OpBuilder &builder, Value source, Type target) {
  auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(
      source.getLoc(), TypeRange{target}, ValueRange{source});
  return castOp.getResult(0);
}
Value castEP2Value(OpBuilder &builder, Value source) {
  return castEP2Value(builder, source, builder.getType<ep2::AnyType>());
}

auto parseFunctionName(StringRef name) {
  if (name.startswith("EXT__")) // this is a extern function. ignore this for now
    name = name.drop_front(5);
  return name.split("__");
}

// canonicalize the LLVM input
void handlerCanonicalize(OpBuilder &builder, ep2::FuncOp funcOp) {
  funcOp->setAttr("type", builder.getStringAttr("handler"));

  auto name = funcOp.getName();
  if (name.startswith("EXT__")) {
    // drop the prefix "EXT__"
    name = name.drop_front(5);
    funcOp->setAttr("extern", builder.getBoolAttr(true));
  }

  auto [event, atom] = parseFunctionName(name);

  funcOp->setAttr("event", builder.getStringAttr(event));
  funcOp->setAttr("atom", builder.getStringAttr(atom));

  auto canonName = "__handler_" + event.str() + "_" + atom.str();
  funcOp.setName(canonName);
}

struct FuncRewrite : public OpRewritePattern<LLVM::LLVMFuncOp> {
  TypeConverter &converter;
  FuncRewrite(MLIRContext *context, TypeConverter &converter)
      : OpRewritePattern(context), converter(converter) {}

  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp op,
                                PatternRewriter &rewriter) const final {

    // EP2 parameter conversion
    for (auto &block : op.getBlocks()) {
      for (auto &arg : block.getArguments()) {
        if (isa<LLVM::LLVMPointerType>(arg.getType())) {
          // only for BA, this is a buffer type
          arg.setType(rewriter.getType<ep2::BufferType>());
        } else // else we convert the type
          arg.setType(liftLLVMType(rewriter, arg.getType()));
      }
    }

    auto entryBlock = &op.front();
    auto newFunc = rewriter.create<ep2::FuncOp>(op.getLoc(), op.getName(),
      rewriter.getFunctionType(entryBlock->getArgumentTypes(), {}));
    newFunc.getRegion().takeBody(op.getRegion());

    handlerCanonicalize(rewriter, newFunc);

    rewriter.eraseOp(op);
    return success();
  }
};

// Rewrite for Calls
struct CallRewrite : public OpRewritePattern<LLVM::CallOp> {
  TypeConverter &converter;
  CallRewrite(MLIRContext *context, TypeConverter &converter) : OpRewritePattern(context), converter(converter) {};

  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &rewriter) const final {
    auto funcName = op.getCallee();

    auto loc = op.getLoc();
    auto args = op.getArgOperands();

    // auto anyTableType = rewriter.getType<ep2::TableType>(
    //     rewriter.getType<ep2::AnyType>(), rewriter.getType<ep2::AnyType>(), 0);

    if (!funcName.has_value()) {
      // TODO(zhiyuang): dynamic function address. why?
      // TODO(zhiyuang): WE assume this is an init op. need to be fixed!!!
      rewriter.replaceOpWithNewOp<ep2::InitOp>(op, rewriter.getType<ep2::BufferType>());
    } else if (funcName == "bufextract") {
      auto buf =  castEP2Value(rewriter, args[0], rewriter.getType<ep2::BufferType>());
      auto header = castEP2Value(rewriter, args[1], convertLLVMPointer(rewriter, args[1]));

      auto extractOp = rewriter.create<ep2::ExtractOp>(loc, header.getType(), buf);
      rewriter.replaceOpWithNewOp<ep2::AssignOp>(op, header, extractOp);
      return success();
    } else if (funcName == "bufemit") {
      auto buf =  castEP2Value(rewriter, args[0], rewriter.getType<ep2::BufferType>());
      auto header = castEP2Value(rewriter, args[1]);

      rewriter.replaceOpWithNewOp<ep2::EmitOp>(op, buf, header);
      return success();
    } else if (funcName == "bufinit") {
      rewriter.replaceOpWithNewOp<ep2::InitOp>(op, rewriter.getType<ep2::BufferType>());
      return success();
    } else if (funcName == "table_lookup") {
      // TODO(zhiyuang): do we need an extra cast here?
      auto table = castEP2Value(rewriter, args[0]);
      auto key = castEP2Value(rewriter, args[1]);
      auto value = castEP2Value(rewriter, args[2]);

      auto lookupOp = rewriter.create<ep2::LookupOp>(loc, value.getType(), table, key);
      rewriter.replaceOpWithNewOp<ep2::AssignOp>(op, value, lookupOp);
      return success();
    } else if (funcName == "table_update") {
      auto table = castEP2Value(rewriter, args[0]);
      auto key = castEP2Value(rewriter, args[1]);
      auto value = castEP2Value(rewriter, args[2]);

      rewriter.replaceOpWithNewOp<ep2::UpdateOp>(op, table, key, value);
      return success();
    } else {
      // this is a generation call. Here we use the raw format, and use another pass to call it later
      auto callee = op.getCallee();
      // TODO(zhiyuang): check this? what's a format of value call
      auto argValues = llvm::to_vector(args);
      auto [event, atom] = parseFunctionName(*callee);
      auto [_, returnOp] = createGenerate(rewriter, op.getLoc(), event, atom, argValues);

      rewriter.replaceOp(op, returnOp);
      return success();
    }

  }
};

class LoadRewrite : public OpRewritePattern<LLVM::LoadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::LoadOp op, PatternRewriter &rewriter) const final {
    auto gepOp = dyn_cast_if_present<LLVM::GEPOp>(op.getAddr().getDefiningOp());
    if (!gepOp)
      return rewriter.notifyMatchFailure(op, "unsupported load type");

    // TODO(zhiyuang): error check the logic
    // GEP to StructUpdate
    auto elemType = gepOp.getElemType();
    auto structType = liftLLVMType(rewriter, *elemType);

    auto offset = gepOp.getIndices()[1].get<IntegerAttr>();

    // TODO: use adaptor?
    auto converted = castEP2Value(rewriter, gepOp.getOperand(0), structType);
    auto accessOp = rewriter.create<ep2::StructAccessOp>(op.getLoc(), converted, offset.getInt());
    Value result = accessOp.getResult();
    // If struct field type is narrower than the LLVM load type, sign-extend to
    // restore the original width so downstream LLVM arithmetic (e.g. xor) stays
    // type-consistent.
    auto origIntTy = dyn_cast<IntegerType>(op.getType());
    auto newIntTy  = dyn_cast<IntegerType>(result.getType());
    if (origIntTy && newIntTy && newIntTy.getWidth() < origIntTy.getWidth())
      result = rewriter.create<LLVM::ZExtOp>(op.getLoc(), origIntTy, result);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class StoreRewrite : public OpRewritePattern<LLVM::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(LLVM::StoreOp op, PatternRewriter &rewriter) const final {
    auto gepOp = dyn_cast_if_present<LLVM::GEPOp>(op.getAddr().getDefiningOp());
    if (!gepOp)
      return rewriter.notifyMatchFailure(op, "unsupported store type");

    // GEP to StructUpdate
    auto elemType = gepOp.getElemType();
    auto structType = liftLLVMType(rewriter, *elemType);

    auto offset = gepOp.getIndices()[1].get<IntegerAttr>();
    auto newStruct = rewriter.create<ep2::StructUpdateOp>(
        op.getLoc(), structType, gepOp.getOperand(0), offset.getInt(), op.getValue());

    rewriter.replaceOpWithNewOp<ep2::AssignOp>(op, gepOp.getBase(), newStruct);
    return success();
  }
};

// convert alloca of struct to init
struct InitRewrite : public OpRewritePattern<LLVM::AllocaOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::AllocaOp op, PatternRewriter &rewriter) const final {
    auto elemType = dyn_cast_if_present<LLVM::LLVMStructType>(op.getElemType().value_or(nullptr));
    if (!elemType)
      return rewriter.notifyMatchFailure(op, "non-target alloca");

    // "buf_tag" is the underlying struct for buf_t (buf_t = struct buf_tag *).
    // Allocating a struct buf_tag creates a local buffer; treat it as Buffer.
    if (elemType.getName() == "struct.buf_tag") {
      rewriter.replaceOpWithNewOp<ep2::InitOp>(op, rewriter.getType<ep2::BufferType>());
      return success();
    }

    auto structType = liftLLVMType(rewriter, elemType);
    rewriter.replaceOpWithNewOp<ep2::InitOp>(op, structType);
    return success();
  }
};

struct ReturnRewrite : public OpRewritePattern<LLVM::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::ReturnOp op, PatternRewriter &rewriter) const final {
    // TODO: check is no value
    rewriter.replaceOpWithNewOp<ep2::TerminateOp>(op);
    return success();
  }
};

} // namespace

static LogicalResult preAnalysisRewrite(Operation *op) {
  auto builder = OpBuilder(op);

  TypeConverter converter;
  populateTypeConversion(converter, builder);

  // apply rules
  mlir::RewritePatternSet patterns(builder.getContext());
  patterns.add<FuncRewrite, CallRewrite>(builder.getContext(), converter);
  patterns.add<LoadRewrite, StoreRewrite, InitRewrite, ReturnRewrite>(builder.getContext());

  FrozenRewritePatternSet patternSet(std::move(patterns));
  return applyPatternsAndFoldGreedily(op, patternSet);
}

namespace {

// rewrite patterns
class ConstantConversion : public OpConversionPattern<LLVM::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(LLVM::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    auto valueAttr = op.getValue();
    return TypeSwitch<Attribute, LogicalResult>(valueAttr)
        .Case([&](IntegerAttr attr) {
          rewriter.replaceOpWithNewOp<ep2::ConstantOp>(op, attr.getValue().getBitWidth(), attr.getInt());
          return success();
        })
        .Default([&](Attribute attr) {
          // TODO(zhiyuang): check this. we remove for now.
          rewriter.eraseOp(op);
          return success();
          // return rewriter.notifyMatchFailure(op, "unsupported constant type");
        });
  }
};

class InsertValueConversion : public OpConversionPattern<LLVM::InsertValueOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(LLVM::InsertValueOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

std::optional<Type> tableIdentification (OpBuilder &builder, LLVM::LLVMStructType structType) {
  auto types = structType.getBody();
  if (types.size() < 3)
    return std::nullopt;

  auto keyType = liftLLVMType(builder, types[0]);
  auto valueType = liftLLVMType(builder, types[1]);

  auto sizeType = types[2].dyn_cast<LLVM::LLVMArrayType>();
  if (!sizeType)
    return std::nullopt;
  
  auto size = sizeType.getNumElements();
  return builder.getType<ep2::TableType>(keyType, valueType, size);
}

class AddressOfConversion : public OpConversionPattern<LLVM::AddressOfOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(LLVM::AddressOfOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    OpBuilder builder(op);
    auto moduleOp = op->getParentOfType<ModuleOp>();
    SymbolTable symbolTable(moduleOp);

    auto table = symbolTable.lookup<LLVM::GlobalOp>(op.getGlobalName());
    if (!table)
      return rewriter.notifyMatchFailure(op, "cannot find global variable");

    auto tableType = table.getType();
    return TypeSwitch<Type, LogicalResult>(tableType)
        // This is a table...
        .Case<LLVM::LLVMStructType>([&](LLVM::LLVMStructType structType) {

          auto tableType = tableIdentification(rewriter, structType);
          if (!tableType.has_value())
            return rewriter.notifyMatchFailure(op, "unsupported table type");

          rewriter.replaceOpWithNewOp<ep2::GlobalImportOp>(op, *tableType, op.getGlobalName());
          return success();
        })
        .Default([&](Type type) {
          return rewriter.notifyMatchFailure(op, "unsupported type");
        });
  }
};

class GloablOpConversion : public OpConversionPattern<LLVM::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(LLVM::GlobalOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    
    auto elemType = dyn_cast_if_present<LLVM::LLVMStructType>(op.getType());
    if (!elemType) {
      rewriter.eraseOp(op);
      return success();
    }
    
    auto tableType = tableIdentification(rewriter, elemType);
    if (!tableType.has_value()) {
      rewriter.eraseOp(op);
      return success();
    }
      
    rewriter.create<ep2::GlobalOp>(op.getLoc(), *tableType, op.getName());
    rewriter.eraseOp(op);

    return success();
  }
};

class AddConversion : public OpConversionPattern<LLVM::AddOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(LLVM::AddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ep2::AddOp>(op, op.getResult().getType(), adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

class SubConversion : public OpConversionPattern<LLVM::SubOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(LLVM::SubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ep2::SubOp>(op, op.getResult().getType(), adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

class SExtConversion : public OpConversionPattern<LLVM::SExtOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(LLVM::SExtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ep2::BitCastOp>(op, op.getResult().getType(), adaptor.getArg());
    return success();
  }
};

class TruncConversion : public OpConversionPattern<LLVM::TruncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(LLVM::TruncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ep2::BitCastOp>(op, op.getResult().getType(), adaptor.getArg());
    return success();
  }
};

class ZExtConversion : public OpConversionPattern<LLVM::ZExtOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(LLVM::ZExtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ep2::BitCastOp>(op, op.getResult().getType(), adaptor.getArg());
    return success();
  }
};

// Map LLVM icmp predicate to EP2 CmpOp predicate char value.
// EP2 uses tok_cmp_eq=-40 (stored as 40), tok_cmp_le=-41 (41), tok_cmp_ge=-42 (42),
// '<'=60 for slt, '>'=62 for sgt.  Use 43 as NE (not in existing EP2 tok set).
static std::optional<int> llvmICmpToEP2Pred(LLVM::ICmpPredicate pred) {
  switch (pred) {
    case LLVM::ICmpPredicate::eq:  return 40;   // -tok_cmp_eq
    case LLVM::ICmpPredicate::ne:  return 43;   // NE extension
    case LLVM::ICmpPredicate::slt:
    case LLVM::ICmpPredicate::ult: return 60;   // '<'
    case LLVM::ICmpPredicate::sle:
    case LLVM::ICmpPredicate::ule: return 41;   // -tok_cmp_le
    case LLVM::ICmpPredicate::sgt:
    case LLVM::ICmpPredicate::ugt: return 62;   // '>'
    case LLVM::ICmpPredicate::sge:
    case LLVM::ICmpPredicate::uge: return 42;   // -tok_cmp_ge
    default: return std::nullopt;
  }
}

class ICmpConversion : public OpConversionPattern<LLVM::ICmpOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(LLVM::ICmpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    auto ep2Pred = llvmICmpToEP2Pred(op.getPredicate());
    if (!ep2Pred)
      return rewriter.notifyMatchFailure(op, "unsupported icmp predicate");
    rewriter.replaceOpWithNewOp<ep2::CmpOp>(op, (char)*ep2Pred,
                                             adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

// Convert LLVM CF branch ops to CF dialect branch ops.
class LLVMBrConversion : public OpConversionPattern<LLVM::BrOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(LLVM::BrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(),
                                               adaptor.getDestOperands());
    return success();
  }
};

class LLVMCondBrConversion : public OpConversionPattern<LLVM::CondBrOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(LLVM::CondBrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(),
        op.getTrueDest(), adaptor.getTrueDestOperands(),
        op.getFalseDest(), adaptor.getFalseDestOperands());
    return success();
  }
};

} // namespace

static LogicalResult postAnalysisRewrite(Operation *op) {
  auto builder = OpBuilder(op);

  // type conversion
  TypeConverter typeConverter;
  populateTypeConversion(typeConverter, builder);

  ConversionTarget target(*builder.getContext());
  target.addLegalDialect<ep2::EP2Dialect>();
  target.addIllegalOp<LLVM::AddressOfOp, LLVM::GlobalOp, LLVM::InsertValueOp, LLVM::ConstantOp>();
  // arith
  target.addIllegalOp<LLVM::AddOp, LLVM::SExtOp, LLVM::TruncOp, LLVM::ZExtOp>();
  // comparisons and control flow
  target.addIllegalOp<LLVM::ICmpOp, LLVM::BrOp, LLVM::CondBrOp>();
  target.addLegalDialect<cf::ControlFlowDialect>();

  mlir::RewritePatternSet patterns(builder.getContext());
  patterns.add<ConstantConversion, AddressOfConversion, InsertValueConversion,
               GloablOpConversion>(typeConverter, builder.getContext());
  patterns.add<AddConversion, SubConversion, SExtConversion, TruncConversion,
               ZExtConversion, ICmpConversion,
               LLVMBrConversion, LLVMCondBrConversion>(typeConverter, builder.getContext());

  FrozenRewritePatternSet patternSet(std::move(patterns));

  return applyPartialConversion(op, target, patternSet);
}

void LiftLLVMPasses::runOnOperation() {
  OpBuilder builder(getOperation());

  // process for json array
  if (!structDesc.hasValue()) {
    llvm::errs() << "struct description is required\n";
    return signalPassFailure();
  }

  auto jsonBuffer = llvm::MemoryBuffer::getFile(structDesc.getValue());
  if (jsonBuffer.getError()) {
    llvm::errs() << "cannot open file\n";
    return signalPassFailure();
  }

  auto json = llvm::json::parse(jsonBuffer.get()->getBuffer());
  if (auto err = json.takeError()) {
    llvm::errs() << "cannot parse json\n";
    return signalPassFailure();
  }

  tableDesc.clear();
  llvm::json::Path::Root root;
  llvm::json::Path path(root);
  llvm::json::fromJSON(json.get(), tableDesc, path);

  // Set all functions to public; remove main and external declarations.
  // Also collect which functions have at least one call site.
  {
    llvm::DenseSet<StringRef> calledFunctions;
    getOperation()->walk([&](LLVM::CallOp call) {
      if (auto callee = call.getCallee())
        calledFunctions.insert(*callee);
    });

    OperatorRemoveGuard toRemove;
    getOperation()->walk([&](LLVM::LLVMFuncOp func) {
      // make sure everything is public and cannot be ignored
      if (!func.isDeclaration())
          func.setPublic();
      // remove the main function
      if (func.getName() == "main")
        toRemove.add(func);
      // clear external lib functions
      if (func.isExternal())
        toRemove.add(func);
      // remove internal/private helper functions with no remaining call sites
      // (already inlined by the MLIR --inline pass)
      auto lnk = func.getLinkage();
      if (!func.isDeclaration() &&
          (lnk == LLVM::Linkage::Internal || lnk == LLVM::Linkage::Private) &&
          calledFunctions.find(func.getName()) == calledFunctions.end())
        toRemove.add(func);
    });
  }

  // Remove lifetime intrinsics before preAnalysisRewrite: InitRewrite converts
  // alloca pointers to struct values, which breaks the pointer-typed operand of
  // lifetime.start/end.
  {
    OperatorRemoveGuard toRemove;
    getOperation()->walk([&](Operation *op) {
      if (isa<LLVM::LifetimeStartOp, LLVM::LifetimeEndOp>(op))
        toRemove.add(op);
    });
  }

  if (failed(preAnalysisRewrite(getOperation())))
    return signalPassFailure();

  if (failed(postAnalysisRewrite(getOperation())))
    return signalPassFailure();

  DataFlowSolver solver;
  // must have this two lines, to help with liveness
  solver.load<SparseConstantPropagation>();
  solver.load<DeadCodeAnalysis>();
  solver.load<StackVariableAnalysis>();
  // currently we cannot go with "blackbox" function calls
  // this is fixed within newer MLIR versions

  if (solver.initializeAndRun(getOperation()).failed())
    return signalPassFailure();

  // replace!
  {
    OperatorRemoveGuard toRemove;
    getOperation()->walk([&](Operation *op) {
      // since its dense, we always have state
      auto state = solver.lookupState<StackSlotValue>(op);

      // replace the operator with memory slots
      for (unsigned i = 0; i < op->getNumOperands(); i++) {
        auto arg = op->getOperand(i);
        op->setOperand(i, state->lookup(arg));
      }

      // speciali handling after the replacement
      TypeSwitch<Operation *>(op)
          .Case([&](LLVM::StoreOp storeOp) {
            // TODO(zhiyuang): how to cast TypedValue to Value?
            if (storeOp.getOperand(0) == storeOp.getOperand(1))
              toRemove.add(storeOp);
          })
          .Case([&](ep2::AssignOp assignOp) {
            if (assignOp.getLhs() == assignOp.getRhs())
              toRemove.add(assignOp);
          })
          .Case([&](ep2::LookupOp lookupOp) {
            auto valueType = lookupOp.getTable().getType().getValueType();
            lookupOp.getResult().setType(valueType);
          })
          .Case([&](LLVM::LoadOp loadOp){
            // if we find load op try to load a value.. eliminate it
            auto addr = loadOp.getOperand();
            if (!isa<LLVM::LLVMPointerType>(addr.getType())) {
              loadOp.replaceAllUsesWith(addr);
              toRemove.add(loadOp);
            }
          });

    });
  }

  // dce
  OperatorRemoveGuard::until([&](OperatorRemoveGuard &toRemove) {
    getOperation()->walk([&](Operation *op) {
      // Never remove terminators — they're needed for block structure even if
      // they have no SSA result uses.
      if (op->hasTrait<OpTrait::IsTerminator>())
        return;
      if (op->use_empty() && (isPure(op) || isa<LLVM::AllocaOp>(op)))
        toRemove.add(op);
    });
  });
}


} // namespace ep2
} // namespace mlir

#undef DEBUG_TYPE
