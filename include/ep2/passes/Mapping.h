#ifndef EP2_MAPPING_H
#define EP2_MAPPING_H

#include "ep2/dialect/Dialect.h"
#include "ep2/Utilities.h"
#include "ep2/passes/WorkloadSpec.h"

#include <map>
#include <string>
#include <vector>

namespace mlir {
namespace ep2 {

// handler a pipeline and insert controller
using HandlerPipeline = llvm::SmallVector<ep2::FuncOp>;

void simpleMapping(HandlerPipeline &pipeline, llvm::SmallVector<int> *replications = nullptr);
void simpleGlobalMapping(HandlerPipeline &pipeline, int localTableNumber = 0);

void preMappingCanonicalize(HandlerPipeline &pipeline, llvm::StringRef mode);
void insertController(HandlerPipeline &pipeline);


// netronome specific passes
void bufferToRef(HandlerPipeline &pipeline);
void contextIdentification(HandlerPipeline &pipeline);

// handler split and pipeline
struct PipelineResult {
  double sourceWeight;
  llvm::DenseSet<mlir::Operation*> sinkOps;
  llvm::DenseSet<mlir::Value> sinkValues;
  std::string err;
  bool dumpFile;
};
struct PipelinePolicy {
  double sourceWeight;
  double tolerance;
  // options
  bool done = false, dumpCuts = true;

  PipelinePolicy(double sourceWeight, double tolerance) : sourceWeight(sourceWeight), tolerance(tolerance) {}

  virtual int typeTransmitCost(mlir::Type t) = 0;
  virtual int operationWeight(mlir::Operation* op) = 0;
  virtual int valueWeight(mlir::Value v) = 0;

  virtual std::pair<std::shared_ptr<PipelinePolicy>, std::shared_ptr<PipelinePolicy>> splitPolicy(PipelineResult &result) = 0;
  virtual std::pair<std::string, std::string> splitName() = 0;
};
bool pipelineHandler(ep2::FuncOp funcOp, PipelinePolicy* policy, PipelineResult* results);

using PolicyP = std::shared_ptr<PipelinePolicy>;
using SearchPair = std::pair<ep2::FuncOp, PolicyP>;
using SearchDirection = llvm::DenseMap<ep2::FuncOp, PolicyP>;

// A list of pipeline policies
std::pair<bool, SmallVector<ep2::FuncOp>> tableCut(ep2::FuncOp targetFunc,
                                                    llvm::DenseMap<mlir::Operation*, int> tableMemMap,
                                                    double avgPktBytes = 64.0);
bool isTableClean(ep2::FuncOp funcOp);

void kcutPolicy(Operation * moduleOp, int k, FuncOp targetFunc);
void bfsSearchPolicy(Operation * moduleOp);
void weightPolicy(FuncOp targetFunc, PolicyP weightPolicy);


// ---------------------------------------------------------------------------
// Netronome hardware spec (loaded from JSON or constructed from defaults)
// ---------------------------------------------------------------------------

struct MemLayerSpec {
  std::string id;       // "LMEM" | "CLS" | "CTM" | "EMEM"
  std::string scope;    // "per_me" | "per_island" | "chip"
  int64_t sizeBytes;
  int latencyCycles;
};

struct NetronomeSpec {
  int frequencyMhz    = 800;
  int latencyTarget   = 100;
  int intraIslandCost = 0;   // cycles to send event within same island
  int interIslandCost = 0;   // cycles to send event across islands

  std::vector<std::string>      computeUnitIds; // ordered ME id list
  std::map<std::string, int>    meIsland;       // me_id -> island index
  std::map<std::string, int>    instrLatency;   // op_name -> cycles
  std::vector<MemLayerSpec>     memoryLayers;

  // Instruction latency lookup (falls back to "default" key, then 1)
  int getInstrLatency(const std::string &opName) const;

  // Memory layer queries by layer id string ("LMEM", "CLS", ...)
  int     getMemoryLatency(const std::string &layerId) const;
  int64_t getMemorySize(const std::string &layerId) const;

  // Factory: parse from JSON file; falls back to defaults() on any error
  static NetronomeSpec load(llvm::StringRef path);
  // Factory: exact replica of the original hardcoded NetronomePerformanceModel
  static NetronomeSpec defaults();
};

// performance model
class PerformanceModel {
 public:
  using UnitMap = std::map<int, std::vector<std::string>>;
  struct MappingResult {
    int latency;
    int bottleneckIndex;
    UnitMap unitMap;

    bool operator<(const MappingResult &other) const {
      return latency < other.latency;
    }
  };

  virtual int getAccessOverhead(ep2::GlobalOp globalOp) = 0;
  // Returns Time_Instr + Time_Mem for the handler.
  // Time_Comm is excluded — it is mapping-dependent and computed separately
  // via getCommunicationCost.
  virtual int getLatency(ep2::FuncOp funcOp) = 0;
  virtual int getCommunicationCost(std::vector<std::string> &froms,
                                   std::vector<std::string> &tos) = 0;
  virtual int getLatencyTarget() = 0;
  virtual std::vector<std::string> getComputeUnits() = 0;
  virtual llvm::DenseMap<mlir::Operation*, int> getTableMemMap(ep2::FuncOp) { return {}; }

  // This function provides a simple, greedy mapping method for a sequence of handlers
  virtual MappingResult
  getMapping(llvm::SmallVector<ep2::FuncOp> &ops);
};

// FPGA Model
class FPGAPerformanceModel : public PerformanceModel {
  const int numComputeUnits = 128;
 public:
   int getAccessOverhead(ep2::GlobalOp globalOp) override { return 0; }
   int getLatency(ep2::FuncOp funcOp) override {
     int latency = 1;
     funcOp.walk([&](Operation *op) {
       if (isa<ep2::LookupOp, ep2::UpdateOp>(op))
         latency = 3;
     });
     return latency;
   }
   int getCommunicationCost(std::vector<std::string> &froms,
                            std::vector<std::string> &tos) override {
     return 0;
   }
   int getLatencyTarget() override { return 1; }
   std::vector<std::string> getComputeUnits() override {
    std::vector<std::string> computeUnits;
    for (int i = 0; i < numComputeUnits; i++)
      computeUnits.push_back("vcu" + std::to_string(i));
    return computeUnits;
   }
};

// Netronome Model — reads from NetronomeSpec (JSON or hardcoded defaults)
class NetronomePerformanceModel : public PerformanceModel {
  NetronomeSpec spec_;
  WorkloadSpec workload_;
 public:
  // Default constructor: preserves original hardcoded behaviour
  NetronomePerformanceModel()
      : spec_(NetronomeSpec::defaults()), workload_(WorkloadSpec::none()) {}
  // Spec-file constructor: loads JSON, falls back to defaults on error
  explicit NetronomePerformanceModel(llvm::StringRef specPath)
      : spec_(NetronomeSpec::load(specPath)), workload_(WorkloadSpec::none()) {}
  // Spec + workload constructor
  NetronomePerformanceModel(llvm::StringRef specPath, llvm::StringRef workloadPath)
      : spec_(specPath.empty() ? NetronomeSpec::defaults() : NetronomeSpec::load(specPath)),
        workload_(WorkloadSpec::load(workloadPath)) {}

  int getAccessOverhead(ep2::GlobalOp globalOp) override { return 0; }

  // Returns the size of one table entry in bytes from a GlobalImportOp.
  int64_t getTableBytes(ep2::GlobalImportOp importOp) {
    auto tableType =
        importOp.getOutput().getType().dyn_cast<ep2::TableType>();
    if (!tableType)
      return 0;
    int numEntries = tableType.getSize();
    int entryBits = 0;
    auto addBits = [&](mlir::Type ty) {
      if (auto intTy = ty.dyn_cast<mlir::IntegerType>())
        entryBits += intTy.getWidth();
      else if (auto structTy = ty.dyn_cast<ep2::StructType>())
        for (auto elemTy : structTy.getElementTypes())
          if (auto iTy = elemTy.dyn_cast<mlir::IntegerType>())
            entryBits += iTy.getWidth();
    };
    addBits(tableType.getKeyType());
    addBits(tableType.getValueType());
    return static_cast<int64_t>(numEntries) * ((entryBits + 7) / 8);
  }

  // Assigns each unique table in funcOp to LMEM or CLS, mirroring the
  // Netronome backend's binary placement (isLocal → __lmem, else → __cls).
  //
  // Placement priority (which tables enter LMEM first):
  //   hotKeyRatio == 0: sort by size ascending (small tables fill LMEM first).
  //   hotKeyRatio >  0: sort by access density descending (opCount / sizeBytes).
  //     Skewed traffic concentrates accesses on fewer tables; prioritising
  //     high-density tables maximises the latency benefit of fast LMEM.
  //
  // Returns empty map when memoryLayers is empty (defaults() path → 0 cost).
  llvm::DenseMap<mlir::Operation *, int>
  buildTableMemMap(ep2::FuncOp funcOp, double hotKeyRatio = 0.0) {
    llvm::DenseMap<mlir::Operation *, int> result;

    // Find LMEM (per_me) and CLS (first per_island) layer latencies
    int lmemLatency = 0, clsLatency = 0;
    int64_t lmemSize = 0;
    for (auto &layer : spec_.memoryLayers) {
      if (layer.scope == "per_me" && lmemSize == 0) {
        lmemLatency = layer.latencyCycles;
        lmemSize    = layer.sizeBytes;
      } else if (layer.scope == "per_island" && clsLatency == 0) {
        clsLatency = layer.latencyCycles;
      }
    }
    if (lmemSize == 0 && clsLatency == 0)
      return result; // no layer info (defaults() path)

    // Collect unique tables with size and op-access count
    struct TableEntry {
      ep2::GlobalImportOp importOp;
      int64_t sizeBytes;
      int opCount; // number of lookup/update ops referencing this table
    };
    llvm::SmallVector<TableEntry> tables;
    llvm::DenseMap<mlir::Operation *, int> opCountMap;
    funcOp.walk([&](Operation *op) {
      ep2::GlobalImportOp importOp = nullptr;
      if (auto lookup = dyn_cast<ep2::LookupOp>(op))
        importOp = lookup.getTable().getDefiningOp<ep2::GlobalImportOp>();
      else if (auto update = dyn_cast<ep2::UpdateOp>(op))
        importOp = update.getTable().getDefiningOp<ep2::GlobalImportOp>();
      if (!importOp)
        return;
      opCountMap[importOp]++;
    });
    llvm::DenseSet<mlir::Operation *> seen;
    for (auto &[op, cnt] : opCountMap) {
      auto importOp = cast<ep2::GlobalImportOp>(op);
      tables.push_back({importOp, getTableBytes(importOp), cnt});
    }

    if (hotKeyRatio > 0.0) {
      // Skewed workload: prioritise tables with the highest access density
      // (ops per byte) so the most-accessed data lands in fast LMEM.
      llvm::sort(tables, [](const TableEntry &a, const TableEntry &b) {
        // Higher density first (descending); break ties by smaller size
        double da = (double)a.opCount / (a.sizeBytes + 1);
        double db = (double)b.opCount / (b.sizeBytes + 1);
        return da != db ? da > db : a.sizeBytes < b.sizeBytes;
      });
    } else {
      // No workload info: sort by size ascending (small tables fill LMEM first)
      llvm::sort(tables, [](const TableEntry &a, const TableEntry &b) {
        return a.sizeBytes < b.sizeBytes;
      });
    }

    // Assign to LMEM if it fits, otherwise CLS
    int64_t lmemRemaining = lmemSize;
    for (auto &entry : tables) {
      int latency = clsLatency; // default: CLS
      if (entry.sizeBytes <= lmemRemaining) {
        lmemRemaining -= entry.sizeBytes;
        latency = lmemLatency;
        llvm::errs() << "[TimeMem] table " << entry.sizeBytes << "B (ops="
                     << entry.opCount << ") -> LMEM (" << latency << "c)\n";
      } else {
        llvm::errs() << "[TimeMem] table " << entry.sizeBytes << "B (ops="
                     << entry.opCount << ") -> CLS (" << latency << "c)\n";
      }
      result[entry.importOp] = latency;
    }
    return result;
  }

  // Returns Time_Instr + Time_Mem for the handler.
  // Time_Comm is excluded — it is mapping-dependent and computed separately
  // via getCommunicationCost.
  int getLatency(ep2::FuncOp funcOp) override {
    int latency = 0;
    double avgPkt = workload_.avgPktBytes;  // default 64
    double hot    = workload_.hotKeyRatio;  // default 0.0

    // hotKeyRatio influences placement priority inside buildTableMemMap:
    // hot > 0 → rank by access density (ops/byte) so frequently-used tables
    // get LMEM first; hot == 0 → rank by size (current default).
    auto tableMemMap = buildTableMemMap(funcOp, hot);

    funcOp.walk([&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op)
          .Case<ep2::LookupOp>([&](ep2::LookupOp lookupOp) {
            int instrCost = spec_.getInstrLatency("lookup");
            int memCost = 0;
            if (auto importOp =
                    lookupOp.getTable().getDefiningOp<ep2::GlobalImportOp>())
              memCost = tableMemMap.lookup(importOp);
            latency += instrCost + memCost;
          })
          .Case<ep2::UpdateOp>([&](ep2::UpdateOp updateOp) {
            int instrCost = spec_.getInstrLatency("update");
            int memCost = 0;
            if (auto importOp =
                    updateOp.getTable().getDefiningOp<ep2::GlobalImportOp>())
              memCost = tableMemMap.lookup(importOp);
            latency += instrCost + memCost;
          })
          .Case<ep2::ExtractOp>([&](Operation *) {
            // Larger packets require more cycles to move (1c per 8B overhead)
            latency += static_cast<int>(spec_.getInstrLatency("extract") + avgPkt / 8.0);
          })
          .Case<ep2::EmitOp>([&](Operation *) {
            latency += static_cast<int>(spec_.getInstrLatency("emit") + avgPkt / 8.0);
          })
          .Case<ep2::AddOp>([&](Operation *) {
            latency += spec_.getInstrLatency("add");
          })
          .Case<ep2::SubOp>([&](Operation *) {
            latency += spec_.getInstrLatency("sub");
          });
    });
    return latency;
  }

  int getCommunicationCost(std::vector<std::string> &froms,
                           std::vector<std::string> &tos) override {
    if (froms.empty() || tos.empty())
      return 0;
    if (spec_.meIsland.empty())
      return 0;
    int total = 0, count = 0;
    for (auto &f : froms) {
      auto fi = spec_.meIsland.find(f);
      if (fi == spec_.meIsland.end()) {
        llvm::errs() << "[NetronomeSpec] island info missing for ME: " << f << "\n";
        continue;
      }
      for (auto &t : tos) {
        auto ti = spec_.meIsland.find(t);
        if (ti == spec_.meIsland.end()) {
          llvm::errs() << "[NetronomeSpec] island info missing for ME: " << t << "\n";
          continue;
        }
        total += (fi->second == ti->second) ? spec_.intraIslandCost
                                            : spec_.interIslandCost;
        count++;
      }
    }
    return (count == 0) ? 0 : total / count;
  }

  int getLatencyTarget() override {
    if (workload_.pps > 0) {
      int target = static_cast<int>(spec_.frequencyMhz * 1e6 / workload_.pps);
      return std::max(1, target);
    }
    return spec_.latencyTarget;
  }

  std::vector<std::string> getComputeUnits() override {
    return spec_.computeUnitIds;
  }

  llvm::DenseMap<mlir::Operation*, int> getTableMemMap(ep2::FuncOp funcOp) override {
    return buildTableMemMap(funcOp, workload_.hotKeyRatio);
  }

  // Memory layer accessors for future traffic-aware state placement
  int     getMemoryLatency(const std::string &layerId) {
    return spec_.getMemoryLatency(layerId);
  }
  int64_t getMemorySize(const std::string &layerId) {
    return spec_.getMemorySize(layerId);
  }
};
// Generic json model

// Loop based searching
static const int PIPELINE_EXTRA_SEARCH = 1;

// RAII guard to ensure that pipeline is mapped
class PipelineMapper {
  public:
    // the optimal result
    std::unique_ptr<PerformanceModel> performanceModel;
    HandlerPipeline functions;
    PerformanceModel::MappingResult mappingResult;

    // init the result to a large value
    PipelineMapper(std::unique_ptr<PerformanceModel> model)
        : performanceModel(std::move(model)),
          mappingResult{performanceModel->getLatencyTarget() + 1000} {}

     std::pair<bool, PerformanceModel::MappingResult>
     tryMap(HandlerPipeline &pipeline) {
        auto result = performanceModel->getMapping(pipeline);
        bool updated = false;
        // better if <
        if (result < mappingResult) {
            mappingResult = result;
            functions = pipeline;
            updated = true;
        }
        return {updated, mappingResult};
    }

    // remove intermidiate functions from it
    void finalize(ModuleOp containerOp) {
      OperatorRemoveGuard guard;

      containerOp.walk([&](ep2::FuncOp funcOp) {
        if (funcOp.isExtern() || !funcOp.isHandler())
          return;
        if (llvm::find(functions, funcOp) == functions.end())
          guard.add(funcOp);
      });

      // adding the instance attribute
      OpBuilder builder(containerOp);
      for (int i = 0; i < functions.size(); i++) {
        auto funcOp = functions[i];
        auto &units = mappingResult.unitMap[i];

        auto refVec = llvm::map_to_vector(units, [&](std::string &str) {
          return StringRef(str);
        });

        auto instances = builder.getStrArrayAttr(refVec);
        funcOp->setAttr("instances", instances);
      }
    }
};

// a searcher, decides the next pipeline to explore
class PipelineCutExplorer {
  public:
    virtual std::vector<HandlerPipeline> next(HandlerPipeline &pipeline, int bottleneckIndex) = 0;
};

class BottleneckExplorer : public PipelineCutExplorer {
  public:
    double avgPktBytes = 64.0;
    PerformanceModel* model = nullptr;

    BottleneckExplorer() = default;
    BottleneckExplorer(double avgPkt, PerformanceModel* m)
        : avgPktBytes(avgPkt), model(m) {}

    std::vector<HandlerPipeline> next(HandlerPipeline &pipeline, int bottleneckIndex) override {
        // first try table cut, if it is not working, try kcut
        auto tableMemMap = model ? model->getTableMemMap(pipeline[bottleneckIndex])
                                 : llvm::DenseMap<mlir::Operation*, int>{};
        auto [success, newFuncs] = tableCut(pipeline[bottleneckIndex], tableMemMap, avgPktBytes);
        if (success) {
            auto newPipeline = pipeline;

            auto eraseIt = newPipeline.begin() + bottleneckIndex;
            auto insertIt = newPipeline.erase(eraseIt);
            newPipeline.insert(insertIt, newFuncs.begin(), newFuncs.end());

            return {newPipeline};
        }

        // TODO(zhiyuang): add fallback policies
        return {};
    }
};

void optimizationLoop(FuncOp funcOp, PipelineMapper &mapper, PipelineCutExplorer &explorer);

} // namespace ep2
} // namespace mlir 





#endif // _MAPPING_H_