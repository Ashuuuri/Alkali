#ifndef EP2_MAPPING_H
#define EP2_MAPPING_H

#include "ep2/dialect/Dialect.h"
#include "ep2/Utilities.h"

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
std::pair<bool, SmallVector<ep2::FuncOp>> tableCut(ep2::FuncOp targetFunc);
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
  virtual int getLatency(ep2::FuncOp funcOp) = 0;
  virtual int getCommunicationCost(std::vector<std::string> &froms,
                                   std::vector<std::string> &tos) = 0;
  virtual int getLatencyTarget() = 0;
  virtual std::vector<std::string> getComputeUnits() = 0;


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
 public:
  // Default constructor: preserves original hardcoded behaviour
  NetronomePerformanceModel() : spec_(NetronomeSpec::defaults()) {}
  // Spec-file constructor: loads JSON, falls back to defaults on error
  explicit NetronomePerformanceModel(llvm::StringRef specPath)
      : spec_(NetronomeSpec::load(specPath)) {}

  int getAccessOverhead(ep2::GlobalOp globalOp) override { return 0; }

  int getLatency(ep2::FuncOp funcOp) override {
    int latency = 0;
    funcOp.walk([&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op)
          .Case<ep2::LookupOp>(
              [&](Operation *) { latency += spec_.getInstrLatency("lookup"); })
          .Case<ep2::UpdateOp>(
              [&](Operation *) { latency += spec_.getInstrLatency("update"); })
          .Case<ep2::AddOp>(
              [&](Operation *) { latency += spec_.getInstrLatency("add"); })
          .Case<ep2::SubOp>(
              [&](Operation *) { latency += spec_.getInstrLatency("sub"); })
          .Case<ep2::EmitOp>(
              [&](Operation *) { latency += spec_.getInstrLatency("emit"); })
          .Case<ep2::ExtractOp>(
              [&](Operation *) { latency += spec_.getInstrLatency("extract"); });
    });
    return latency;
  }

  int getCommunicationCost(std::vector<std::string> &froms,
                           std::vector<std::string> &tos) override {
    if (froms.empty() || tos.empty())
      return 0;
    auto fi = spec_.meIsland.find(froms[0]);
    auto ti = spec_.meIsland.find(tos[0]);
    if (fi == spec_.meIsland.end() || ti == spec_.meIsland.end())
      return 0;
    return (fi->second == ti->second) ? spec_.intraIslandCost
                                      : spec_.interIslandCost;
  }

  int getLatencyTarget() override { return spec_.latencyTarget; }

  std::vector<std::string> getComputeUnits() override {
    return spec_.computeUnitIds;
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
static const int PIPELINE_EXTRA_SEARCH = 10;

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
    std::vector<HandlerPipeline> next(HandlerPipeline &pipeline, int bottleneckIndex) override {
        // first try table cut, if it is not working, try kcut
        auto [success, newFuncs] = tableCut(pipeline[bottleneckIndex]);
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