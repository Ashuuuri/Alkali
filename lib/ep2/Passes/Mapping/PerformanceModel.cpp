#include "ep2/passes/Mapping.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace ep2 {

// ---------------------------------------------------------------------------
// NetronomeSpec implementation
// ---------------------------------------------------------------------------

NetronomeSpec NetronomeSpec::defaults() {
  NetronomeSpec s;
  s.frequencyMhz    = 800;
  s.latencyTarget   = 100;
  s.intraIslandCost = 0;
  s.interIslandCost = 0;

  // Preserve the exact original 22-CU hardcoded list so the fallback path
  // produces identical results to the old NetronomePerformanceModel.
  static const char *kOriginalCUs[] = {
    "cu0",  "cu1",  "cu2",  "cu3",  "cu4",  "cu5",  "cu6",
    "cu9",  "cu10", "cu11", "cu12", "cu13", "cu14", "cu15",
    "cu16", "cu17", "cu18", "cu19", "cu20", "cu21", "cu22", "cu23",
  };
  for (const char *id : kOriginalCUs)
    s.computeUnitIds.push_back(id);

  // No island mapping in the original — all unknown, getCommunicationCost → 0
  s.instrLatency = {
    {"lookup",  100}, {"update",  100},
    {"add",       1}, {"sub",       1},
    {"emit",      1}, {"extract",   1},
    {"default",   1},
  };

  // Memory layers matching real Agilio CX hardware (used for future queries)
  s.memoryLayers = {
    {"LMEM", "per_me",    4096LL,         4},
    {"CLS",  "per_island", 65536LL,       16},
    {"CTM",  "per_island", 262144LL,      40},
    {"EMEM", "chip",       4294967296LL, 300},
  };

  return s;
}

NetronomeSpec NetronomeSpec::load(llvm::StringRef path) {
  if (path.empty()) {
    llvm::errs() << "[NetronomeSpec] No spec path given, using defaults.\n";
    return defaults();
  }

  auto bufOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufOrErr) {
    llvm::errs() << "[NetronomeSpec] Cannot open '" << path
                 << "', using defaults.\n";
    return defaults();
  }

  auto jsonOrErr = llvm::json::parse((*bufOrErr)->getBuffer());
  if (!jsonOrErr) {
    llvm::errs() << "[NetronomeSpec] JSON parse error in '" << path
                 << "': " << llvm::toString(jsonOrErr.takeError())
                 << ", using defaults.\n";
    return defaults();
  }

  const llvm::json::Object *root = jsonOrErr->getAsObject();
  if (!root) {
    llvm::errs() << "[NetronomeSpec] JSON root is not an object, using defaults.\n";
    return defaults();
  }

  // Missing keys retain the member-initializer defaults (matching the
  // original hardcoded values: 800 MHz, 100-cycle latency target, etc.).
  NetronomeSpec s;

  if (auto v = root->getInteger("frequency_mhz"))
    s.frequencyMhz = (int)*v;

  if (auto v = root->getInteger("latency_target"))
    s.latencyTarget = (int)*v;

  // compute_units: [{"id": "cu0", "island": 0}, ...]
  if (const llvm::json::Array *cus = root->getArray("compute_units")) {
    for (const auto &elem : *cus) {
      const llvm::json::Object *cu = elem.getAsObject();
      if (!cu) continue;
      auto id     = cu->getString("id");
      auto island = cu->getInteger("island");
      if (!id) continue;
      s.computeUnitIds.push_back(id->str());
      if (island)
        s.meIsland[id->str()] = (int)*island;
    }
  }

  // instruction_latencies: {"lookup": 100, ...}
  if (const llvm::json::Object *il = root->getObject("instruction_latencies")) {
    for (const auto &kv : *il) {
      if (auto v = kv.second.getAsInteger())
        s.instrLatency[kv.first.str()] = (int)*v;
    }
  }

  // communication_cost: {"intra_island": 20, "inter_island": 100}
  if (const llvm::json::Object *cc = root->getObject("communication_cost")) {
    if (auto v = cc->getInteger("intra_island"))
      s.intraIslandCost = (int)*v;
    if (auto v = cc->getInteger("inter_island"))
      s.interIslandCost = (int)*v;
  }

  // memory_layers: [{"id": "LMEM", "scope": "per_me", "size_bytes": 4096,
  //                  "latency_cycles": 4}, ...]
  if (const llvm::json::Array *layers = root->getArray("memory_layers")) {
    for (const auto &elem : *layers) {
      const llvm::json::Object *ml = elem.getAsObject();
      if (!ml) continue;
      MemLayerSpec layer;
      if (auto v = ml->getString("id"))     layer.id    = v->str();
      if (auto v = ml->getString("scope"))  layer.scope = v->str();
      if (auto v = ml->getInteger("size_bytes"))     layer.sizeBytes    = *v;
      if (auto v = ml->getInteger("latency_cycles")) layer.latencyCycles = (int)*v;
      s.memoryLayers.push_back(std::move(layer));
    }
  }

  llvm::errs() << "[NetronomeSpec] Loaded '" << path << "': "
               << s.computeUnitIds.size() << " MEs, "
               << s.memoryLayers.size() << " memory layers.\n";
  return s;
}

int NetronomeSpec::getInstrLatency(const std::string &opName) const {
  auto it = instrLatency.find(opName);
  if (it != instrLatency.end())
    return it->second;
  auto def = instrLatency.find("default");
  return (def != instrLatency.end()) ? def->second : 1;
}

int NetronomeSpec::getMemoryLatency(const std::string &layerId) const {
  for (const auto &ml : memoryLayers)
    if (ml.id == layerId)
      return ml.latencyCycles;
  return 0;
}

int64_t NetronomeSpec::getMemorySize(const std::string &layerId) const {
  for (const auto &ml : memoryLayers)
    if (ml.id == layerId)
      return ml.sizeBytes;
  return 0;
}

// ---------------------------------------------------------------------------

namespace {

int getLatencyForUnit(PerformanceModel &model, int idx,
                      llvm::SmallVector<ep2::FuncOp> &ops,
                      PerformanceModel::UnitMap &unitMap) {
  auto func = ops[idx];
  auto commLatency = 0;
  if (idx < ops.size() - 1) {
    commLatency = model.getCommunicationCost(unitMap[idx], unitMap[idx + 1]);
  }
  auto totalLatency = model.getLatency(func) + commLatency;
  return totalLatency;
}

} // namespace

// Simple greedy default mapping
// This function provides a simple, greedy mapping method
PerformanceModel::MappingResult
PerformanceModel::getMapping(llvm::SmallVector<ep2::FuncOp> &ops) {
  // try to assign compute units one by one, and make sure each get enough latency
  auto latencyTarget = getLatencyTarget();
  auto units = getComputeUnits();
  PerformanceModel::UnitMap unitMap;

  for (int idx = 0; idx < ops.size(); idx++)
    unitMap[idx] = {};

  for (auto unit : units) {
    bool assigned = false;
    for (int idx = 0; idx < ops.size(); idx++) {
      auto totalLatency = getLatencyForUnit(*this, idx, ops, unitMap);

      // we need to add more units
      if (totalLatency > latencyTarget * unitMap[idx].size()) {

        // constraint on the number of units
        // TODO(zhiyuang): if its not tbale clean, we cannot replicate. check this.
        if (!isTableClean(ops[idx]) && unitMap[idx].size() >= 1)
          continue;

        unitMap[idx].push_back(unit);

        assigned = true;
        break;
      }
    }


    // if we do not need more unit, we can stop
    if (!assigned)
      break;
  }

  int targetIndex = 0, maxLatency = 0;
  for (int idx = 0; idx < ops.size(); idx++) {
    auto totalLatency = getLatencyForUnit(*this, idx, ops, unitMap) / unitMap[idx].size();
    if (totalLatency > maxLatency) {
      maxLatency = totalLatency;
      targetIndex = idx;
    }
  }

  // we run out of units ...
  return {maxLatency, targetIndex, unitMap};
}

// NetroNome is a simple performance model that assumes a core execution

// int SimpleNetronomePerformanceModel::getAccessOverhead() {
//   // get the diameter of the global
//   return 1;
// }
//
// int SimpleNetronomePerformanceModel::getLatency(ep2::FuncOp funcOp) {
//   // get the diameter of the function
//
//   for (auto &block : funcOp.getBlocks()) {
//     int cycles = 0;
//     for (auto &op : block) {
//       cycles += TypeSwitch<Operation *, int>(&op)
//         // zero width ops
//         .Case<ep2::InitOp, ep2::TerminateOp, ep2::ExtractOp>([](auto op) { return 0; })
//         .Case<ep2::EmitOp>([](auto op) { return 1; })
//         .Default([](auto op) { return 1; });
//         ;
//     }
//     return cycles;
//   }
// }

} // namespace ep2
} // namespace mlir