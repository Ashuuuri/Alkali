#ifndef EP2_WORKLOADSPEC_H
#define EP2_WORKLOADSPEC_H

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace mlir {
namespace ep2 {

struct WorkloadSpec {
  double pps = 0;           // target packets per second (0 = not specified)
  double avgPktBytes = 64;  // average packet size in bytes
  int activeFlows = 1024;   // number of active flows
  double hotKeyRatio = 0.0; // fraction of lookups hitting fast memory

  static WorkloadSpec load(llvm::StringRef path);
  static WorkloadSpec none();  // returns default (pps=0, no workload info)
};

} // namespace ep2
} // namespace mlir

#endif // EP2_WORKLOADSPEC_H
