# Experiment: Netronome Greedy Mapping Sampling Matrix
**Date:** 2026-04-05  
**Branch:** feature/netronome-spec-externalize @ ec3a40d  
**Config:** PIPELINE_EXTRA_SEARCH=10, getCommunicationCost wired up  
**Spec:** tests/specs/netronome.json (48 MEs, 4 islands, intra=20, inter=100)

---

## §1 Setup

Two configurations per benchmark:
- **no-spec**: `ep2-pipeline-handler="mode=loop target=netronome"` (defaults fallback, 22 CUs, empty meIsland → getCommunicationCost returns 0)
- **with-spec**: `ep2-pipeline-handler="mode=loop target=netronome spec=tests/specs/netronome.json"` (48 MEs, island costs active)

Runs bypass `netronome_compile.sh` and call `ep2c-opt` directly on pre-computed `commonopt.mlir` to avoid `ninja -j32` rebuilds. Same binary used for all runs.

---

## §2 Reference: nfchain.c

*Previously collected on same branch.*

### no-spec (22 CUs) — 5 rounds (PIPELINE_EXTRA_SEARCH=10)

| Run | Latency | Stages | CU total | Per-stage |
|-----|---------|--------|----------|-----------|
| 1   | 76      | 4      | 14       | 3+3+3+5   |
| 2   | 76      | 4      | 14       | 3+3+5+3   |
| 3   | 82      | 4      | 13       | 3+3+3+4   |
| 4   | **71**  | **5**  | 15       | 3+3+3+3+3 |
| 5   | 81      | 4      | 13       | 3+3+4+3   |

Summary: latency {71, 76, 76, 81, 82}, CU range 13–15. Run 4 found a 5-stage pipeline — PIPELINE_EXTRA_SEARCH=10 enabled deeper exploration vs the old value of 1 (best was 76).

### with-spec (48 MEs, JSON active) — 5 rounds

| Run | Latency | Stages | CU total | Per-stage |
|-----|---------|--------|----------|-----------|
| 1   | 85      | 4      | 13       | 5+3+3+2   |
| 2   | 81      | 4      | 13       | 4+3+3+3   |
| 3   | 86      | 4      | 13       | 5+3+3+2   |
| 4   | 82      | 4      | 13       | 3+3+2+5   |
| 5   | 81      | 4      | 13       | 4+3+3+3   |

Summary: latency {81, 81, 82, 85, 86}, CU always **13**. Latency is higher than no-spec because getCommunicationCost (intra-island=20 cycles) is now added to effective stage latency. CU usage does not increase beyond 13 — greedy stops when all stages satisfy latencyTarget=100.

**Key finding:** with-spec uses cu7, cu8 (not in the defaults 22-CU list), confirming the 48-ME JSON is active. But CU pool is still underutilised — 35 of 48 MEs never assigned.

---

## §3 Benchmark status after fixing C source issues

Four additional benchmarks were fixed to compile through the Netronome pipeline:

| Benchmark | Fix applied |
|-----------|------------|
| transport_rx.c | Replaced `struct buf_tag packet_out` + `bufemit(&packet_out, packet)` with reusing input `packet` directly |
| firewall.c | Removed `if/else` branches (ep2-lift-llvm doesn't support SCF); always does lookup |
| load_balancer.c | Fixed XOR of 16-bit struct fields (struct JSON type mismatch); removed `if` branch; removed dead backend-init code |
| l3fwd.c | `static inline` not inlined at -O0 → second ep2.func → assertion; inlined body manually |

Two benchmarks remain unviable:
- **l2_echo.c**: latency=0 (no LookupOp/EmitOp), 0 CUs assigned, degenerate
- **bridge.c**: same as l2_echo

---

## §4 Extended sampling matrix

### §4.1 transport_rx.c

**no-spec (22 CUs) — 10 rounds**

| Run | Latency | Stages | CU total | Per-stage |
|-----|---------|--------|----------|-----------|
| 1–10 | 69 | 1 | 3 | 3 |

**with-spec (48 MEs) — 10 rounds**

| Run | Latency | Stages | CU total | Per-stage |
|-----|---------|--------|----------|-----------|
| 1–10 | 69 | 1 | 3 | 3 |

Summary: fully deterministic. Handler latency=207 cycles (1 LookupOp + 1 UpdateOp + emit/extract ops), latencyTarget=100. Greedy allocates ceil(207/100)=3 CUs. Table cut always fails (Valid:0) — handler is too monolithic for tableCut to split. No randomness.

no-spec vs with-spec identical: with-spec adds intra-island comm cost but since there's only 1 stage, getCommunicationCost is never called (no adjacent stages).

---

### §4.2 firewall.c

**no-spec (22 CUs) — 10 rounds**

| Run | Latency | Stages | CU total | Per-stage |
|-----|---------|--------|----------|-----------|
| 1–10 | 100 | 1 | 1 | 1 |

**with-spec (48 MEs) — 10 rounds**

| Run | Latency | Stages | CU total | Per-stage |
|-----|---------|--------|----------|-----------|
| 1–10 | 100 | 1 | 1 | 1 |

Summary: fully deterministic. Handler latency=100 cycles (exactly 1 LookupOp). Greedy allocates 1 CU (100 ≤ 100×1). Table cut never succeeds. No randomness. With-spec identical to no-spec (single stage, no comm cost path).

---

### §4.3 load_balancer.c

**no-spec (22 CUs) — 10 rounds**

| Run | Latency | Stages | CU total | Per-stage |
|-----|---------|--------|----------|-----------|
| 1   | 100     | 2      | 5        | 3+2       |
| 2   | 100     | 2      | 5        | 2+3       |
| 3   | 100     | 2      | 5        | 3+2       |
| 4   | 100     | 2      | 5        | 2+3       |
| 5   | 100     | 2      | 5        | 2+3       |
| 6   | 100     | 2      | 5        | 3+2       |
| 7   | 100     | 2      | 5        | 2+3       |
| 8   | 100     | 2      | 5        | 2+3       |
| 9   | 100     | 2      | 5        | 3+2       |
| 10  | 100     | 2      | 5        | 3+2       |

Summary: latency always 100, stages always 2, CU total always **5**. Per-stage distribution randomises between 3+2 and 2+3 (min-cut assigns the bottleneck stage differently). No latency variation.

**with-spec (48 MEs) — 10 rounds**

| Run | Latency | Stages | CU total | Per-stage |
|-----|---------|--------|----------|-----------|
| 1   | 100     | 2      | 6        | 4+2       |
| 2   | 100     | 2      | 6        | 4+2       |
| 3   | 100     | 2      | 6        | 3+3       |
| 4   | 100     | 2      | 6        | 4+2       |
| 5   | 100     | 2      | 6        | 4+2       |
| 6   | 100     | 2      | 6        | 4+2       |
| 7   | 100     | 2      | 6        | 3+3       |
| 8   | 100     | 2      | 6        | 4+2       |
| 9   | 100     | 2      | 6        | 4+2       |
| 10  | 100     | 2      | 6        | 4+2       |

Summary: latency always 100, stages always 2, CU total always **6** (vs 5 no-spec). with-spec allocates 1 extra CU to compensate for intra-island comm cost (20 cycles) added to the bottleneck stage's effective latency. Distribution randomises between 4+2 and 3+3.

**Notable:** This is the only benchmark where with-spec CU total > no-spec CU total (6 vs 5). The communication cost is actively changing the allocation decision.

---

### §4.4 l3fwd.c

**no-spec (22 CUs) — 10 rounds**

| Run | Latency | Stages | CU total | Per-stage |
|-----|---------|--------|----------|-----------|
| 1   | 713     | 3      | 7        | 2+2+3     |
| 2   | **100** | **4**  | **11**   | 2+3+3+3   |
| 3   | 713     | 3      | 7        | 2+2+3     |
| 4   | **816** | **1**  | **2**    | 2         |
| 5   | 713     | 3      | 7        | 2+2+3     |
| 6   | 713     | 3      | 7        | 2+3+2     |
| 7   | 713     | 3      | 7        | 2+3+2     |
| 8   | 713     | 3      | 7        | 2+3+2     |
| 9   | **100** | **4**  | **11**   | 4+2+3+2   |
| 10  | 713     | 3      | 7        | 2+3+2     |

Summary: latency {100, 100, 713×8, 816}, CU range 2–11, stage count 1–4. High variance. The dominant outcome (8/10) is 3-stage latency=713 (still above latencyTarget=100 — greedy runs out of CUs). Occasionally (2/10) finds a 4-stage solution reaching latency=100. Once (1/10) fails to cut at all (1-stage, latency=816).

**with-spec (48 MEs) — 10 rounds**

| Run | Latency | Stages | CU total | Per-stage |
|-----|---------|--------|----------|-----------|
| 1   | 733     | 3      | 7        | 2+2+3     |
| 2   | 713     | 3      | 7        | 2+3+2     |
| 3   | **816** | **1**  | **2**    | 2         |
| 4   | 713     | 3      | 7        | 2+3+2     |
| 5   | 733     | 3      | 7        | 2+2+3     |
| 6   | 713     | 3      | 7        | 2+3+2     |
| 7   | 713     | 3      | 7        | 2+3+2     |
| 8   | 713     | 3      | 7        | 2+3+2     |
| 9   | 733     | 3      | 7        | 2+2+3     |
| 10  | 713     | 3      | 7        | 2+3+2     |

Summary: latency {713×7, 733×3, 816×1} (note: no run reached 100). CU total always **7** (same as no-spec's dominant outcome). with-spec never found the 4-stage solution that no-spec found twice (runs 2, 9). The extra comm cost makes the 4-stage pipeline appear worse to the greedy evaluator, so it is never selected as the best.

**Notable:** with-spec *hurts* l3fwd — with-spec never reaches latency=100 while no-spec does (20% of runs). The 48-ME pool makes no difference since CU saturation point is 7 regardless.

---

## §5 Cross-benchmark comparison

| Benchmark    | Config     | Latency set        | CU total | Typical stages |
|--------------|------------|--------------------|----------|----------------|
| nfchain      | no-spec    | {71,76,76,81,82}   | 13–15    | 4 (once 5)     |
| nfchain      | with-spec  | {81,81,82,85,86}   | 13       | 4              |
| transport_rx | no-spec    | {69}               | 3        | 1              |
| transport_rx | with-spec  | {69}               | 3        | 1              |
| firewall     | no-spec    | {100}              | 1        | 1              |
| firewall     | with-spec  | {100}              | 1        | 1              |
| load_balancer| no-spec    | {100}              | 5        | 2              |
| load_balancer| with-spec  | {100}              | **6**    | 2              |
| l3fwd        | no-spec    | {100,713,816}      | 2–11     | 1–4            |
| l3fwd        | with-spec  | {713,733,816}      | 2–7      | 1–3            |

---

## §6 Observations and analysis

### Q1: Does with-spec CU total increase compared to no-spec?

Only in **load_balancer**: CU 5→6. For every other benchmark, CU total is identical (transport_rx: 3=3, firewall: 1=1, nfchain: both ~13, l3fwd: both 7 in dominant case). The 48-ME pool is never the binding constraint.

### Q2: Is 13 CU a nfchain-specific saturation point?

Yes and no. 13 is specific to nfchain's handler latency + latencyTarget=100 arithmetic. The general pattern holds across all benchmarks: CU saturation = ceil(stage_latency / latencyTarget) summed across stages. Different benchmarks saturate at different points (1 for firewall, 3 for transport_rx, 5–6 for load_balancer, 7–11 for l3fwd). The underlying mechanism is universal.

### Q3: Does with-spec ever utilise MEs beyond what no-spec would?

Only load_balancer (by 1 CU). The comm cost (intra=20 cycles) effectively raises the latency threshold of the bottleneck stage, requiring 1 extra replica. For all other benchmarks, either: (a) handler latency is small enough that comm cost doesn't shift the ceiling (transport_rx, firewall), or (b) the handler is so heavy that even adding comm cost doesn't change which CU count satisfies the threshold (l3fwd).

### Q4: Does with-spec improve or hurt quality?

- **nfchain**: with-spec latency is higher (81–86 vs 71–82) — because the reported metric now includes comm cost overhead; the mapping itself isn't worse, but the metric changed
- **load_balancer**: identical latency (100), 1 more CU — neutral to slightly worse (uses more resources for same latency)
- **l3fwd**: with-spec **never** reaches latency=100, while no-spec does 20% of the time — with-spec hurts l3fwd
- **transport_rx / firewall**: no effect (single stage, comm cost never invoked)

### Q5: Root cause diagnosis

The 48-ME pool is systematically underutilised across all benchmarks. The binding constraint is **latencyTarget=100 (hardcoded)**. Greedy stops allocating CUs as soon as every stage satisfies `stage_latency ≤ latencyTarget × num_replicas`. With typical handler latencies of 69–816 cycles, this is satisfied with 1–11 CUs — far below 48. Making the mapping workload-aware requires either:
1. Making latencyTarget dynamic (derived from target packet rate and pipeline depth), or
2. Changing the greedy objective from "meet latency target" to "maximise throughput given a fixed CU budget"
