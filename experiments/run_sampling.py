#!/usr/bin/env python3
"""
Automated sampling runner for Alkali/Netronome mapping experiments.

Usage:
    python3 experiments/run_sampling.py \
        --bench nfchain \
        --workloads default,8mpps,20mpps,40mpps \
        --runs 10

    python3 experiments/run_sampling.py \
        --bench nfchain,transport_rx,l3fwd \
        --workloads default,20mpps,40mpps \
        --runs 5
"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "results")
ALKALIC      = os.path.join(REPO_ROOT, "scripts", "alkalic")
WORKLOAD_DIR = os.path.join(REPO_ROOT, "tests", "specs")

WORKLOAD_TARGET = {
    "default": 100,
    "8mpps":   100,
    "20mpps":   40,
    "40mpps":   20,
    "20mpps_64B":   40,
    "20mpps_256B":  40,
    "20mpps_1500B": 40,
    "20mpps_hot50": 40,
    "20mpps_hot90": 40,
    "20mpps_4flows":  40,
    "20mpps_16flows": 40,
    "20mpps_64flows": 40,
}

ME_FREQ = 800_000_000  # 800 MHz

WORKLOAD_TARGET_PPS = {
    "default":        0,
    "8mpps":          8e6,
    "20mpps":        20e6,
    "40mpps":        40e6,
    "20mpps_64B":    20e6,
    "20mpps_256B":   20e6,
    "20mpps_1500B":  20e6,
    "20mpps_hot50":  20e6,
    "20mpps_hot90":  20e6,
    "20mpps_4flows": 20e6,
    "20mpps_16flows":20e6,
    "20mpps_64flows":20e6,
}

CSV_FIELDS = ["benchmark", "workload", "run", "latency", "stages", "cu_total",
              "distribution", "est_pps", "meets_target"]

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_final_mapping(output: str):
    """
    From a run's combined output, extract the LAST mapping result:
    returns (latency, stages, cu_total, distribution_str)
    or raises ValueError on failure.

    The solver prints repeated intermediate results; we want the last one.
    """
    lines = output.splitlines()

    last_lat_idx = -1
    last_lat = None
    for i, line in enumerate(lines):
        m = re.search(r"Result Latency:\s*(\d+)", line)
        if m:
            last_lat_idx = i
            last_lat = int(m.group(1))

    if last_lat_idx < 0:
        raise ValueError("No 'Result Latency' found in output")

    # Collect Function lines starting just after the latency line.
    # Skip "Result Mapping:" header and blank lines; stop at the first
    # non-blank non-Function non-header line.
    funcs = []
    for line in lines[last_lat_idx + 1:]:
        m = re.match(r"^Function:\s+\S+\s+Unit:\s*(.*)\s*$", line)
        if m:
            units = m.group(1).split()
            funcs.append(len(units))
        elif re.match(r"^Result Mapping:", line) or not line.strip():
            continue   # skip "Result Mapping:" header and blank lines
        else:
            break      # any other non-empty line ends the group

    if not funcs:
        raise ValueError("No 'Function ... Unit:' lines after last Result Latency")

    stages    = len(funcs)
    cu_total  = sum(funcs)
    dist      = "+".join(str(c) for c in funcs)
    return last_lat, stages, cu_total, dist


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(bench: str, workload: str, run_idx: int) -> dict:
    """Run one alkalic invocation and return a result dict."""
    # Use repo-relative path: run_c.sh prepends "./" so absolute paths break it.
    bench_rel = f"tests/experiments_c/{bench}.c"
    env = dict(os.environ)
    if workload != "default":
        wpath = os.path.join(WORKLOAD_DIR, f"workload_{workload}.json")
        if not os.path.isfile(wpath):
            return _error_row(bench, workload, run_idx, f"workload file not found: {wpath}")
        env["WORKLOAD_SPEC"] = wpath
    else:
        env.pop("WORKLOAD_SPEC", None)

    try:
        proc = subprocess.run(
            [ALKALIC, bench_rel, "--target", "netronome"],
            capture_output=True, text=True, env=env,
            cwd=REPO_ROOT,
        )
        output = proc.stdout + proc.stderr
        # Note: non-zero exit is tolerated — the emit step may fail after
        # mapping completes. We still attempt to parse the mapping result.
        lat, stages, cu_total, dist = parse_final_mapping(output)
        est_pps    = ME_FREQ / lat if lat > 0 else 0
        target_pps = WORKLOAD_TARGET_PPS.get(workload, 0)
        if target_pps > 0:
            meets = "yes" if est_pps >= target_pps else "no"
        else:
            meets = "-"
        return {
            "benchmark":    bench,
            "workload":     workload,
            "run":          run_idx,
            "latency":      lat,
            "stages":       stages,
            "cu_total":     cu_total,
            "distribution": dist,
            "est_pps":      f"{est_pps/1e6:.1f}M",
            "meets_target": meets,
        }
    except ValueError as exc:
        return _error_row(bench, workload, run_idx, str(exc))
    except Exception as exc:
        return _error_row(bench, workload, run_idx, f"unexpected: {exc}")


def _error_row(bench, workload, run_idx, reason):
    print(f"  [ERROR] {bench}/{workload} run {run_idx}: {reason}", file=sys.stderr)
    return {
        "benchmark":    bench,
        "workload":     workload,
        "run":          run_idx,
        "latency":      "ERROR",
        "stages":       "ERROR",
        "cu_total":     "ERROR",
        "distribution": reason,
        "est_pps":      "ERROR",
        "meets_target": "ERROR",
    }


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], bench: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{bench}_sampling.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _stats(values):
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return "N/A", "N/A", "N/A"
    return min(nums), max(nums), round(statistics.median(nums), 1)


def write_markdown(rows: list[dict], bench: str, workloads: list[str]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{bench}_sampling.md")
    lines = [f"# {bench} sampling results\n"]

    # Per-workload tables
    for wload in workloads:
        target = WORKLOAD_TARGET.get(wload, "?")
        wrows  = [r for r in rows if r["workload"] == wload]
        lines.append(f"## {wload}  (target={target})\n")
        lines.append("| Run | Latency | Stages | CU total | Distribution | Est. pps | Meets? |")
        lines.append("|-----|---------|--------|----------|--------------|----------|--------|")
        for r in wrows:
            lines.append(
                f"| {r['run']} | {r['latency']} | {r['stages']} "
                f"| {r['cu_total']} | {r['distribution']} "
                f"| {r['est_pps']} | {r['meets_target']} |"
            )
        lats  = [r["latency"]  for r in wrows]
        cus   = [r["cu_total"] for r in wrows]
        lmin, lmax, lmed = _stats(lats)
        cmin, cmax, cmed = _stats(cus)
        lines.append(f"\n**Summary:** latency min={lmin} max={lmax} median={lmed}; "
                     f"CU min={cmin} max={cmax} median={cmed}\n")

    # Cross-workload summary table
    lines.append("## Cross-workload summary\n")
    lines.append("| Workload | target | Latency (median) | CU (median) | Stages (median) |")
    lines.append("|----------|--------|-----------------|-------------|-----------------|")
    for wload in workloads:
        target = WORKLOAD_TARGET.get(wload, "?")
        wrows  = [r for r in rows if r["workload"] == wload]
        _, _, lmed = _stats([r["latency"]  for r in wrows])
        _, _, cmed = _stats([r["cu_total"] for r in wrows])
        _, _, smed = _stats([r["stages"]   for r in wrows])
        lines.append(f"| {wload} | {target} | {lmed} | {cmed} | {smed} |")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def _est_pps_str(lat_med):
    """Convert median latency (number or 'N/A') to est pps string."""
    if not isinstance(lat_med, (int, float)):
        return "N/A"
    pps = ME_FREQ / lat_med if lat_med > 0 else 0
    return f"{pps/1e6:.1f}M"


def print_summary(all_rows: list[dict], benches: list[str], workloads: list[str]):
    """Print cross-benchmark summary table to stdout."""
    col_w = [12, 16, 7, 14, 11, 7, 10, 7]
    header = ["Benchmark", "Workload", "target", "Latency (med)", "CU (med)",
              "Stages", "Est. pps", "Meets?"]
    sep    = "  ".join("-" * w for w in col_w)
    row_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)

    print()
    print("=" * (sum(col_w) + 2 * (len(col_w) - 1)))
    print("SAMPLING SUMMARY")
    print("=" * (sum(col_w) + 2 * (len(col_w) - 1)))
    print(row_fmt.format(*header))
    print(sep)
    for bench in benches:
        for wload in workloads:
            target = WORKLOAD_TARGET.get(wload, "?")
            wrows  = [r for r in all_rows
                      if r["benchmark"] == bench and r["workload"] == wload]
            _, _, lmed = _stats([r["latency"]  for r in wrows])
            _, _, cmed = _stats([r["cu_total"] for r in wrows])
            _, _, smed = _stats([r["stages"]   for r in wrows])
            est        = _est_pps_str(lmed)
            tpps       = WORKLOAD_TARGET_PPS.get(wload, 0)
            if tpps > 0 and isinstance(lmed, (int, float)) and lmed > 0:
                meets = "yes" if ME_FREQ / lmed >= tpps else "no"
            else:
                meets = "-"
            print(row_fmt.format(bench, wload, str(target),
                                 str(lmed), str(cmed), str(smed), est, meets))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Alkali/Netronome sampling matrix and produce CSV + Markdown reports."
    )
    parser.add_argument(
        "--bench", required=True,
        help="Comma-separated benchmark names, e.g. nfchain,transport_rx,l3fwd"
    )
    parser.add_argument(
        "--workloads", default="default,8mpps,20mpps,40mpps",
        help="Comma-separated workload names (default: default,8mpps,20mpps,40mpps)"
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of runs per (benchmark, workload) combination (default: 10)"
    )
    args = parser.parse_args()

    benches   = [b.strip() for b in args.bench.split(",") if b.strip()]
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    n_runs    = args.runs

    total = len(benches) * len(workloads) * n_runs
    print(f"Running {len(benches)} bench × {len(workloads)} workloads × {n_runs} runs = {total} total")

    all_rows = []
    done = 0
    for bench in benches:
        bench_rows = []
        for wload in workloads:
            for i in range(1, n_runs + 1):
                done += 1
                print(f"  [{done:3d}/{total}] {bench:12s} {wload:8s} run {i}", end="", flush=True)
                row = run_one(bench, wload, i)
                bench_rows.append(row)
                all_rows.append(row)
                lat  = row["latency"]
                cu   = row["cu_total"]
                epps = row["est_pps"]
                mt   = row["meets_target"]
                print(f"  → latency={lat}  CU={cu}  est_pps={epps}  meets={mt}")

        csv_path = write_csv(bench_rows, bench)
        md_path  = write_markdown(bench_rows, bench, workloads)
        print(f"  Wrote {csv_path}")
        print(f"  Wrote {md_path}")

    print_summary(all_rows, benches, workloads)


if __name__ == "__main__":
    main()
