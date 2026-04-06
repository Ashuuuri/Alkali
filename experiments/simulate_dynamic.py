#!/usr/bin/env python3
"""
Dynamic recompilation simulation for Alkali/Netronome mapping.

Simulates a day's traffic pattern: for each scenario the compiler
re-optimises the mapping; a static baseline (default workload, fixed
once) is shown for comparison.

Usage:
    python3 experiments/simulate_dynamic.py
Output:
    experiments/dynamic_recompilation.png
    summary table printed to stdout
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
ALKALIC     = os.path.join(REPO_ROOT, "scripts", "alkalic")
BENCH_REL   = "tests/experiments_c/nfchain.c"
OUT_PNG     = os.path.join(SCRIPT_DIR, "dynamic_recompilation.png")

# ---------------------------------------------------------------------------
# Workload time-series
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "label":         "t=0\nmorning\nlow traffic",
        "pps":           8_000_000,
        "avg_pkt_bytes": 64,
        "active_flows":  1024,
        "hot_key_ratio": 0.0,
    },
    {
        "label":         "t=1\nramp up",
        "pps":           20_000_000,
        "avg_pkt_bytes": 64,
        "active_flows":  1024,
        "hot_key_ratio": 0.0,
    },
    {
        "label":         "t=2\npeak\nlarge packets",
        "pps":           20_000_000,
        "avg_pkt_bytes": 1500,
        "active_flows":  4096,
        "hot_key_ratio": 0.0,
    },
    {
        "label":         "t=3\nhot-key\nburst",
        "pps":           40_000_000,
        "avg_pkt_bytes": 64,
        "active_flows":  8192,
        "hot_key_ratio": 0.9,
    },
    {
        "label":         "t=4\nfew flows",
        "pps":           20_000_000,
        "avg_pkt_bytes": 64,
        "active_flows":  4,
        "hot_key_ratio": 0.0,
    },
    {
        "label":         "t=5\nnight\nlow traffic",
        "pps":           8_000_000,
        "avg_pkt_bytes": 64,
        "active_flows":  1024,
        "hot_key_ratio": 0.0,
    },
]

# ---------------------------------------------------------------------------
# Parsing (mirrors run_sampling.parse_final_mapping)
# ---------------------------------------------------------------------------

def parse_final_mapping(output: str):
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

    funcs = []
    for line in lines[last_lat_idx + 1:]:
        m = re.match(r"^Function:\s+\S+\s+Unit:\s*(.*)\s*$", line)
        if m:
            units = m.group(1).split()
            funcs.append(len(units))
        elif re.match(r"^Result Mapping:", line) or not line.strip():
            continue
        else:
            break

    if not funcs:
        raise ValueError("No 'Function ... Unit:' lines after last Result Latency")

    stages   = len(funcs)
    cu_total = sum(funcs)
    dist     = "+".join(str(c) for c in funcs)
    return last_lat, stages, cu_total, dist

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_scenario(workload_dict: dict | None) -> dict:
    """
    Run alkalic for nfchain with the given workload dict (or no workload
    if workload_dict is None).  Returns a result dict with latency/stages/
    cu_total/distribution, or raises on failure.
    """
    env = dict(os.environ)
    env.pop("WORKLOAD_SPEC", None)

    if workload_dict is not None:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="alkali_wl_"
        )
        json.dump(workload_dict, tmp)
        tmp.close()
        env["WORKLOAD_SPEC"] = tmp.name
    else:
        tmp = None

    try:
        proc = subprocess.run(
            [ALKALIC, BENCH_REL, "--target", "netronome"],
            capture_output=True, text=True, env=env, cwd=REPO_ROOT,
        )
        output = proc.stdout + proc.stderr
        lat, stages, cu_total, dist = parse_final_mapping(output)
        return {"latency": lat, "stages": stages, "cu_total": cu_total,
                "distribution": dist}
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

# ---------------------------------------------------------------------------
# Workload summary string for plot annotation
# ---------------------------------------------------------------------------

def wl_summary(s: dict) -> str:
    pps_m = s["pps"] / 1e6
    parts = [f"{pps_m:.0f}M"]
    if s["avg_pkt_bytes"] != 64:
        parts.append(f"{s['avg_pkt_bytes']}B")
    if s["hot_key_ratio"] > 0:
        parts.append(f"hot={s['hot_key_ratio']}")
    if s["active_flows"] <= 4:
        parts.append(f"{s['active_flows']}flows")
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n = len(SCENARIOS)
    x = list(range(n))
    labels = [s["label"] for s in SCENARIOS]

    # --- Run static baseline (no WORKLOAD_SPEC) ---
    print("Running static baseline ...", flush=True)
    static = run_scenario(None)
    print(f"  static → latency={static['latency']}  CU={static['cu_total']}")

    # --- Run each dynamic scenario ---
    dynamic_results = []
    for i, sc in enumerate(SCENARIOS):
        wl = {k: sc[k] for k in ("pps", "avg_pkt_bytes", "active_flows", "hot_key_ratio")}
        print(f"Running scenario {i} ({sc['label'].replace(chr(10),' ')}) ...", flush=True)
        res = run_scenario(wl)
        dynamic_results.append(res)
        print(f"  → latency={res['latency']}  CU={res['cu_total']}  dist={res['distribution']}")

    dyn_cu  = [r["cu_total"] for r in dynamic_results]
    dyn_lat = [r["latency"]  for r in dynamic_results]

    # --- Terminal summary table ---
    col_w = [5, 22, 11, 12, 10, 11]
    hdr   = ["Time", "Workload", "Dynamic CU", "Dynamic Lat", "Static CU", "Static Lat"]
    fmt   = "  ".join(f"{{:<{w}}}" for w in col_w)
    sep   = "  ".join("-" * w for w in col_w)
    print()
    print("=" * (sum(col_w) + 2 * (len(col_w) - 1)))
    print("DYNAMIC RECOMPILATION SUMMARY")
    print("=" * (sum(col_w) + 2 * (len(col_w) - 1)))
    print(fmt.format(*hdr))
    print(sep)
    for i, (sc, res) in enumerate(zip(SCENARIOS, dynamic_results)):
        pps_m = sc["pps"] / 1e6
        wl_str = f"{pps_m:.0f}M, {sc['avg_pkt_bytes']}B, hot={sc['hot_key_ratio']}"
        print(fmt.format(
            f"t={i}", wl_str,
            str(res["cu_total"]), str(res["latency"]),
            str(static["cu_total"]), str(static["latency"]),
        ))
    print()
    max_diff = max(abs(d - static["cu_total"]) for d in dyn_cu)
    print(f"Max CU difference (dynamic vs static): {max_diff}")
    print()

    # --- Plot ---
    PALE_BLUE = "#4878CF"
    PALE_RED  = "#D65F5F"

    fig, (ax_cu, ax_lat) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle("nfchain — Dynamic recompilation simulation",
                 fontsize=14, fontweight="bold", y=0.97)

    # ── CU subplot ──────────────────────────────────────────────────────────
    ax_cu.plot(x, dyn_cu, color=PALE_BLUE, marker="o", linewidth=2,
               markersize=7, label="Dynamic mapping")
    ax_cu.axhline(static["cu_total"], color=PALE_RED, linestyle="--",
                  linewidth=1.8, label=f"Static baseline ({static['cu_total']} CU)")

    for xi, (sc, cu) in enumerate(zip(SCENARIOS, dyn_cu)):
        annot = wl_summary(sc)
        ax_cu.annotate(annot, xy=(xi, cu),
                       xytext=(0, 12), textcoords="offset points",
                       ha="center", va="bottom", fontsize=7.5,
                       color=PALE_BLUE)

    ax_cu.set_ylabel("CU total", fontsize=11)
    ax_cu.legend(fontsize=9, loc="upper left")
    ax_cu.set_ylim(0, max(max(dyn_cu), static["cu_total"]) * 1.35)
    ax_cu.grid(axis="y", linestyle=":", alpha=0.5)
    ax_cu.spines[["top", "right"]].set_visible(False)

    # ── Latency subplot ─────────────────────────────────────────────────────
    ax_lat.plot(x, dyn_lat, color=PALE_BLUE, marker="o", linewidth=2,
                markersize=7, label="Dynamic mapping")
    ax_lat.axhline(static["latency"], color=PALE_RED, linestyle="--",
                   linewidth=1.8, label=f"Static baseline ({static['latency']} cy)")

    ax_lat.set_ylabel("Latency (cycles)", fontsize=11)
    ax_lat.set_xlabel("Time point", fontsize=11)
    ax_lat.legend(fontsize=9, loc="upper left")
    ax_lat.set_ylim(0, max(max(dyn_lat), static["latency"]) * 1.25)
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(labels, fontsize=8.5)
    ax_lat.grid(axis="y", linestyle=":", alpha=0.5)
    ax_lat.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
