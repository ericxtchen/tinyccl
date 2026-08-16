#!/usr/bin/env python3
"""
Benchmark driver for the ring-allreduce binary, driven through run_ring.sh.

Sweeps --size across a log2 range, runs run_ring.sh once per (size, trial),
and parses the RESULT lines main.cu prints out of ./logs/rank_<i>.log to
build a CSV that plot_results.py can chart.
"""

import argparse
import csv
import os
import subprocess
import sys


def parse_result_line(line):
    parts = line.strip().split(",")
    if not parts or parts[0] != "RESULT":
        return None
    return {
        "rank": int(parts[1]),
        "world_size": int(parts[2]),
        "size": int(parts[3]),
        "mode": parts[4],
        "total_ms": float(parts[5]),
        "staging_ms": float(parts[6]),
        "transfer_ms": float(parts[7]),
        "compute_ms": float(parts[8]),
    }


def run_once(run_script, binary, world_size, size, gpu, logs_dir):
    cmd = [run_script]
    if gpu:
        cmd.append("--gpu")
    cmd += ["--size", str(size), str(world_size), binary]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[warn] run_ring.sh exited {proc.returncode} for size={size}\n{proc.stderr}",
              file=sys.stderr)
        return []

    results = []
    for rank in range(world_size):
        log_path = os.path.join(logs_dir, f"rank_{rank}.log")
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            for line in f:
                parsed = parse_result_line(line)
                if parsed:
                    results.append(parsed)
    return results


def run_sweep(run_script, binary, world_size, sizes, gpu, trials, warmup, logs_dir):
    rows = []
    for size in sizes:
        for _ in range(warmup):
            run_once(run_script, binary, world_size, size, gpu, logs_dir)  # discard

        for trial in range(trials):
            results = run_once(run_script, binary, world_size, size, gpu, logs_dir)
            if not results:
                print(f"[warn] no RESULT lines for size={size} trial={trial}", file=sys.stderr)
                continue
            # the collective isn't "done" until the slowest rank finishes
            total_ms = max(r["total_ms"] for r in results)
            staging_ms = max(r["staging_ms"] for r in results)
            transfer_ms = max(r["transfer_ms"] for r in results)
            compute_ms = max(r["compute_ms"] for r in results)
            rows.append({
                "size": size, "world_size": world_size, "mode": "gpu" if gpu else "cpu",
                "trial": trial, "total_ms": total_ms, "staging_ms": staging_ms,
                "transfer_ms": transfer_ms, "compute_ms": compute_ms,
            })
            print(f"size={size:>10} trial={trial} total_ms={total_ms:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-script", default="./run_ring.sh")
    ap.add_argument("--binary", default="./build/myapp")
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--min-log2", type=int, default=10, help="smallest size = 2**min_log2 floats")
    ap.add_argument("--max-log2", type=int, default=24, help="largest size = 2**max_log2 floats")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--logs-dir", default="logs", help="matches run_ring.sh's LOG_DIR")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    sizes = [2 ** i for i in range(args.min_log2, args.max_log2 + 1)]
    rows = run_sweep(args.run_script, args.binary, args.world_size, sizes,
                      args.gpu, args.trials, args.warmup, args.logs_dir)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "size", "world_size", "mode", "trial",
            "total_ms", "staging_ms", "transfer_ms", "compute_ms",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()