#!/usr/bin/env python3
"""
Turn results.csv (from benchmark_allreduce.py) into three panels:
  1. total time vs size (log-log)
  2. staging vs transfer vs compute breakdown vs size (stacked)
  3. achieved bandwidth vs size

Usage: python plot_results.py results.csv --out results.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--out", default="results.png")
    ap.add_argument("--mode", choices=["cpu", "gpu"], default=None,
                     help="filter to one mode if the CSV has both")
    ap.add_argument("--bytes-per-elem", type=int, default=4, help="4 for float32")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    if args.mode:
        df = df[df["mode"] == args.mode]

    summary = df.groupby("size").agg(
        total_mean=("total_ms", "mean"), total_std=("total_ms", "std"),
        staging_mean=("staging_ms", "mean"),
        transfer_mean=("transfer_ms", "mean"),
        compute_mean=("compute_ms", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.errorbar(summary["size"], summary["total_mean"], yerr=summary["total_std"],
                marker="o", capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Array size (elements)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Total time vs. size")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.stackplot(summary["size"], summary["staging_mean"], summary["transfer_mean"],
                 summary["compute_mean"],
                 labels=["staging (memcpy)", "transfer (socket)", "compute (reduce_add)"])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Array size (elements)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Where the time goes")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    bandwidth_gbps = (summary["size"] * args.bytes_per_elem) / (summary["total_mean"] / 1000) / 1e9
    ax.plot(summary["size"], bandwidth_gbps, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Array size (elements)")
    ax.set_ylabel("Achieved bandwidth (GB/s)")
    ax.set_title("Bandwidth vs. size")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")

    avg_staging_frac = (df["staging_ms"] / df["total_ms"]).mean()
    print(f"avg fraction of time in memcpy staging: {avg_staging_frac:.1%}")


if __name__ == "__main__":
    main()