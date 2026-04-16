#!/usr/bin/env python3
"""
Verify re-computed IoU against reported metrics.

Gate (Day-2 step after paper_fast OBJ rerun):
  rerun mean IoU and bad-rate must match the reported numbers within tolerance,
  otherwise the original metric pipeline itself is suspect and downstream
  comparisons should be halted.

Usage:
    python scripts/verify_metrics.py \
        --rerun   results/comparison/baseline_iou_rerun/baseline_iou.json \
        --reported results/comparison/baseline_iou/baseline_iou.json \
        --tolerance 0.005

    # Or compare against literal numbers
    python scripts/verify_metrics.py \
        --rerun results/comparison/baseline_iou_rerun/baseline_iou.json \
        --reported-mean 0.7945 --reported-bad-rate 0.11 --tolerance 0.005

Exit codes:
  0 = within tolerance
  1 = drift detected (STOP downstream comparison, audit metric pipeline)
  2 = file/format error
"""

import argparse
import json
import os
import sys
from statistics import mean


def _load(path):
    if not os.path.exists(path):
        print(f"ERROR: not found: {path}", file=sys.stderr)
        sys.exit(2)
    with open(path) as f:
        d = json.load(f)
    def _pick(entry, src_path):
        if isinstance(entry, dict):
            if "mean" in entry:
                return float(entry["mean"])
            if "mean_iou" in entry:
                return float(entry["mean_iou"])
            print(f"ERROR: dict entry in {src_path} has neither 'mean' nor 'mean_iou': keys={list(entry.keys())}",
                  file=sys.stderr)
            sys.exit(2)
        return float(entry)

    if isinstance(d, dict):
        vals = [_pick(v, path) for v in d.values()]
    elif isinstance(d, list):
        vals = [_pick(e, path) for e in d]
    else:
        print(f"ERROR: unexpected JSON shape in {path}", file=sys.stderr)
        sys.exit(2)
    return vals


def _stats(vals, bad_threshold=0.7):
    return {
        "n": len(vals),
        "mean": mean(vals) if vals else 0.0,
        "min": min(vals) if vals else 0.0,
        "max": max(vals) if vals else 0.0,
        "bad_rate": sum(1 for v in vals if v < bad_threshold) / max(len(vals), 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rerun", required=True)
    p.add_argument("--reported", default="", help="Reported JSON path (paired comparison)")
    p.add_argument("--reported-mean", type=float, default=None)
    p.add_argument("--reported-bad-rate", type=float, default=None)
    p.add_argument("--bad-threshold", type=float, default=0.7)
    p.add_argument("--tolerance", type=float, default=0.005)
    args = p.parse_args()

    rerun_vals = _load(args.rerun)
    rerun = _stats(rerun_vals, args.bad_threshold)
    print(f"Rerun:    n={rerun['n']} mean={rerun['mean']:.4f} bad_rate={rerun['bad_rate']:.4f}"
          f" min={rerun['min']:.4f} max={rerun['max']:.4f}")

    if args.reported:
        reported_vals = _load(args.reported)
        reported = _stats(reported_vals, args.bad_threshold)
        print(f"Reported: n={reported['n']} mean={reported['mean']:.4f} bad_rate={reported['bad_rate']:.4f}")
        d_mean = abs(rerun["mean"] - reported["mean"])
        d_bad = abs(rerun["bad_rate"] - reported["bad_rate"])
    elif args.reported_mean is not None:
        d_mean = abs(rerun["mean"] - args.reported_mean)
        d_bad = abs(rerun["bad_rate"] - args.reported_bad_rate) if args.reported_bad_rate is not None else 0.0
        print(f"Reported (literal): mean={args.reported_mean} bad_rate={args.reported_bad_rate}")
    else:
        print("ERROR: provide --reported or --reported-mean", file=sys.stderr)
        sys.exit(2)

    print(f"|Δ mean|    = {d_mean:.4f}  (tolerance {args.tolerance})")
    print(f"|Δ bad_rate|= {d_bad:.4f}  (tolerance {args.tolerance})")

    if d_mean > args.tolerance or d_bad > args.tolerance:
        print("\n🔴 DRIFT — metric pipeline reproduction broken. Halt downstream comparisons, audit first.")
        sys.exit(1)
    print("\n✅ Within tolerance. Proceed with side-by-side comparison.")


if __name__ == "__main__":
    main()
