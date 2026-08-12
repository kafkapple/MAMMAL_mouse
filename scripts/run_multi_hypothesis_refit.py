#!/usr/bin/env python3
"""Run multi-hypothesis refit pilots for selected hard frames.

The script is intentionally conservative:
- it never deletes Hydra output directories;
- each hypothesis writes copied OBJ files into a separate output folder;
- `--dry-run` prints commands without running them.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


HYPOTHESES: dict[str, list[str]] = {
    "accurate_default": ["optim=accurate"],
    "mask_heavy": ["optim=accurate", "loss_weights.mask_step2=5000"],
    "keypoint_heavy": [
        "optim=accurate",
        "loss_weights.2d=0.4",
        "loss_weights.3d=4.0",
    ],
    "smooth_heavy": [
        "optim=accurate",
        "loss_weights.stretch=2.0",
        "loss_weights.temp=0.5",
        "loss_weights.temp_d=0.4",
        "loss_weights.chest_deformer=0.2",
    ],
    "long_step2": [
        "optim=accurate",
        "optim.solve_step2_iters=100",
    ],
}


def read_frames(path: Path, limit: int) -> list[int]:
    frames = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    return frames[:limit] if limit > 0 else frames


def newest_fitting_dir(root: Path, since: float) -> Path | None:
    candidates = [
        p
        for p in root.glob("markerless_mouse_1_nerf_v012345_kp22_2026*")
        if p.is_dir() and p.stat().st_mtime >= since
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_frame(frame: int, hypothesis: str, overrides: list[str], output: Path, dry_run: bool) -> int:
    frame_pad = f"{frame:06d}"
    hyp_dir = output / hypothesis
    obj_dir = hyp_dir / "obj"
    log_dir = hyp_dir / "logs"
    obj_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_obj = obj_dir / f"step_2_frame_{frame_pad}.obj"
    if out_obj.exists():
        print(f"[skip] {hypothesis} frame {frame}: {out_obj} exists")
        return 0

    frame_end = frame + 5
    cmd = [
        "./run_experiment.sh",
        "baseline_6view_keypoint",
        f"fitter.start_frame={frame}",
        f"fitter.end_frame={frame_end}",
        "fitter.interval=5",
        *overrides,
    ]
    print("[run]", " ".join(cmd))
    if dry_run:
        return 0

    log_path = log_dir / f"frame_{frame_pad}.log"
    since = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[fail] {hypothesis} frame {frame}: command failed, see {log_path}")
        return proc.returncode

    latest = newest_fitting_dir(Path("results/fitting"), since)
    if latest is None:
        print(f"[fail] {hypothesis} frame {frame}: no new Hydra output dir")
        return 2
    src_obj = latest / "obj" / f"step_2_frame_{frame_pad}.obj"
    if not src_obj.exists():
        print(f"[fail] {hypothesis} frame {frame}: missing {src_obj}")
        return 3
    shutil.copy2(src_obj, out_obj)
    (hyp_dir / "source_dirs.txt").open("a").write(f"{frame}\t{latest}\n")
    print(f"[ok] {hypothesis} frame {frame}: {out_obj}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="conf/frames/pilots/refit_next20_hard_cases.txt")
    parser.add_argument("--output", default="results/fitting/multi_hypothesis_refit_260520/")
    parser.add_argument("--hypotheses", nargs="+", default=list(HYPOTHESES))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    frames = read_frames(Path(args.frames), args.limit)
    output = Path(args.output)
    rc = 0
    for hyp in args.hypotheses:
        if hyp not in HYPOTHESES:
            raise SystemExit(f"unknown hypothesis {hyp}; choices={sorted(HYPOTHESES)}")
        for frame in frames:
            rc = max(rc, run_frame(frame, hyp, HYPOTHESES[hyp], output, args.dry_run))
    return rc


if __name__ == "__main__":
    sys.exit(main())
