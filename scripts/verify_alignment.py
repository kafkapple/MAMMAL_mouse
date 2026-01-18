#!/usr/bin/env python3
"""
Verify MAMMAL-pose-splatter frame alignment.

Usage:
    python verify_alignment.py <mammal_result_dir> [--posesplatter-config <path>]

Example:
    python verify_alignment.py results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260118_123456
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def load_mammal_config(result_dir: Path) -> dict:
    """Load MAMMAL result config."""
    config_path = result_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def count_mammal_frames(result_dir: Path) -> dict:
    """Count MAMMAL output files."""
    obj_dir = result_dir / "obj"
    params_dir = result_dir / "params"

    obj_files = list(obj_dir.glob("step_2_frame_*.obj")) if obj_dir.exists() else []
    param_files = list(params_dir.glob("step_*_frame_*.pkl")) if params_dir.exists() else []

    # Extract frame indices from obj files
    frame_indices = []
    for f in obj_files:
        # step_2_frame_000000.obj -> 0
        name = f.stem  # step_2_frame_000000
        idx = int(name.split("_")[-1])
        frame_indices.append(idx)

    return {
        "obj_count": len(obj_files),
        "param_count": len(param_files),
        "frame_indices": sorted(frame_indices),
        "min_frame": min(frame_indices) if frame_indices else None,
        "max_frame": max(frame_indices) if frame_indices else None,
    }


def load_posesplatter_config(config_path: Path) -> dict:
    """Load pose-splatter config."""
    with open(config_path) as f:
        return json.load(f)


def verify_alignment(mammal_dir: str, posesplatter_config: str = None):
    """Verify frame alignment between MAMMAL and pose-splatter."""

    mammal_path = Path(mammal_dir)
    if not mammal_path.exists():
        print(f"❌ MAMMAL result directory not found: {mammal_path}")
        sys.exit(1)

    print("=" * 60)
    print("MAMMAL-pose-splatter Frame Alignment Verification")
    print("=" * 60)
    print(f"\nMAMMAL result: {mammal_path.name}")
    print()

    # Load MAMMAL config
    try:
        mammal_config = load_mammal_config(mammal_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Extract frame settings
    fitter = mammal_config.get("fitter", {})
    start_frame = fitter.get("start_frame", 0)
    end_frame = fitter.get("end_frame", "?")
    interval = fitter.get("interval", 1)

    print("📋 MAMMAL Frame Settings:")
    print(f"   start_frame: {start_frame}")
    print(f"   end_frame:   {end_frame}")
    print(f"   interval:    {interval}")
    print()

    # Count actual frames
    frame_info = count_mammal_frames(mammal_path)
    print(f"📊 MAMMAL Output Files:")
    print(f"   obj files:   {frame_info['obj_count']}")
    print(f"   param files: {frame_info['param_count']}")
    if frame_info['frame_indices']:
        print(f"   frame range: {frame_info['min_frame']} - {frame_info['max_frame']}")
    print()

    # pose-splatter settings
    ps_frame_jump = 5  # Default
    if posesplatter_config:
        ps_path = Path(posesplatter_config)
        if ps_path.exists():
            ps_config = load_posesplatter_config(ps_path)
            ps_frame_jump = ps_config.get("frame_jump", 5)

    print(f"📋 pose-splatter Settings:")
    print(f"   frame_jump:  {ps_frame_jump}")
    print()

    # Alignment check
    print("=" * 60)
    print("🔍 Alignment Check")
    print("=" * 60)
    print()

    aligned = (interval == ps_frame_jump)

    if aligned:
        print(f"✅ ALIGNED: MAMMAL interval ({interval}) = pose-splatter frame_jump ({ps_frame_jump})")
        print()
        print("   Frame mapping (keypoint_frame_ratio = 1):")
        print("   ┌─────────────┬───────────┬────────────────────┐")
        print("   │ MAMMAL idx  │ Raw frame │ pose-splatter idx  │")
        print("   ├─────────────┼───────────┼────────────────────┤")

        # Show first 5 and last 2 frames
        indices = frame_info['frame_indices']
        show_indices = indices[:5] + ["..."] + indices[-2:] if len(indices) > 7 else indices

        for idx in show_indices:
            if idx == "...":
                print("   │     ...     │    ...    │        ...         │")
            else:
                raw_frame = idx * interval
                ps_idx = idx  # Because aligned, 1:1 mapping
                print(f"   │ {idx:^11} │ {raw_frame:^9} │ {ps_idx:^18} │")

        print("   └─────────────┴───────────┴────────────────────┘")
        print()
        print("   ✅ keypoint_frame_ratio = 1 (perfect alignment)")

    else:
        print(f"❌ MISALIGNED: MAMMAL interval ({interval}) ≠ pose-splatter frame_jump ({ps_frame_jump})")
        print()
        print("   Problem:")
        print(f"   - MAMMAL frame 1 → raw frame {interval}")
        print(f"   - pose-splatter sample 1 → raw frame {ps_frame_jump}")
        print(f"   - Temporal mismatch: {abs(interval - ps_frame_jump)} frames!")
        print()
        print("   Solution:")
        print(f"   - Re-run MAMMAL with interval={ps_frame_jump}")
        print(f"   - Or use keypoint_frame_ratio={ps_frame_jump}/{interval} (not recommended)")

    print()
    print("=" * 60)

    # Coverage check
    if frame_info['obj_count'] > 0:
        print()
        print(f"📈 Coverage:")
        print(f"   MAMMAL frames available: {frame_info['obj_count']}")
        print(f"   pose-splatter samples covered: 0 - {frame_info['max_frame']}")

        # Assuming 3600 total pose-splatter samples
        total_ps_samples = 3600
        coverage_pct = (frame_info['obj_count'] / total_ps_samples) * 100
        print(f"   Coverage: {coverage_pct:.1f}% of total ({frame_info['obj_count']}/{total_ps_samples})")

        if frame_info['obj_count'] < 100:
            print(f"   ⚠️  Warning: Only {frame_info['obj_count']} frames. Consider running longer.")

    print()

    # Return status
    return aligned


def main():
    parser = argparse.ArgumentParser(description="Verify MAMMAL-pose-splatter frame alignment")
    parser.add_argument("mammal_result_dir", help="Path to MAMMAL result directory")
    parser.add_argument("--posesplatter-config", "-p",
                        help="Path to pose-splatter config (optional)")

    args = parser.parse_args()

    aligned = verify_alignment(args.mammal_result_dir, args.posesplatter_config)

    sys.exit(0 if aligned else 1)


if __name__ == "__main__":
    main()
