#!/usr/bin/env python3
"""E3: P1 do_opt=true crash diagnosis.

Hypothesis: original failures on bori RTX 3060 (12GB VRAM, conda env=mammal_stable)
likely CUDA OOM or env-specific bug. gpu03 H100 (80GB, env=mammal_blackwell) should
reveal whether it's memory or code.

Strategy:
1. Minimal single-run with opt_iters=5 (fast smoke test)
2. Verbose logging at every iteration (loss, grad_norm, memory)
3. Catch + log any exception with full trace

Outputs:
    results/p1_debug/run_TIMESTAMP/
      stdout.log, params.json, if-crash: exception_trace.txt
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir",
                    default="results/fitting/production_3600_canon",
                    help="Existing fitting result directory for UV texture optimization. "
                         "Original failed run used markerless_mouse_1_nerf_v012345_kp22_20251206_165254 "
                         "(only exists on bori). production_3600_canon is current on gpu03.")
    ap.add_argument("--output-base", default="results/p1_debug/")
    ap.add_argument("--opt-iters", type=int, default=5,
                    help="Keep low for smoke test. Original failure config used 30-100.")
    ap.add_argument("--uv-size", type=int, default=512)
    ap.add_argument("--visibility-threshold", type=float, default=0.193)
    ap.add_argument("--w-tv", type=float, default=1e-4)
    ap.add_argument("--fusion-method", default="average")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_base) / f"run_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    params = {k: v for k, v in vars(args).items()}
    params["timestamp"] = ts
    with open(out / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        import torch
        print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
              f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            print(f"[env] GPU mem free={free/1e9:.1f}GB total={total/1e9:.1f}GB")

        print(f"[cfg] result_dir={args.result_dir}")
        print(f"[cfg] opt_iters={args.opt_iters}, uv_size={args.uv_size}, fusion={args.fusion_method}")
        print(f"[cfg] vis_thr={args.visibility_threshold}, w_tv={args.w_tv}")

        # Import UV pipeline
        from mammal_ext.uvmap.uv_pipeline import UVMapPipeline, UVPipelineConfig

        cfg = UVPipelineConfig(
            result_dir=args.result_dir,
            output_dir=str(out),
            uv_size=args.uv_size,
            visibility_threshold=args.visibility_threshold,
            use_visibility_weighting=True,
            do_optimization=True,
            opt_iters=args.opt_iters,
            opt_w_tv=args.w_tv,
        )
        print(f"[pipeline] config: {cfg}")

        pipeline = UVMapPipeline(cfg)

        # Execute with verbose tracking
        t0 = time.time()
        print(f"[run] starting pipeline.run() at {time.strftime('%H:%M:%S')}")
        result = pipeline.run()
        elapsed = time.time() - t0
        print(f"[run] completed in {elapsed:.1f}s")
        print(f"[result] keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")

        # Save success indicator
        with open(out / "SUCCESS", "w") as f:
            f.write(f"elapsed={elapsed:.1f}s\n")
        return 0

    except Exception as e:
        # Capture full trace
        trace = traceback.format_exc()
        print(f"\n[CRASH] {type(e).__name__}: {e}")
        print(trace)
        with open(out / "exception_trace.txt", "w") as f:
            f.write(f"Exception: {type(e).__name__}\n")
            f.write(f"Message: {e}\n\n")
            f.write(trace)
        # Also log memory state on crash
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                alloc = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                with open(out / "gpu_memory_on_crash.txt", "w") as f:
                    f.write(f"free={free/1e9:.2f}GB total={total/1e9:.2f}GB "
                            f"allocated={alloc:.2f}GB reserved={reserved:.2f}GB\n")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
