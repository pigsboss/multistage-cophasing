# -*- coding: utf-8 -*-
"""
MCPC Visualizer CLI

Usage:
    python -m utils.visualizers --time "2026-04-10T12:00:00" [--vedo]
    python -m utils.visualizers --time "2026-04-10T12:00:00" --duration 86400 --step 3600 [--vedo] [--output frames/]
"""

import argparse
import sys
from pathlib import Path
import os

# Ensure the project root is importable (in case running from any directory)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.visualizers.base import SceneBuilder, LogScale
from utils.visualizers.backends.debug import DebugRenderer

# Import MCPC high precision ephemeris
try:
    from mission_sim.core.spacetime.ephemeris.high_precision import (
        HighPrecisionEphemeris, EphemerisMode, EphemerisConfig
    )
except ImportError as e:
    print("Fatal: could not import MCPC high precision ephemeris module.", file=sys.stderr)
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="MCPC 3D Scene Visualizer (Sun-Earth-Moon demo)")
    parser.add_argument("--time", type=str, default="2026-04-10T12:00:00",
                        help="UTC start time in ISO format (default: 2026-04-10T12:00:00)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Total simulation duration in seconds (e.g., 86400 for one day). "
                             "If not given, a single frame is rendered.")
    parser.add_argument("--step", type=float, default=3600,
                        help="Time step between frames in seconds (default: 3600). "
                             "Only used when --duration is specified.")
    parser.add_argument("--vedo", action="store_true",
                        help="Use vedo for 3D output (single frame interactive, or multi‑frame image sequence)")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save frames when using --vedo with --duration. "
                             "If not set, frames are not saved.")
    args = parser.parse_args()

    if args.duration is not None and args.duration <= 0:
        print("Error: --duration must be positive.", file=sys.stderr)
        sys.exit(1)
    if args.step <= 0:
        print("Error: --step must be positive.", file=sys.stderr)
        sys.exit(1)

    # Configure HighPrecisionEphemeris in SPICE mode – kernel discovery is automatic
    config = EphemerisConfig(
        mode=EphemerisMode.SPICE,
        verbose=False
    )

    eph = HighPrecisionEphemeris(config=config)

    # Convert UTC string to ephemeris time (ET) using SPICE (or fallback if SPICE unavailable)
    try:
        start_epoch = eph.utc_to_et(args.time)
    except Exception as exc:
        print(f"Failed to convert UTC to ET: {exc}", file=sys.stderr)
        eph.shutdown()
        sys.exit(1)

    print(f"Initialized SPICE, start epoch = {start_epoch:.3f} s", file=sys.stdout)

    # Build time sequence
    if args.duration is not None:
        times = [start_epoch + i * args.step
                 for i in range(int(args.duration / args.step) + 1)]
    else:
        times = [start_epoch]

    # Prepare output directory if needed
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Scene builder (single instance, could change scale function)
    builder = SceneBuilder(scale_function=LogScale(linear_threshold=3.8e8, compression=5e8))

    # Renderer (vedo backend will handle single/multi‑frame)
    renderer = DebugRenderer(use_vedo=args.vedo)

    total_frames = len(times)
    for idx, epoch in enumerate(times):
        scene = builder.build_solar_system_demo(epoch, eph)
        print(f"\n--- Frame {idx+1}/{total_frames} : epoch = {epoch:.3f} s ---")

        renderer.render(
            scene,
            frame_index=idx,
            total_frames=total_frames,
            output_dir=str(output_dir) if output_dir else None
        )

    # Clean up
    eph.shutdown()
    print("\nDone.", file=sys.stdout)


if __name__ == "__main__":
    main()
