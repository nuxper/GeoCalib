#!/usr/bin/env python3
"""Command-line interface for 360° pitch/roll estimation using GeoCalib."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

from core import MAX_SAMPLE_COUNT, download_image_bytes, predict_360


def main():
    parser = argparse.ArgumentParser(
        description="Estimate pitch and roll of a 360° equirectangular image."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", "-f", type=Path, help="Path to the input image file.")
    source.add_argument("--url", "-u", type=str, help="URL of the input image.")

    parser.add_argument(
        "--fov",
        type=float,
        default=70.0,
        metavar="DEG",
        help="Field of view in degrees for perspective crops (default: 70, range: 1–179).",
    )
    parser.add_argument(
        "--inlier-threshold-deg",
        type=float,
        default=2.0,
        metavar="DEG",
        help="Max combined error (°) for a sample to be an inlier (default: 2.0).",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=18,
        metavar="N",
        help=f"Number of perspective crops (default: 18, range: 4–{MAX_SAMPLE_COUNT}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output the full result as JSON instead of a human-readable summary.",
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Omit per-sample details from JSON output.",
    )

    args = parser.parse_args()

    if not (1.0 <= args.fov <= 179.0):
        parser.error("--fov must be between 1 and 179 degrees.")
    if not (0.01 <= args.inlier_threshold_deg <= 90.0):
        parser.error("--inlier-threshold-deg must be between 0.01 and 90.")
    if not (4 <= args.sample_count <= MAX_SAMPLE_COUNT):
        parser.error(f"--sample-count must be between 4 and {MAX_SAMPLE_COUNT}.")

    if args.file:
        path = args.file.expanduser().resolve()
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        image_bytes = path.read_bytes()
    else:
        try:
            image_bytes = download_image_bytes(args.url)
        except (ValueError, HTTPError, URLError, TimeoutError) as e:
            print(f"Error downloading image: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        result = predict_360(
            image_bytes,
            fov_deg=args.fov,
            inlier_threshold_deg=args.inlier_threshold_deg,
            sample_count=args.sample_count,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output_json:
        if args.no_samples:
            result = {k: v for k, v in result.items() if k != "samples"}
        print(json.dumps(result, indent=2))
    else:
        roll = result["roll"]
        pitch = result["pitch"]
        inlier_count = result["inlier_count"]
        inlier_ratio = result["inlier_ratio"]
        mae = result["mae_inlier_deg"]
        rmse = result["rmse_inlier_deg"]
        roll_str = f"{roll:.3f}°" if roll is not None else "N/A"
        pitch_str = f"{pitch:.3f}°" if pitch is not None else "N/A"
        ratio_str = f"{inlier_ratio:.1%}" if inlier_ratio is not None else "N/A"
        mae_str = f"{mae:.3f}°" if mae is not None else "N/A"
        rmse_str = f"{rmse:.3f}°" if rmse is not None else "N/A"
        print(f"Roll:         {roll_str}")
        print(f"Pitch:        {pitch_str}")
        print(f"Inliers:      {inlier_count}/{result['sample_count']} ({ratio_str})")
        print(f"MAE (inlier): {mae_str}")
        print(f"RMSE(inlier): {rmse_str}")


if __name__ == "__main__":
    main()
