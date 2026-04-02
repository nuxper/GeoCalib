#!/usr/bin/env python3
"""Command-line interface for 360° pitch/roll estimation using GeoCalib."""

from __future__ import annotations

import argparse
import glob as glob_module
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

from core import MAX_SAMPLE_COUNT, download_image_bytes, predict_360

_GLOB_CHARS = frozenset("*?[")


def _resolve_paths(raw_paths: list[Path]) -> list[Path]:
    """Expand any glob patterns in raw_paths and return sorted, deduplicated paths."""
    resolved = []
    for p in raw_paths:
        if _GLOB_CHARS & set(str(p)):
            matches = sorted(glob_module.glob(str(p), recursive=True))
            if not matches:
                print(f"Error: no files matched '{p}'", file=sys.stderr)
                sys.exit(1)
            resolved.extend(Path(m) for m in matches)
        else:
            resolved.append(p.expanduser().resolve())
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in resolved:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _process_file(path: Path, args: argparse.Namespace) -> dict:
    """Run predict_360 on a single file and return the result dict."""
    return predict_360(
        path.read_bytes(),
        fov_deg=args.fov,
        inlier_threshold_deg=args.inlier_threshold_deg,
        sample_count=args.sample_count,
    )


def _print_summary(result: dict, label: str | None = None) -> None:
    """Print a human-readable summary of a single prediction result."""
    if label:
        print(f"\n=== {label} ===")
    roll, pitch = result["roll"], result["pitch"]
    inlier_ratio, mae, rmse = result["inlier_ratio"], result["mae_inlier_deg"], result["rmse_inlier_deg"]
    print(f"Roll:         {f'{roll:.3f}°' if roll is not None else 'N/A'}")
    print(f"Pitch:        {f'{pitch:.3f}°' if pitch is not None else 'N/A'}")
    print(f"Inliers:      {result['inlier_count']}/{result['sample_count']} ({f'{inlier_ratio:.1%}' if inlier_ratio is not None else 'N/A'})")
    print(f"MAE (inlier): {f'{mae:.3f}°' if mae is not None else 'N/A'}")
    print(f"RMSE(inlier): {f'{rmse:.3f}°' if rmse is not None else 'N/A'}")


def _strip_samples(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "samples"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate pitch and roll of a 360° equirectangular image."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--file", "-f",
        type=Path,
        nargs="+",
        metavar="PATH",
        help='One or more image files or patterns (e.g. "images/*.jpg", "**/*.jpg").',
    )
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
        help="Output results as JSON instead of a human-readable summary.",
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

    # --- URL mode ---
    if args.url:
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
            print(json.dumps(_strip_samples(result) if args.no_samples else result, indent=2))
        else:
            _print_summary(result)
        return

    # --- File mode (single paths or patterns) ---
    paths = _resolve_paths(args.file)
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"Error: file not found: {p}", file=sys.stderr)
        sys.exit(1)

    multi = len(paths) > 1
    all_results = []
    exit_code = 0

    for path in paths:
        try:
            result = _process_file(path, args)
        except ValueError as e:
            print(f"Error [{path.name}]: {e}", file=sys.stderr)
            exit_code = 1
            if args.output_json:
                all_results.append({"file": str(path), "status": "error", "error": str(e)})
            continue

        if args.output_json:
            entry = _strip_samples(result) if args.no_samples else result
            all_results.append({"file": str(path), **entry})
        else:
            _print_summary(result, label=path.name if multi else None)

    if args.output_json:
        output = all_results if multi else all_results[0]
        print(json.dumps(output, indent=2))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
