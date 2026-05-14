#!/usr/bin/env python3
"""Web viewer for 360° panoramas with roll/pitch correction and keyboard navigation.

Usage:
    python viewer.py /path/to/images [--port 5001] [--results geocalib_results.json]

    # In Docker (images mounted at /data):
    docker run -v /my/panos:/data -p 5001:5001 geocalib-viewer /data

Keyboard shortcuts:
  ← / → (or A / D)  navigate between images
  Space (or C)       toggle correction on/off
  Home / End         jump to first / last image
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

_EXIF_ROLL_TAG_WRITE = "XMP-GPano:PoseRollDegrees"   # tag name used when writing
_EXIF_ROLL_TAG_READ  = "XMP:PoseRollDegrees"          # tag name returned when reading
_EXIF_PITCH_TAG_WRITE = "XMP-GPano:PosePitchDegrees"
_EXIF_PITCH_TAG_READ  = "XMP:PosePitchDegrees"

def _read_exif_batch(paths: list[Path]) -> dict[str, dict]:
    """Return {filename: {roll, pitch}} for all paths, in one exiftool call."""
    try:
        from exiftool import ExifToolHelper
    except ImportError:
        return {}
    result = {}
    with ExifToolHelper() as et:
        tags_list = et.get_tags(
            [str(p) for p in paths],
            tags=[_EXIF_ROLL_TAG_WRITE, _EXIF_PITCH_TAG_WRITE],
        )
        for path, tags in zip(paths, tags_list):
            roll  = tags.get(_EXIF_ROLL_TAG_READ)
            pitch = tags.get(_EXIF_PITCH_TAG_READ)
            result[path.name] = {
                "roll":  float(roll)  if roll  is not None else None,
                "pitch": float(pitch) if pitch is not None else None,
            }
    return result


def _load_results_json(json_path: Path) -> dict[str, dict]:
    """Return {filename: {inlier_ratio, ...}} from a geocalib_results.json."""
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return {}
    if isinstance(data, dict):
        data = [data]
    result = {}
    for entry in data:
        name = Path(entry.get("file", "")).name
        if name:
            result[name] = entry
    return result


def _build_image_list(
    paths: list[Path],
    exif: dict[str, dict],
    results: dict[str, dict],
) -> list[dict]:
    images = []
    for p in paths:
        e = exif.get(p.name, {})
        r = results.get(p.name, {})
        # Prefer JSON results (already has roll/pitch from inference); fall back to EXIF
        roll  = r.get("roll")  if r else e.get("roll")
        pitch = r.get("pitch") if r else e.get("pitch")
        if roll is None:
            roll = e.get("roll")
        if pitch is None:
            pitch = e.get("pitch")
        images.append({
            "name":         p.name,
            "url":          f"/image/{p.name}",
            "roll":         roll,
            "pitch":        pitch,
            "inlier_ratio": r.get("inlier_ratio"),
        })
    return images


def _build_app(directory: Path, images: list[dict]) -> Flask:
    _root = Path(__file__).parent
    app = Flask(__name__,
                template_folder=str(_root / "templates"),
                static_folder=str(_root / "static"))
    name_to_path = {Path(img["url"]).name: directory / Path(img["url"]).name for img in images}

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/images")
    def api_images():
        return jsonify(images)

    @app.route("/image/<path:name>")
    def image(name):
        path = name_to_path.get(name)
        if not path or not path.exists():
            abort(404)
        return send_file(path)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Web viewer for 360° panoramas with roll/pitch correction."
    )
    parser.add_argument("directory", type=Path, help="Directory containing equirectangular JPEGs.")
    parser.add_argument("--port", type=int, default=5001, metavar="PORT", help="HTTP port (default: 5001).")
    parser.add_argument("--host", default="127.0.0.1", metavar="HOST",
                        help="Host to bind to (default: 127.0.0.1; use 0.0.0.0 to expose on the network).")
    parser.add_argument(
        "--results", type=Path, default=None, metavar="JSON",
        help="Optional geocalib_results.json to show inlier ratios.",
    )
    parser.add_argument("--start", type=int, default=0, metavar="N", help="Start at image index N (default: 0).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab automatically.")
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    paths = sorted(
        p for p in args.directory.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg"}
    )
    if not paths:
        print(f"Error: no JPEG files found in {args.directory}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading EXIF from {len(paths)} images…", flush=True)
    exif = _read_exif_batch(paths)

    results: dict[str, dict] = {}
    if args.results:
        results = _load_results_json(args.results)
    else:
        auto = args.directory / "geocalib_results.json"
        if auto.exists():
            results = _load_results_json(auto)
            print(f"Loaded results from {auto}", flush=True)

    images = _build_image_list(paths, exif, results)
    app = _build_app(args.directory, images)

    url = f"http://localhost:{args.port}?start={args.start}"
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    print(f"Serving {len(images)} images at {url}  (Ctrl-C to stop)")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
