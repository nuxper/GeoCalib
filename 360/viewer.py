#!/usr/bin/env python3
"""Web viewer for 360° panoramas with roll/pitch correction and keyboard navigation.

Usage:
    uv run viewer.py /path/to/images [--port 5001] [--results geocalib_results.json]

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

from flask import Flask, abort, request, send_file

_EXIF_ROLL_TAG_WRITE = "XMP-GPano:PoseRollDegrees"   # tag name used when writing
_EXIF_ROLL_TAG_READ  = "XMP:PoseRollDegrees"          # tag name returned when reading
_EXIF_PITCH_TAG_WRITE = "XMP-GPano:PosePitchDegrees"
_EXIF_PITCH_TAG_READ  = "XMP:PosePitchDegrees"

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>360° Viewer</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/@photo-sphere-viewer/core@5/index.min.css">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { height: 100%; background: #111; color: #eee; font-family: monospace; }
    body { display: flex; flex-direction: column; }

    #viewer-wrap { position: relative; flex: 1; min-height: 0; }
    #viewer      { width: 100%; height: 100%; }

    /* Fixed crosshair — valid when looking straight ahead (pitch = 0) */
    #horizon, #vertical {
      position: absolute; pointer-events: none; z-index: 100;
    }
    #horizon {
      top: 50%; left: 0; right: 0; height: 0;
      border-top: 2px solid rgba(255, 220, 0, 0.85);
    }
    #vertical {
      left: 50%; top: 0; bottom: 0; width: 0;
      border-left: 2px solid rgba(255, 220, 0, 0.85);
    }

    /* Bottom bar */
    #bar {
      display: flex; align-items: center; gap: 16px;
      padding: 7px 14px; background: #1a1a1a;
      font-size: 13px; flex-shrink: 0;
    }
    #nav { display: flex; gap: 6px; }
    button {
      background: #333; color: #ddd; border: 1px solid #444;
      padding: 3px 11px; cursor: pointer; border-radius: 3px; font-size: 13px;
    }
    button:hover { background: #4a4a4a; }
    #btn-toggle               { border-color: #666; }
    #btn-toggle.on            { background: #1a4a1a; border-color: #5f5; color: #5f5; }
    #btn-toggle.off           { background: #4a1a1a; border-color: #f55; color: #f55; }

    #filename { font-weight: bold; }
    #pose      { margin-left: 14px; color: #aaa; }
    #inlier    { margin-left: 14px; }
    #counter   { margin-left: auto; color: #888; }

    .tag-ok   { color: #5f5; }
    .tag-warn { color: #fa0; }
    .tag-bad  { color: #f55; }
    .tag-na   { color: #888; }
  </style>
</head>
<body>
  <div id="viewer-wrap">
    <div id="viewer"></div>
    <div id="horizon"></div>
    <div id="vertical"></div>
  </div>
  <div id="bar">
    <div id="nav">
      <button id="btn-prev">&#8592; Prev</button>
      <button id="btn-next">Next &#8594;</button>
      <button id="btn-toggle" class="on">Correction ON</button>
    </div>
    <span id="filename">—</span>
    <span id="pose"></span>
    <span id="inlier"></span>
    <span id="counter"></span>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/three@0.150.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@photo-sphere-viewer/core@5/index.min.js"></script>
  <script>
    const images = __IMAGES__;
    let current = 0;
    let corrected = true;

    const viewer = new PhotoSphereViewer.Viewer({
      container: document.getElementById('viewer'),
      panorama: images[0].url,
      panoData: poseData(images[0]),
      defaultPitch: 0,
      defaultYaw: 0,
      navbar: false,
      loadingImg: null,
      touchmoveTwoFingers: false,
    });

    function poseData(img) {
      if (!corrected) return { poseRoll: 0, posePitch: 0 };
      return { poseRoll: img.roll ?? 0, posePitch: img.pitch ?? 0 };
    }

    function inlierTag(ratio) {
      if (ratio == null) return '<span class="tag-na">inliers N/A</span>';
      const pct = (ratio * 100).toFixed(0) + '%';
      const cls = ratio >= 0.6 ? 'tag-ok' : ratio >= 0.4 ? 'tag-warn' : 'tag-bad';
      return `<span class="${cls}">inliers ${pct}</span>`;
    }

    function updateToggleButton() {
      const btn = document.getElementById('btn-toggle');
      if (corrected) { btn.textContent = 'Correction ON';  btn.className = 'on'; }
      else           { btn.textContent = 'Correction OFF'; btn.className = 'off'; }
    }

    function applyPose(yaw = 0, pitch = 0) {
      const img = images[current];
      viewer.setPanorama(img.url, {
        panoData: poseData(img),
        transition: false,
        showLoader: false,
      });
      viewer.rotate({ yaw, pitch });
    }

    function navigate(index) {
      current = ((index % images.length) + images.length) % images.length;
      const img = images[current];
      applyPose(0, 0);
      document.getElementById('filename').textContent = img.name;
      const r = img.roll  != null ? (img.roll  >= 0 ? '+' : '') + img.roll.toFixed(2)  + '°' : 'N/A';
      const p = img.pitch != null ? (img.pitch >= 0 ? '+' : '') + img.pitch.toFixed(2) + '°' : 'N/A';
      document.getElementById('pose').textContent = `roll ${r}   pitch ${p}`;
      document.getElementById('inlier').innerHTML = inlierTag(img.inlier_ratio);
      document.getElementById('counter').textContent = `${current + 1} / ${images.length}`;
    }

    function toggleCorrection() {
      const pos = viewer.getPosition();
      const img = images[current];
      // Compensate the pitch offset so the same scene point stays centred.
      // In corrected mode the sphere is rotated by posePitch, so switching modes
      // shifts the apparent pitch by ±posePitch.
      const pitchDelta = (img.pitch ?? 0) * Math.PI / 180;
      corrected = !corrected;
      updateToggleButton();
      applyPose(pos.yaw, pos.pitch + (corrected ? pitchDelta : -pitchDelta));
    }

    document.getElementById('btn-prev').onclick   = () => navigate(current - 1);
    document.getElementById('btn-next').onclick   = () => navigate(current + 1);
    document.getElementById('btn-toggle').onclick = () => toggleCorrection();

    document.addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT') return;
      if (e.key === 'ArrowLeft'  || e.key === 'a') navigate(current - 1);
      if (e.key === 'ArrowRight' || e.key === 'd') navigate(current + 1);
      if (e.key === ' ' || e.key === 'c')          { e.preventDefault(); toggleCorrection(); }
      if (e.key === 'Home') navigate(0);
      if (e.key === 'End')  navigate(images.length - 1);
    });

    navigate(0);
  </script>
</body>
</html>
"""


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
    app = Flask(__name__)
    name_to_path = {Path(img["url"]).name: directory / Path(img["url"]).name for img in images}
    images_json = json.dumps(images)
    html = _HTML.replace("__IMAGES__", images_json)

    @app.route("/")
    def index():
        start = request.args.get("start", "0")
        # inject starting index via a tiny inline script
        page = html.replace("navigate(0);", f"navigate({start});")
        return page

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
    parser.add_argument(
        "--results", type=Path, default=None, metavar="JSON",
        help="Optional geocalib_results.json to show inlier ratios.",
    )
    parser.add_argument("--start", type=int, default=0, metavar="N", help="Start at image index N (default: 0).")
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
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    print(f"Serving {len(images)} images at {url}  (Ctrl-C to stop)")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
