# geocalib-viewer

Web viewer for 360° equirectangular panoramas with roll/pitch correction.

Displays a collection of JPEGs side by side with a sidebar, lets you toggle the
GeoCalib correction on/off, and adjust roll/pitch manually with sliders.

## Usage

### Local

```bash
# with uv (no install needed)
uv run viewer.py /path/to/images

# or after install
pip install .
geocalib-viewer /path/to/images
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--port PORT` | `5001` | HTTP port |
| `--host HOST` | `127.0.0.1` | Bind address |
| `--results JSON` | auto | Path to `geocalib_results.json` |
| `--start N` | `0` | Open at image index N |
| `--no-browser` | off | Don't open a browser tab |

### Docker

```bash
docker build -t geocalib-viewer .
docker run -v /path/to/images:/data -p 5001:5001 geocalib-viewer
```

Then open `http://localhost:5001` in your browser.

To pass a `geocalib_results.json` alongside the images, place it inside the
mounted directory — it is picked up automatically.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `←` / `→` or `A` / `D` | Previous / next image |
| `Space` or `C` | Toggle correction on/off |
| `Home` / `End` | First / last image |

## Input format

- **Images:** equirectangular JPEGs in the target directory.
- **Results (optional):** a `geocalib_results.json` file (produced by
  `geocalib-360`) placed in the same directory, or passed via `--results`.
  Provides `roll`, `pitch`, and `inlier_ratio` per image.
- **EXIF fallback:** if no JSON is found, roll/pitch are read from
  `XMP-GPano:PoseRollDegrees` / `XMP-GPano:PosePitchDegrees` tags via
  `exiftool` (optional — gracefully skipped if not installed).
