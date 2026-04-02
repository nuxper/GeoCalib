from __future__ import annotations

import io
import logging
import random
import traceback
import uuid

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.middleware.proxy_fix import ProxyFix

from core import (
    MAX_SAMPLE_COUNT,
    ImageCache,
    download_image_bytes,
    load_panoramax_ids,
    predict_360,
)

app = Flask(__name__)

app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_CORS_ORIGINS = {
    "https://leveltest.seen.one",
}

image_cache = ImageCache(max_size=300)
panoramax_ids = []


def error_response(message, status_code=400):
    return jsonify({"status": "error", "error": message}), status_code


@app.route("/")
@app.route("/predict_360")
def predict_360_page():
    return render_template("index.html")


@app.route("/viewer_360")
def viewer_360():
    return render_template("viewer_360.html")


@app.route("/api/random_id")
def get_random_id():
    global panoramax_ids
    if not panoramax_ids:
        panoramax_ids = load_panoramax_ids()
    if panoramax_ids:
        return jsonify({"id": random.choice(panoramax_ids)})
    return error_response("No IDs found", 404)


@app.route("/mem_image/<image_id>")
def get_mem_image(image_id):
    item = image_cache.get(image_id)
    if item:
        return send_file(io.BytesIO(item["data"]), mimetype=item["content_type"])
    return "Image not found", 404


@app.route("/api/predict_360", methods=["POST"])
def api_predict_360():
    from urllib.error import HTTPError, URLError

    if "file" not in request.files and "url" not in request.form:
        return error_response("No file or URL part")

    image_bytes = None
    file = request.files.get("file")
    url = (request.form.get("url") or "").strip()

    if file and file.filename:
        image_bytes = file.read()
    elif url:
        try:
            image_bytes = download_image_bytes(url)
        except (ValueError, HTTPError, URLError, TimeoutError) as e:
            logger.error("Error downloading image from URL: %s", e)
            return error_response(f"Failed to download image: {e}")
        except Exception as e:
            logger.error("Unexpected URL download error: %s", e)
            return error_response(f"Failed to download image: {e}")

    if not image_bytes:
        return error_response("No selected file or valid URL")

    try:
        fov_deg = float(request.form.get("fov", 70))
        inlier_threshold_deg = float(request.form.get("inlier_threshold_deg", 2.0))
        sample_count = int(request.form.get("sample_count", 18))
    except (TypeError, ValueError):
        return error_response("Invalid numeric request parameter")

    if sample_count < 4:
        return error_response("sample_count should be >= 4")
    if sample_count > MAX_SAMPLE_COUNT:
        return error_response(f"sample_count should be <= {MAX_SAMPLE_COUNT}")
    if not (1.0 <= fov_deg <= 179.0):
        return error_response("fov should be between 1 and 179 degrees")
    if not (0.01 <= inlier_threshold_deg <= 90.0):
        return error_response("inlier_threshold_deg should be between 0.01 and 90")

    main_image_id = str(uuid.uuid4())
    image_cache.set(main_image_id, image_bytes)

    try:
        result = predict_360(image_bytes, fov_deg, inlier_threshold_deg, sample_count)
    except ValueError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error("Error processing 360 image: %s", e)
        logger.error(traceback.format_exc())
        return error_response(str(e), 500)

    # Attach cached image IDs for web display (not needed by the core logic)
    import cv2
    import numpy as np

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    import py360convert
    from core import GEOCALIB_VIEW_HW

    sample_yaws = np.linspace(0, 360, sample_count, endpoint=False, dtype=np.float32)
    for entry in result["samples"]:
        idx = entry["index"]
        sample_img = py360convert.e2p(
            img_bgr,
            fov_deg=fov_deg,
            u_deg=float(sample_yaws[idx]),
            v_deg=0.0,
            in_rot_deg=0.0,
            out_hw=GEOCALIB_VIEW_HW,
            mode="bilinear",
        )
        success, buffer = cv2.imencode(".jpg", sample_img)
        if not success:
            return error_response("Failed to encode sampled image", 500)
        sample_image_id = str(uuid.uuid4())
        image_cache.set(sample_image_id, buffer.tobytes())
        entry["image_id"] = sample_image_id

    return jsonify({"status": "success", "main_id": main_image_id, **result})


def apply_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin in ALLOWED_CORS_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        requested_headers = request.headers.get("Access-Control-Request-Headers")
        response.headers["Access-Control-Allow-Headers"] = (
            requested_headers if requested_headers else "Content-Type, Authorization"
        )
        response.headers["Access-Control-Max-Age"] = "86400"
        response.headers.add("Vary", "Origin")
    return response


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        return apply_cors_headers(app.make_default_options_response())
    return None


@app.after_request
def add_header(response):
    response.headers["X-UA-Compatible"] = "IE=Edge,chrome=1"
    response.headers["Cache-Control"] = "public, max-age=0"
    return apply_cors_headers(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
