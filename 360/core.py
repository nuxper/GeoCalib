from __future__ import annotations

import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from urllib.request import Request, urlopen

from typing import Any

import cv2
import numpy as np
import py360convert
import torch
from geocalib import GeoCalib

logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = GeoCalib().to(DEVICE)
MODEL.eval()
MODEL_LOCK = Lock()
PANORAMAX_IDS_PATH = Path(__file__).resolve().parent / "random_panoramax_bike_ids.txt"
MAX_SAMPLE_COUNT = 144
INFERENCE_MICROBATCH_SIZE = 16


def _get_geocalib_view_hw(model: GeoCalib) -> tuple[int, int]:
    """Return the (H, W) input resolution expected by the GeoCalib image processor."""
    resize = getattr(model.image_processor.conf, "resize", 320)
    if isinstance(resize, int):
        return (int(resize), int(resize))
    if isinstance(resize, (tuple, list)) and len(resize) == 2:
        return (int(resize[0]), int(resize[1]))
    logger.warning("Unexpected GeoCalib resize config %s. Falling back to 320x320.", resize)
    return (320, 320)


GEOCALIB_VIEW_HW = _get_geocalib_view_hw(MODEL)


def _finite_float_or_none(value: float) -> float | None:
    """Return value as float, or None if it is not finite (NaN / ±inf)."""
    value = float(value)
    return value if np.isfinite(value) else None


class ImageCache:
    """LRU in-memory cache for raw image bytes, keyed by arbitrary string IDs."""

    def __init__(self, max_size: int = 50) -> None:
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.max_size = max_size

    def set(self, key: str, data: bytes, content_type: str = "image/jpeg") -> None:
        """Insert or refresh an entry, evicting the oldest if the cache is full."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = {"data": data, "content_type": content_type}
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached entry for key, or None if not present."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None


def load_panoramax_ids() -> list[str]:
    """Load and return Panoramax image IDs from the bundled text file."""
    if not PANORAMAX_IDS_PATH.exists():
        logger.warning("Panoramax ID file not found at %s", PANORAMAX_IDS_PATH)
        return []
    with PANORAMAX_IDS_PATH.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def download_image_bytes(url: str) -> bytes:
    """Download an image from url and return its raw bytes.

    Raises ValueError if the response body is empty.
    """
    req = Request(url, headers={"User-Agent": "GeoCalib-360/1.0"})
    with urlopen(req, timeout=30) as response:
        data = response.read()
        if not data:
            raise ValueError("Downloaded response was empty.")
        return data


def numpy_rgb_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    """Convert an HxWx3 uint8 RGB array to a normalised CxHxW float tensor in [0, 1]."""
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape [H, W, 3].")
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    image_rgb = np.ascontiguousarray(image_rgb)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255.0)
    return tensor


def _expand_or_fill(tensor: torch.Tensor | None, size: int, fill_value: float) -> np.ndarray:
    """Convert a radian uncertainty tensor to a degree array of length size.

    If tensor is None or has an unexpected shape, returns an array filled with fill_value.
    """
    if tensor is None:
        return np.full(size, fill_value, dtype=np.float32)

    values = torch.rad2deg(tensor.detach().cpu()).reshape(-1).numpy().astype(np.float32)
    if values.size == size:
        return values
    if values.size == 1:
        return np.full(size, float(values[0]), dtype=np.float32)
    logger.warning("Unexpected uncertainty shape %s for %d samples.", values.shape, size)
    return np.full(size, fill_value, dtype=np.float32)


def estimate_rp_batch(
    sample_bgr_images: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run GeoCalib inference on a list of BGR crops.

    Returns (predicted_rolls, predicted_pitches, roll_unc_deg, pitch_unc_deg),
    all as float32 arrays of length len(sample_bgr_images), in degrees.
    """
    sample_tensors = [
        numpy_rgb_to_tensor(cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB)) for sample_bgr in sample_bgr_images
    ]
    sample_batch = torch.stack(sample_tensors, dim=0).to(DEVICE, non_blocking=True)
    sample_count = sample_batch.shape[0]

    predicted_roll_chunks = []
    predicted_pitch_chunks = []
    roll_unc_chunks = []
    pitch_unc_chunks = []

    with MODEL_LOCK:
        with torch.no_grad():
            for start in range(0, sample_count, INFERENCE_MICROBATCH_SIZE):
                end = min(start + INFERENCE_MICROBATCH_SIZE, sample_count)
                micro_batch = sample_batch[start:end]
                result = MODEL.calibrate(
                    micro_batch, camera_model="pinhole", shared_intrinsics=True
                )

                rp_deg = torch.rad2deg(result["gravity"].rp.detach().cpu()).numpy().astype(np.float32)
                rp_deg = np.nan_to_num(rp_deg, nan=0.0, posinf=89.9, neginf=-89.9)
                predicted_roll_chunks.append(rp_deg[:, 0])
                predicted_pitch_chunks.append(rp_deg[:, 1])
                roll_unc = _expand_or_fill(result.get("roll_uncertainty"), end - start, 3.0)
                roll_unc = np.nan_to_num(roll_unc, nan=3.0, posinf=30.0, neginf=3.0)
                pitch_unc = _expand_or_fill(result.get("pitch_uncertainty"), end - start, 3.0)
                pitch_unc = np.nan_to_num(pitch_unc, nan=3.0, posinf=30.0, neginf=3.0)
                roll_unc_chunks.append(np.maximum(roll_unc, 0.1))
                pitch_unc_chunks.append(np.maximum(pitch_unc, 0.1))

                del result, micro_batch

    predicted_rolls = np.concatenate(predicted_roll_chunks, axis=0)
    predicted_pitches = np.concatenate(predicted_pitch_chunks, axis=0)
    roll_unc_deg = np.concatenate(roll_unc_chunks, axis=0)
    pitch_unc_deg = np.concatenate(pitch_unc_chunks, axis=0)
    predicted_rolls = np.nan_to_num(predicted_rolls, nan=0.0, posinf=89.9, neginf=-89.9)
    predicted_pitches = np.nan_to_num(predicted_pitches, nan=0.0, posinf=89.9, neginf=-89.9)
    roll_unc_deg = np.maximum(np.nan_to_num(roll_unc_deg, nan=3.0, posinf=30.0, neginf=3.0), 0.1)
    pitch_unc_deg = np.maximum(np.nan_to_num(pitch_unc_deg, nan=3.0, posinf=30.0, neginf=3.0), 0.1)
    return predicted_rolls, predicted_pitches, roll_unc_deg, pitch_unc_deg


def _normalize_rad(rad: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle (in radians) to the range (−π, π]."""
    return (rad + np.pi) % (2 * np.pi) - np.pi


def _angular_diff_deg_vec(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Element-wise shortest angular distance between two degree arrays, in [0, 180]."""
    return np.abs((a_deg - b_deg + 180.0) % 360.0 - 180.0).astype(np.float32)


def _model_roll_deg(alpha_rad: float, beta_rad: float, yaw_rads: np.ndarray) -> np.ndarray:
    """Predicted roll (degrees) for a camera tilted by (alpha, beta) at each yaw angle."""
    return np.rad2deg(np.arctan(np.tan(beta_rad) * np.cos(yaw_rads - alpha_rad)))


def _model_pitch_deg(alpha_rad: float, beta_rad: float, yaw_rads: np.ndarray) -> np.ndarray:
    """Predicted pitch (degrees) for a camera tilted by (alpha, beta) at each yaw angle."""
    return np.rad2deg(-np.arctan(np.tan(beta_rad) * np.sin(yaw_rads - alpha_rad)))


def _extract_samples(img_bgr: np.ndarray, sample_yaws: np.ndarray, fov_deg: float) -> list[np.ndarray]:
    """Extract perspective crops from an equirectangular BGR image at each yaw in sample_yaws."""
    sample_count = len(sample_yaws)

    def extract_one(idx, yaw_deg):
        return idx, py360convert.e2p(
            img_bgr,
            fov_deg=fov_deg,
            u_deg=float(yaw_deg),
            v_deg=0.0,
            in_rot_deg=0.0,
            out_hw=GEOCALIB_VIEW_HW,
            mode="bilinear",
        )

    samples = [None] * sample_count
    with ThreadPoolExecutor(max_workers=min(8, sample_count)) as executor:
        for idx, img in executor.map(lambda args: extract_one(*args), enumerate(sample_yaws)):
            samples[idx] = img
    return samples


def _evaluate_hypothesis(
    alpha_rad: float,
    beta_rad: float,
    yaw_rads: np.ndarray,
    predicted_rolls: np.ndarray,
    predicted_pitches: np.ndarray,
    roll_unc_deg: np.ndarray,
    pitch_unc_deg: np.ndarray,
    inlier_threshold_deg: float,
) -> dict[str, Any]:
    """Score a tilt hypothesis (alpha, beta) against GeoCalib predictions.

    alpha_rad is the azimuth of the tilt axis; beta_rad is the tilt magnitude.
    Returns a dict with modeled angles, per-sample errors, inlier mask, and
    aggregate statistics (mae, rmse, weighted_mse).
    """
    modeled_rolls = np.nan_to_num(
        _model_roll_deg(alpha_rad, beta_rad, yaw_rads).astype(np.float32),
        nan=0.0, posinf=90.0, neginf=-90.0,
    )
    modeled_pitches = np.nan_to_num(
        _model_pitch_deg(alpha_rad, beta_rad, yaw_rads).astype(np.float32),
        nan=0.0, posinf=90.0, neginf=-90.0,
    )
    roll_errors = np.nan_to_num(
        _angular_diff_deg_vec(predicted_rolls, modeled_rolls), nan=180.0, posinf=180.0, neginf=180.0
    )
    pitch_errors = np.nan_to_num(
        _angular_diff_deg_vec(predicted_pitches, modeled_pitches), nan=180.0, posinf=180.0, neginf=180.0
    )
    combined_errors = np.nan_to_num(
        np.sqrt((roll_errors**2 + pitch_errors**2) / 2.0).astype(np.float32),
        nan=180.0, posinf=180.0, neginf=180.0,
    )
    weighted_sq_errors = np.nan_to_num(
        0.5 * ((roll_errors / roll_unc_deg) ** 2 + (pitch_errors / pitch_unc_deg) ** 2),
        nan=1e6, posinf=1e6, neginf=1e6,
    ).astype(np.float32)
    inliers = combined_errors <= inlier_threshold_deg
    inlier_count = int(np.sum(inliers))
    mae = float(np.mean(combined_errors[inliers])) if inlier_count > 0 else float("inf")
    rmse = float(np.sqrt(np.mean(np.square(combined_errors[inliers])))) if inlier_count > 0 else float("inf")
    weighted_mse = float(np.mean(weighted_sq_errors[inliers])) if inlier_count > 0 else float("inf")
    return {
        "alpha_rad": float(_normalize_rad(alpha_rad)),
        "beta_rad": float(beta_rad),
        "modeled_rolls": modeled_rolls,
        "modeled_pitches": modeled_pitches,
        "roll_errors": roll_errors,
        "pitch_errors": pitch_errors,
        "weighted_errors": np.sqrt(weighted_sq_errors).astype(np.float32),
        "errors": combined_errors,
        "inliers": inliers,
        "inlier_count": inlier_count,
        "mae": mae,
        "rmse": rmse,
        "weighted_mse": weighted_mse,
    }


def _ransac_best_hypothesis(
    yaw_rads: np.ndarray,
    predicted_rolls: np.ndarray,
    predicted_pitches: np.ndarray,
    roll_unc_deg: np.ndarray,
    pitch_unc_deg: np.ndarray,
    inlier_threshold_deg: float,
) -> dict[str, Any] | None:
    """Generate all pairwise hypotheses and return the best one.

    Each pair of samples (i, j) analytically determines up to two (alpha, beta)
    candidates. The best hypothesis maximises inlier count, then minimises
    weighted MSE, then MAE. Returns None if no valid pair was found.
    """
    tiny = 1e-6
    tan_rolls = np.tan(np.deg2rad(predicted_rolls))
    sample_count = len(yaw_rads)
    best = None

    for i in range(sample_count):
        for j in range(i + 1, sample_count):
            ai, aj = tan_rolls[i], tan_rolls[j]
            theta_i, theta_j = yaw_rads[i], yaw_rads[j]
            if not np.isfinite(ai) or not np.isfinite(aj):
                continue
            p = ai * np.cos(theta_j) - aj * np.cos(theta_i)
            q = ai * np.sin(theta_j) - aj * np.sin(theta_i)
            if not np.isfinite(p) or not np.isfinite(q) or (abs(p) < tiny and abs(q) < tiny):
                continue

            alpha_candidate_1 = np.arctan2(-p, q)
            for alpha_candidate in (alpha_candidate_1, alpha_candidate_1 + np.pi):
                if not np.isfinite(alpha_candidate):
                    continue
                cos_term = np.cos(theta_i - alpha_candidate)
                if abs(cos_term) < tiny:
                    continue
                tan_beta = ai / cos_term
                if not np.isfinite(tan_beta):
                    continue
                beta_candidate = np.arctan(tan_beta)
                if not np.isfinite(beta_candidate):
                    continue

                h = _evaluate_hypothesis(
                    alpha_candidate, beta_candidate,
                    yaw_rads, predicted_rolls, predicted_pitches, roll_unc_deg, pitch_unc_deg, inlier_threshold_deg,
                )
                if best is None:
                    best = h
                elif h["inlier_count"] > best["inlier_count"]:
                    best = h
                elif h["inlier_count"] == best["inlier_count"] and h["weighted_mse"] < best["weighted_mse"]:
                    best = h
                elif (
                    h["inlier_count"] == best["inlier_count"]
                    and np.isclose(h["weighted_mse"], best["weighted_mse"])
                    and h["mae"] < best["mae"]
                ):
                    best = h

    return best


def _refine_hypothesis(
    best: dict[str, Any],
    yaw_rads: np.ndarray,
    predicted_rolls: np.ndarray,
    predicted_pitches: np.ndarray,
    roll_unc_deg: np.ndarray,
    pitch_unc_deg: np.ndarray,
    inlier_threshold_deg: float,
) -> dict[str, Any]:
    """Refine the RANSAC hypothesis with a two-pass grid search over inliers.

    Pass 1: ±6° grid at 0.5° steps. Pass 2: ±1° grid at 0.1° steps.
    Returns best unchanged if there are fewer than 2 inliers.
    """
    inlier_indices = np.where(best["inliers"])[0]
    if len(inlier_indices) < 2:
        return best

    inlier_roll_unc = roll_unc_deg[inlier_indices]
    inlier_pitch_unc = pitch_unc_deg[inlier_indices]

    def objective(alpha_rad, beta_rad):
        roll_errors = np.nan_to_num(
            _angular_diff_deg_vec(predicted_rolls[inlier_indices], _model_roll_deg(alpha_rad, beta_rad, yaw_rads[inlier_indices])),
            nan=180.0, posinf=180.0, neginf=180.0,
        )
        pitch_errors = np.nan_to_num(
            _angular_diff_deg_vec(predicted_pitches[inlier_indices], _model_pitch_deg(alpha_rad, beta_rad, yaw_rads[inlier_indices])),
            nan=180.0, posinf=180.0, neginf=180.0,
        )
        val = np.mean(0.5 * ((roll_errors / inlier_roll_unc) ** 2 + (pitch_errors / inlier_pitch_unc) ** 2))
        return float(val) if np.isfinite(val) else 1e6

    alpha_center, beta_center = best["alpha_rad"], best["beta_rad"]
    best_obj = objective(alpha_center, beta_center)

    for step, half_range in ((0.5, 6.0), (0.1, 1.0)):
        grid = np.deg2rad(np.arange(-half_range, half_range + 1e-9, step))
        for da in grid:
            for db in grid:
                alpha_try = _normalize_rad(alpha_center + da)
                beta_try = np.clip(beta_center + db, np.deg2rad(-89.0), np.deg2rad(89.0))
                obj = objective(alpha_try, beta_try)
                if obj < best_obj:
                    best_obj = obj
                    alpha_center, beta_center = alpha_try, beta_try

    return _evaluate_hypothesis(
        alpha_center, beta_center,
        yaw_rads, predicted_rolls, predicted_pitches, roll_unc_deg, pitch_unc_deg, inlier_threshold_deg,
    )


def _build_result(
    best: dict[str, Any],
    sample_yaws: np.ndarray,
    predicted_rolls: np.ndarray,
    predicted_pitches: np.ndarray,
    fov_deg: float,
    inlier_threshold_deg: float,
) -> dict[str, Any]:
    """Assemble the final result dict from the refined hypothesis."""
    sample_count = len(sample_yaws)
    tan_beta = np.tan(best["beta_rad"])
    roll_deg = float(np.rad2deg(np.arctan(tan_beta * np.cos(best["alpha_rad"]))))
    pitch_deg = float(np.rad2deg(np.arctan(tan_beta * np.sin(best["alpha_rad"]))))

    samples = [
        {
            "index": idx,
            "yaw_deg": _finite_float_or_none(sample_yaws[idx]),
            "predicted_roll_deg": _finite_float_or_none(predicted_rolls[idx]),
            "modeled_roll_deg": _finite_float_or_none(best["modeled_rolls"][idx]),
            "predicted_pitch_deg": _finite_float_or_none(predicted_pitches[idx]),
            "modeled_pitch_deg": _finite_float_or_none(best["modeled_pitches"][idx]),
            "roll_error_deg": _finite_float_or_none(best["roll_errors"][idx]),
            "pitch_error_deg": _finite_float_or_none(best["pitch_errors"][idx]),
            "weighted_error": _finite_float_or_none(best["weighted_errors"][idx]),
            "error_deg": _finite_float_or_none(best["errors"][idx]),
            "inlier": bool(best["inliers"][idx]),
        }
        for idx in range(sample_count)
    ]

    return {
        "roll": _finite_float_or_none(roll_deg),
        "pitch": _finite_float_or_none(pitch_deg),
        "alpha_deg": _finite_float_or_none(float(np.rad2deg(best["alpha_rad"]))),
        "beta_deg": _finite_float_or_none(float(np.rad2deg(best["beta_rad"]))),
        "sample_count": sample_count,
        "fov_deg": fov_deg,
        "inlier_threshold_deg": inlier_threshold_deg,
        "inlier_count": int(best["inlier_count"]),
        "inlier_ratio": _finite_float_or_none(best["inlier_count"] / sample_count),
        "mae_inlier_deg": _finite_float_or_none(best["mae"]),
        "rmse_inlier_deg": _finite_float_or_none(best["rmse"]),
        "samples": samples,
    }


def predict_360(
    image_bytes: bytes,
    fov_deg: float = 70.0,
    inlier_threshold_deg: float = 2.0,
    sample_count: int = 18,
) -> dict[str, Any]:
    """
    Estimate the pitch and roll of a 360° equirectangular image.

    Returns a dict with keys: roll, pitch, alpha_deg, beta_deg,
    inlier_count, inlier_ratio, mae_inlier_deg, rmse_inlier_deg, samples.

    Raises ValueError on invalid input or failure to find a stable hypothesis.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image")

    sample_yaws = np.linspace(0, 360, sample_count, endpoint=False, dtype=np.float32)
    sample_images = _extract_samples(img_bgr, sample_yaws, fov_deg)

    predicted_rolls, predicted_pitches, roll_unc_deg, pitch_unc_deg = estimate_rp_batch(sample_images)
    yaw_rads = np.deg2rad(sample_yaws)

    best = _ransac_best_hypothesis(yaw_rads, predicted_rolls, predicted_pitches, roll_unc_deg, pitch_unc_deg, inlier_threshold_deg)
    if best is None:
        raise ValueError("Unable to estimate a stable hypothesis")

    best = _refine_hypothesis(best, yaw_rads, predicted_rolls, predicted_pitches, roll_unc_deg, pitch_unc_deg, inlier_threshold_deg)

    return _build_result(best, sample_yaws, predicted_rolls, predicted_pitches, fov_deg, inlier_threshold_deg)
