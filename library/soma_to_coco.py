"""SOMA-77 → COCO-18 projection utilities.

Vendored from the deleted ``gateway/poser_soma.py`` so the projection
pipeline (used to produce the ``frames_flat`` preview-render field in
text-to-pose responses) lives inside the inference-comfyui container.

Pure numpy — no torch, no matplotlib, no PIL. Operates on numpy arrays.

SOMA-77 is a superset of COCO-18 (77 joints vs 18). 16 of the 18 COCO
joints have direct SOMA counterparts. The two ear joints do not exist
in SOMA-77 and are extrapolated from the eye + head geometry.

Coordinate conventions:
  SOMA frame: +X = subject's left, +Y = up, +Z = forward (subject faces +Z)
  Image frame: 0,0 = top-left, +X = right, +Y = down
  Subject's right = image LEFT (COCO convention)
"""
from __future__ import annotations

from typing import Iterable, Literal

import numpy as np

from .skeleton import (
    DIRECT_COCO_TO_SOMA_INDICES,
    SOMA77_IDX,
)


# ── Ear extrapolation ──────────────────────────────────────────────────────
# SOMA-77 has no ear joints. We synthesize them from the eye geometry.
# Adult human ears sit ~7cm behind and slightly below the eyes, at roughly
# half the interocular distance lateral to the eye center. We use the
# interocular distance as a scale reference so the extrapolation adapts
# to different body sizes.
EAR_LATERAL_RATIO: float = 0.65
EAR_VERTICAL_DROP_RATIO: float = 0.20


def extrapolate_ears(joints_3d: np.ndarray) -> np.ndarray:
    """Compute right and left ear 3D positions from eye + head geometry.

    Args:
        joints_3d: (J, 3) or (T, J, 3) SOMA-77 joint positions, in meters.

    Returns:
        (2, 3) or (T, 2, 3) — [right_ear, left_ear] 3D positions.
    """
    single = joints_3d.ndim == 2
    if single:
        joints_3d = joints_3d[np.newaxis, ...]

    left_eye = joints_3d[:, SOMA77_IDX["LeftEye"]]
    right_eye = joints_3d[:, SOMA77_IDX["RightEye"]]

    interocular = right_eye - left_eye
    interocular_dist = np.linalg.norm(interocular, axis=-1, keepdims=True)
    interocular_dist = np.maximum(interocular_dist, 1e-6)
    eye_unit = interocular / interocular_dist

    head = joints_3d[:, SOMA77_IDX["Head"]]
    neck1 = joints_3d[:, SOMA77_IDX["Neck1"]]
    head_up = head - neck1
    head_up_norm = np.maximum(np.linalg.norm(head_up, axis=-1, keepdims=True), 1e-6)
    head_up_unit = head_up / head_up_norm

    lateral = eye_unit * (EAR_LATERAL_RATIO * interocular_dist)
    downward = -head_up_unit * (EAR_VERTICAL_DROP_RATIO * interocular_dist)

    right_ear = right_eye + lateral + downward
    left_ear = left_eye - lateral + downward

    result = np.stack([right_ear, left_ear], axis=1)

    if single:
        return result[0]
    return result


def project_soma77_to_coco18_3d(joints_3d: np.ndarray) -> np.ndarray:
    """Project SOMA-77 3D joints to COCO-18 3D joints (view-independent)."""
    if joints_3d.ndim not in (2, 3):
        raise ValueError(f"Expected (J,3) or (T,J,3), got shape {joints_3d.shape}")
    if joints_3d.shape[-1] != 3:
        raise ValueError(f"Last dim must be 3, got {joints_3d.shape[-1]}")
    if joints_3d.shape[-2] != 77:
        raise ValueError(f"Expected 77 SOMA joints, got {joints_3d.shape[-2]}")

    single = joints_3d.ndim == 2
    if single:
        joints_3d = joints_3d[np.newaxis, ...]

    T = joints_3d.shape[0]
    out = np.zeros((T, 18, 3), dtype=np.float32)
    for coco_idx, soma_idx in enumerate(DIRECT_COCO_TO_SOMA_INDICES):
        out[:, coco_idx, :] = joints_3d[:, soma_idx, :]
    ears = extrapolate_ears(joints_3d)
    out[:, 16, :] = ears[:, 0, :]
    out[:, 17, :] = ears[:, 1, :]

    if single:
        return out[0]
    return out


CameraView = Literal["front", "side", "three-quarter"]

_CAMERA_PARAMS: dict[CameraView, tuple[float, float]] = {
    "front":          (0.0,   0.0),
    "side":           (0.0,  90.0),
    "three-quarter": (10.0,  35.0),
}


def _rotation_matrix(elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = np.radians(elev_deg)
    azim = np.radians(azim_deg)
    ce, se = np.cos(elev), np.sin(elev)
    ca, sa = np.cos(azim), np.sin(azim)
    Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]], dtype=np.float32)
    Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float32)
    return Ry @ Rx


def project_3d_to_2d(
    joints_3d: np.ndarray,
    width: int = 1024,
    height: int = 1024,
    camera_view: CameraView = "front",
    pad_fraction: float = 0.08,
) -> np.ndarray:
    """Project 3D joints to 2D image coordinates via orthographic projection."""
    if camera_view not in _CAMERA_PARAMS:
        raise ValueError(
            f"Unknown camera_view {camera_view!r}; expected one of {list(_CAMERA_PARAMS)}"
        )

    single = joints_3d.ndim == 2
    if single:
        joints_3d = joints_3d[np.newaxis, ...]

    elev, azim = _CAMERA_PARAMS[camera_view]
    R = _rotation_matrix(elev, azim)

    rotated = joints_3d @ R.T
    xy = rotated[..., :2]

    xy_min = xy.min(axis=(0, 1))
    xy_max = xy.max(axis=(0, 1))
    span = np.maximum(xy_max - xy_min, 1e-6)
    avail_w = width * (1 - 2 * pad_fraction)
    avail_h = height * (1 - 2 * pad_fraction)
    scale = min(avail_w / span[0], avail_h / span[1])

    centered = (xy - (xy_min + xy_max) / 2) * scale
    px = (centered[..., 0] + width / 2).astype(np.float32)
    py = (height / 2 - centered[..., 1]).astype(np.float32)
    out = np.stack([px, py], axis=-1)

    if single:
        return out[0]
    return out


def build_coco18_keypoints(
    coco18_3d: np.ndarray,
    width: int = 1024,
    height: int = 1024,
    camera_view: CameraView = "front",
    confidence: float = 0.9,
    ear_confidence: float = 0.5,
) -> np.ndarray:
    """Project COCO-18 3D joints to flat [x, y, c, ...] keypoint arrays."""
    single = coco18_3d.ndim == 2
    if single:
        coco18_3d = coco18_3d[np.newaxis, ...]

    xy = project_3d_to_2d(coco18_3d, width=width, height=height, camera_view=camera_view)

    T = coco18_3d.shape[0]
    out = np.zeros((T, 18, 3), dtype=np.float32)
    out[..., :2] = xy
    out[:, :16, 2] = confidence
    out[:, 16:, 2] = ear_confidence

    flat = out.reshape(T, -1)

    if single:
        return flat[0]
    return flat


FrameStrategy = Literal["auto_peak", "auto_mid", "manual", "all"]


def pick_frame(
    motion_3d: np.ndarray,
    strategy: FrameStrategy = "auto_peak",
    manual_indices: Iterable[int] | None = None,
) -> list[int]:
    """Select frame indices from a motion sequence per a strategy."""
    if motion_3d.ndim != 3:
        raise ValueError(f"Expected (T, J, 3) motion, got shape {motion_3d.shape}")
    T = motion_3d.shape[0]
    if T == 0:
        raise ValueError("Empty motion — 0 frames")
    if T == 1:
        return [0]
    if strategy == "all":
        return list(range(T))
    if strategy == "manual":
        if not manual_indices:
            raise ValueError("strategy='manual' requires manual_indices")
        idxs = sorted(set(int(i) for i in manual_indices))
        for i in idxs:
            if not (0 <= i < T):
                raise ValueError(f"Manual frame index {i} out of range [0, {T})")
        return idxs
    if strategy == "auto_mid":
        return [T // 2]
    if strategy == "auto_peak":
        lo = max(1, int(T * 0.2))
        hi = min(T - 1, int(T * 0.8))
        if hi <= lo:
            return [T // 2]
        window = motion_3d[lo:hi + 1]
        centroids = window.mean(axis=1, keepdims=True)
        dispersion = ((window - centroids) ** 2).sum(axis=(1, 2))
        peak = lo + int(np.argmax(dispersion))
        return [peak]
    raise ValueError(f"Unknown frame strategy: {strategy!r}")


def motion_to_coco18(
    soma_motion: np.ndarray,
    width: int = 1024,
    height: int = 1024,
    camera_view: CameraView = "front",
    confidence: float = 0.9,
    ear_confidence: float = 0.5,
) -> dict:
    """Project a full SOMA-77 motion sequence to COCO-18 keypoints."""
    if soma_motion.ndim != 3:
        raise ValueError(f"Expected (T, 77, 3), got shape {soma_motion.shape}")
    if soma_motion.shape[1] != 77 or soma_motion.shape[2] != 3:
        raise ValueError(f"Expected (T, 77, 3), got shape {soma_motion.shape}")

    coco18_3d = project_soma77_to_coco18_3d(soma_motion)
    keypoints_flat = build_coco18_keypoints(
        coco18_3d,
        width=width, height=height,
        camera_view=camera_view,
        confidence=confidence,
        ear_confidence=ear_confidence,
    )
    coco18_2d = keypoints_flat.reshape(-1, 18, 3)[..., :2]
    peak = pick_frame(soma_motion, strategy="auto_peak")[0]

    return {
        "coco18_3d": coco18_3d,
        "coco18_2d": coco18_2d,
        "keypoints_flat": keypoints_flat,
        "peak_frame_idx": peak,
        "frame_count": int(soma_motion.shape[0]),
    }
