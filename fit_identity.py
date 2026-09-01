"""Chamfer-distance identity fitter — TRELLIS mesh → Anny phenotype coeffs.

Takes an arbitrary-topology target mesh (e.g. TRELLIS output) and finds the
11 Anny phenotype coefficients that produce the closest-shaped SOMA/Anny
mesh.  Uses pytorch3d.loss.chamfer_distance (GPU) as the loss and
scipy.optimize.minimize (Nelder-Mead) as the optimizer — no gradients
needed through the Anny forward pass (which internally uses torch.no_grad).

All GPU work happens here, inside the ComfyUI container.  The server wraps
calls in asyncio.to_thread().
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)

N_ANNY_PHENOTYPE = 11


def _center_normalize(verts: np.ndarray) -> np.ndarray:
    """Center at origin and scale to unit bounding-sphere radius."""
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    vc = verts - center
    radius = float(np.max(np.linalg.norm(vc, axis=1)))
    if radius < 1e-6:
        return vc
    return vc / radius


def _get_shaped_verts_cm(engine: Any) -> np.ndarray:
    """Extract the identity-shaped vertices (in cm) from the engine after
    ``set_identity`` was called."""
    rest = getattr(engine._model, "_cached_rest_shape", None)
    if rest is None:
        # Fallback: use the stored vertices
        return engine.vertices.copy()
    if hasattr(rest, "detach"):
        rest = rest.detach().cpu().numpy()
    rest = np.asarray(rest, dtype=np.float32)
    if rest.ndim == 3 and rest.shape[0] == 1:
        rest = rest[0]
    # _cached_rest_shape is in meters → centimeters
    return (rest * 100.0).astype(np.float32)


def fit_identity(
    target_vertices: np.ndarray,
    *,
    identity_model: str = "anny",
    max_iter: int = 200,
    n_target_samples: int = 5000,
) -> dict[str, Any]:
    """Fit Anny phenotype coefficients to a target mesh via Chamfer distance.

    Args:
        target_vertices: (M, 3) float32 — arbitrary-topology mesh (e.g. TRELLIS).
        identity_model: "anny" (11 phenotype dims) or "mhr" (45).
        max_iter: Maximum optimizer iterations.
        n_target_samples: Subsample target to this many points for speed.

    Returns:
        {"identity_coeffs": [...], "chamfer_loss": float, "iterations": int,
         "elapsed_s": float}
    """
    from pytorch3d.loss import chamfer_distance
    from scipy.optimize import minimize
    from .skin import get_engine, _PHENOTYPE_DIMS

    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_phen = _PHENOTYPE_DIMS.get(identity_model, N_ANNY_PHENOTYPE)

    # ── Prepare target ────────────────────────────────────────────────
    target_np = np.asarray(target_vertices, dtype=np.float32)
    if target_np.ndim != 2 or target_np.shape[1] != 3:
        raise ValueError(f"target_vertices must be (M, 3), got {target_np.shape}")

    # Subsample if too many points (Chamfer is O(n*m))
    if len(target_np) > n_target_samples:
        idx = np.random.choice(len(target_np), n_target_samples, replace=False)
        target_np = target_np[idx]

    target_norm = _center_normalize(target_np)
    target_t = torch.from_numpy(target_norm).to(device).unsqueeze(0)

    # ── Get engine ────────────────────────────────────────────────────
    engine = get_engine(identity_model)
    _center_normalize(_get_shaped_verts_cm(engine))

    # ── Loss function ─────────────────────────────────────────────────
    eval_count = [0]

    def loss_fn(coeffs_np: np.ndarray) -> float:
        eval_count[0] += 1
        coeffs_clipped = np.clip(coeffs_np, 0.0, 1.0)
        try:
            engine.set_identity(coeffs_clipped.tolist())
        except Exception as e:
            log.warning("set_identity failed during fit: %s", e)
            return 1e6
        shaped = _get_shaped_verts_cm(engine)
        shaped_norm = _center_normalize(shaped)
        shaped_t = torch.from_numpy(shaped_norm).to(device).unsqueeze(0)
        with torch.no_grad():
            loss_val, _ = chamfer_distance(shaped_t, target_t)
        return float(loss_val.item())

    # ── Run optimizer ─────────────────────────────────────────────────
    x0 = np.full(n_phen, 0.5, dtype=np.float64)  # neutral midpoint

    result = minimize(
        loss_fn,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": max_iter,
            "xatol": 0.01,
            "fatol": 1e-5,
            "adaptive": True,
        },
    )

    best_coeffs = np.clip(result.x, 0.0, 1.0).tolist()
    elapsed = time.perf_counter() - t0

    log.info(
        "fit_identity(%s): %d evals, chamfer=%.6f, %d iters, %.1fs",
        identity_model, eval_count[0], result.fun, result.nit, elapsed,
    )

    return {
        "identity_coeffs": best_coeffs,
        "chamfer_loss": float(result.fun),
        "iterations": int(result.nit),
        "evaluations": eval_count[0],
        "elapsed_s": round(elapsed, 2),
    }
