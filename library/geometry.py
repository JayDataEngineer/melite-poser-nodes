"""Shared geometry helpers for the poser renderer pipeline.

Historically the bounding-sphere math (center = bbox centroid,
radius = max vertex distance from center) was inlined in THREE places:

  1. ``rasterize.py::render_passes`` — the depth/normal/rgb renderer.
  2. ``nodes.py::PoserRenderOpenPose.render`` — three branches: mesh input,
     template mesh, joints-only.
  3. ``fit_identity.py::_center_normalize`` — Chamfer fitter (NOT refactored;
     applies normalization inline and returns the centered+scaled verts,
     so it stays as a one-off).

That duplication caused the original A-pose / 67%-pixel-alignment bug
(2026-07-31): ``PoserRenderOpenPose`` used joints×1.15 for the bounding
sphere while ``PoserRender`` used mesh vertices — the two renders came
out at DIFFERENT SCALES and the openpose + depth ControlNets fought each
other. Extracting the math into ONE function makes that class of bug
structurally impossible: there is only one bbox-centroid→radius pipeline.

NOTE: this is the "bounding sphere of the axis-aligned bounding box",
NOT the optimal minimum enclosing sphere (Welzl's algorithm). The
existing camera framing factors are calibrated against THIS
approximation — switching algorithms would silently shift pixel
alignment between depth and OpenPose renders.
"""
from __future__ import annotations

import numpy as np


def bounding_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Center + radius of the bounding sphere enclosing ``points``.

    Returns the centroid of the axis-aligned bounding box as ``center``
    and the maximum Euclidean distance from that center to any point as
    ``radius``. Caller is responsible for the degenerate-input check
    (``radius < 1e-6``) so the error message can include caller-specific
    context (which renderer, which input branch).

    Args:
        points: (N, D) array. Works for any dimensionality but the
            renderers always pass (N, 3) joint/vertex positions.

    Returns:
        (center, radius):
          * ``center`` — (D,) ndarray, the bbox centroid.
          * ``radius`` — Python float, the max distance from center to
            any point. ~0.0 when all points coincide (degenerate).

    Examples:
        >>> import numpy as np
        >>> pts = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]],
        ...                dtype=np.float32)
        >>> center, radius = bounding_sphere(pts)
        >>> center.tolist()
        [1.0, 1.0, 0.0]
        >>> round(radius, 6)
        1.414214
    """
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_max + bbox_min) / 2.0
    centered = points - center
    radius = float(np.max(np.linalg.norm(centered, axis=1)))
    return center, radius
