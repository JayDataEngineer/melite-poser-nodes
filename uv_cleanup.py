"""xatlas UV re-unwrap engine — clean non-overlapping atlas.

TRELLIS's o_voxel ``uv_unwrap`` (cone-clustering) produces an atlas that
overlaps ~29x (measured). nvdiffrast UV-space texture projection then
assigns each texel to an ARBITRARY triangle among the ~5 overlapping
candidates → salt-and-pepper noise. xatlas produces a clean atlas
(total |UV area| ≈ 0.5, no overlap) so each texel maps to exactly one
triangle → coherent projection (VLM-verified 2026-07-25).

This is pure CPU work (xatlas + trimesh) — no model loading, no VRAM.
Lives here so the RayXatlasUnwrap ComfyUI node can call it via relative
import. The deleted HTTP route /poser/trellis/xatlas_unwrap was the
previous wrapper (removed 2026-07-26 per the comfyui-script-only rule).
"""
from __future__ import annotations

import io
import logging

import numpy as np

log = logging.getLogger(__name__)


def run_xatlas_unwrap(glb_bytes: bytes) -> bytes:
    """Re-unwrap a GLB's UVs with xatlas.

    Args:
        glb_bytes: input GLB (any UV state — will be replaced).

    Returns:
        New GLB bytes — same mesh geometry, clean xatlas UVs, a neutral
        4×4 baseColorTexture (the projection step overwrites it).

    Raises:
        RuntimeError: if xatlas fails or the GLB is invalid.
    """
    import trimesh
    import xatlas
    from PIL import Image
    from .texture_project import _normalize_glb_buffers

    mesh = trimesh.load(
        io.BytesIO(_normalize_glb_buffers(glb_bytes)),
        force="mesh", process=False, file_type="glb",
    )
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    atlas = xatlas.Atlas()
    atlas.add_mesh(verts, faces)
    atlas.generate()
    vm, fm, uv = atlas[0]  # vm: new-vert → original-vert; fm, uv: xatlas output

    clean = trimesh.Trimesh(
        vertices=verts[vm], faces=fm, process=False,
        visual=trimesh.visual.TextureVisuals(
            uv=uv.astype(np.float32),
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.new("RGB", (4, 4), (200, 200, 200)),
            ),
        ),
    )

    # Compute atlas quality metric (total UV area, no overlap ⇒ ≈ 0.5)
    uv_faces = uv[fm]
    area = float(abs(
        ((uv_faces[:, 1] - uv_faces[:, 0])[:, 0] * (uv_faces[:, 2] - uv_faces[:, 0])[:, 1]) -
        ((uv_faces[:, 2] - uv_faces[:, 0])[:, 0] * (uv_faces[:, 1] - uv_faces[:, 0])[:, 1])
    ).sum() / 2)
    log.info(
        "[xatlas] %d→%d verts, |UV|=%.3f (TRELLIS native ≈ 29.0)",
        len(verts), len(vm), area,
    )

    out = io.BytesIO()
    clean.export(out, file_type="glb")
    return out.getvalue()
