"""Texture PROJECTION helpers (post-migration — 2026-07-28).

This module previously held the hand-rolled nvdiffrast UV-space texture
projection algorithm (``project_views_to_texture``). That algorithm is
DELETED — replaced by the upstream Comfy3D (MrForExample)
``[Comfy3D] ExplicitTarget Color Projection`` node, driven by
``media/comfyui/families/texture_project.py``.

WHY THE MIGRATION:
  The hand-rolled algorithm had a camera-framing bug — ``orbit_radius=1.75``
  made the unit-normalized mesh fill only 65% of the camera frustum, but
  the qwen-edit source body images had the character filling 100% of the
  frame. The projection sampled chest pixels where face pixels were
  expected (face was completely missing from the output texture). The
  upstream node with ``orbit_radius=1.14`` (= ``0.5 / tan(fovy/2)``) makes
  the mesh fill the frustum vertically the same way the source images
  fill their frames, producing a correct projection.

WHAT REMAINS HERE:
  * ``_normalize_glb_buffers`` — used by ``uv_cleanup.py`` (xatlas path,
    which itself is deprecated but still imported by some old code paths).
  * ``_view_camera``, ``dr_rasterize``, ``dr_interpolate``,
    ``_image_to_tensor``, ``_fg_bbox``, ``render_textured_views`` — used
    by ``server.py``'s debug render route (``/poser/multiview/render``)
    to render a textured GLB from arbitrary angles. This is a SEPARATE
    use case from projection (we're rendering FROM the textured mesh,
    not projecting ONTO it), so it stays.

Reuses melite-poser-nodes/rasterize.py for the nvdiffrast context + camera math.
All synchronous GPU work — the server wraps calls in asyncio.to_thread().
"""
from __future__ import annotations

import io
import json
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

from .rasterize import (
    _dev, _get_glctx, _look_at, _perspective,
    _vertex_normals, _consolidate_normals,
)

log = logging.getLogger(__name__)


def _normalize_glb_buffers(glb_bytes: bytes) -> bytes:
    """Fix GLB BIN-chunk / bufferView length mismatches.

    SOMAX and the motion-bake export GLBs whose BIN chunk can be shorter than
    the bufferViews declare (trailing-accessor padding stripped, or buffer
    length metadata stale). trimesh's ``_read_buffers`` hard-asserts
    ``len(slice) == view['byteLength']`` and crashes on these. This function
    pads the BIN chunk with zeros so every bufferView fits, and refreshes the
    ``buffers[0].byteLength`` in the JSON chunk to match.
    """
    import struct as _struct
    if glb_bytes[:4] != b'glTF':
        return glb_bytes  # not a GLB — let trimesh handle it
    version, total_len = _struct.unpack_from('<II', glb_bytes, 4)
    # Parse chunks
    offset = 12
    json_chunk = None
    bin_chunk = None
    chunks_meta = []  # (data_offset, data_len, chunk_type)
    while offset < total_len:
        clen, ctype = _struct.unpack_from('<II', glb_bytes, offset)
        data_off = offset + 8
        chunks_meta.append((data_off, clen, ctype))
        if ctype == 0x4E4F534A:  # JSON
            json_chunk = json.loads(glb_bytes[data_off:data_off + clen])
        elif ctype == 0x004E4942:  # BIN
            bin_chunk = glb_bytes[data_off:data_off + clen]
        offset += 8 + clen + ((4 - (clen % 4)) % 4)  # 4-byte aligned
    if json_chunk is None or bin_chunk is None:
        return glb_bytes  # can't fix — let trimesh fail with a clear error
    # Find max extent across all bufferViews
    buffer_views = json_chunk.get('bufferViews', [])
    if not buffer_views:
        return glb_bytes
    max_extent = max(
        bv.get('byteOffset', 0) + bv.get('byteLength', 0)
        for bv in buffer_views
    )
    if len(bin_chunk) >= max_extent:
        return glb_bytes  # already fine
    # Pad the BIN chunk
    padded_bin = bin_chunk + b'\x00' * (max_extent - len(bin_chunk))
    # Align to 4 bytes
    pad_align = (4 - (len(padded_bin) % 4)) % 4
    padded_bin += b'\x00' * pad_align
    # Update buffers[0].byteLength
    if json_chunk.get('buffers'):
        json_chunk['buffers'][0]['byteLength'] = max_extent
    # Rebuild the GLB
    json_data = json.dumps(json_chunk, separators=(',', ':')).encode('utf-8')
    # Pad JSON to 4-byte alignment with spaces (valid JSON whitespace)
    json_pad = (4 - (len(json_data) % 4)) % 4
    json_data += b' ' * json_pad
    new_total = 12 + 8 + len(json_data) + 8 + len(padded_bin)
    out = bytearray()
    out += b'glTF'
    out += _struct.pack('<II', 2, new_total)
    out += _struct.pack('<II', len(json_data), 0x4E4F534A)
    out += json_data
    out += _struct.pack('<II', len(padded_bin), 0x004E4942)
    out += padded_bin
    log.info("[texture_project] padded BIN chunk %d → %d bytes (bufferView fix)",
             len(bin_chunk), len(padded_bin))
    return bytes(out)


# ── REMOVED 2026-07-28 ──────────────────────────────────────────────────
# The hand-rolled texture projection algorithm lived here:
#   * project_views_to_texture (nvdiffrast UV-rasterization)
#   * _unweld_uvs, _normalize_unit_sphere (mesh prep)
#   * _pad4, _surgical_texture_swap, _surgical_add_normal_texture
#     (GLB byte surgery, now in media/comfyui/glb_surgical.py)
#
# Replaced by the upstream Comfy3D (MrForExample) ExplicitTargetColorProjection
# node, driven by media/comfyui/families/texture_project.py. The hand-rolled
# implementation had a camera-framing bug (orbit_radius=1.75 made the mesh fill
# only 65% of the camera frustum — chest pixels sampled where face pixels were
# expected). The upstream node, with orbit_radius=1.14, produces correct output.
#
# KEPT in this file:
#   * _normalize_glb_buffers (used by uv_cleanup.py)
#   * _view_camera, dr_rasterize, dr_interpolate, _image_to_tensor,
#     _fg_bbox, render_textured_views (used by server.py for debug renders)


# ── Mesh prep utilities (kept for render_textured_views debug renders) ───

def _normalize_unit_sphere(verts: np.ndarray) -> np.ndarray:
    """Scale vertices so the farthest one lies on the unit sphere."""
    norms = np.linalg.norm(verts, axis=-1)
    max_norm = float(norms.max())
    if max_norm > 1e-8:
        return verts / max_norm
    return verts


def _unweld_uvs(verts, faces, uv_pc):
    """Unweld mesh so each face-vertex has a unique UV entry.

    Returns (new_verts, new_faces, new_uvs) as numpy arrays.
    Each face-vertex pair becomes a new vertex; faces are re-indexed
    into the flat list.  This is the standard pre-step for per-pixel
    UV rasterization (nvdiffrast grid_sample).
    """
    n_faces = faces.shape[0]
    flat = faces.reshape(-1)  # (F*3,) vertex indices
    new_verts = verts[flat].copy()  # (F*3, 3)
    new_faces = np.arange(n_faces * 3, dtype=np.int32).reshape(-1, 3)
    new_uvs = uv_pc.reshape(-1, 2).astype(np.float32)  # (F*3, 2)
    return new_verts, new_faces, new_uvs


# ── Camera (mirrors rasterize.render_passes framing exactly) ──────────────

def _view_camera(az_deg, el_deg, fov_deg, aspect, framing, target_y, dev):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    cam_dist = 1.0 / math.tan(math.radians(fov_deg) / 2.0) * framing
    eye = torch.tensor(
        [cam_dist * math.cos(el) * math.sin(az),
         cam_dist * math.sin(el),
         cam_dist * math.cos(el) * math.cos(az)],
        device=dev, dtype=torch.float32,
    )
    target = torch.tensor([0.0, target_y, 0.0], device=dev, dtype=torch.float32)
    up = torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=torch.float32)
    view = _look_at(eye, target, up)
    near, far = max(cam_dist - 2.0, 0.1), cam_dist + 2.0
    proj = _perspective(fov_deg, aspect, near, far)
    view_proj = (proj @ view).to(dev)
    return eye, view, view_proj


# ── small nvdiffrast wrappers (imported lazily so module import is cheap) ─

def dr_rasterize(glctx, verts_clip, faces, resolution):
    import nvdiffrast.torch as dr
    return dr.rasterize(glctx, verts_clip, faces, resolution)


def dr_interpolate(attr, rast, faces):
    import nvdiffrast.torch as dr
    return dr.interpolate(attr, rast, faces)


def _image_to_tensor(img: Image.Image, dev) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)  # (H, W, 3)
    return torch.from_numpy(arr).to(dev)


def _fg_bbox(img_t: torch.Tensor) -> tuple[float, float, float, float]:
    """Foreground (character) bbox of an image in [0,1]², via background threshold.

    Reference images place the character on a light background; the mesh must be
    aligned to the character's bbox (not the full frame) or the face maps to the
    background.

    Algorithm: sample the full image border (not just corners — corners can be
    occupied by hair/hands/props) to estimate the background brightness via
    MEDIAN (robust against small foreground bleed into the border). The
    foreground is anything darker than bg − max(25, 15%·bg) — a relative
    threshold that adapts to both bright studios (bg≈240) and darker scenes
    (bg≈80). Outlier trimming at p5/p95 prevents a single hair strand at the
    edge from tightening the bbox too much.

    Returns (x0, y0, x1, y1) in [0,1] (x=width frac, y=row frac, top=0).
    Also returns confidence: True if the bbox is trustworthy for silhouette
    alignment, False if it should be skipped (border contaminated by character,
    too small, or too large to be useful).
    """
    f = img_t.float().mean(-1)  # (H, W)
    H, W = f.shape
    border = max(H // 10, 12)  # 10% of each edge — generous margin vs. 6%
    # Sample the full border ring (top/bottom strips + left/right strips)
    bg_samples = torch.cat([
        f[:border, :].flatten(),         # top strip
        f[-border:, :].flatten(),        # bottom strip
        f[:, :border].flatten(),         # left strip
        f[:, -border:].flatten(),        # right strip
    ])
    bg = float(bg_samples.median().item())
    # Relative threshold: at least 25 below bg, or 15% of bg — whichever is larger.
    # This adapts to both bright studios (bg≈240 → threshold≈36) and dark scenes
    # (bg≈80 → threshold≈25).
    thresh = bg - max(25.0, 0.15 * bg)
    mask = f < thresh
    fg_px = int(mask.sum().item())
    total_px = H * W
    fg_frac = fg_px / total_px

    # Confidence gate: skip remapping if bbox is unreliable.
    #   <5% foreground → character is tiny → bbox fragile (hair strand etc.)
    #   >98% foreground → border IS character → background sample contaminated
    if fg_px < 50 or fg_frac < 0.05 or fg_frac > 0.98:
        return 0.0, 0.0, 1.0, 1.0

    # Outlier-trimmed bbox: use p5/p95 instead of min/max. A single strand of
    # hair extending to x=995 (on a 1024-wide image) shouldn't tighten the
    # entire bbox to [0..995/1024] → face would be squeezed off-center.
    ys_t, xs_t = torch.where(mask)
    x_sort = xs_t.float().sort().values
    y_sort = ys_t.float().sort().values
    n_fg = x_sort.numel()
    lo = max(int(n_fg * 0.02), 0)
    hi = max(int(n_fg * 0.98), n_fg - 1)
    return (
        float(x_sort[lo].item() / W), float(y_sort[lo].item() / H),
        float(x_sort[hi].item() / W), float(y_sort[hi].item() / H),
    )


# ── Textured view renderer ────────────────────────────────────────────────
# Renders the mesh from arbitrary angles, sampling the GLB's own baseColor
# texture via the interpolated UVs — the texture is guaranteed to show,
# unlike Blender 4.2 which renders these GLBs flat gray (TRELLIS emits
# invalid texture bytes; even valid PNGs don't display through Blender's
# glTF importer). This is the reliable "see the actual character" path.

def render_textured_views(
    glb_bytes: bytes,
    views: list[dict] | None = None,
    resolution: int = 800,
    fov: float = 30.0,
    framing: float = 1.2,
    bg_color: tuple[float, float, float] = (0.06, 0.06, 0.07),
) -> list[dict]:
    """Render a textured GLB from multiple angles via nvdiffrast + UV sampling.

    Args:
        glb_bytes: GLB with UVs + a baseColorTexture (e.g. a projected GLB).
        views: list of {azimuth, elevation, name} (deg). Default 4 angles.
        resolution: square render resolution.
        fov: camera field of view (deg).
        framing: camera distance multiplier (higher = further).
        bg_color: RGB background in [0,1].

    Returns:
        List of {name, png_bytes}.
    """
    if views is None:
        views = [
            {"azimuth": 0, "elevation": 5, "name": "front"},
            {"azimuth": 90, "elevation": 5, "name": "right"},
            {"azimuth": 180, "elevation": 5, "name": "back"},
            {"azimuth": 270, "elevation": 5, "name": "left"},
        ]

    mesh = trimesh.load(io.BytesIO(_normalize_glb_buffers(glb_bytes)),
                        force="mesh", process=False, file_type="glb")
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    vis = getattr(mesh, "visual", None)
    uvs_v = getattr(vis, "uv", None)
    if uvs_v is None:
        raise ValueError("mesh has no UVs — textured render needs an atlas")
    uv_pc = np.asarray(uvs_v)[faces].astype(np.float32)

    mat = getattr(vis, "material", None)
    base_img = getattr(mat, "baseColorTexture", None)
    if base_img is None:
        raise ValueError("mesh has no baseColorTexture — nothing to render")
    if not isinstance(base_img, Image.Image):
        base_img = Image.fromarray(np.asarray(base_img))
    base_arr = np.asarray(base_img.convert("RGB"), dtype=np.uint8)

    dev = _dev()
    glctx = _get_glctx()

    verts_n = _normalize_unit_sphere(verts)
    v2, f2, uv2 = _unweld_uvs(verts_n, faces, uv_pc)
    v2t = torch.from_numpy(v2).to(dev)
    f2t = torch.from_numpy(np.ascontiguousarray(f2)).to(dev)
    uv2t = torch.from_numpy(np.ascontiguousarray(uv2.astype(np.float32))).to(dev)
    vn = _consolidate_normals(v2t, _vertex_normals(v2t, f2t))
    facing0 = (vn * v2t).sum(-1, keepdim=True) > 0
    vn = torch.where(facing0, vn, -vn)

    # Texture as (1, 3, H, W) float for grid_sample, row 0 = top of image
    # (as PIL stores it). Sampling convention handled in the grid (see below).
    tex = torch.from_numpy(base_arr.astype(np.float32) / 255.0).to(dev)
    tex = tex.permute(2, 0, 1).unsqueeze(0).contiguous()  # (1,3,H,W)

    res = int(resolution)
    bg = torch.tensor(bg_color, device=dev, dtype=torch.float32)
    # Flat, bright lighting (ambient-heavy) so texture detail isn't hidden in
    # shadow — the goal is to SEE the texture, not dramatic shading. Two lights
    # (key + fill) + a high ambient floor → near-wraparound, ~1.0 on the lit
    # front, ~0.7 on the shadowed side.
    light_key = F.normalize(torch.tensor([0.5, 0.7, 0.5], device=dev), dim=0)
    light_fill = F.normalize(torch.tensor([-0.5, 0.3, 0.6], device=dev), dim=0)

    renders: list[dict] = []
    for vw in views:
        eye, _view, vp = _view_camera(
            vw.get("azimuth", 0.0), vw.get("elevation", 5.0),
            fov, 1.0, framing, vw.get("target_y", 0.0), dev,
        )
        ph = torch.cat([v2t, torch.ones(v2t.shape[0], 1, device=dev)], dim=-1)
        clip = (vp @ ph.T).T                                   # (V,4)
        rast, _ = dr_rasterize(glctx, clip.unsqueeze(0).contiguous(), f2t, [res, res])
        uv_map, _ = dr_interpolate(uv2t.unsqueeze(0).contiguous(), rast, f2t)  # (1,res,res,2)
        nrm_map, _ = dr_interpolate(vn.unsqueeze(0).contiguous(), rast, f2t)   # (1,res,res,3)
        cov = rast[..., 3:4] > 0                               # (1,res,res,1)

        # Sample texture at interpolated UVs. grid_sample expects xy in [-1,1]
        # with align_corners=True: x=-1→col 0 (left), y=-1→row 0 (top).
        # glTF: V=0 addresses the BOTTOM of the image (last row), V=1 the top
        # (row 0). So grid_y = 1 - 2V maps V=0→+1(last row), V=1→-1(row 0). ✓
        gx = uv_map[..., 0:1] * 2.0 - 1.0
        gy = 1.0 - uv_map[..., 1:2] * 2.0
        grid = torch.cat([gx, gy], dim=-1)                     # (1,res,res,2)
        color = torch.nn.functional.grid_sample(
            tex, grid, mode="bilinear", padding_mode="border", align_corners=True,
        ).permute(0, 2, 3, 1)                                  # (1,res,res,3)

        # Near-full-bright flat lighting — the goal is to SEE the texture at
        # near-true color, not dramatic shading. High ambient + gentle diffuse
        # + a gamma lift so midtones read clearly on the dark gallery theme.
        nrm = F.normalize(nrm_map, dim=-1, eps=1e-8)
        dk = (nrm * light_key).sum(-1, keepdim=True).clamp(min=0.0)
        df = (nrm * light_fill).sum(-1, keepdim=True).clamp(min=0.0)
        shade = (0.88 + 0.12 * dk + 0.06 * df)
        shaded = color * shade
        shaded = shaded.pow(0.85)  # gamma lift on midtones

        out = torch.where(cov, shaded, bg.expand_as(shaded))
        # nvdiffrast emits the framebuffer bottom-row-first (OpenGL origin at
        # bottom-left). Flip vertically so row 0 = top of the image (PNG/PIL
        # convention) → character renders right-side-up.
        out = torch.flip(out.clamp(0, 1), [1])[0].cpu().numpy()
        out = (out * 255).astype(np.uint8)
        png = io.BytesIO()
        Image.fromarray(out).save(png, format="PNG")
        renders.append({"name": vw.get("name", f"az{vw.get('azimuth',0)}"),
                        "png_bytes": png.getvalue()})
        log.info("[render_textured] %s: %d covered px",
                 vw.get("name"), int(cov.sum().item()))
    return renders

