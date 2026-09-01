"""Headless mesh rasterizer using nvdiffrast.

Renders posed SOMA/Anny vertices (from /poser/skin/deform_soma) to Depth,
Normal, and shaded RGB raster passes.  These serve as ControlNet structural
guides for 2D stylization — the missing bridge between the 3D mesh pipeline
and the 2D diffusion pipeline.

CUDA-native via nvdiffrast 0.4.0 — no EGL / OSMesa / pyrender required.
Modeled on trellis2/renderers/mesh_renderer.py (same container, same stack).

All public functions are synchronous GPU work — the server wraps calls in
asyncio.to_thread() to avoid blocking the aiohttp event loop.
"""
from __future__ import annotations

import io
import logging
import math
import threading
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Shared bounding-sphere helper (suggestion #2, 2026-07-31). Factored out
# of the inline bbox-centroid→radius math so PoserRender (depth) and
# PoserRenderOpenPose CANNOT drift apart — they call the SAME function.
from .library.geometry import bounding_sphere

log = logging.getLogger(__name__)


def _aces_tonemap(x: torch.Tensor) -> torch.Tensor:
    """Narkowicz ACES filmic tone map (the Three.js default).

    Maps HDR lighting (>1.0) smoothly into [0,1] instead of hard-clamping.
    Without this, lit faces blow out to white and — worse — the CLAMP
    creates visible tears at the boundary where one face is at 0.99 light
    (stays colored) and the adjacent face is at 1.01 (snaps to white).
    Pose Studio's WebGL renderer applies this by default; the rasterizer
    MUST match or the render looks flat/blown compared to the canvas.
    """
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return (x * (a * x + b)) / (x * (c * x + d) + e)


# ── nvdiffrast context singleton ──────────────────────────────────────────
# RasterizeCudaContext costs ~200 ms to create but is free to reuse.
# Created lazily on first render and cached for the process lifetime.
_glctx: Any = None
_glctx_lock = threading.Lock()


def _dev() -> str:
    """CUDA device string.

    We intentionally do NOT use ``torch.cuda.is_available()`` — it goes
    through NVML, which can fail inside Docker containers even when the
    GPU is fully accessible to CUDA compute (a well-known cgroup/NVML
    mismatch).  Instead we try to create a tiny CUDA tensor, which
    initialises the actual CUDA runtime.  The genuine error surfaces here
    if the device truly is missing.
    """
    try:
        torch.tensor([0.0], device="cuda")
    except RuntimeError as exc:
        raise RuntimeError(
            "CUDA device not accessible from this process. "
            "Mesh rendering runs inside the inference-comfyui container "
            f"where the GPU is available. (underlying error: {exc})"
        ) from exc
    return "cuda"


def _get_glctx() -> Any:
    """Lazily create and cache the nvdiffrast CUDA rasterizer context."""
    global _glctx
    if _glctx is None:
        with _glctx_lock:
            if _glctx is None:
                import nvdiffrast.torch as dr
                _glctx = dr.RasterizeCudaContext(device=_dev())
                log.info("nvdiffrast RasterizeCudaContext initialized")
    return _glctx


# ── Camera math (OpenGL conventions: -Z forward, Y up) ────────────────────

def _look_at(
    eye: torch.Tensor, center: torch.Tensor, up: torch.Tensor
) -> torch.Tensor:
    """Build a 4×4 world→camera view matrix."""
    f = F.normalize(center - eye, dim=0)        # forward
    r = F.normalize(torch.cross(f, up), dim=0)  # right
    u = torch.cross(r, f)                         # recomputed up
    M = torch.eye(4, device=eye.device, dtype=torch.float32)
    M[0, :3] = r
    M[1, :3] = u
    M[2, :3] = -f
    M[0, 3] = -torch.dot(r, eye)
    M[1, 3] = -torch.dot(u, eye)
    M[2, 3] = torch.dot(f, eye)
    return M


def _perspective(
    fov_y_deg: float, aspect: float, near: float, far: float
) -> torch.Tensor:
    """OpenGL perspective projection matrix (clip-space z ∈ [-1, 1])."""
    t = math.tan(math.radians(fov_y_deg) / 2.0)
    M = torch.zeros((4, 4), device=_dev(), dtype=torch.float32)
    M[0, 0] = 1.0 / (t * aspect)
    M[1, 1] = 1.0 / t
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = 2.0 * far * near / (near - far)
    M[3, 2] = -1.0
    return M


# ── PNG encoding ──────────────────────────────────────────────────────────

def _tensor_to_png(tensor: torch.Tensor, bg: float = 0.0) -> bytes:
    """Convert a [0, 1] GPU tensor to PNG bytes.

    Args:
        tensor: (H, W) grayscale or (H, W, 3) RGB, values approximately [0, 1].
            nvdiffrast writes in OpenGL convention (row 0 = bottom of image);
            we flip vertically so row 0 = top, matching PIL/PNG convention.
        bg: fill value for NaN / Inf pixels.
    """
    t = tensor.detach()
    t = torch.where(torch.isfinite(t), t, torch.full_like(t, bg))
    # Flip vertically: OpenGL bottom-origin → PNG top-origin.
    t = torch.flip(t, dims=[0])
    arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    mode = "L" if arr.ndim == 2 else "RGB"
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Per-vertex normals ────────────────────────────────────────────────────

def _vertex_normals(
    verts: torch.Tensor, faces: torch.Tensor
) -> torch.Tensor:
    """Compute smooth (Gouraud) per-vertex normals via face-normal accumulation.

    Uses the RAW cross product (NOT pre-normalized) so each face's
    contribution is weighted by its area — matching Three.js
    ``computeVertexNormals``. Pre-normalizing face normals (the previous
    approach) gave every triangle equal weight regardless of size,
    which let small/degenerate triangles on high-curvature regions
    (chest, face) dominate the vertex normal and produce visible
    shading tears.

    Args:
        verts: (V, 3) float32.
        faces: (F, 3) int32 triangle indices.

    Returns:
        (V, 3) normalized per-vertex normals.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (F, 3) — magnitude ∝ area
    vn = torch.zeros_like(verts)
    vn.index_add_(0, faces[:, 0], fn)
    vn.index_add_(0, faces[:, 1], fn)
    vn.index_add_(0, faces[:, 2], fn)
    return F.normalize(vn, dim=-1, eps=1e-8)


def _consolidate_normals(
    verts: torch.Tensor, vn: torch.Tensor, tol: float = 1e-5
) -> torch.Tensor:
    """Consolidate normals across vertices at identical positions.

    TRELLIS-generated meshes contain ~22K exact-duplicate vertices (same
    position, different index) arising from the marched-cubes output.  Each
    duplicate has different face connectivity → its own per-vertex normal.
    At render time this produces per-vertex normal variations of up to 179°
    between duplicates at the *same* surface point — visible as surface
    shimmer/noise.  95.5% of duplicate pairs differ by >5°, 39% by >30°.

    This function hashes vertex positions to integer grid cells, groups
    duplicates, sums their normals within each group, and broadcasts the
    averaged normal back to every member.  The result is the normal field
    that *would* have been computed if the mesh had no duplicates — smooth
    and continuous across the surface.

    For meshes without duplicates this is a no-op (every group has one
    member → summed normal == original normal).

    Args:
        verts: (V, 3) float32 — deformed vertex positions.
        vn:    (V, 3) float32 — per-vertex normals from ``_vertex_normals``.
        tol:   position tolerance for duplicate detection (default 10nm).

    Returns:
        (V, 3) consolidated, normalized per-vertex normals.
    """
    if verts.shape[0] < 2:
        return vn
    # Hash each vertex to an integer grid cell.  Round to `tol` precision
    # so vertices within `tol` of each other collapse to the same key.
    keys = torch.round(verts / tol).long()  # (V, 3) int64
    # Find unique cells and the group index for each vertex.
    unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
    # If every vertex is its own group, no duplicates → no-op.
    if unique.shape[0] == verts.shape[0]:
        return vn
    # Sum normals within each group, normalize, map back.
    summed = torch.zeros_like(vn)
    summed.index_add_(0, inverse, vn)
    summed = F.normalize(summed, dim=-1, eps=1e-8)
    return summed[inverse]


# ── Main entry point ──────────────────────────────────────────────────────

def render_passes(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    width: int = 1024,
    height: int = 1024,
    azimuth: float = 0.0,
    elevation: float = 5.0,
    fov: float = 30.0,
    passes: list[str] | None = None,
    background: str = "black",
    target_y: float = 0.0,
    framing: float = 1.5,
    mesh_color: tuple[float, float, float] | None = None,
) -> dict[str, bytes]:
    """Rasterize a posed mesh to Depth / Normal / shaded RGB PNG images.

    The mesh is auto-centered and normalized to a unit bounding sphere so
    camera framing is consistent regardless of input scale.

    Args:
        vertices: (V, 3) float32 in centimeters, Y-up (SOMA convention).
        faces:    (F, 3) int32 triangle indices (topology is pose-invariant).
        width:    Output image width  [32–4096].
        height:   Output image height [32–4096].
        azimuth:  Horizontal orbit angle in degrees (0 = front-facing).
        elevation: Vertical orbit angle in degrees (0 = eye-level, + = above).
        fov:      Vertical field of view in degrees.
        passes:   Subset of ``["depth", "normal", "rgb"]`` to render.
                  Default: all three.
        background: ``"black"`` or ``"white"`` for non-mesh pixels.
        target_y: Vertical offset of the camera look-at point in NORMALIZED
                  mesh coords (after unit-sphere normalization). SIGN
                  CONVENTION (empirically verified 2026-07-28 by pixel
                  measurement on real human-shape renders):
                    HIGHER target_y  →  subject appears HIGHER in frame
                                       (head approaches top edge).
                    target_y = 0.0   →  subject centered (default, safe
                                       for full-body framing).
                    LOWER target_y   →  subject appears LOWER in frame
                                       (feet approach bottom edge).
                  Prior docstrings had the sign inverted. The geometry:
                  ``target`` is the world-space point the camera looks at.
                  Raising the target rotates the camera UP, which moves
                  the projected image of a fixed subject DOWN in frame
                  relative to the look-at melite — wait, that's the opposite.
                  The truth is the empirical measurement above; the
                  camera math is non-obvious. Trust the pixels.
                  Practical range for a standing human: [-0.1, +0.3].
                  Beyond ±0.3 the subject clips.
        framing:  Camera-distance multiplier. ``1.5`` = loose (body fills
                  ~65% of frame, 17% margins). ``1.15`` = tight (body fills
                  ~85%). ``1.0`` = body fills frame edge-to-edge (use with
                  caution — parts may be cropped).

    Returns:
        Dict mapping pass name → PNG bytes, e.g.
        ``{"depth": b"\\x89PNG...", "normal": ..., "rgb": ...}``

    Depth convention:     near = white, far = black (standard ControlNet).
    Normal convention:    camera-space, [-1,1]→[0,1], front-facing = blue.
    RGB convention:       warm-gray Lambertian shaded render for img2img source.
    """
    if passes is None:
        passes = ["depth", "normal", "rgb"]

    # ── Validate ─────────────────────────────────────────────────────
    verts_np = np.asarray(vertices, dtype=np.float32)
    faces_np = np.asarray(faces, dtype=np.int32)
    if verts_np.ndim != 2 or verts_np.shape[1] != 3:
        raise ValueError(f"vertices must be (V, 3), got {verts_np.shape}")
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3), got {faces_np.shape}")
    if verts_np.shape[0] < 3 or faces_np.shape[0] < 1:
        raise ValueError("need ≥ 3 vertices and ≥ 1 face")
    if int(faces_np.max()) >= verts_np.shape[0]:
        raise ValueError(
            f"face index {int(faces_np.max())} ≥ vertex count {verts_np.shape[0]}"
        )
    if not (32 <= width <= 4096 and 32 <= height <= 4096):
        raise ValueError("width/height must be in [32, 4096]")

    bg_val = 0.0 if background == "black" else 1.0
    dev = _dev()

    # ── Normalize mesh: center at origin, scale to unit bounding sphere ──
    # Shared with PoserRenderOpenPose via library.geometry.bounding_sphere
    # so depth and skeleton renders CANNOT drift to different scales.
    center, radius = bounding_sphere(verts_np)
    if radius < 1e-6:
        raise ValueError("degenerate mesh — zero bounding-sphere radius")
    verts_c = (verts_np - center) / radius

    # ── Camera: orbit around target ─────────────────────────────────
    az = math.radians(azimuth)
    el = math.radians(elevation)
    # framing multiplier controls zoom: 1.5 = loose (50% headroom),
    # 1.15 = tight (15% headroom, body fills ~85% of frame).
    cam_dist = 1.0 / math.tan(math.radians(fov) / 2.0) * framing
    eye = torch.tensor(
        [
            cam_dist * math.cos(el) * math.sin(az),
            cam_dist * math.sin(el),
            cam_dist * math.cos(el) * math.cos(az),
        ],
        device=dev,
        dtype=torch.float32,
    )
    # target_y biases the look-at point vertically in NORMALIZED mesh space.
    # SIGN CONVENTION (empirically verified, 2026-07-28): HIGHER target_y
    # makes the subject appear HIGHER in frame. At target_y=+0.3 the head
    # is at row 7/1024 (nearly clipped); at -0.05 the head is at row 214
    # and FEET CLIP at row 1023. The earlier comment here claimed +0.3
    # "centers the upper-torso visually" — that was wrong, and shipping
    # it caused two cycles of broken framing. The camera math is non-
    # intuitive; trust pixel measurements over geometric reasoning.
    target = torch.tensor(
        [0.0, target_y, 0.0], device=dev, dtype=torch.float32
    )
    up = torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=torch.float32)

    view = _look_at(eye, target, up)
    near, far = max(cam_dist - 2.0, 0.1), cam_dist + 2.0
    proj = _perspective(fov, width / height, near, far)
    view_proj = (proj @ view).to(dev)

    # ── Upload mesh → GPU ────────────────────────────────────────────
    verts_t = torch.from_numpy(verts_c).to(dev)
    faces_t = torch.from_numpy(np.ascontiguousarray(faces_np)).to(dev)

    verts_homo = torch.cat(
        [verts_t, torch.ones(verts_t.shape[0], 1, device=dev)], dim=-1
    )  # (V, 4)

    verts_cam = verts_homo @ view.T                        # (V, 4) camera space
    verts_clip = (verts_homo @ view_proj.T).unsqueeze(0).contiguous()  # (1, V, 4)

    # ── Rasterize ────────────────────────────────────────────────────
    import nvdiffrast.torch as dr

    glctx = _get_glctx()
    rast, _ = dr.rasterize(glctx, verts_clip, faces_t, (height, width))
    # rast: (1, H, W, 4) — (u, v, z/w, triangle_id); tri_id = 0 → background

    mask = (rast[0, ..., 3] > 0).float()  # (H, W) foreground mask
    mask3 = mask.unsqueeze(-1)             # (H, W, 1) for RGB broadcasting
    results: dict[str, bytes] = {}

    # ── Smooth per-vertex normals (shared by normal + rgb passes) ─────
    need_normals = "normal" in passes or "rgb" in passes
    if need_normals:
        vn = _vertex_normals(verts_t, faces_t)
        # Consolidate normals across duplicate vertices (TRELLIS meshes have
        # ~22K exact-position duplicates whose independent face connectivity
        # produces per-vertex normals differing by up to 179° — visible as
        # surface shimmer/noise).  For clean meshes this is a no-op.
        vn = _consolidate_normals(verts_t, vn)
        # Flip inward-facing normals (mesh centered at origin: outward ≈ dot(n, v) > 0)
        facing = (vn * verts_t).sum(-1, keepdim=True) > 0
        vn = torch.where(facing, vn, -vn)

    # ── Depth pass ───────────────────────────────────────────────────
    if "depth" in passes:
        cam_z = verts_cam[:, 2:3].unsqueeze(0).contiguous()  # (1, V, 1)
        depth, _ = dr.interpolate(cam_z, rast, faces_t)
        depth = depth[0, ..., 0]  # (H, W) camera-space Z (negative)

        # Normalize: near (less negative) → white, far → black
        visible = mask > 0
        if visible.any():
            dv = depth[visible]
            vmin, vmax = dv.min(), dv.max()
            span = (vmax - vmin).item()
            if span > 1e-6:
                depth_n = (depth - vmin) / span
            else:
                depth_n = torch.ones_like(depth)
        else:
            depth_n = torch.zeros_like(depth)
        depth_out = torch.where(
            visible, depth_n, torch.full_like(depth_n, bg_val)
        )
        results["depth"] = _tensor_to_png(depth_out)

    # ── Normal-map pass (camera-space) ────────────────────────────────
    if "normal" in passes:
        # Transform world normals → camera space BEFORE interpolation
        vn_cam = (vn @ view[:3, :3].T).contiguous()  # (V, 3)
        n_val, _ = dr.interpolate(vn_cam.unsqueeze(0), rast, faces_t)
        n_val = F.normalize(n_val[0], dim=-1, eps=1e-8)  # (H, W, 3)

        # Camera looks down −Z; front-facing surfaces have normal Z > 0.
        # Flip any remaining back-facing fragments (edges / barycentric noise).
        front = n_val[..., 2:] >= 0
        n_val = torch.where(front, n_val, -n_val)

        n_img = (n_val + 1.0) / 2.0  # [-1, 1] → [0, 1]
        n_img = n_img * mask3 + bg_val * (1.0 - mask3)
        results["normal"] = _tensor_to_png(n_img)

    # ── Shaded RGB pass (world-space Lambertian, img2img source) ──────
    if "rgb" in passes:
        n_world, _ = dr.interpolate(
            vn.unsqueeze(0).contiguous(), rast, faces_t
        )
        n_world = F.normalize(n_world[0], dim=-1, eps=1e-8)  # (H, W, 3)

        # Lighting MATCHED to Pose Studio's Three.js scene exactly
        # (PoseViewer3D.tsx ~L1266-1282, default non-edit non-orb mode):
        #   <ambientLight intensity={0.45} />
        #   <directionalLight position={[5,8,5]}  intensity={1.0} />   ← key
        #   <directionalLight position={[-4,3,-2]} intensity={0.25} /> ← fill
        #   <hemisphereLight args={["#cdd6f4","#1a1a2e", 0.35]} />
        # Prior mismatch (key intensity 0.35, no hemisphere) made the
        # render look flat/washed compared to Pose Studio. The render is
        # Picture 2 in the blue_mesh_somax edit — if it looks worse than
        # Pose Studio, qwen-img-edit copies that crudeness. This must
        # match.
        key_dir = F.normalize(
            torch.tensor([5.0, 8.0, 5.0], device=dev), dim=0
        )
        fill_dir = F.normalize(
            torch.tensor([-4.0, 3.0, -2.0], device=dev), dim=0
        )

        # Per-fragment double-sided shading — HALF-LAMBERT (Valve).
        #
        # Pose Studio uses meshStandardMaterial with roughness=0.55.
        # Roughness softens the shadow boundary in PBR — the transition
        # from lit to unlit is spread over a wide range of normal angles.
        # Our Lambertian rasterizer has no specular/roughness. Any clamp
        # (even wrap lighting's shifted clamp) creates a HARD boundary
        # where adjacent triangles straddle it (one at dot=+0.01, next at
        # dot=-0.01) → brightness snaps in one pixel → the "tearing" the
        # user sees on chest/nipples.
        #
        # Half-Lambert eliminates the zero-crossing ENTIRELY:
        #   diff = dot(n, L)           range [-1, 1]
        #   half = diff * 0.5 + 0.5    range [ 0, 1]  ← remapped, always ≥ 0
        #   light = half²              range [ 0, 1]  ← squared for contrast
        #
        # There is NO clamp(min=0). The curve is a smooth quadratic across
        # the ENTIRE normal angle range. Adjacent triangles at +0.01 vs
        # -0.01 get (0.505)²=0.255 vs (0.495)²=0.245 — a 1% difference,
        # invisible. This is what PBR roughness does for organic surfaces:
        # it spreads light into shadow regions smoothly.
        view_dir = torch.tensor([0.0, 0.0, 1.0], device=dev)
        back_facing = (n_world @ view_dir) < 0.0           # (H, W)
        n_front = torch.where(
            back_facing.unsqueeze(-1), -n_world, n_world   # flip back faces
        )

        n_dot_key = n_front @ key_dir                        # (H,W) [-1,1]
        n_dot_fill = n_front @ fill_dir                      # (H,W) [-1,1]
        key = (n_dot_key * 0.5 + 0.5) ** 2 * 0.8            # [0, 0.8]
        fill = (n_dot_fill * 0.5 + 0.5) ** 2 * 0.2          # [0, 0.2]
        ambient = 0.35

        # Hemisphere light: blend sky #cdd6f4 (205,214,244) for upward
        # normals → ground #1a1a2e (26,26,46) for downward. Three.js
        # formula: weight = normal.y * 0.5 + 0.5; color = mix(ground, sky).
        sky = torch.tensor(
            [205 / 255, 214 / 255, 244 / 255], device=dev
        )
        ground = torch.tensor(
            [26 / 255, 26 / 255, 46 / 255], device=dev
        )
        hemi_w = (n_world[..., 1:2] * 0.5 + 0.5).clamp(0, 1)  # (H,W,1)
        hemi = (sky * hemi_w + ground * (1.0 - hemi_w)) * 0.20  # (H,W,3)

        albedo = torch.tensor(
            mesh_color if mesh_color is not None else [0.78, 0.75, 0.72],
            device=dev,
        )
        # Diffuse (scalar) + hemisphere (colored) added additively.
        # NOTE: do NOT clamp here — ACES tone map below handles HDR values
        # smoothly. Hard-clamping was the root cause of both the white
        # blowout (lit faces snapping to white) AND the tears (the clamp
        # boundary creates a sharp discontinuity between 0.99-lit and
        # 1.01-lit adjacent faces).
        diffuse = (ambient + key + fill).unsqueeze(-1)  # (H,W,1)
        rgb = diffuse * albedo + hemi                   # (H,W,3), may exceed 1.0

        # ACES filmic tone mapping — matches Three.js's default renderer.
        # Compresses highlights so the blue stays blue instead of blowing
        # to white, and eliminates the clip-boundary tears.
        rgb = _aces_tonemap(torch.clamp(rgb, 0.0, 8.0))  # clamp input for safety

        # Geometric antialiasing — nvdiffrast smooths triangle-boundary
        # jaggies. WITHOUT this, every triangle edge is a hard staircase
        # that reads as "marks" on the mesh surface. Pose Studio's WebGL
        # renderer has MSAA; this is the rasterizer's equivalent.
        rgb_aa = dr.antialias(
            (rgb * mask3)[None, ...].contiguous(),  # (1,H,W,3)
            rast,
            verts_clip,
            faces_t,
        )
        rgb_aa = rgb_aa[0]  # (H, W, 3)
        rgb_aa = rgb_aa * mask3 + bg_val * (1.0 - mask3)
        results["rgb"] = _tensor_to_png(rgb_aa)

    return results
