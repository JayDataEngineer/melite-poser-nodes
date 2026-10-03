"""HTTP route handlers for the melite-poser-nodes ComfyUI pack.

Registers /poser/* routes on ComfyUI's PromptServer so Pose Studio and
Body Studio can fetch bind meshes, identity metadata, and SOMA-77 joint
positions WITHOUT a separate poser-skin container. Everything runs inside
inference-comfyui — one GPU container, one place for mesh logic.

The handlers are thin wrappers around the cached SkinEngine (skin.py).
Heavy work (set_identity, skin) runs in a thread so the aiohttp event
loop stays responsive for other ComfyUI traffic.

VRAM management: TRUST COMFYUI. ComfyUI's ModelPatcher is refcounted —
when a node output goes out of scope, the model auto-unloads. We do NOT
call /free, we do NOT call unload_all_models(), we do NOT monkey-patch
anything. Calling /free from outside the prompt_worker races with it and
crashes the worker (the "stuck on z-image" failure mode). If a future
stage OOMs, the right fix is to find the reference leak in OUR code, not
to paper over it with /free calls.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
from typing import Any

import numpy as np
from aiohttp import web

from server import PromptServer

from .rasterize import render_passes
from .skin import get_engine, identity_space

log = logging.getLogger(__name__)

# BVH upload safety limits.
_MAX_BVH_BYTES = 10 * 1024 * 1024       # 10 MB max POST body
_MAX_BVH_FRAMES_DEFAULT = 500           # default max_frames if not specified
_MAX_BVH_FRAMES_HARD = 5000             # hard cap on max_frames


# ── Route registration ──────────────────────────────────────────────────────
routes = PromptServer.instance.routes


@routes.get("/poser/skin/mesh")
async def get_skin_mesh(request: web.Request) -> web.Response:
    """GET /poser/skin/mesh[?identity_model=soma|mhr|anny] — bind mesh.

    Returns the NEUTRAL bind mesh (default_vertices) — not affected by
    previous set_identity calls. Use POST /poser/skin/mesh for shaped meshes.
    """
    identity_model = (request.query.get("identity_model") or "soma").lower()
    request.query.get("preview_pose")  # accepted but ignored
    try:
        engine = await asyncio.to_thread(get_engine, identity_model)
    except FileNotFoundError as e:
        return _err(str(e), 404)
    except (ImportError, ValueError) as e:
        return _err(str(e), 400)
    except Exception:
        log.exception("Failed to load skin mesh (%s)", identity_model)
        return _err("internal error", 500)

    return _json({
        "vertices": engine.default_vertices.tolist(),
        "faces": engine.default_faces().tolist(),
        "bind_joints": engine.default_bind_joints.tolist(),
        "identity_model": identity_model,
    })


@routes.post("/poser/skin/mesh")
async def post_skin_mesh(request: web.Request) -> web.Response:
    """POST /poser/skin/mesh — SHAPED bind mesh.

    Body: {identity_model, identity_coeffs, local_changes, custom_targets}
    Applies the identity morphs (~30 ms) and returns the shaped mesh.
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _err("invalid JSON body", 400)

    identity_model = (body.get("identity_model") or "soma").lower()
    coeffs = body.get("identity_coeffs")
    local_changes = body.get("local_changes")
    custom_targets = body.get("custom_targets")

    try:
        engine = await asyncio.to_thread(get_engine, identity_model)
        # set_identity mutates engine.vertices in place; the response
        # reads them back from the live engine state.
        await asyncio.to_thread(
            engine.set_identity, coeffs, local_changes, custom_targets
        )
    except FileNotFoundError as e:
        return _err(str(e), 404)
    except (ImportError, ValueError) as e:
        return _err(str(e), 400)
    except Exception:
        log.exception("Failed to shape mesh (%s)", identity_model)
        return _err("internal error", 500)

    return _json({
        "vertices": engine.vertices.tolist(),
        "faces": engine.default_faces().tolist(),
        "bind_joints": engine.bind_joints.tolist(),
        "identity_model": identity_model,
    })


@routes.get("/poser/skin/identity-space")
async def get_identity_space(request: web.Request) -> web.Response:
    """GET /poser/skin/identity-space?identity_model=X — coefficient metadata.

    Accepts both ``identity_model`` and legacy ``model`` query params.
    Pure metadata — dims, labels, defaults, range. Triggers a one-time
    model load for anny/mhr so num_shape_components is accurate.
    """
    identity_model = (
        request.query.get("identity_model") or request.query.get("model") or "soma"
    ).lower()
    # For anny/mhr, ensure the engine is cached so identity_space() can
    # report the real num_shape_components. Wrap in try/except — if the
    # model load fails, fall back to the default dim count.
    if identity_model in {"anny", "mhr"}:
        try:
            await asyncio.to_thread(get_engine, identity_model)
        except Exception as e:
            log.warning("identity-space: engine load failed (%s): %s", identity_model, e)
    return _json(identity_space(identity_model))


@routes.get("/poser/skin/bind_joints")
async def get_bind_joints(request: web.Request) -> web.Response:
    """GET /poser/skin/bind_joints[?identity_model=X] — neutral bind joints."""
    identity_model = (request.query.get("identity_model") or "soma").lower()
    try:
        engine = await asyncio.to_thread(get_engine, identity_model)
    except Exception as e:
        return _err(str(e), 400)
    return _json({
        "joints": engine.default_bind_joints.tolist(),
        "identity_model": identity_model,
    })


# Cached canonical arm angles — loaded ONCE from skin_standard.npz.
# Body-type-INVARIANT (canonical pose is the same for every identity).
_CANONICAL_ARM_ANGLES_CACHE: tuple[float, float] | None = None


def _load_canonical_arm_angles_from_npz() -> tuple[float, float]:
    """Read canonical L/R arm angles from skin_standard.npz.

    NO FALLBACK: raises if the npz is missing or malformed. The canonical
    pose is the GROUND TRUTH the SOMAX weight transfer consumer expects;
    silently hardcoding 44.36/44.40 masks deployment issues (npz not
    mounted into melite-head) and causes wrong arm angles in production.
    """
    global _CANONICAL_ARM_ANGLES_CACHE
    if _CANONICAL_ARM_ANGLES_CACHE is not None:
        return _CANONICAL_ARM_ANGLES_CACHE

    import math
    npz_path = "/opt/kimodo/kimodo/assets/skeletons/somaskel77/skin_standard.npz"
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"skin_standard.npz not found at {npz_path!r}. This file is the "
            f"GROUND TRUTH for the 44° canonical A-pose. The kimodo skeleton "
            f"assets must be installed in inference-comfyui."
        )
    d = np.load(npz_path, allow_pickle=True)
    if "bind_rig_transform" not in d.files:
        raise KeyError(
            f"skin_standard.npz has no 'bind_rig_transform' key "
            f"(found: {d.files!r}). The npz is corrupt or incompatible."
        )
    joints = np.asarray(d["bind_rig_transform"], dtype=np.float32)[:, :3, 3]

    def _angle(sh: int, wr: int) -> float:
        v = joints[wr] - joints[sh]
        horizontal = math.sqrt(float(v[0]) ** 2 + float(v[2]) ** 2)
        below = -float(v[1])
        return float(math.degrees(math.atan2(below, horizontal)))

    L, R = _angle(12, 14), _angle(40, 42)
    _CANONICAL_ARM_ANGLES_CACHE = (L, R)
    log.info("Canonical arm angles cached: L=%.4f° R=%.4f°", L, R)
    return L, R


@routes.get("/poser/skin/canonical_arm_angles")
async def get_canonical_arm_angles(request: web.Request) -> web.Response:
    """GET /poser/skin/canonical_arm_angles — the ground-truth 44° L/R angles.

    Returns the canonical SOMAX arm angles (degrees below horizontal)
    measured from ``bind_rig_transform`` in ``skin_standard.npz``. These
    are the angles the SOMAX weight-transfer consumer expects; the
    A-pose correction in ``_pose_helpers._correct_to_canonical_apose``
    rotates each arm chain to match them.

    Body-type-INVARIANT — same canonical pose for every identity. Cached
    after first call.

    NO FALLBACK: raises if the npz is missing. The caller (melite-head) MUST
    propagate this error rather than silently using hardcoded values.
    """
    try:
        L, R = await asyncio.to_thread(_load_canonical_arm_angles_from_npz)
    except Exception as e:
        return _err(f"canonical arm angles unavailable: {e}", 500)
    return _json({"angle_l_deg": L, "angle_r_deg": R, "source": "skin_standard.npz"})


@routes.post("/poser/skin/bind_joints")
async def post_bind_joints(request: web.Request) -> web.Response:
    """POST /poser/skin/bind_joints — SHAPED bind joints.

    Body: {identity_model, identity_coeffs, local_changes, custom_targets}
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _err("invalid JSON body", 400)

    identity_model = (body.get("identity_model") or "soma").lower()
    try:
        engine = await asyncio.to_thread(get_engine, identity_model)
        await asyncio.to_thread(
            engine.set_identity,
            body.get("identity_coeffs"),
            body.get("local_changes"),
            body.get("custom_targets"),
        )
    except Exception as e:
        return _err(str(e), 400)
    return _json({
        "joints": engine.bind_joints.tolist(),
        "identity_model": identity_model,
    })


@routes.post("/poser/skin/bind_pose")
async def post_bind_pose(request: web.Request) -> web.Response:
    """POST /poser/skin/bind_pose — SHAPED bind pose (positions + rotations).

    The principled upstream-truth source for ControlNet conditioning.
    Returns the body-type-aware SOMA-77 bind pose:

    - **positions** (77, 3): joint positions regressed from the morphed
      mesh via ``skeleton_transfer.fit()``. These ARE body-type-specific
      (child has shorter limbs, woman has wider hips, etc.).
    - **rotations** (77, 3, 3): canonical bind-pose rotations from
      ``bind_pose_local`` (loaded ONCE from ``skin_standard.npz``).
      Body-type-INVARIANT — one canonical pose for all bodies.

    Combined, this is the "proper A-pose per body type" — the same
    canonical 44° arm drop applied to body-type-scaled proportions. Zero
    deviation from SOMAX expectations (it IS the SOMA bind pose).

    Body: {identity_model, identity_coeffs, local_changes, custom_targets,
           units?: "cm"|"m" (default "cm" — matches the deform_soma contract)}.

    Returns: {positions: [[x,y,z]×77], rotations: [[3×3]×77],
              identity_model, units}
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _err("invalid JSON body", 400)

    identity_model = (body.get("identity_model") or "soma").lower()
    units = (body.get("units") or "cm").lower()
    if units not in ("cm", "m"):
        return _err(f"units must be 'cm' or 'm', got {units!r}", 400)

    try:
        engine = await asyncio.to_thread(get_engine, identity_model)
        await asyncio.to_thread(
            engine.set_identity,
            body.get("identity_coeffs"),
            body.get("local_changes"),
            body.get("custom_targets"),
        )
    except Exception as e:
        return _err(str(e), 400)

    # Full 4×4 world-space transforms — body-type-aware positions +
    # canonical rotations in one tensor.
    transforms = engine._extract_soma_x_transforms()  # (77, 4, 4), meters
    positions_m = transforms[:, :3, 3]
    rotations = transforms[:, :3, :3]

    scale = 100.0 if units == "cm" else 1.0
    return _json({
        "positions": (positions_m * scale).tolist(),
        "rotations": rotations.tolist(),
        "identity_model": identity_model,
        "units": units,
    })


@routes.post("/poser/skin/deform_soma")
async def post_deform_soma(request: web.Request) -> web.Response:
    """POST /poser/skin/deform_soma — SOMA-77 positions → deformed vertices.

    Body: {
        identity_model: str,
        frames_soma77_3d: [[x,y,z]*77] or [[[x,y,z]*77]*T],
        frames_soma77_rotmat?: (T, 77, 3, 3) optional,
        identity_coeffs?: [...], local_changes?: {...}
    }

    The frontend sends joint positions in CENTIMETERS (matching the
    frames_soma77_3d field from the generate response). deform() works in
    METERS (py-soma-x's native unit), so we convert cm → m on entry and
    m → cm on exit.
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _err("invalid JSON body", 400)

    identity_model = (body.get("identity_model") or "soma").lower()
    # Accept both frames_soma77_3d (frontend convention) and the legacy
    # frames_soma77_pos field name.
    pos = body.get("frames_soma77_3d") or body.get("frames_soma77_pos")
    rot = body.get("frames_soma77_rotmat")
    if pos is None:
        return _err("frames_soma77_3d required", 400)

    try:
        pos_np = np.asarray(pos, dtype=np.float32)
        rot_np = np.asarray(rot, dtype=np.float32) if rot is not None else None
        engine = await asyncio.to_thread(get_engine, identity_model)

        # Apply identity morphs first (same as POST /poser/skin/mesh).
        coeffs = body.get("identity_coeffs")
        local = body.get("local_changes")
        custom = body.get("custom_targets")
        if coeffs is not None or local or custom:
            await asyncio.to_thread(engine.set_identity, coeffs, local, custom)

        # deform() accepts METERS. Frontend sends cm (matching the generate
        # response's frames_soma77_3d). Convert cm → m.
        pos_m = pos_np / 100.0
        deformed_m = await asyncio.to_thread(engine.deform, pos_m, rot_np)
        # Convert meters → centimeters for the frontend.
        deformed_cm = (deformed_m * 100.0).astype(np.float32)
    except ValueError as e:
        return _err(str(e), 400)
    except Exception as e:
        log.exception("deform_soma failed")
        return _err(str(e), 500)

    # Binary mode: return base64-encoded float32 (the frontend's
    # deformSomaJoints expects vertices_b64 for lossless transfer).
    if body.get("binary"):
        import base64 as _b64
        verts_b64 = _b64.b64encode(deformed_cm.tobytes()).decode("ascii")
        return _json({
            "vertices_b64": verts_b64,
            "vertex_count": int(deformed_cm.shape[1]) if deformed_cm.ndim == 3 else int(deformed_cm.shape[0]),
            "identity_model": identity_model,
        })

    return _json({
        "vertices": deformed_cm.tolist(),
        "identity_model": identity_model,
    })


@routes.post("/poser/skin/deform_npz")
async def post_deform_npz(request: web.Request) -> web.Response:
    """POST /poser/skin/deform_npz — Kimodo NPZ → deformed vertices (all frames).

    Accepts a Kimodo motion NPZ (base64-encoded) containing posed_joints
    (T, 77, 3) and global_rot_mats (T, 77, 3, 3), runs LBS via the skin
    engine, and returns base64-encoded deformed vertices for all frames.

    Body: {
        npz_b64: str,                  # base64-encoded .npz binary
        identity_model?: str,          # default "soma"
        identity_coeffs?: [...],       # optional identity morph
        local_changes?: {...},         # optional
        custom_targets?: {...},        # optional
    }

    Response (binary mode): {
        vertices_b64: str,             # base64 float32 (T, V, 3)
        frame_count: int,
        vertex_count: int,
        fps: float,
        identity_model: str,
    }
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _err("invalid JSON body", 400)

    npz_b64 = body.get("npz_b64")
    if not npz_b64:
        return _err("npz_b64 required", 400)

    identity_model = (body.get("identity_model") or "soma").lower()

    try:
        import base64 as _b64
        import io

        npz_bytes = _b64.b64decode(npz_b64)
        motion = np.load(io.BytesIO(npz_bytes), allow_pickle=False)

        # Extract joint data from the NPZ
        posed_joints = motion.get("posed_joints")
        global_rot_mats = motion.get("global_rot_mats")

        if posed_joints is None:
            # Try alternative key names
            posed_joints = motion.get("frames_soma77_3d")
        if global_rot_mats is None:
            global_rot_mats = motion.get("frames_soma77_rotmat")

        if posed_joints is None:
            available = list(motion.keys())
            return _err(
                f"NPZ missing joint data. Available keys: {available}. "
                f"Expected 'posed_joints' or 'frames_soma77_3d'.",
                400,
            )

        pos_np = np.asarray(posed_joints, dtype=np.float32)
        rot_np = np.asarray(global_rot_mats, dtype=np.float32) if global_rot_mats is not None else None

        # Ensure correct shape: pos should be (T, 77, 3)
        if pos_np.ndim == 2:
            # Single frame: (77, 3) → (1, 77, 3)
            pos_np = pos_np[np.newaxis, ...]
        if pos_np.shape[-1] != 3 or pos_np.shape[-2] != 77:
            return _err(
                f"Unexpected posed_joints shape {pos_np.shape}. Expected (T, 77, 3).",
                400,
            )

        T = pos_np.shape[0]

        # FPS from NPZ or default
        fps = float(motion.get("fps", 30.0)) if "fps" in motion.files else 30.0

        engine = await asyncio.to_thread(get_engine, identity_model)

        # Apply identity morphs
        coeffs = body.get("identity_coeffs")
        local = body.get("local_changes")
        custom = body.get("custom_targets")
        if coeffs is not None or local or custom:
            await asyncio.to_thread(engine.set_identity, coeffs, local, custom)

        # deform() accepts METERS. NPZ joints are in centimeters.
        pos_m = pos_np / 100.0
        deformed_m = await asyncio.to_thread(engine.deform, pos_m, rot_np)
        # Convert meters → centimeters for the frontend.
        deformed_cm = (deformed_m * 100.0).astype(np.float32)

        verts_b64 = _b64.b64encode(deformed_cm.tobytes()).decode("ascii")
        return _json({
            "vertices_b64": verts_b64,
            "frame_count": T,
            "vertex_count": int(deformed_cm.shape[1]),
            "fps": fps,
            "identity_model": identity_model,
        })

    except ValueError as e:
        return _err(str(e), 400)
    except Exception as e:
        log.exception("deform_npz failed")
        return _err(str(e), 500)


# ── Static catalogs (no GPU, no model load) ─────────────────────────────────
# Registry format MUST match the frontend's RegistryTree interface
# (web/editor/src/lib/text-to-pose.ts):
#   {
#     datasets: string[],
#     tree: { [dataset]: { skeletons: RegistrySkeleton[] } },
#     default_dataset, default_skeleton, default_version
#   }
# Each RegistrySkeleton = { label, key, versions: RegistryVersion[] }
# Each RegistryVersion = { display, short_key, local, is_g1 }

_DATASET_ID = "kimodo-soma-rp"
_SKELETON_LABEL = "SOMA 77-joint"
_SKELETON_KEY = "soma-77"
_VERSION_DISPLAY = "Kimodo-SOMA-RP-v1.1"
_VERSION_KEY = "v1.1"

_REGISTRY = {
    "datasets": [_DATASET_ID],
    "tree": {
        _DATASET_ID: {
            "skeletons": [
                {
                    "label": _SKELETON_LABEL,
                    "key": _SKELETON_KEY,
                    "versions": [
                        {
                            "display": _VERSION_DISPLAY,
                            "short_key": _VERSION_KEY,
                            "local": True,
                            "is_g1": False,
                        },
                    ],
                },
            ],
        },
    },
    "default_dataset": _DATASET_ID,
    "default_skeleton": _SKELETON_LABEL,
    "default_version": _VERSION_KEY,
}

_MODELS = [
    {"id": _VERSION_DISPLAY, "label": "Kimodo SOMA-RP v1.1", "dataset": _DATASET_ID},
]


@routes.get("/poser/text-to-pose/registry")
async def get_registry(request: web.Request) -> web.Response:
    return _json(_REGISTRY)


@routes.get("/poser/text-to-pose/models")
async def get_models(request: web.Request) -> web.Response:
    return _json({"models": _MODELS})


@routes.get("/poser/health")
async def poser_health(request: web.Request) -> web.Response:
    """Health check — used by the frontend to confirm the poser routes live."""
    return _json({"status": "ok", "service": "melite-poser-nodes"})


@routes.get("/poser/queue_health")
async def poser_queue_health(request: web.Request) -> web.Response:
    """Inspect the ComfyUI prompt queue + prompt_worker liveness.

    READ-ONLY observability — no VRAM-management side effects. Reads
    ``PromptServer.instance.prompt_queue.get_queue_remaining()`` natively.

    Surfaces the "stuck on z-image" failure mode (pending > 0, running == 0)
    so the frontend/backend can react instead of hanging silently. Native
    ComfyUI exposes the queue via ``get_queue_remaining()`` which returns
    ``(currently_running, pending)``.

    Response shape:
      {
        "pending": int, "running": int,
        "verdict": "healthy" | "idle" | "stalled",
        "vram_free": int | None,   # from comfy.model_management.get_free_memory
        "vram_total": int | None,
      }
    """
    pending = 0
    running = 0
    try:
        pq = getattr(PromptServer.instance, "prompt_queue", None)
        if pq is not None and hasattr(pq, "get_queue_remaining"):
            remaining = pq.get_queue_remaining()
            if isinstance(remaining, (list, tuple)) and len(remaining) >= 2:
                running = len(remaining[0])
                pending = len(remaining[1])
    except Exception as e:
        log.debug("[queue_health] get_queue_remaining error: %s", e)

    if running > 0:
        verdict = "healthy"
    elif pending > 0:
        verdict = "stalled"
    else:
        verdict = "idle"

    vram_free = None
    vram_total = None
    try:
        import torch
        if torch.cuda.is_available():
            vram_free, vram_total = torch.cuda.mem_get_info()
    except Exception:
        pass

    return _json({
        "pending": pending,
        "running": running,
        "verdict": verdict,
        "vram_free": vram_free,
        "vram_total": vram_total,
    })


# ── Text-to-pose async job endpoints ─────────────────────────────────────────
# These were previously in gateway/poser_text.py (deleted in the forge purge).
# Reimplemented here so Pose Studio's TextToPoseTab can generate Kimodo motion
# without the old forge layer. The pipeline:
#   1. KimodoPoseService.infer() — run diffusion → SOMA-77 NPZ
#   2. Parse NPZ → posed_joints (T, 77, 3) + global_rot_mats (T, 77, 3, 3)
#   3. SkinEngine.deform() — skin the ANNY mesh for every frame
#   4. Build PoseMotion response (the format TextToPoseTab expects)

# NOTE: POST /poser/text-to-pose + GET/DELETE /poser/text-to-pose/jobs/{job_id}
# were DELETED 2026-07-26. The Kimodo GPU diffusion pipeline they wrapped is
# now exposed as proper ComfyUI custom nodes (KimodoLoader + KimodoTextToPose
# in custom_nodes/melite-kimodo-nodes/nodes.py) using PipelinePatcher — ComfyUI
# auto-manages the ~17GB model's VRAM. NO async job-polling, NO module-global
# singleton (_kimodo_svc), NO base64 HTTP wrapper. The static GET endpoints
# /poser/text-to-pose/registry + /poser/text-to-pose/models remain (they
# serve dataset/model catalogs used by Pose Studio's frontend).


@routes.get("/poser/library/corpus")
async def library_corpus_list(request: web.Request) -> web.Response:
    """GET /poser/library/corpus — list all built-in poses (metadata only).

    Returns:
        {"poses": [{slug, name, description, category, fps, frame_count,
                    is_motion, tags, source}, ...], "count": N}
    """
    from .library.pose_corpus import corpus_summary
    try:
        poses = await asyncio.to_thread(corpus_summary)
    except FileNotFoundError as e:
        return _err(str(e), 503)
    except Exception:
        log.exception("corpus_summary failed")
        return _err("internal error loading corpus", 500)
    return _json({"poses": poses, "count": len(poses)})


@routes.get("/poser/library/corpus/{slug}")
async def library_corpus_detail(request: web.Request) -> web.Response:
    """GET /poser/library/corpus/{slug} — full pose including (T, 77, 3) frames."""
    from .library.pose_corpus import get_pose
    slug = request.match_info.get("slug", "")
    try:
        pose = await asyncio.to_thread(get_pose, slug)
    except FileNotFoundError as e:
        return _err(str(e), 503)
    except Exception:
        log.exception("get_pose failed")
        return _err("internal error loading pose", 500)
    if pose is None:
        return _err(f"Unknown pose slug: {slug!r}", 404)
    return _json(pose.to_dict())


@routes.post("/poser/library/import_bvh")
async def library_import_bvh(request: web.Request) -> web.Response:
    """POST /poser/library/import_bvh — retarget BVH text → SOMA-77 positions.

    Body:
        {
            "bvh": "<BVH file contents as UTF-8 string>",
            "max_frames": 240,      // optional cap (default 240, hard cap 720)
            "start_at_hips": true   // optional (default true)
        }
    Returns:
        {
            "frames": [[[x, y, z], ...], ...],   // (T, 77, 3) in meters
            "frame_count": T,
            "joint_count": 77,
            "fps": 30.0
        }
    """
    try:
        raw = await request.read()
    except Exception as e:
        return _err(f"Failed to read body: {e}", 400)
    if len(raw) > _MAX_BVH_BYTES:
        return _err(f"BVH payload exceeds {_MAX_BVH_BYTES} bytes", 413)
    try:
        body = json.loads(raw)
    except Exception:
        return _err("Invalid JSON body", 400)

    bvh_text = body.get("bvh")
    if not isinstance(bvh_text, str) or not bvh_text.strip():
        return _err("Missing or empty 'bvh' field (must be a non-empty string)", 400)

    try:
        max_frames = int(body.get("max_frames", _MAX_BVH_FRAMES_DEFAULT))
    except (TypeError, ValueError):
        max_frames = _MAX_BVH_FRAMES_DEFAULT
    max_frames = max(1, min(max_frames, _MAX_BVH_FRAMES_HARD))
    start_at_hips = bool(body.get("start_at_hips", True))

    def _do_retarget():
        from .library.bvh_retarget import retarget_bvh_text
        return retarget_bvh_text(bvh_text, max_frames=max_frames,
                                 start_at_hips=start_at_hips)

    try:
        frames = await asyncio.to_thread(_do_retarget)
    except ValueError as e:
        return _err(str(e), 422)
    except Exception:
        log.exception("BVH import failed unexpectedly")
        return _err("Internal error during retargeting", 500)

    return _json({
        "frames": np.round(frames.astype(np.float64), 5).tolist(),
        "frame_count": int(frames.shape[0]),
        "joint_count": int(frames.shape[1]),
        "fps": 30.0,
    })


@routes.get("/poser/library/cmu_catalog")
async def library_cmu_catalog(request: web.Request) -> web.Response:
    """GET /poser/library/cmu_catalog — browse the full upstream CMU catalog.

    Query params: q, category, limit (≤2000), offset
    """
    from .library.pose_corpus import list_cmu_catalog
    q = request.query.get("q")
    category = request.query.get("category")
    try:
        limit = int(request.query.get("limit", 200))
    except ValueError:
        limit = 200
    limit = max(1, min(limit, 2000))
    try:
        offset = int(request.query.get("offset", 0))
    except ValueError:
        offset = 0
    try:
        result = await asyncio.to_thread(
            list_cmu_catalog, q, category, limit, offset
        )
    except FileNotFoundError as e:
        return _err(str(e), 503)
    except Exception:
        log.exception("CMU catalog list failed")
        return _err("internal error loading CMU catalog", 500)
    return _json(result)


@routes.post("/poser/library/cmu_fetch")
async def library_cmu_fetch(request: web.Request) -> web.Response:
    """POST /poser/library/cmu_fetch — retarget one CMU clip on demand.

    Body: {"subject": 13, "trial": 17, "max_frames": 240}
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _err("Invalid JSON body", 400)
    try:
        subject = int(body["subject"])
        trial = int(body["trial"])
    except (KeyError, TypeError, ValueError):
        return _err("Body must include integer 'subject' and 'trial'", 400)
    max_frames = int(body.get("max_frames", 240))
    max_frames = max(1, min(max_frames, 720))

    def _do_fetch():
        from .library.pose_corpus import fetch_cmu_motion
        return fetch_cmu_motion(subject, trial, max_frames=max_frames)

    try:
        frames = await asyncio.to_thread(_do_fetch)
    except Exception as e:
        log.exception("CMU on-demand fetch failed")
        return _err(f"CMU fetch/retarget failed: {e}", 502)

    return _json({
        "frames": np.round(frames.astype(np.float64), 5).tolist(),
        "frame_count": int(frames.shape[0]),
        "joint_count": int(frames.shape[1]),
        "fps": 30.0,
        "subject": subject,
        "trial": trial,
        "source": (f"CMU Graphics Lab Motion Capture Database "
                   f"(mocap.cs.cmu.edu) — subject {subject:02d}, trial {trial:02d}"),
    })


# ── 2D pose render endpoint ────────────────────────────────────────────────
# Renders COCO-18 keypoints (flat [x, y, c, ...] array) to a PNG stick figure.
# Vendored from the deleted gateway /poser/render pipeline — but rendered
# with PIL (always available in ComfyUI) instead of matplotlib, so we don't
# add a heavy dep to the inference container.

# COCO-18 skeleton pairs (same as the client-side OpenPose renderer +
# the old gateway render endpoint).
_SKELETON_PAIRS = [
    (0, 1), (0, 14), (0, 15), (1, 2), (1, 5),
    (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (9, 10), (11, 12), (12, 13),
    (14, 16), (15, 17), (1, 8), (1, 11), (8, 11),
]


def _render_skeleton_png(
    keypoints: np.ndarray,
    width: int = 1024,
    height: int = 1024,
    line_width: int = 5,
    point_radius: int = 6,
    background: str = "#0e0e1a",
    limb_color: str = "#98bdf7",
    joint_color: str = "#00ffff",
) -> bytes:
    """Render a single COCO-18 keypoint set to PNG bytes (RGBA).

    keypoints: (54,) flat [x, y, c, ...] — x/y in pixel coords, c = confidence.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(img)

    xs = keypoints[0::3]
    ys = keypoints[1::3]
    cs = keypoints[2::3]

    # Limbs
    for a, b in _SKELETON_PAIRS:
        if cs[a] < 0.1 or cs[b] < 0.1:
            continue
        draw.line(
            [(float(xs[a]), float(ys[a])), (float(xs[b]), float(ys[b]))],
            fill=limb_color, width=line_width,
        )

    # Joints
    for i in range(18):
        if cs[i] < 0.1:
            continue
        x = float(xs[i]); y = float(ys[i])
        draw.ellipse(
            [x - point_radius, y - point_radius,
             x + point_radius, y + point_radius],
            fill=joint_color,
        )

    import io as _io
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@routes.post("/poser/render")
async def poser_render(request: web.Request) -> web.Response:
    """POST /poser/render — render COCO-18 keypoints to a PNG stick figure.

    Body:
        {
            "keypoints": [x0, y0, c0, x1, y1, c1, ...],  // (54,) flat
            "width":  1024,  // optional (default 1024)
            "height": 1024,  // optional (default 1024)
            "line_width": 5,
            "point_radius": 6,
            "background": "#0e0e1a"  // or "transparent"
        }
    Returns: image/png
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _err("Request body must be JSON", 400)

    raw_kp = body.get("keypoints")
    if not isinstance(raw_kp, list) or len(raw_kp) != 54:
        return _err("keypoints must be a list of 54 numbers (18 COCO joints × 3)", 400)
    try:
        kp = np.asarray(raw_kp, dtype=np.float32)
    except Exception:
        return _err("keypoints contains non-numeric values", 400)

    width = int(body.get("width", 1024))
    height = int(body.get("height", 1024))
    line_width = int(body.get("line_width", 5))
    point_radius = int(body.get("point_radius", 6))
    background = str(body.get("background", "#0e0e1a"))

    if width < 32 or width > 4096 or height < 32 or height > 4096:
        return _err("width/height must be in [32, 4096]", 400)

    try:
        png_bytes = await asyncio.to_thread(
            _render_skeleton_png, kp, width, height, line_width,
            point_radius, background,
        )
    except Exception:
        log.exception("skeleton PNG render failed")
        return _err("internal error during render", 500)

    return web.Response(body=png_bytes, content_type="image/png")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _json(obj: Any) -> web.Response:
    """aiohttp JSON response with numpy-aware serialization."""
    return web.json_response(_to_native(obj))


def _err(msg: str, status: int = 400) -> web.Response:
    return web.json_response({"error": msg}, status=status)


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# ── Mesh rasterization endpoint ────────────────────────────────────────────

@routes.post("/poser/skin/render")
async def post_skin_render(request: web.Request) -> web.Response:
    """POST /poser/skin/render — rasterize deformed mesh → depth/normal/RGB.

    Accepts posed vertices (from /poser/skin/deform_soma) + faces and
    renders them to ControlNet-ready raster passes using nvdiffrast.

    Body:
        {
            "vertices_b64": "...",   # base64 float32 (V×3), from deform_soma
            "faces": [[0,1,2], ...], # (F,3) int triangle indices
            "width":  1024,          # optional [32–4096]
            "height": 1024,
            "azimuth": 0.0,          # orbit angle degrees (0 = front)
            "elevation": 5.0,        # vertical angle degrees
            "fov": 30.0,             # vertical field of view degrees
            "passes": ["depth", "normal", "rgb"],  # subset
            "background": "black"    # "black" or "white"
        }

    Returns JSON:
        { "depth": "<base64 PNG>", "normal": "...", "rgb": "..." }
    """
    import base64 as _b64

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _err("Request body must be JSON", 400)

    verts_b64 = body.get("vertices_b64")
    raw_faces = body.get("faces")
    if not verts_b64:
        return _err("vertices_b64 is required (base64 float32 from deform_soma)", 400)
    if not isinstance(raw_faces, list) or len(raw_faces) == 0:
        return _err("faces must be a non-empty list of [v0, v1, v2] triangles", 400)

    try:
        verts_bytes = _b64.b64decode(verts_b64)
        verts = np.frombuffer(verts_bytes, dtype=np.float32)
        if verts.shape[0] % 3 != 0:
            return _err("vertices_b64 length is not a multiple of 3", 400)
        verts = verts.reshape(-1, 3)
        faces = np.asarray(raw_faces, dtype=np.int32)
    except Exception as exc:
        return _err(f"failed to decode mesh: {exc}", 400)

    width = int(body.get("width", 1024))
    height = int(body.get("height", 1024))
    azimuth = float(body.get("azimuth", 0.0))
    elevation = float(body.get("elevation", 5.0))
    fov = float(body.get("fov", 30.0))
    passes = body.get("passes", ["depth", "normal", "rgb"])
    background = str(body.get("background", "black"))
    target_y = float(body.get("target_y", 0.0))
    framing = float(body.get("framing", 1.5))

    if not isinstance(passes, list) or not passes:
        return _err("passes must be a non-empty list", 400)
    invalid = set(passes) - {"depth", "normal", "rgb"}
    if invalid:
        return _err(f"unknown pass type(s): {invalid}", 400)

    try:
        pngs = await asyncio.to_thread(
            render_passes,
            verts,
            faces,
            width=width,
            height=height,
            azimuth=azimuth,
            elevation=elevation,
            fov=fov,
            passes=passes,
            background=background,
            target_y=target_y,
            framing=framing,
        )
    except ValueError as exc:
        return _err(str(exc), 400)
    except RuntimeError as exc:
        log.exception("mesh render failed")
        return _err(str(exc), 500)
    except Exception:
        log.exception("unexpected error during mesh render")
        return _err("internal error during render", 500)

    return _json({name: _b64.b64encode(data).decode("ascii") for name, data in pngs.items()})


# ── TRELLIS image-to-3D endpoint ────────────────────────────────────────────

# NOTE: /poser/trellis/image-to-3d was DELETED 2026-07-26.
#
# It violated the comfyui-script-only architecture policy: pipeline
# code MUST go through media.comfyui.workflows (ComfyScript node graph)
# → submit_comfy_workflow → ComfyUI's prompt_worker, NOT through HTTP
# side-routes that load models into module globals ComfyUI can't see.
#
# The HTTP route loaded TRELLIS via trellis_bridge._pipeline_cache (a
# module-global dict invisible to ComfyUI's ModelPatcher refcounting).
# The pipeline's second TRELLIS call (hair/outfit asset generation) then
# OOM'd at decode_shape_slat because ComfyUI didn't know to unload the
# first one. The "fix" of _unload_all_custom_caches was a band-aid over
# this architectural bug — also removed.
#
# The replacement: core/pipelines/character.py:stage3_trellis now
# calls media.comfyui.workflows.trellis.image_to_glb, which wires
# the registered Trellis2Loader + Trellis2ImageTo3D ComfyScript nodes.
# ComfyUI owns the pipeline; refcounting works; no leak; no manual
# cleanup needed.
#
# trellis_bridge.py is also gone. If you need TRELLIS, use the
# ComfyScript node (custom_nodes/melite-trellis-nodes/nodes.py).


# NOTE: /poser/trellis/project_texture was DELETED 2026-07-26.
#
# It violated the same comfyui-script-only policy as /poser/trellis/image-to-3d
# (deleted above). Texture projection went through two more revisions after that:
#
#   2026-07-26 — 2026-07-28: RayProjectViewsToTexture custom node (hand-rolled
#               nvdiffrast UV-rasterization in custom_nodes/melite-poser-nodes/
#               texture_project.py). Worked, but had a camera-framing mismatch:
#               orbit_radius=1.75 made the unit-normalized mesh fill only 65%
#               of the camera frustum while qwen-edit source images had the
#               character filling 100% of the frame → projection sampled
#               chest pixels where face pixels were expected.
#
#   2026-07-28 — present: media/comfyui/families/texture_project.py uses
#               the UPSTREAM Comfy3D (MrForExample) `[Comfy3D] ExplicitTarget
#               Color Projection` node with orbit_radius=1.14 (=
#               0.5 / tan(fovy/2)) so mesh + source frames match. Surgical
#               swap preserves rig (SOMAX skins/joints). RayProjectViewsToTexture
#               and project_views_to_texture are deleted.
#
# The DRY contract persists: ONE texture technique for figure, hair, AND outfit.
# All three callers go through media/comfyui/families/texture_project.py
# (Layer-4 brick, registered as family "texture_project"). No base64 HTTP route,
# no in-process nvdiffrast on melite-head, no parallel implementation.


# NOTE: 4 GPU/side routes were DELETED 2026-07-26 per the comfyui-script-only
# architecture policy. Each is now a registered ComfyUI custom node in
# custom_nodes/melite-poser-nodes/nodes.py, driven through ComfyScript →
# submit_comfy_workflow. NO HTTP base64, NO module-global caches.
#
#   /poser/trellis/xatlas_unwrap     → RayXatlasUnwrap node (CPU, uv_cleanup.py)
#   /poser/segment/hair_clothing     → RaySAMSegmentHairClothing node (GPU,
#                                       SAM wrapped in PipelinePatcher via
#                                       sam_segment.py — ComfyUI-tracked VRAM)
#   /poser/multiview/unique3d        → DELETED (dead code: the CLI
#                                       --proj-method choices no longer include
#                                       'unique3d'; Unique3D repo also no
#                                       longer installed in the container).
#   /poser/free_vram                 → DELETED (no callers; was the manual-
#                                       cleanup anti-pattern. PipelinePatcher
#                                       wrapping of SAM/TRELLIS/anigen means
#                                       ComfyUI's free_memory handles eviction
#                                       natively — no manual cache dropping).
#
# Also deleted (manual model-management anti-patterns made obsolete by
# PipelinePatcher wrapping in sam_segment.py):
#   _unload_all_custom_caches()  — was called by unique3d + free_vram routes
#   _unload_u3d_modules()        — was called by unique3d route
#   _sam_predictor module global — replaced by PipelinePatcher-tracked cache
#   _get_sam(), _sam_checkpoint_path(), _SAM_CKPT_CANDIDATES  — moved into
#       sam_segment.py (engine module owned by the RaySAMSegmentHairClothing
#       node).


@routes.post("/poser/skin/fit_identity")
async def post_fit_identity(request: web.Request) -> web.Response:
    """POST /poser/skin/fit_identity — fit Anny coeffs to a target mesh.

    Body: {vertices_b64, identity_model?, max_iter?, n_target_samples?}
    Returns: {identity_coeffs, chamfer_loss, iterations, elapsed_s}

    The target mesh is typically a TRELLIS output (arbitrary topology).
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _err("invalid JSON body", 400)

    verts_b64 = body.get("vertices_b64", "")
    if not verts_b64:
        return _err("vertices_b64 required", 400)

    import base64 as _b64
    try:
        verts_raw = _b64.b64decode(verts_b64)
        target_verts = np.frombuffer(verts_raw, dtype=np.float32).reshape(-1, 3)
    except Exception:
        return _err("invalid vertices_b64 (expected base64 float32 Nx3)", 400)

    identity_model = str(body.get("identity_model", "anny")).lower()
    max_iter = int(body.get("max_iter", 200))
    n_samples = int(body.get("n_target_samples", 5000))

    try:
        from .fit_identity import fit_identity
        result = await asyncio.to_thread(
            fit_identity,
            target_verts,
            identity_model=identity_model,
            max_iter=max_iter,
            n_target_samples=n_samples,
        )
        return _json(result)
    except Exception as exc:
        log.exception("fit_identity failed")
        return _err(str(exc), 500)


### ── Body Studio "From Photo" — one-shot image→TRELLIS→Anny ───────────────
# Chained endpoint that the BodyStudioTab "From Photo" button calls.
# Runs TRELLIS image-to-3D (15-90s) then Chamfer identity fitting (~7s)
# in a single HTTP call so the frontend doesn't have to orchestrate.

@routes.post("/poser/body-fit/image-to-anny")
async def post_body_fit_image_to_anny(request: web.Request) -> web.Response:
    """POST /poser/body-fit/image-to-anny — image → TRELLIS mesh → Anny coeffs.

    Body: {image_b64, mode?: 'quick'|'accurate'}
    Returns: {persons: [{phenotype_coeffs: [...]}], vertex_count,
              chamfer_loss, trellis_elapsed_s, fit_elapsed_s}

    The 'quick' mode uses 512-resolution TRELLIS (~15s); 'accurate' uses
    1024 (~60s).  Both then run the Chamfer optimizer (200 iterations, ~7s).
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _err("invalid JSON body", 400)

    image_b64 = body.get("image_b64", "")
    if not image_b64:
        return _err("image_b64 required", 400)

    mode = str(body.get("mode", "quick")).lower()
    resolution = "512" if mode != "accurate" else "1024"

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return _err("invalid image_b64", 400)

    try:
        # ── Step 1: TRELLIS image → mesh ──────────────────────────────
        from .trellis_bridge import image_to_mesh
        mesh_result = await asyncio.to_thread(
            image_to_mesh, image_bytes,
            seed=42, resolution=resolution, decimation=200_000,
        )
        vertices = mesh_result["vertices"]
        log.info(
            "[body-fit] TRELLIS: %d verts in %.1fs",
            mesh_result["vertex_count"], mesh_result["elapsed_s"],
        )

        # ── Step 2: Chamfer fit mesh → Anny coeffs ─────────────────────
        from .fit_identity import fit_identity
        fit_result = await asyncio.to_thread(
            fit_identity, vertices,
            identity_model="anny", max_iter=200, n_target_samples=5000,
        )
        log.info(
            "[body-fit] Chamfer: loss=%.6f, %d evals in %.1fs",
            fit_result["chamfer_loss"], fit_result["evaluations"],
            fit_result["elapsed_s"],
        )

        return _json({
            "persons": [{
                "phenotype_coeffs": fit_result["identity_coeffs"],
            }],
            "vertex_count": mesh_result["vertex_count"],
            "chamfer_loss": fit_result["chamfer_loss"],
            "trellis_elapsed_s": mesh_result["elapsed_s"],
            "fit_elapsed_s": fit_result["elapsed_s"],
        })
    except Exception as exc:
        log.exception("body-fit/image-to-anny failed")
        return _err(str(exc), 500)


# SkinEngine.default_faces helper — monkey-patched onto the class for now
# (avoids changing skin.py just to rename faces → default_faces).
def _engine_default_faces(self):
    return self.faces

from .skin import SkinEngine  # noqa: E402
SkinEngine.default_faces = _engine_default_faces


# ── Custom SOMAX GLB loading + deform (Pose Studio integration) ────────────
#
# Allows loading external SOMAX meshes (from the TRELLIS → SOMA weight transfer
# pipeline) into Pose Studio. The GLB contains glTF skinning data (JOINTS_0,
# WEIGHTS_0, IBM) that we use for manual LBS deformation.

# SOMA-77 → Mixamo bone name mapping (subset — main body joints)
_SOMA_TO_MIXAMO = {
    0: "Hips", 1: "Spine", 2: "Spine1", 3: "Spine2",
    4: "Neck", 5: "Neck1", 6: "Head",
    11: "LeftShoulder", 12: "LeftArm", 13: "LeftForeArm", 14: "LeftHand",
    39: "RightShoulder", 40: "RightArm", 41: "RightForeArm", 42: "RightHand",
    67: "LeftUpLeg", 68: "LeftLeg", 69: "LeftFoot", 70: "LeftToeBase",
    72: "RightUpLeg", 73: "RightLeg", 74: "RightFoot", 75: "RightToeBase",
}

# Reverse: Mixamo bone name → SOMA-77 index
_MIXAMO_TO_SOMA = {v: k for k, v in _SOMA_TO_MIXAMO.items()}

# Cache for loaded custom meshes: {cache_key: {vertices, faces, joints, weights, ibm, soma_map}}
_custom_mesh_cache: dict[str, dict] = {}


def _load_glb_skinning(glb_path: str) -> dict:
    """Load a SOMAX GLB and extract full skinning data for LBS deformation.

    Extracts: vertices, faces, per-vertex joint indices (JOINTS_0),
    per-vertex weights (WEIGHTS_0), inverse bind matrices (IBM), and
    the GLB-joint → SOMA-77 index mapping.

    The SOMA weight transfer (soma_weight_transfer.py) writes Mixamo-named
    bones. We reverse SOMA_IDX_TO_MIXAMO_NAME to map each GLB joint to its
    SOMA-77 index, then reorder IBMs into SOMA-77 index space so deform_custom
    can index directly by SOMA-77 joint.
    """
    import json
    import struct

    # ── Parse GLB binary ONCE — extract geometry + skinning from the same
    # buffer so vertex indices line up with JOINTS_0/WEIGHTS_0. (trimesh
    # returns a Scene for rigged GLBs and would renumber vertices.)
    with open(glb_path, "rb") as f:
        magic = f.read(4)
        if magic != b"glTF":
            return {"vertices": np.zeros((0, 3), np.float32),
                    "faces": np.array([], np.int32), "has_skinning": False}
        f.read(8)  # version + length
        json_len = struct.unpack("<I", f.read(4))[0]
        f.read(4)  # chunk type
        gltf_json = json.loads(f.read(json_len))
        bin_len = struct.unpack("<I", f.read(4))[0]
        f.read(4)  # chunk type
        bin_data = f.read(bin_len)

    def _read_accessor(acc_idx):
        acc = gltf_json["accessors"][acc_idx]
        bv = gltf_json["bufferViews"][acc["bufferView"]]
        comp_types = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
                       5123: np.uint16, 5125: np.uint32, 5126: np.float32}
        type_comps = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
        dt = comp_types[acc["componentType"]]
        nc = type_comps[acc["type"]]
        offset = (bv.get("byteOffset", 0) or 0) + (acc.get("byteOffset", 0) or 0)
        count = acc["count"]
        arr = np.frombuffer(bin_data, dtype=dt, count=count * nc, offset=offset)
        return arr.reshape(count, nc).copy()

    # ── Extract geometry from first mesh primitive ──
    meshes = gltf_json.get("meshes", [])
    if not meshes:
        return {"vertices": np.zeros((0, 3), np.float32),
                "faces": np.array([], np.int32), "has_skinning": False}
    prim = meshes[0]["primitives"][0]
    attrs = prim.get("attributes", {})

    pos_acc = attrs.get("POSITION")
    if pos_acc is None:
        return {"vertices": np.zeros((0, 3), np.float32),
                "faces": np.array([], np.int32), "has_skinning": False}
    vertices = _read_accessor(pos_acc).astype(np.float32)

    # Faces (indices) — if missing, synthesize a trivial 0..N-1 list
    if "indices" in prim:
        idx_acc = prim["indices"]
        idx = _read_accessor(idx_acc).astype(np.int32).reshape(-1)
        faces = idx.reshape(-1, 3).astype(np.int32)
    else:
        v_count = vertices.shape[0]
        faces = np.arange(v_count, dtype=np.int32).reshape(-1, 3)

    result: dict = {"vertices": vertices, "faces": faces, "has_skinning": False}

    # ── Skinning accessors (JOINTS_0 / WEIGHTS_0 / IBM) parsed from the
    # same binary we already opened above — no re-read needed.
    skins = gltf_json.get("skins", [])
    meshes = gltf_json.get("meshes", [])
    if not skins or not meshes:
        return result

    skin = skins[0]
    prim = meshes[0]["primitives"][0]
    attrs = prim.get("attributes", {})
    joints_acc = attrs.get("JOINTS_0")
    weights_acc = attrs.get("WEIGHTS_0")
    if joints_acc is None or weights_acc is None:
        return result

    joints_raw = _read_accessor(joints_acc).astype(np.int32)    # (V, 4)
    weights_raw = _read_accessor(weights_acc).astype(np.float32) # (V, 4)

    ibm_acc = skin.get("inverseBindMatrices")
    if ibm_acc is not None:
        ibm_raw = _read_accessor(ibm_acc).astype(np.float32)     # (J, 16)
        # GLTF MAT4 is COLUMN-MAJOR (per spec §5.25): for ONE matrix, the 16
        # floats lay out its 4 columns contiguously. Reading them with
        # `reshape(4, 4, order="F")` is correct.
        #
        # BUT — for an ARRAY of J matrices, `reshape(J, 4, 4, order="F")` does
        # NOT give "J column-major matrices". The `order` flag applies to the
        # WHOLE tensor, making the FIRST axis (J) vary fastest in the flat
        # representation. That interleaves all J matrices' [0,0] elements,
        # then all [1,0] elements, etc. — scrambling every matrix.
        #
        # This was the root cause of the SOMAX body-horror bug: the IBM was
        # scrambled, so bind_world = inv(IBM) had joints at nonsense positions
        # (e.g. Hips at Y=-67cm, L Foot at X=3552cm). The LBS deformation then
        # sent every vertex to the wrong place, the bind-pose round-trip was
        # off by 25cm mean / 878cm max, and MIMO saw "skeleton collapses into
        # incoherent vertical line" (because the scrambled IBM produced a
        # single dominant eigenvector along Y).
        #
        # Correct method: reshape row-major (J matrices, each contiguous),
        # then transpose the INNER 4x4 to swap row/column interpretation.
        ibm_glb = ibm_raw.reshape(-1, 4, 4).transpose(0, 2, 1).copy()
    else:
        num_joints = len(skin["joints"])
        ibm_glb = np.tile(np.eye(4, dtype=np.float32), (num_joints, 1, 1))

    # ── Map GLB joint node names → SOMA-77 indices ──
    nodes = gltf_json.get("nodes", [])
    joint_node_idxs = skin["joints"]
    joint_names = []
    for ni in joint_node_idxs:
        nm = nodes[ni].get("name", "")
        joint_names.append(nm.replace("mixamorig:", ""))

    # Reverse lookup: Mixamo short-name → SOMA-77 index. The table
    # lives in the SIBLING pack melite-autorig-nodes (its
    # soma_weight_transfer.py) — resolved through the runtime's
    # custom_nodes roots, never a hardcoded host path (the audit's
    # blocker: /root/... resolved on no install anywhere).
    try:
        from soma_weight_transfer import SOMA_IDX_TO_MIXAMO_NAME
    except ImportError:
        import sys
        from pathlib import Path
        import folder_paths
        for root in folder_paths.get_folder_paths("custom_nodes"):
            candidate = Path(root) / "melite-autorig-nodes"
            if (candidate / "soma_weight_transfer.py").is_file():
                sys.path.insert(0, str(candidate))
                break
        else:
            raise RuntimeError(
                "soma_weight_transfer not found: install the "
                "melite-autorig-nodes pack (it owns the SOMA joint table "
                "the skinning reverse-lookup needs)"
            )
        from soma_weight_transfer import SOMA_IDX_TO_MIXAMO_NAME
    mixamo_to_soma = {v: k for k, v in SOMA_IDX_TO_MIXAMO_NAME.items()}

    glb_to_soma = np.zeros(len(joint_names), dtype=np.int32)
    import re
    soma_name_re = re.compile(r"^SOMA_(\d+)$")
    unmapped_count = 0
    for i, nm in enumerate(joint_names):
        # Try direct Mixamo name lookup first
        soma_idx = mixamo_to_soma.get(nm, -1)
        if soma_idx < 0:
            # Handle End Site bones named "SOMA_N" — the SOMA weight transfer
            # writes these for joints that have no Mixamo equivalent (indices
            # 23, 28, 33, 38, 51, 56, 61, 66). The index is encoded in the
            # name. Mapping these to 0 (Hips) by default produces body horror:
            # vertices weighted to End Site bones stay pinned at the hips
            # while the rest of the body moves.
            m = soma_name_re.match(nm)
            if m:
                soma_idx = int(m.group(1))
            else:
                soma_idx = 0  # truly unknown — fall back to Hips
                unmapped_count += 1
                log.warning("load_custom: unmapped joint '%s' → Hips(0)", nm)
        if soma_idx > 76:
            soma_idx = 0
        glb_to_soma[i] = soma_idx
    if unmapped_count:
        log.warning("load_custom: %d joint(s) could not be mapped", unmapped_count)

    # Remap per-vertex joints from GLB indices → SOMA-77 indices
    joints_soma = glb_to_soma[joints_raw]  # (V, 4)

    # Reorder IBM into SOMA-77 index space (77 joints)
    ibm_soma = np.tile(np.eye(4, dtype=np.float32), (77, 1, 1))
    for glb_idx, soma_idx in enumerate(glb_to_soma):
        if glb_idx < ibm_glb.shape[0]:
            ibm_soma[soma_idx] = ibm_glb[glb_idx]

    # ── Optional T_align: SOMA-template → source-mesh coordinate frame ──
    # The IBM lives in source-mesh space (because soma_weight_transfer applied
    # T_align to soma_bind before computing IBM). But Kimodo emits motion in
    # SOMA-template space. deform_custom uses t_align (if present) to map
    # incoming positions/rotations into source-mesh space so the LBS World
    # matrices share a frame with the IBM. Without this, the LBS mixes two
    # coordinate frames and produces body horror (mesh stretched 1.7× in Y,
    # feet 30cm from any joint, MIMO sees bones collapse).
    t_align = np.eye(4, dtype=np.float32)
    t_align_extras = skin.get("extras") or {}
    raw = t_align_extras.get("somaT_alignRowMajor")
    if isinstance(raw, list) and len(raw) == 16:
        t_align = np.asarray(raw, dtype=np.float32).reshape(4, 4)
        log.info("load_custom: T_align detected (uniform_scale=%.4f)",
                 float(np.linalg.norm(t_align[0, :3])))
    else:
        log.warning("load_custom: no T_align in skin extras — motion will be "
                    "in SOMA template space, mismatched with source-mesh IBM")

    result["joints"] = joints_soma.astype(np.int32)   # (V, 4) SOMA-77 indices
    result["weights"] = weights_raw                    # (V, 4)
    result["ibm"] = ibm_soma                           # (77, 4, 4)
    result["t_align"] = t_align                        # (4, 4) SOMA→source frame
    result["has_skinning"] = True

    # Detect vertex unit: TRELLIS GLB is meters (~1.7), SOMA template is
    # centimeters (~170). Store the unit so deform_custom can convert input
    # positions (always meters from Kimodo NPZ) to match. We do NOT rescale
    # the vertices or IBM — keeping them in their native unit preserves the
    # IBM↔vertex relationship the GLB was authored with.
    vert_extent = float(np.max(np.linalg.norm(vertices, axis=1)))
    if vert_extent < 10.0:
        result["unit"] = "m"        # meters
        result["scale_factor"] = 1.0
    else:
        result["unit"] = "cm"       # centimeters
        result["scale_factor"] = 1.0

    log.info("load_custom: %d verts, %d faces, skinning=%s, scale=%.1fx",
             vertices.shape[0], faces.shape[0] if faces.size > 0 else 0,
             result["has_skinning"], result["scale_factor"])
    return result


@routes.post("/poser/skin/load_custom")
async def post_load_custom_mesh(request: web.Request) -> web.Response:
    """POST /poser/skin/load_custom — load a SOMAX GLB for Pose Studio.

    Body: {glb_path: str} or {glb_b64: str, cache_key: str}

    Caches the skinning data for subsequent deform_custom calls.
    Returns vertex_count and cache_key.
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _err("invalid JSON body", 400)

    glb_path = body.get("glb_path", "")
    glb_b64 = body.get("glb_b64", "")
    cache_key = body.get("cache_key") or str(uuid.uuid4())

    if glb_b64:
        # Write base64 GLB to a temp file
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        tmp.write(base64.b64decode(glb_b64))
        tmp.close()
        glb_path = tmp.name

    if not glb_path:
        return _err("glb_path or glb_b64 required", 400)

    try:
        data = await asyncio.to_thread(_load_glb_skinning, glb_path)
    except Exception as e:
        log.exception("load_custom failed")
        return _err(f"Failed to load GLB: {e}", 500)

    _custom_mesh_cache[cache_key] = data

    # Return T_align + unit so clients can transform SOMA-space motion joints
    # into the same source-mesh frame as the deformed vertices (needed for
    # skeleton-overlay rendering that aligns with the mesh silhouette).
    resp: dict[str, Any] = {
        "cache_key": cache_key,
        "vertex_count": int(data["vertices"].shape[0]),
        "face_count": int(data["faces"].shape[0]) if data["faces"].size > 0 else 0,
        "unit": data.get("unit", "cm"),
    }
    t_align = data.get("t_align")
    if t_align is not None:
        resp["t_align_rowmajor"] = [float(x) for x in np.asarray(t_align).reshape(16)]
    return _json(resp)


@routes.post("/poser/skin/deform_custom")
async def post_deform_custom_mesh(request: web.Request) -> web.Response:
    """POST /poser/skin/deform_custom — deform a loaded SOMAX mesh.

    Body: {
        cache_key: str,
        frames_soma77_3d: [[x,y,z]*77],
        frames_soma77_rotmat?: (T, 77, 3, 3),
        binary?: bool
    }

    Returns deformed vertices (base64 float32 if binary, else JSON list).
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _err("invalid JSON body", 400)

    cache_key = body.get("cache_key", "")
    if cache_key not in _custom_mesh_cache:
        return _err(f"cache_key '{cache_key}' not found. Call /poser/skin/load_custom first.", 400)

    data = _custom_mesh_cache[cache_key]
    pos = body.get("frames_soma77_3d")
    if pos is None:
        return _err("frames_soma77_3d required", 400)

    # ── Linear Blend Skinning ─────────────────────────────────────────
    # LBS formula: deformed_v[i] = Σ_k weights[i,k] * (D[joints[i,k]] @ v_homo[i])
    # where D[j] = World[j] @ IBM[j], World[j] = [R_j | p_j; 0 | 1]
    #
    # UNIT HANDLING — Kimodo NPZ positions are METERS; frontend skeleton
    # positions are CENTIMETERS; GLB vertices+IBM are in their native unit
    # (meters for TRELLIS, cm for SOMA template). We:
    #   1. Detect input position unit (max|coord| < 10 → meters).
    #   2. Convert to the GLB vertex unit (data["unit"]) so LBS math agrees.
    #   3. Run LBS in the vertex unit.
    #   4. Convert output to centimeters to match deform_soma's contract.
    verts_native = data["vertices"]                    # (V, 3) in data["unit"]
    V = verts_native.shape[0]
    glb_unit = data.get("unit", "cm")                  # "m" or "cm"

    pos_np = np.asarray(pos, dtype=np.float32)         # (T, 77, 3) or (77, 3)
    if pos_np.ndim == 2:
        pos_np = pos_np[None, ...]
    T = pos_np.shape[0]
    pos_np = pos_np[:, :77].reshape(T, 77, 3)

    # ── Auto-detect INPUT position unit and normalize to the GLB unit ──
    pos_max_abs = float(np.max(np.abs(pos_np))) if pos_np.size else 0.0
    if pos_max_abs < 1e-3:
        return _err("frames_soma77_3d looks empty", 400)
    input_is_meters = pos_max_abs < 10.0
    if input_is_meters and glb_unit == "cm":
        pos_np = (pos_np * 100.0).astype(np.float32)   # m → cm
    elif not input_is_meters and glb_unit == "m":
        pos_np = (pos_np / 100.0).astype(np.float32)   # cm → m
    log.debug("deform_custom: input_meters=%s glb_unit=%s pos_max=%.2f",
              input_is_meters, glb_unit, pos_max_abs)

    rot_np = body.get("frames_soma77_rotmat")
    if rot_np is not None:
        rot_np = np.asarray(rot_np, dtype=np.float32)
        if rot_np.ndim == 3:                          # (77, 3, 3) → (T, 77, 3, 3)
            rot_np = np.broadcast_to(rot_np[None, ...], (T, 77, 3, 3))
        elif rot_np.ndim == 4:
            rot_np = rot_np[:, :77].reshape(T, 77, 3, 3)
        else:
            rot_np = np.broadcast_to(np.eye(3, dtype=np.float32), (T, 77, 3, 3))
    else:
        rot_np = np.broadcast_to(np.eye(3, dtype=np.float32), (T, 77, 3, 3))

    # ── Transform motion from SOMA-template space → source-mesh space ──
    # Kimodo emits (pos_np, rot_np) in SOMA template space; the IBM was built
    # in source-mesh space (after T_align was applied to soma_bind). Without
    # this transform, the LBS mixes two coordinate frames → body horror.
    #
    # T_align is affine: T_align @ [R | p; 0 | 1] @ inv(T_align) gives the
    # frame in source-mesh space. For positions, that's T_align @ p_homo.
    # For rotations: T_align[:3: 3] @ R @ inv(T_align[:3: 3]).
    # _aabb_align_uniform uses uniform scale (s) + axis-permutation (P), so
    # T_align[:3: 3] = s * P with P orthogonal → R_aligned = P @ R @ P.T.
    #
    # Caller can pass `space: "source"` to opt out (e.g., when replaying
    # bind-pose data extracted from inv(IBM), which is already in source space).
    motion_space = body.get("space", "soma")
    t_align = data.get("t_align")
    if (t_align is not None
            and not np.allclose(t_align, np.eye(4))
            and motion_space != "source"):
        A = t_align[:3, :3]                       # linear part = s * P
        b = t_align[:3, 3]                        # translation
        # Solve for orthogonal P (use polar decomposition — for our uniform-
        # scale + permutation case, U = s*I, P = A/s; this generalizes safely).
        # We need P such that P @ P.T = I and A = s * P for some scalar s.
        # Take s = cubic root of |det(A)| (uniform scale).
        s = float(np.cbrt(abs(np.linalg.det(A)))) if np.linalg.det(A) != 0 else 1.0
        if s < 1e-9:
            s = 1.0
        P = A / s
        # Apply: positions get full affine; rotations get conjugated by P.
        # positions: (T, 77, 3). New = A @ p + b broadcast.
        pos_np = (pos_np @ A.T) + b[None, None, :]    # equivalent to A @ p + b
        # rotations: R_new = P @ R @ P.T  per joint per frame.
        # rot_np: (T, 77, 3, 3). Use einsum for batched conjugation.
        rot_np = np.einsum("ij,tajk,kl->tail", P, rot_np, P.T).astype(np.float32)
        log.debug("deform_custom: applied T_align (s=%.4f)", s)

    # ── No skinning data → fallback to bind-pose tile ──
    if not data.get("has_skinning"):
        deformed = np.tile(verts_native[None, ...], (T, 1, 1)).astype(np.float32)
    else:
        joints = data["joints"]                        # (V, 4) SOMA-77 indices
        weights = data["weights"]                      # (V, 4)
        ibm = data["ibm"]                              # (77, 4, 4)

        # Build per-frame, per-joint world transforms (T, 77, 4, 4).
        # pos_np is now in the GLB vertex unit, matching the IBM.
        world = np.zeros((T, 77, 4, 4), dtype=np.float32)
        world[:, :, :3, :3] = rot_np
        world[:, :, :3, 3] = pos_np
        world[:, :, 3, 3] = 1.0

        # Deformation matrices D[j] = World[j] @ IBM[j]
        # ibm is (77, 4, 4); broadcast-matmul against (T, 77, 4, 4)
        D = np.matmul(world, ibm[None, ...])           # (T, 77, 4, 4)

        # Split D into rotation (3x3) and translation (3) parts so we can
        # use einsum on (T, V, 3, 3) instead of (T, V, 4, 4) — half the RAM.
        D_rot = D[:, :, :3, :3]                        # (T, 77, 3, 3)
        D_trans = D[:, :, :3, 3]                       # (T, 77, 3)

        deformed = np.zeros((T, V, 3), dtype=np.float32)
        for k in range(4):
            jk = joints[:, k]                          # (V,) SOMA-77 index for influence k
            wk = weights[:, k].astype(np.float32)      # (V,) weight for influence k
            # Gather per-vertex joint matrices: (T, V, 3, 3) and (T, V, 3)
            R_k = D_rot[:, jk]                         # (T, V, 3, 3)
            t_k = D_trans[:, jk]                       # (T, V, 3)
            # Rotate vertex, add translation: R_k @ verts_native[v] + t_k
            rotated = np.einsum("tvij,vj->tvi", R_k, verts_native)
            transformed = rotated + t_k                # (T, V, 3)
            deformed += wk[None, :, None] * transformed

    # ── Convert output to centimeters to match deform_soma's contract ──
    if glb_unit == "m":
        deformed = (deformed * 100.0).astype(np.float32)

    if body.get("binary"):
        verts_b64 = base64.b64encode(deformed.astype(np.float32).tobytes()).decode("ascii")
        return _json({
            "vertices_b64": verts_b64,
            "vertex_count": V,
            "frames": T,
        })

    return _json({
        "vertices": deformed.tolist(),
        "vertex_count": V,
        "frames": T,
    })
# NOTE: /poser/somax/bake was DELETED 2026-07-26 — replaced by the
# RaySomaxBake ComfyUI custom node (custom_nodes/melite-poser-nodes/nodes.py).
# Pure CPU (numpy rotations + GLB IO) — no GPU work. Same engine
# (somax_bake.bake_motion), now driven through ComfyScript →
# submit_comfy_workflow per the comfyui-script-only policy.


@routes.post("/poser/render_glb")
async def post_render_glb(request: web.Request) -> web.Response:
    """POST /poser/render_glb — render a GLB from multiple angles via Blender.

    Uses Blender 4.2 headless (BLENDER_EEVEE_NEXT) with 3-point studio
    lighting. This is the explicit, repeatable render path for the harsh
    9-10/10 visual critique of every pipeline stage — no ephemeral
    ``docker exec blender`` calls.

    Body:
        {
          "glb_b64": "...",             # GLB to render
          "views": [                    # optional; default = front/right/back/left
            {"azimuth": 0, "elevation": 5, "name": "front"}
          ],
          "resolution": 800,            # square render resolution
          "samples": 64,                # EEVEE_NEXT TAA samples
          "transparent": false,         # transparent background
          "focal_length": 50,           # mm
          "framing": 1.2               # camera margin multiplier (1.0=exact fit)
        }

    Returns: {"renders": [{"name": "front", "image_b64": "..."}, ...]}
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _err("invalid JSON body", 400)

    import base64 as _b64
    glb_bytes = _b64.b64decode(body.get("glb_b64", ""))
    if not glb_bytes:
        return _err("glb_b64 required", 400)

    views = body.get("views")  # None → default 4 angles
    resolution = int(body.get("resolution", 800))
    width = body.get("width")
    height = body.get("height")
    samples = int(body.get("samples", 64))
    transparent = bool(body.get("transparent", False))
    focal_length = int(body.get("focal_length", 50))
    framing = float(body.get("framing", 1.2))

    try:
        from .blender_render import render_glb
        renders = await asyncio.to_thread(
            render_glb, glb_bytes, views, resolution, samples,
            transparent, focal_length, framing,
            width if width is not None else resolution,
            height if height is not None else resolution,
        )
    except Exception as e:
        log.exception("render_glb failed")
        return _err(f"render failed: {e}", 500)

    return _json({
        "status": "ok",
        "renders": [
            {"name": r["name"],
             "image_b64": _b64.b64encode(r["png_bytes"]).decode("ascii")}
            for r in renders
        ],
    })




@routes.post("/poser/bridge_creature")
async def post_bridge_creature(request: web.Request) -> web.Response:
    """POST /poser/bridge_creature - bind a static textured mesh onto an
    anyCreature species skeleton so it inherits that species baked
    idle/move animations. Path (a) of GAME-KIT-PROGRESS.md (2026-08-25).
    CPU-only Blender subprocess (creature_rig_bridge.bridge_creature).

    Body:
        {"creature_b64": "...",   # anyCreature GLB (armature + anims)
         "mesh_b64": "...",       # static textured mesh GLB (e.g. TRELLIS)
         "height_ratio": 1.0,
         "max_influences": 3,
         "strip_body": true,
         "weight_falloff": "linear" | "inv_power",
         "weight_power": 3.0,            # exponent for inv_power
         "weight_window": "none" | "smoothstep",
         "yaw_align": true},             # align mesh spine onto rig spine

    Returns: {"status": "ok", "glb_b64": "...", "bones": N, "scale": f}
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _err("invalid JSON body", 400)

    import base64 as _b64
    import os as _os
    import shutil as _shutil
    import tempfile as _tf
    creature_bytes = _b64.b64decode(body.get("creature_b64", ""))
    mesh_bytes = _b64.b64decode(body.get("mesh_b64", ""))
    if not creature_bytes or not mesh_bytes:
        return _err("creature_b64 and mesh_b64 required", 400)

    td = _tf.mkdtemp(prefix="bridge_http_")
    try:
        from .creature_rig_bridge import bridge_creature
        out_path = _os.path.join(td, "bridged.glb")
        stats = await asyncio.to_thread(
            bridge_creature, creature_bytes, mesh_bytes, out_path,
            float(body.get("height_ratio", 1.0)),
            int(body.get("max_influences", 3)),
            bool(body.get("strip_body", True)),
            str(body.get("weight_falloff", "linear")),
            float(body.get("weight_power", 2.0)),
            str(body.get("weight_window", "none")),
            bool(body.get("yaw_align", True)),
        )
        with open(out_path, "rb") as f:
            glb_out = f.read()
    except Exception as e:
        log.exception("bridge_creature failed")
        return _err(f"bridge failed: {e}", 500)
    finally:
        _shutil.rmtree(td, ignore_errors=True)

    return _json({
        "status": "ok",
        "glb_b64": _b64.b64encode(glb_out).decode("ascii"),
        "bones": stats.get("bones"),
        "scale": stats.get("scale"),
    })

@routes.post("/poser/clean_mesh")
async def post_clean_mesh(request: web.Request) -> web.Response:
    """POST /poser/clean_mesh - §4.42 mono mesh stage: weld -> repair ->
    decimate-to-budget (UV-preserving quadric decimation via trimesh).

    Pure-Python GLB surgery — NO Blender subprocess needed (runs anywhere
    trimesh is installed, including the ComfyUI container). Body:
        {"glb_b64": "...", "target_faces": 5000, "weld_threshold": 0.0001}

    Returns {"status": "ok", "glb_b64": "...", "stats": {...}} where stats
    carries faces/verts before->after, budget compliance, weld delta.
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _err("invalid JSON body", 400)
    import base64 as _b64
    import os as _os
    import shutil as _shutil
    import tempfile as _tf
    glb_bytes = _b64.b64decode(body.get("glb_b64", ""))
    if not glb_bytes:
        return _err("glb_b64 required", 400)
    target = int(body.get("target_faces", 15000))
    weld = float(body.get("weld_threshold", 1e-4))
    td = _tf.mkdtemp(prefix="mesh_clean_http_")
    try:
        from .mesh_clean import clean_mesh
        out_path = _os.path.join(td, "cleaned.glb")
        stats = await asyncio.to_thread(
            clean_mesh, glb_bytes, out_path, target, weld)
        with open(out_path, "rb") as f:
            glb_out = f.read()
    except Exception as e:
        log.exception("clean_mesh failed")
        return _err(f"clean_mesh failed: {e}", 500)
    finally:
        _shutil.rmtree(td, ignore_errors=True)
    return _json({
        "status": "ok",
        "glb_b64": _b64.b64encode(glb_out).decode("ascii"),
        "stats": stats,
    })

@routes.post("/poser/render_textured")
async def post_render_textured(request: web.Request) -> web.Response:
    """POST /poser/render_textured — render a GLB's own texture via nvdiffrast.

    Unlike ``/poser/render_glb`` (Blender), this samples the baseColorTexture
    directly through the interpolated UVs, so the texture ALWAYS shows.
    Blender 4.2 renders these GLBs flat gray (TRELLIS emits invalid texture
    bytes + Blender's glTF importer mishandles the valid PNGs). This is the
    reliable "see the actual character" path.

    Body:
        {
          "glb_b64": "...",
          "views": [{"azimuth": 0, "elevation": 5, "name": "front"}, ...],
          "resolution": 800, "fov": 30, "framing": 1.2,
          "bg_color": [0.06, 0.06, 0.07]
        }
    Returns: {"renders": [{"name": "front", "image_b64": "..."}, ...]}
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _err("invalid JSON body", 400)

    import base64 as _b64
    glb_bytes = _b64.b64decode(body.get("glb_b64", ""))
    if not glb_bytes:
        return _err("glb_b64 required", 400)

    raw_views = body.get("views")
    resolution = int(body.get("resolution", 800))
    fov = float(body.get("fov", 30.0))
    framing = float(body.get("framing", 1.2))
    bg = body.get("bg_color", [0.06, 0.06, 0.07])
    # Accept the same preset-name shorthand as render_glb.
    _PRESETS = {"front": (0, 5), "back": (180, 5), "left": (270, 5),
                "right": (90, 5), "3q_left": (315, 5), "3q_right": (45, 5),
                "front_left": (315, 5), "front_right": (45, 5)}
    views = None
    if raw_views:
        views = []
        for v in raw_views:
            if isinstance(v, str):
                az, el = _PRESETS.get(v.strip().lower(), (0, 5))
                views.append({"azimuth": az, "elevation": el, "name": v})
            elif isinstance(v, dict):
                views.append(v)

    try:
        from .texture_project import render_textured_views
        renders = await asyncio.to_thread(
            render_textured_views, glb_bytes, views, resolution, fov, framing,
            tuple(float(c) for c in bg),
        )
    except Exception as e:
        log.exception("render_textured failed")
        return _err(f"render failed: {e}", 500)

    return _json({
        "status": "ok",
        "renders": [
            {"name": r["name"],
             "image_b64": _b64.b64encode(r["png_bytes"]).decode("ascii")}
            for r in renders
        ],
    })
