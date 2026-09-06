"""ComfyUI workflow nodes for mesh cleanup — invoked via ComfyScript.

Architecture: melite-head is a pure HTTP orchestrator. ALL mesh work happens
inside the ComfyUI container. These nodes wrap the bytes-based operations
in ``mesh_cleanup.py`` so they can be called from a ComfyScript-built
workflow graph (``nodes.MeshFixWinding(...)`` → ``api_format()`` →
``submit_comfy_workflow(...)``).

Each node:
  1. Reads a GLB from ComfyUI's ``input/`` folder (melite-head uploads it via
     ``ComfyUIClient.upload_file`` before submitting the workflow).
  2. Calls the corresponding bytes-based op in ``mesh_cleanup.py``.
  3. Writes the result to ComfyUI's ``output/`` folder.
  4. Emits a ``ui.three_model`` entry so ``ComfyUIClient.extract_media`` /
     ``_download_image`` can fetch the result via GET /view.

NO HTTP routes are added — the strict rule is "ComfyUI is used only through
ComfyScript". The previous ``/poser/mesh/*`` HTTP routes were removed; these
nodes are the ComfyScript-compliant replacement.
"""
from __future__ import annotations

import logging
import os
import time
from functools import lru_cache as _lru_cache

from .mesh_cleanup import fix_winding as _fix_winding_bytes
from .mesh_cleanup import dedup as _dedup_bytes
from .character_composite import composite_character as _composite_bytes
from .fit_nodes import FitProp, FitRow
from .skin_nodes import SkinApply, SkinPack, SkinSidecar
# Shared bounding-sphere helper (suggestion #2, 2026-07-31). PoserRender
# (depth, in rasterize.py) and PoserRenderOpenPose (skeleton, below) both
# call this so their renders stay pixel-aligned by construction.
from .library.geometry import bounding_sphere as _bounding_sphere

log = logging.getLogger(__name__)


def _input_path(filename: str) -> str | None:
    """Resolve an uploaded filename to an absolute path in ComfyUI input/."""
    import folder_paths  # ComfyUI core
    # NOTE: ``folder_paths.get_full_path("input", ...)`` returns None on this
    # ComfyUI build — "input" isn't in ``folder_names_and_paths`` (only
    # "output"/"temp"/model-folders are). ``get_input_directory()`` does
    # resolve correctly, so we join manually.
    if os.path.isabs(filename):
        return filename if os.path.isfile(filename) else None
    full = os.path.join(folder_paths.get_input_directory(), filename)
    return full if os.path.isfile(full) else None


def _output_path(prefix: str, ext: str = ".glb") -> str:
    """Build a unique output path in ComfyUI's output/ folder."""
    import folder_paths  # ComfyUI core
    out_dir = folder_paths.get_output_directory()
    name = f"{prefix}_{int(time.time() * 1000)}_{os.getpid()}{ext}"
    return os.path.join(out_dir, name)


def _image_tensor_to_png_bytes(image) -> bytes:
    """Convert a ComfyUI IMAGE tensor (B,H,W,C float32 0-1) to PNG bytes."""
    import io

    import numpy as np
    from PIL import Image

    # Handle batch dimension — take the first image.
    if hasattr(image, "ndim") and image.ndim == 4:
        img = image[0]
    else:
        img = image
    arr = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _image_tensor_to_temp_png(image, prefix: str = "comfy_view") -> str:
    """Save a ComfyUI IMAGE tensor to a temp PNG file, return the path."""
    import tempfile

    import folder_paths

    png_bytes = _image_tensor_to_png_bytes(image)
    tmp_dir = folder_paths.get_temp_directory()
    os.makedirs(tmp_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix=f"{prefix}_", dir=tmp_dir)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(png_bytes)
    return tmp_path


def _ui_entry(path: str) -> dict:
    """Build a ComfyUI UI descriptor for a 3D output file."""
    return {
        "filename": os.path.basename(path),
        "subfolder": "",
        "type": "output",
    }


# ════════════════════════════════════════════════════════════════════════
# Node: MeshFixWinding
# ════════════════════════════════════════════════════════════════════════
class MeshFixWinding:
    """Fix inconsistent face winding (inward normals) in a GLB.

    TRELLIS exports meshes with ~25–45% of faces having INWARD-pointing
    normals, which causes flat-shading spikes, inverted normal maps, and
    VLM evaluation scores of 1–2/10 ("severe mesh corruption").

    For each face, if its geometric normal points toward the mesh centroid,
    swap v1↔v2 to flip it outward. When ``preserve_rig`` is False (default
    for fresh TRELLIS output), also re-encodes WebP textures as PNG so
    downstream consumers see valid PNG chunks. When ``preserve_rig`` is
    True (animated/rigged GLBs), textures are left untouched.

    Input/Output:
      glb_path — filename in ComfyUI input/ → filename in ComfyUI output/
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_path": ("STRING", {"multiline": False, "forceInput": True}),
                "preserve_rig": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "fix"
    CATEGORY = "TechNoir/Mesh"
    OUTPUT_NODE = True

    def fix(self, glb_path: str, preserve_rig: bool = False):
        src = _input_path(glb_path)
        if not src or not os.path.isfile(src):
            raise RuntimeError(
                f"MeshFixWinding: input '{glb_path}' not found in ComfyUI input/. "
                f"Did the caller upload it via ComfyUIClient.upload_file?"
            )
        with open(src, "rb") as f:
            glb_bytes = f.read()
        log.info(
            "[MeshFixWinding] input=%s (%d bytes), preserve_rig=%s",
            src, len(glb_bytes), preserve_rig,
        )
        t0 = time.perf_counter()
        out_bytes = _fix_winding_bytes(glb_bytes, preserve_rig=preserve_rig)
        out_path = _output_path("mesh_fix_winding")
        with open(out_path, "wb") as f:
            f.write(out_bytes)
        log.info(
            "[MeshFixWinding] output=%s (%d bytes) in %.2fs",
            out_path, len(out_bytes), time.perf_counter() - t0,
        )
        return {
            "result": (out_path,),
            "ui": {"three_model": [_ui_entry(out_path)]},
        }


# ════════════════════════════════════════════════════════════════════════
# Node: MeshDedup
# ════════════════════════════════════════════════════════════════════════
class MeshDedup:
    """Merge coincident vertices (within ``tolerance``) in a GLB.

    Uses ``scipy.spatial.cKDTree`` + union-find to identify vertices closer
    than ``tolerance`` (default 0.5mm) and merges them. Reduces vertex
    count by 30–50% on typical TRELLIS output without changing visible
    geometry, eliminating Z-fighting and reducing file size.

    Input/Output:
      glb_path — filename in ComfyUI input/ → filename in ComfyUI output/
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_path": ("STRING", {"multiline": False, "forceInput": True}),
                "tolerance": ("FLOAT", {
                    "default": 5e-4,
                    "min": 1e-5,
                    "max": 1e-2,
                    "step": 1e-5,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "dedup"
    CATEGORY = "TechNoir/Mesh"
    OUTPUT_NODE = True

    def dedup(self, glb_path: str, tolerance: float = 5e-4):
        src = _input_path(glb_path)
        if not src or not os.path.isfile(src):
            raise RuntimeError(
                f"MeshDedup: input '{glb_path}' not found in ComfyUI input/. "
                f"Did the caller upload it via ComfyUIClient.upload_file?"
            )
        with open(src, "rb") as f:
            glb_bytes = f.read()
        log.info(
            "[MeshDedup] input=%s (%d bytes), tolerance=%g",
            src, len(glb_bytes), tolerance,
        )
        t0 = time.perf_counter()
        out_bytes = _dedup_bytes(glb_bytes, tol=tolerance)
        out_path = _output_path("mesh_dedup")
        with open(out_path, "wb") as f:
            f.write(out_bytes)
        log.info(
            "[MeshDedup] output=%s (%d bytes) in %.2fs",
            out_path, len(out_bytes), time.perf_counter() - t0,
        )
        return {
            "result": (out_path,),
            "ui": {"three_model": [_ui_entry(out_path)]},
        }


# ════════════════════════════════════════════════════════════════════════
# Node: CleanMesh — §4.42 mono mesh decimation stage
# ════════════════════════════════════════════════════════════════════════
class CleanMesh:
    """§4.42 mono mesh pipeline: weld → repair → QEM decimate to budget.

    Pure-Python via trimesh (NO Blender). Preserves UVs and PBR textures.

    Input/Output:
      glb_path — filename in ComfyUI input/ → filename in ComfyUI output/
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_path": ("STRING", {"multiline": False, "forceInput": True}),
                "target_faces": ("INT", {
                    "default": 5000,
                    "min": 100,
                    "max": 200000,
                    "tooltip": "Target face count after QEM decimation.",
                }),
                "weld_threshold": ("FLOAT", {
                    "default": 1e-4,
                    "min": 1e-6,
                    "max": 1e-1,
                    "step": 1e-5,
                    "tooltip": "Vertex merge distance threshold.",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "clean"
    CATEGORY = "TechNoir/Mesh"
    OUTPUT_NODE = True

    def clean(self, glb_path: str, target_faces: int = 5000,
              weld_threshold: float = 1e-4):
        src = _input_path(glb_path)
        if not src or not os.path.isfile(src):
            raise RuntimeError(
                f"CleanMesh: input '{glb_path}' not found in ComfyUI input/. "
                f"Did the caller upload it via ComfyUIClient.upload_file?"
            )
        with open(src, "rb") as f:
            glb_bytes = f.read()
        log.info(
            "[CleanMesh] input=%s (%d bytes), target=%d faces, weld=%.1e",
            src, len(glb_bytes), target_faces, weld_threshold,
        )
        t0 = time.perf_counter()
        out_path = _output_path("clean_mesh")
        from .mesh_clean import clean_mesh
        stats = clean_mesh(glb_bytes, out_path, target_faces, weld_threshold)
        log.info(
            "[CleanMesh] output=%s in %.2fs — %d→%d faces (%.0f%% reduction)",
            out_path, time.perf_counter() - t0,
            stats["faces_before"], stats["faces_after"],
            (1.0 - stats["faces_after"] / max(stats["faces_before"], 1)) * 100,
        )
        if stats.get("over_budget"):
            log.warning("[CleanMesh] OVER BUDGET: %d > %d faces",
                        stats["faces_after"], stats["budget"])
        return {
            "result": (out_path,),
            "ui": {"three_model": [_ui_entry(out_path)]},
        }


# ════════════════════════════════════════════════════════════════════════
# Node: CompositeCharacter
# ════════════════════════════════════════════════════════════════════════
class CompositeCharacter:
    """Merge separate 3D GLB assets into one rigged character GLB.

    The character pipeline generates the body, hair, and outfit as separate
    TRELLIS meshes. This node composes the final character:

      • HAIR (helmet method) — static mesh parented to ``mixamorig:Head``.
        Local matrix = ``inverse(head_world_at_bind)`` so the hair stays in
        its model-space location at bind pose and follows the head during
        animation. NO texture projection — it's a real 3D wig.

      • OUTFIT (skinned sibling) — the outfit's mesh primitive (already
        rigged via ``transfer_skin_weights``) is appended to the body's
        mesh, sharing the body's skeleton. Joint indices must already
        reference the same logical joints.

    Every failure mode raises ``RuntimeError`` with a concrete message.
    No silent crashes, no partial output.

    Inputs:
      body_glb_path   — rigged + animated body GLB in ComfyUI input/
      hair_glb_path   — (optional) static hair GLB in input/
      outfit_glb_path — (optional) rigged outfit GLB in input/

    Output: path to the composite GLB in ComfyUI output/.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "body_glb_path": ("STRING", {"multiline": False, "forceInput": True}),
            },
            "optional": {
                "hair_glb_path": ("STRING", {"default": "", "forceInput": True}),
                "outfit_glb_path": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "compose"
    CATEGORY = "TechNoir/Mesh"
    OUTPUT_NODE = True

    def compose(self, body_glb_path: str,
                hair_glb_path: str = "",
                outfit_glb_path: str = ""):
        body_path = _input_path(body_glb_path)
        if not body_path or not os.path.isfile(body_path):
            raise RuntimeError(
                f"CompositeCharacter: body '{body_glb_path}' not found in input/. "
                f"Upload it via ComfyUIClient.upload_file before submitting."
            )
        with open(body_path, "rb") as f:
            body_bytes = f.read()
        log.info("[CompositeCharacter] body=%s (%d bytes)", body_path, len(body_bytes))

        hair_bytes = None
        if hair_glb_path:
            hp = _input_path(hair_glb_path)
            if not hp or not os.path.isfile(hp):
                raise RuntimeError(
                    f"CompositeCharacter: hair '{hair_glb_path}' not found in input/. "
                    f"Upload it before submitting, or pass empty string to skip hair."
                )
            with open(hp, "rb") as f:
                hair_bytes = f.read()
            log.info("[CompositeCharacter] hair=%s (%d bytes)", hp, len(hair_bytes))

        outfit_bytes = None
        if outfit_glb_path:
            op = _input_path(outfit_glb_path)
            if not op or not os.path.isfile(op):
                raise RuntimeError(
                    f"CompositeCharacter: outfit '{outfit_glb_path}' not found in input/. "
                    f"Upload it before submitting, or pass empty string to skip outfit."
                )
            with open(op, "rb") as f:
                outfit_bytes = f.read()
            log.info("[CompositeCharacter] outfit=%s (%d bytes)", op, len(outfit_bytes))

        if not hair_bytes and not outfit_bytes:
            # (2026-09-05 descope) body-only is legal — the anny card
            # runs the body alone; composite_character validates it and
            # passes through (a composite of one).
            log.info("[CompositeCharacter] body-only passthrough "
                     "(descope 2026-09-05)")

        t0 = time.perf_counter()
        out_bytes = _composite_bytes(body_bytes, hair_bytes, outfit_bytes)
        out_path = _output_path("composite_character")
        with open(out_path, "wb") as f:
            f.write(out_bytes)
        log.info(
            "[CompositeCharacter] output=%s (%d bytes) in %.2fs",
            out_path, len(out_bytes), time.perf_counter() - t0,
        )
        return {
            "result": (out_path,),
            "ui": {"three_model": [_ui_entry(out_path)]},
        }


# ════════════════════════════════════════════════════════════════════════
# Node: RayProjectViewsToTexture — REMOVED 2026-07-28
# ════════════════════════════════════════════════════════════════════════
# Replaced by the upstream Comfy3D (MrForExample) ExplicitTargetColorProjection
# node, driven by media/comfyui/families/texture_project.py. The hand-rolled
# nvdiffrast implementation had a camera-framing mismatch (orbit_radius=1.75
# made the mesh fill only 65% of the camera frustum, so projection sampled
# chest pixels where face pixels were expected). The upstream node, with
# orbit_radius=1.14 (= 0.5 / tan(fovy/2)) so the unit-normalized mesh fills
# the frustum vertically the same way qwen-edit body images fill their
# frames, produces a correct projection (face on face, hair on back).
#
# The new brick also handles:
#   * Save3DMesh's missing UI output entry (uses deterministic UUID filename
#     + direct /view?filename=... fetch).
#   * Rig preservation (Save3DMesh's trimesh round-trip drops skins/joints;
#     the brick does a surgical baseColorTexture swap to preserve them).
#
# The old algorithm code (project_views_to_texture, _surgical_texture_swap,
# _surgical_add_normal_texture) was deleted from texture_project.py.
# Kept: _normalize_glb_buffers (used by uv_cleanup), render_textured_views
# (used by server.py for debug rendering), and their helpers.


# ════════════════════════════════════════════════════════════════════════
# Node: RayXatlasUnwrap (CPU UV cleanup)
# ════════════════════════════════════════════════════════════════════════
class RayXatlasUnwrap:
    """Re-unwrap a GLB's UVs with xatlas → clean non-overlapping atlas.

    TRELLIS's o_voxel ``uv_unwrap`` (cone-clustering) produces an atlas that
    overlaps ~29x. nvdiffrast UV-space texture projection then assigns each
    texel to an ARBITRARY triangle among the ~5 overlapping candidates →
    salt-and-pepper noise. xatlas produces a clean atlas so each texel maps
    to exactly one triangle → coherent projection.

    Pure CPU (xatlas + trimesh) — no model, no VRAM. Replaces the deleted
    HTTP route /poser/trellis/xatlas_unwrap.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Target GLB (any UV state — will be replaced).",
                    "forceInput": True,
                    }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "unwrap"
    CATEGORY = "TechNoir/Mesh"
    OUTPUT_NODE = True

    def unwrap(self, glb_path: str):
        import folder_paths
        from .uv_cleanup import run_xatlas_unwrap

        def _resolve(filename):
            if os.path.isabs(filename) and os.path.exists(filename):
                return filename
            for d in (folder_paths.get_input_directory, folder_paths.get_output_directory):
                p = os.path.join(d(), filename)
                if os.path.exists(p):
                    return p
            raise FileNotFoundError(f"Could not resolve {filename!r}")

        target_path = _resolve(glb_path)
        with open(target_path, "rb") as f:
            glb_bytes = f.read()

        t0 = time.perf_counter()
        new_glb_bytes = run_xatlas_unwrap(glb_bytes)
        out_path = _output_path("xatlas_unwrap")
        with open(out_path, "wb") as f:
            f.write(new_glb_bytes)
        log.info("[RayXatlasUnwrap] %.1fs → %s (%d bytes)",
                 time.perf_counter() - t0, out_path, len(new_glb_bytes))
        return {
            "result": (out_path,),
            "ui": {"three_model": [_ui_entry(out_path)]},
        }


# ════════════════════════════════════════════════════════════════════════
# Node: RaySAMSegmentHairClothing (SAM model, PipelinePatcher-tracked)
# ════════════════════════════════════════════════════════════════════════
class RaySAMSegmentHairClothing:
    """Segment a character image into hair + outfit sprites via SAM.

    Wraps Meta's Segment-Anything ViT-B in a PipelinePatcher so ComfyUI's
    VRAM accounting sees it and auto-evicts under pressure — no module-global
    ``_sam_predictor`` squat, no manual VRAM cleanup. Replaces the deleted
    HTTP route /poser/segment/hair_clothing.

    Returns two PNG filenames in ComfyUI's output/ folder:
      • hair_sprite.png  — hair composited on white
      • outfit_sprite.png — clothing composited on white
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Source character image — connect a LoadImage output here."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hair_sprite",)
    FUNCTION = "segment"
    CATEGORY = "TechNoir/Image"
    OUTPUT_NODE = True

    def segment(self, image):
        import folder_paths
        from .sam_segment import segment_hair_clothing

        image_bytes = _image_tensor_to_png_bytes(image)

        t0 = time.perf_counter()
        hair_png, outfit_png = segment_hair_clothing(image_bytes)

        out_dir = folder_paths.get_output_directory()
        ts = int(time.time() * 1000)
        hair_path = os.path.join(out_dir, f"sam_hair_{ts}.png")
        outfit_path = os.path.join(out_dir, f"sam_outfit_{ts}.png")
        with open(hair_path, "wb") as f:
            f.write(hair_png)
        with open(outfit_path, "wb") as f:
            f.write(outfit_png)

        log.info("[RaySAMSegmentHairClothing] %.1fs → %s, %s",
                 time.perf_counter() - t0, hair_path, outfit_path)
        return {
            "result": (hair_path,),
            "ui": {
                "images": [
                    {"filename": os.path.basename(hair_path),
                     "subfolder": "", "type": "output"},
                    {"filename": os.path.basename(outfit_path),
                     "subfolder": "", "type": "output"},
                ],
            },
        }


# ════════════════════════════════════════════════════════════════════════
# Node: RaySomaxBake (CPU — bake motion NPZ → animated GLB)
# ════════════════════════════════════════════════════════════════════════
class RaySomaxBake:
    """Bake SOMA motion NPZ onto a SOMAX-77 rigged GLB → animated GLB.

    Writes SOMA-77 motion NPZ data as glTF animation channels directly onto
    a SOMAX-77 rigged GLB. No Blender, no retargeting — SOMA output's
    skeleton already matches SOMA-77, so the motion maps 1:1.

    Pure CPU (numpy rotations + GLB IO). Replaces the deleted HTTP route
    /poser/somax/bake.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rigged_glb_path": ("STRING", {"multiline": False,
                    "tooltip": "SOMAX-77 rigged GLB (filename in input/).",
                    "forceInput": True,
                    }),
                "motion_npz_path": ("STRING", {"multiline": False,
                    "tooltip": "SOMA-77 motion NPZ (filename in input/).",
                    "forceInput": True,
                    }),
            },
            "optional": {
                "anim_name": ("STRING", {"default": "motion"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("animated_glb_path",)
    FUNCTION = "bake"
    CATEGORY = "TechNoir/Animation"
    OUTPUT_NODE = True

    def bake(self, rigged_glb_path: str, motion_npz_path: str,
             anim_name: str = "motion", fps: float = 30.0):
        import folder_paths
        from .somax_bake import bake_motion

        def _resolve(filename):
            if os.path.isabs(filename) and os.path.exists(filename):
                return filename
            for d in (folder_paths.get_input_directory, folder_paths.get_output_directory):
                p = os.path.join(d(), filename)
                if os.path.exists(p):
                    return p
            raise FileNotFoundError(f"Could not resolve {filename!r}")

        glb_path = _resolve(rigged_glb_path)
        npz_path = _resolve(motion_npz_path)
        with open(glb_path, "rb") as f:
            glb_bytes = f.read()
        with open(npz_path, "rb") as f:
            npz_bytes = f.read()

        t0 = time.perf_counter()
        out_bytes, diag = bake_motion(glb_bytes, npz_bytes, anim_name, fps)
        out_path = _output_path("somax_bake")
        with open(out_path, "wb") as f:
            f.write(out_bytes)
        log.info(
            "[RaySomaxBake] %.1fs → %s (%d bytes), frames=%s, joints=%s",
            time.perf_counter() - t0, out_path, len(out_bytes),
            diag.get("frame_count", "?"), diag.get("joint_count", "?"),
        )
        return {
            "result": (out_path,),
            "ui": {
                "three_model": [_ui_entry(out_path)],
                "diagnostics": diag,
            },
        }


# ════════════════════════════════════════════════════════════════════════
# Node: RaySwapBaseColorTexture — the 4th zero-deviation node.
# Comfy3D's [Comfy3D] Save 3D Mesh strips the rig (skin/joints/weights).
# This node extracts the projected texture from the derigged Comfy3D output
# and swaps it into the ORIGINAL rigged GLB, preserving the entire rig.
# ComfyUI-node port of media.comfyui.glb_surgical.swap_base_color_texture.
# ════════════════════════════════════════════════════════════════════════
class RaySwapBaseColorTexture:
    """Swap projected baseColorTexture (from derigged GLB) into a rigged GLB.

    ``[Comfy3D] Save 3D Mesh`` does a trimesh round-trip that drops the
    ENTIRE rig (skins, joints, WEIGHTS_0/JOINTS_0, node hierarchy, animation
    samplers). This node:

      1. Reads the projected (derigged, textured) GLB from Comfy3D.
      2. Extracts its baseColorTexture PNG bytes.
      3. Swaps that PNG into the ORIGINAL rigged GLB's baseColorTexture slot,
         preserving every other byte (skin, joints, weights, animations).

    The output is a TEXTURED + RIGGED GLB that downstream bake_motion /
    CompositeCharacter nodes can consume. Replaces the host-side
    ``glb_surgical.swap_base_color_texture`` call that the server-side
    texture_project family does after fetching the Comfy3D output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "projected_glb_path": ("STRING", {"multiline": False,
                    "tooltip": "Derigged textured GLB from [Comfy3D] Save 3D Mesh (filename in input/ or output/, or absolute path).",
                    "forceInput": True,
                    }),
                "rigged_glb_path": ("STRING", {"multiline": False,
                    "tooltip": "ORIGINAL rigged GLB (skin/joints/weights preserved). Its baseColorTexture is replaced.",
                    "forceInput": True,
                    }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("textured_rigged_glb_path",)
    FUNCTION = "swap"
    CATEGORY = "TechNoir/Mesh"
    OUTPUT_NODE = True

    def swap(self, projected_glb_path: str, rigged_glb_path: str):
        import folder_paths
        from .glb_surgery import extract_base_color_texture, swap_base_color_texture

        def _resolve(filename):
            filename = (filename or "").strip()
            if os.path.isabs(filename) and os.path.isfile(filename):
                return filename
            for d in (folder_paths.get_input_directory,
                      folder_paths.get_output_directory):
                p = os.path.join(d(), filename)
                if os.path.isfile(p):
                    return p
            raise FileNotFoundError(
                f"RaySwapBaseColorTexture: could not resolve {filename!r} "
                f"(tried absolute, input/, output/)")

        src_proj = _resolve(projected_glb_path)
        src_rig = _resolve(rigged_glb_path)
        with open(src_proj, "rb") as f:
            projected_bytes = f.read()
        with open(src_rig, "rb") as f:
            rigged_bytes = f.read()

        t0 = time.perf_counter()
        img_bytes, mime = extract_base_color_texture(projected_bytes)
        final_bytes = swap_base_color_texture(rigged_bytes, img_bytes, mime)
        out_path = _output_path("swap_texture")
        with open(out_path, "wb") as f:
            f.write(final_bytes)
        log.info(
            "[RaySwapBaseColorTexture] %.1fs projected=%s (%d B) + rigged=%s "
            "(%d B) → %s (%d B, texture %d B %s, rig preserved)",
            time.perf_counter() - t0, src_proj, len(projected_bytes),
            src_rig, len(rigged_bytes), out_path, len(final_bytes),
            len(img_bytes), mime,
        )
        return {
            "result": (out_path,),
            "ui": {
                "three_model": [_ui_entry(out_path)],
                "textured": [True],
                "source_projected": [os.path.basename(src_proj)],
                "source_rigged": [os.path.basename(src_rig)],
            },
        }


# ════════════════════════════════════════════════════════════════════════
# Registration
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# Poser nodes — ComfyScript path for SOMA mesh → depth/normal ControlNet
# ════════════════════════════════════════════════════════════════════════
# WHY THESE EXIST ALONGSIDE THE /poser/skin/* HTTP ROUTES
#
# The /poser/skin/* HTTP routes (server.py) serve the Pose Studio / Body
# Studio webview frontend — interactive slider drags where <100ms latency
# on a hot-cached engine matters. HTTP is the right shape for that case.
#
# The PIPELINE path (anny_depth → ControlNet conditioning → TRELLIS, etc.)
# is NOT interactive. It runs once per pipeline run. For that path we need
# observability: typed inputs the graph validator checks, visible node
# topology in the ComfyUI UI, and workflow JSON that's logged + replayable.
# These three nodes provide that. Same SkinEngine singleton, different shape.
#
# The non-DRY cost (one node class + one HTTP handler per operation, both
# calling the same underlying engine.set_identity / engine.deform / etc.)
# is the cost of observability. We pay it deliberately. The alternative —
# only HTTP, no nodes — is exactly the silent-rotation-bug failure mode
# that bit us 2026-07-22 → 2026-07-27.
#
# ════════════════════════════════════════════════════════════════════════
# SILENT-FAILURE CONTRACT — READ BEFORE EDITING ANY OF THE THREE NODES
# ════════════════════════════════════════════════════════════════════════
# These are the failure modes that go silent on the HTTP path. Each node
# below guards against them at the boundary AND documents them inline at
# the relevant line. If you add a new silent-failure guard, add it to
# this list.
#
#  1. UNITS — the cm↔m trap.
#     SkinEngine.deform() works in METERS (py-soma-x's native unit).
#     bind_pose returns CENTIMETERS by default (units="cm").
#     deform_soma's HTTP handler converts cm→m on entry, m→cm on exit.
#     render expects vertices in CENTIMETERS (the deform output unit).
#     ⇒ A caller who passes meters to deform thinking "engine native"
#        gets a 100×-scaled mesh with no error. Each node below asserts
#        the unit it expects.
#
#  2. ROTATION MATRIX LAYOUT.
#     SkinEngine._extract_soma_x_transforms() returns (77, 4, 4) WORLD-
#     space transforms in meters, laid out as the standard
#        [ R   t ]         where R is 3×3 rotation, t is 3×1 translation
#        [ 0 0 0 1 ]       (bottom row is the homogeneous row).
#     We extract R = transform[:, :3, :3] and t = transform[:, :3, 3].
#     ⇒ Returning "rotations" alone DROPS the translation. The downstream
#        deform call MUST receive positions (translation) separately, not
#        derive them from rotations. Documented at the relevant line.
#
#  3. ROTATION MATRIX SHAPE.
#     Rotations are (T, 77, 3, 3) — frame × joint × 3×3. NOT (T, 3, 3, 77).
#     PyTorch / numpy will happily reduce along the wrong axis without
#     raising if you transpose by mistake. Guarded with an explicit
#     shape assertion.
#
#  4. SOMA-77 JOINT ORDERING.
#     The 77 joints are in a FIXED CANONICAL ORDER defined by SOMA
#     (supplied by py-soma-x). It is NOT COCO-18, NOT SMPL-24, NOT BVH.
#     Applying rotations intended for one convention to another produces
#     silent garbage — every joint goes to the wrong place. We do not
#     re-order; we transport the (T, 77, ...) layout unchanged.
#
#  5. VERTEX ENCODING.
#     The HTTP path uses base64 float32 LITTLE-ENDIAN for lossless transfer.
#     These nodes pass numpy arrays DIRECTLY through the graph (no base64),
#     so the dtype is asserted at node entry. If a future caller builds a
#     MESH manually with float64 vertices, the assertion catches it.
#
#  6. IDENTITY MODEL STRING.
#     "soma" (default) vs "anny" (our use) vs "child" vs future models.
#     A typo loads the wrong mesh silently. The node accepts any string
#     (the engine raises on unknown), but DOCUMENT which identities are
#     valid in the docstring so a future caller knows.
#
#  7. IDENTITY COEFFICIENTS (body-shape knobs).
#     A list whose length depends on the identity_model's phenotype dims
#     (exposed via /poser/skin/identity-space). Too short = engine pads
#     with zeros (silent neutral). Too long = engine raises. Each coeff
#     is in [0, 1] with 0.5 = neutral adult midpoint (for "anny"; other
#     models may differ). Pass an empty list to use the neutral shape.
# ════════════════════════════════════════════════════════════════════════

# Custom ComfyUI types. ComfyUI's type field is a string label; the values
# flow through the graph as Python objects. Declaring JOINTS and MESH as
# distinct labels means the graph validator rejects wiring IMAGE into MESH,
# or JOINTS into IMAGE — the type of silent mismatch that HTTP JSON cannot
# catch.
#
# Wire format (Python tuples — small enough to pass by value, unlike GLBs):
#   JOINTS = (positions, rotations)
#     positions: numpy float32, shape (T, 77, 3), units = CENTIMETERS
#     rotations: numpy float32, shape (T, 77, 3, 3), world-space rotation matrices
#   MESH = (vertices, faces)
#     vertices: numpy float32, shape (V, 3), units = CENTIMETERS
#     faces:    numpy int32,   shape (F, 3), vertex indices into vertices[]
JOINTS_TYPE = "JOINTS"
MESH_TYPE = "MESH"


def _as_float32_np(arr, name: str):
    """Coerce a graph-passed array to float32 numpy or raise.

    Graph-passed values can be numpy arrays OR torch tensors (ComfyUI
    sometimes boxes things). Accept both; reject anything else loudly.
    """
    import numpy as _np
    if isinstance(arr, _np.ndarray):
        return arr.astype(_np.float32, copy=False)
    # torch tensor — accept and convert.
    if hasattr(arr, "detach") and hasattr(arr, "cpu"):
        return arr.detach().cpu().numpy().astype(_np.float32, copy=False)
    raise TypeError(
        f"{name} must be a numpy array or torch tensor, got {type(arr).__name__}. "
        "The graph wire format is documented above JOINTS_TYPE in nodes.py."
    )


def _as_int32_np(arr, name: str):
    import numpy as _np
    if isinstance(arr, _np.ndarray):
        return arr.astype(_np.int32, copy=False)
    if hasattr(arr, "detach") and hasattr(arr, "cpu"):
        return arr.detach().cpu().numpy().astype(_np.int32, copy=False)
    raise TypeError(
        f"{name} must be a numpy int array or torch tensor, got {type(arr).__name__}."
    )


# ════════════════════════════════════════════════════════════════════════
# SOMAX template path — the EXACT npz the weight transfer uses.
# MUST match soma_weight_transfer.py::SOMA_SKIN_PATH.
# THE IT EXPECTS." This is the canonical reference — not the engine,
# not a computed A-pose, not anny. The npz IS the weight-transfer template.)
#
# FULL POSE REFERENCE: docs/CANONICAL-SOMAX-POSE.md
#   Upper arm drop: 51.2°   Forearm drop: 35.4°   Elbow bend: 25.2°
# ════════════════════════════════════════════════════════════════════════
SOMA_SKIN_PATH = "/opt/kimodo/kimodo/assets/skeletons/somaskel77/skin_standard.npz"


@_lru_cache(maxsize=1)
def _load_template_bind():
    """Load the EXACT SOMAX template bind pose from skin_standard.npz.

    Returns (positions_cm, rotations, vertices_cm, faces) where:
      positions_cm: (77, 3) float32 — joint positions in CENTIMETERS
      rotations:    (77, 3, 3) float32 — joint rotations (world-space)
      vertices_cm:  (V, 3) float32 — bind mesh vertices in CENTIMETERS
      faces:        (F, 3) int32 — triangle indices

    This is the SAME data the weight transfer uses (loaded from the SAME
    file). No engine, no statistical mean, no computed A-pose. The pose is
    ~51° arm drop with ~25° elbow bend — the ACTUAL canonical SOMAX rest
    pose, not a synthetic T-pose rotated 45°.

    CACHED (suggestion #4, 2026-07-31): the npz is a 531 KB read that
    decodes ~300 KB of float32 arrays. ``lru_cache(maxsize=1)`` makes the
    FIRST call pay the I/O + decode cost, and every subsequent call in the
    same ComfyUI process returns the cached tuple in O(1). Multi-angle
    pipelines (anny_depth's 8-angle mode) and the combined
    poser_template_views brick both call this function multiple times per
    run — caching cuts the npz work from N×(531KB read) to 1×(531KB read)
    per process, per session.

    The cache key is empty (the path is a module-level constant), so the
    tuple is genuinely process-singleton. ``functools.lru_cache`` is
    thread-safe under the GIL, so concurrent ComfyUI node executions share
    the cache cleanly. The npz is a vendored read-only asset — if it ever
    changes, the ComfyUI container must be restarted (same as any code
    change), at which point the new process starts with an empty cache.

    Callers MUST NOT mutate the returned arrays — they're shared across
    all callers via the cache. The function already calls ``.copy()`` on
    the rotations slice (so the (77, 3, 3) view doesn't alias the bind
    matrix); positions/vertices/faces are fresh allocations from the
    astype() calls. Treat the return value as immutable.
    """
    import numpy as _np
    soma = _np.load(SOMA_SKIN_PATH, allow_pickle=True)
    bind = soma["bind_rig_transform"].astype(_np.float32)   # (77, 4, 4)
    positions_m = bind[:, :3, 3]                             # (77, 3) meters
    rotations = bind[:, :3, :3].copy()                       # (77, 3, 3)
    # npz is in METERS; the JOINTS/MESH contract is CENTIMETERS.
    positions_cm = (positions_m * 100.0).astype(_np.float32)
    vertices_cm = (soma["bind_vertices"].astype(_np.float32) * 100.0)
    faces = soma["faces"].astype(_np.int32)
    return positions_cm, rotations, vertices_cm, faces


# ════════════════════════════════════════════════════════════════════════
# Node: PoserTemplateBind — 1:1 template JOINTS + MESH from skin_standard.npz
# ════════════════════════════════════════════════════════════════════════
class PoserTemplateBind:
    """Load the EXACT SOMAX template bind pose from skin_standard.npz.

    This is the 1:1 canonical reference — the SAME file the weight transfer
    uses. No engine, no deformation, no anny body model, no computed A-pose.

    Use this instead of PoserBindPose + PoserDeformSoma when you need the
    depth + openpose to EXACTLY match what SOMAX expects. The generated
    character will come out in the template's pose, making the weight
    transfer alignment trivially correct.

    The template pose has ~51° arm drop with ~25° elbow bend (a natural
    resting pose from MoCap data). This is NOT a textbook A-pose — it's
    the ACTUAL pose the weight-transfer mesh is in.

    Outputs:
      joints: (positions, rotations) — JOINTS wire format, units = cm
      mesh:   (vertices, faces) — MESH wire format, units = cm
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = (JOINTS_TYPE, MESH_TYPE)
    RETURN_NAMES = ("joints", "mesh")
    FUNCTION = "load"
    CATEGORY = "TechNoir/Poser"

    def load(self):
        import numpy as _np
        positions_cm, rotations, vertices_cm, faces = _load_template_bind()
        # Wrap as (1, 77, ...) for the JOINTS contract (T=1 single frame)
        positions_t = positions_cm[_np.newaxis, ...]
        rotations_t = rotations[_np.newaxis, ...]
        log.info(
            "[PoserTemplateBind] 1:1 template from %s: 77 joints, "
            "%d verts, %d faces (NO engine, NO deform)",
            SOMA_SKIN_PATH, len(vertices_cm), len(faces),
        )
        return {
            "result": (
                (positions_t, rotations_t),
                (vertices_cm, faces),
            ),
        }


# ════════════════════════════════════════════════════════════════════════
# Node: PoserBindPose
# ════════════════════════════════════════════════════════════════════════
class PoserBindPose:
    """Bind-pose SOMA-77 joints for a (optionally shaped) identity model.

    Returns the canonical A-pose joint set: 77 positions (cm) + 77 world-
    space rotation matrices. This is the ControlNet-conditioning ground
    truth for a body-type-aware Anny mesh — the same data the
    ``/poser/skin/bind_pose`` HTTP route returns, but as a typed graph
    value that flows into ``PoserDeformSoma``.

    UNIT CONTRACT (silent-failure point #1):
      Output positions are in CENTIMETERS (matching the deform input unit).
      Internally, SkinEngine works in METERS — this node does the m→cm
      conversion so the next node (PoserDeformSoma) receives cm and does
      cm→m on its own entry, mirroring the HTTP route's contract.

    ROTATION CONTRACT (silent-failure points #2 + #3 + #4):
      Output rotations are WORLD-space, shape (1, 77, 3, 3), float32. Each
      [77][i] is a 3×3 rotation matrix; the bottom row [3][3] of the
      original 4×4 transform is dropped (homogeneous row, always [0,0,0,1]).
      TRANSLATION IS NOT IN THE ROTATIONS — read it from `positions` instead.

    IDENTITY CONTRACT (silent-failure points #6 + #7):
      Known identity_model values: "soma" (default, neutral adult),
      "anny" (anime-stylized adult female, used by anny_depth), "child".
      identity_coeffs: comma-separated list of body-shape coefficients
      in [0, 1] (0.5 = neutral for anny). Empty string = use neutral.
      local_changes: JSON object of detail morphs → value in [0,1].
        These are the 189+ body-part morphs (hip-width-incr, waist-narrow,
        breast-scale-incr, etc.) — the fine-grained knobs BEYOND the 11
        global phenotype dims. See docs/anny-phenotype-mapping.md.
        Empty string = no detail morphs. Example:
        '{"hip-width-incr":0.7,"waist-narrow":0.5,"breast-scale-incr":0.6}'
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "identity_model": ("STRING", {"default": "anny"}),
                "identity_coeffs": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Comma-separated body-shape coefficients in [0,1]. "
                               "Empty = neutral. Example: '0.7,0.4,0.55'",
                }),
                "units": (["cm", "m"], {"default": "cm"}),
            },
            "optional": {
                "local_changes": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "JSON object of detail morph → value in [0,1]. "
                               "Empty = none. Example: "
                               '{\"hip-width-incr\":0.7,\"waist-narrow\":0.5}',
                }),
            },
        }

    RETURN_TYPES = (JOINTS_TYPE,)
    RETURN_NAMES = ("joints",)
    FUNCTION = "bind"
    CATEGORY = "TechNoir/Poser"

    def bind(self, identity_model: str, identity_coeffs: str, units: str,
             local_changes: str = ""):
        import json as _json
        import numpy as _np
        # Local imports — these pull torch + py-soma-x, only needed at execution.
        from .skin import get_engine

        identity_model = (identity_model or "anny").lower()
        if units not in ("cm", "m"):
            raise ValueError(
                f"PoserBindPose: units must be 'cm' or 'm', got {units!r}. "
                "This is the OUTPUT unit of positions; rotations are unitless."
            )

        # Parse identity_coeffs — empty = None (engine uses neutral).
        coeffs = None
        if identity_coeffs and identity_coeffs.strip():
            try:
                coeffs = [float(x) for x in identity_coeffs.split(",")]
            except ValueError as e:
                raise ValueError(
                    f"PoserBindPose: identity_coeffs must be a comma-separated "
                    f"list of floats (e.g. '0.7,0.4,0.55'). Got {identity_coeffs!r}. "
                    f"Parse error: {e}"
                ) from e

        # Parse local_changes — JSON object of morph → value.
        # Empty = no detail morphs (the 11 global dims still apply via coeffs).
        local_changes_dict: dict | None = None
        if local_changes and local_changes.strip():
            try:
                parsed = _json.loads(local_changes)
            except _json.JSONDecodeError as e:
                raise ValueError(
                    f"PoserBindPose: local_changes must be a JSON object of "
                    f"morph→value (e.g. '{{\"hip-width-incr\":0.7}}'). "
                    f"Got parse error: {e}"
                ) from e
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"PoserBindPose: local_changes must be a JSON OBJECT, "
                    f"got {type(parsed).__name__}."
                )
            local_changes_dict = {
                str(k): float(v) for k, v in parsed.items()
                if v is not None
            }

        # ── Engine call ────────────────────────────────────────────────
        # Synchronous (ComfyUI runs node execute in a worker thread; the
        # async HTTP handlers use asyncio.to_thread — same underlying call).
        engine = get_engine(identity_model)
        if coeffs is not None or local_changes_dict:
            # set_identity(coeffs, local_changes, custom_targets=None).
            # local_changes flow through to AnnySimplified.forward as
            # scale_params — the 189+ body-part morphs (hips, waist, breasts).
            # Previously hardcoded to None, which is why the body was always
            # the androgynous default regardless of what the prompt asked for.
            engine.set_identity(coeffs, local_changes_dict, None)

        # (77, 4, 4) world-space transforms in METERS (py-soma-x native).
        transforms_m = engine._extract_soma_x_transforms()
        positions_m = transforms_m[:, :3, 3]            # (77, 3)
        rotations = transforms_m[:, :3, :3].copy()       # (77, 3, 3)

        # ── Unit conversion: meters → requested unit ───────────────────
        # SILENT-FAILURE GUARD #1: If you flip this scale factor, every
        # downstream vertex coordinate silently becomes 100× larger or
        # 100× smaller. The HTTP route (server.py:298) multiplies by 100.0
        # for cm; we mirror that here.
        scale = 100.0 if units == "cm" else 1.0
        positions_out = (positions_m * scale).astype(_np.float32)
        rotations_out = rotations.astype(_np.float32)

        # ── Shape guard: silent-failure points #3 + #4 ─────────────────
        if positions_out.shape != (77, 3):
            raise RuntimeError(
                f"PoserBindPose: expected positions shape (77, 3), got "
                f"{positions_out.shape}. The SOMA-77 joint count is fixed; "
                f"a different shape means py-soma-x changed its contract "
                f"or the wrong identity_model was loaded."
            )
        if rotations_out.shape != (77, 3, 3):
            raise RuntimeError(
                f"PoserBindPose: expected rotations shape (77, 3, 3), got "
                f"{rotations_out.shape}. Transposed matrices (3, 3, 77) "
                f"would silently broadcast wrong downstream — caught here."
            )

        # ── Wrap as (T=1, 77, ...) — single-frame batch ────────────────
        # deform_soma accepts (T, 77, 3) and (T, 77, 3, 3). We always emit
        # T=1 here; a future PoserDeformSoma from a motion clip would emit
        # T=N. Wrapping explicitly avoids the (77, 3) vs (1, 77, 3) shape
        # ambiguity in deform_soma's input parser.
        positions_t = positions_out[_np.newaxis, ...]   # (1, 77, 3)
        rotations_t = rotations_out[_np.newaxis, ...]   # (1, 77, 3, 3)

        log.info(
            "[PoserBindPose] identity=%s coeffs=%s units=%s → "
            "positions(1,77,3) rotations(1,77,3,3)",
            identity_model,
            f"[{len(coeffs)} coeffs]" if coeffs else "neutral",
            units,
        )

        return {
            "result": ((positions_t, rotations_t),),
            "ui": {
                "poser": [{
                    "op": "bind_pose",
                    "identity_model": identity_model,
                    "coeffs_count": len(coeffs) if coeffs else 0,
                    "units": units,
                    "joint_count": 77,
                }],
            },
        }


# ════════════════════════════════════════════════════════════════════════
# Node: PoserBindMesh
# ════════════════════════════════════════════════════════════════════════
class PoserBindMesh:
    """Load the identity bind-pose mesh directly, no deformation.

    Returns the canonical bind mesh (vertices + faces) for the requested
    identity model. Unlike PoserDeformSoma, this does NOT call
    engine.deform() — it returns the mesh at its rest bind pose.

    This is the mesh equivalent of PoserBindPose (which returns bind-pose
    joints). Use this when you want to render the identity mesh at its
    neutral bind pose without applying any joint deformation.

    Supported identity_model values: "soma" (neutral SOMAX mannequin),
    "anny", "mhr". For "soma", the SOMA NPZ engine loads the mesh at
    bind pose directly — PoserDeformSoma would fail because the SOMA
    engine doesn't support deform. This node handles that case by
    returning the engine's loaded vertices+faces directly.

    OUTPUT CONTRACT:
      MESH tuple — (vertices, faces)
        vertices: float32 (V, 3), units = CENTIMETERS (same as the npz)
        faces:    int32 (F, 3), triangle vertex indices
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "identity_model": (["soma", "anny", "mhr"], {"default": "soma"}),
            },
        }

    RETURN_TYPES = (MESH_TYPE,)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "load"
    CATEGORY = "TechNoir/Poser"

    def load(self, identity_model: str):
        import numpy as _np
        from .skin import get_engine

        identity_model = (identity_model or "soma").lower()
        engine = get_engine(identity_model)

        vertices = _np.asarray(engine.vertices, dtype=_np.float32)
        faces = _np.asarray(engine.faces, dtype=_np.int32)

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise RuntimeError(
                f"PoserBindMesh: engine.vertices has unexpected shape "
                f"{vertices.shape}. Expected (V, 3)."
            )
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise RuntimeError(
                f"PoserBindMesh: engine.faces has unexpected shape "
                f"{faces.shape}. Expected (F, 3)."
            )

        log.info(
            "[PoserBindMesh] identity=%s verts=%d faces=%d (bind-pose, cm)",
            identity_model, vertices.shape[0], faces.shape[0],
        )

        return {
            "result": ((vertices, faces),),
            "ui": {
                "poser": [{
                    "op": "bind_mesh",
                    "identity_model": identity_model,
                    "vertex_count": int(vertices.shape[0]),
                    "face_count": int(faces.shape[0]),
                }],
            },
        }


# ════════════════════════════════════════════════════════════════════════
# Node: PoserDeformSoma
# ════════════════════════════════════════════════════════════════════════
class PoserDeformSoma:
    """Apply SOMA-77 joint positions + rotations to a bind mesh → vertices.

    Wraps SkinEngine.deform() — the same Linear Blend Skinning call the
    ``/poser/skin/deform_soma`` HTTP route uses, but as a typed graph node.

    INPUT CONTRACT (silent-failure points #1, #3, #4, #5):
      joints: JOINTS tuple from PoserBindPose — (positions, rotations)
        positions: float32 (T, 77, 3), units = CENTIMETERS (NOT meters).
        rotations: float32 (T, 77, 3, 3), world-space.
      This node converts cm → m on entry, mirroring the HTTP route.

    OUTPUT CONTRACT:
      MESH tuple — (vertices, faces)
        vertices: float32 (V, 3), units = CENTIMETERS (m → cm on exit,
                                                  mirroring the HTTP route).
        faces: int32 (F, 3), vertex indices (pulled from the bind mesh).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "joints": (JOINTS_TYPE,),
                "identity_model": ("STRING", {"default": "anny"}),
            },
        }

    RETURN_TYPES = (MESH_TYPE,)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "deform"
    CATEGORY = "TechNoir/Poser"

    def deform(self, joints, identity_model: str):
        import numpy as _np
        from .skin import get_engine

        # ── Unwrap + validate JOINTS ───────────────────────────────────
        if not isinstance(joints, tuple) or len(joints) != 2:
            raise TypeError(
                f"PoserDeformSoma: joints must be a (positions, rotations) tuple "
                f"from PoserBindPose. Got {type(joints).__name__} len={len(joints) if hasattr(joints, '__len__') else 'n/a'}. "
                "The JOINTS wire format is documented above JOINTS_TYPE in nodes.py."
            )
        positions_cm, rotations = joints
        positions_cm = _as_float32_np(positions_cm, "positions")
        rotations = _as_float32_np(rotations, "rotations")

        if positions_cm.ndim != 3 or positions_cm.shape[1:] != (77, 3):
            raise RuntimeError(
                f"PoserDeformSoma: positions must have shape (T, 77, 3). "
                f"Got {positions_cm.shape}. A (77, 3) input would silently "
                f"broadcast wrong; the PoserBindPose node wraps it as (1, 77, 3) "
                f"to avoid that — did you bypass it?"
            )
        if rotations.ndim != 4 or rotations.shape[1:] != (77, 3, 3):
            raise RuntimeError(
                f"PoserDeformSoma: rotations must have shape (T, 77, 3, 3). "
                f"Got {rotations.shape}. Transposed (T, 3, 3, 77) silently "
                f"reduces along the wrong axis — this is exactly the silent "
                f"rotation bug the user has been burned by."
            )
        if positions_cm.shape[0] != rotations.shape[0]:
            raise RuntimeError(
                f"PoserDeformSoma: positions and rotations must have the same "
                f"T (frame count). Got positions T={positions_cm.shape[0]}, "
                f"rotations T={rotations.shape[0]}."
            )

        identity_model = (identity_model or "anny").lower()
        engine = get_engine(identity_model)

        # ── Unit conversion: cm → m (silent-failure point #1) ───────────
        # SkinEngine.deform() works in METERS. The JOINTS contract is cm.
        # This division is the single most silent-failure-prone line in
        # the chain — if a future caller passes positions already in m,
        # the deform will be 100× too small and the render downstream
        # will produce a tiny invisible dot (or nothing). There is no
        # way to detect "wrong unit" from the number alone.
        positions_m = positions_cm / 100.0

        # deform signature: deform(pos_m, rot_m) where pos_m is (T, 77, 3)
        # in meters and rot_m is (T, 77, 3, 3) in world-space.
        deformed_m = engine.deform(positions_m, rotations)
        # deformed_m: (V, 3) or (T, V, 3) in meters. For T=1 input the
        # engine returns (V, 3) — squeeze if needed.
        if deformed_m.ndim == 3 and deformed_m.shape[0] == 1:
            deformed_m = deformed_m[0]
        if deformed_m.ndim != 2 or deformed_m.shape[1] != 3:
            raise RuntimeError(
                f"PoserDeformSoma: engine.deform() returned unexpected shape "
                f"{deformed_m.shape}. Expected (V, 3)."
            )

        # ── Unit conversion: m → cm (mirror the HTTP route) ────────────
        vertices_cm = (deformed_m * 100.0).astype(_np.float32)

        # ── Faces: pull from the bind mesh for this identity ───────────
        # The HTTP route does GET /poser/skin/mesh?identity_model=anny
        # to fetch the static face indices. Here we ask the engine directly
        # — same data, no HTTP round-trip.
        #
        # Silent-failure guard: faces MUST come from the SAME identity_model
        # whose vertices we just deformed. Cross-identity face indices would
        # point at nonexistent vertices or wrap silently via numpy indexing.
        # The engine's faces are loaded at construction and are (F, 3) int32
        # already — see skin.py:254 and skin.py:377.
        faces = _np.asarray(engine.faces, dtype=_np.int32)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise RuntimeError(
                f"PoserDeformSoma: engine.faces has unexpected shape "
                f"{faces.shape}. Expected (F, 3) — see skin.py for the "
                f"SkinEngine.faces contract."
            )

        log.info(
            "[PoserDeformSoma] identity=%s verts=%d faces=%d (cm, world-space)",
            identity_model, vertices_cm.shape[0], faces.shape[0],
        )

        return {
            "result": ((vertices_cm, faces),),
            "ui": {
                "poser": [{
                    "op": "deform_soma",
                    "identity_model": identity_model,
                    "vertex_count": int(vertices_cm.shape[0]),
                    "face_count": int(faces.shape[0]),
                }],
            },
        }


# ════════════════════════════════════════════════════════════════════════
# Node: PoserRender
# ════════════════════════════════════════════════════════════════════════
class PoserRender:
    """Rasterize a MESH → depth/normal/rgb ControlNet-ready IMAGE.

    Wraps rasterize.render_passes() — same call the ``/poser/skin/render``
    HTTP route uses. Returns a standard ComfyUI IMAGE tensor (B, H, W, 3)
    float in [0, 1], suitable for piping into any ControlNet preprocessor
    or conditioning node.

    INPUT CONTRACT:
      mesh: MESH tuple from PoserDeformSoma — (vertices, faces)
        vertices: float32 (V, 3), units = CENTIMETERS. The render path
                  assumes cm; do NOT pre-convert to meters here.
        faces: int32 (F, 3).

    OUTPUT CONTRACT:
      IMAGE: float32 tensor (1, H, W, 3) in [0, 1]. The depth pass is
             expanded to 3-channel RGB (the ControlNet contract — single-
             channel conditioning silently fails on most controlnet nodes).

    PARAMS:
      passes: which render passes to compute. "depth" is the ControlNet-
              conditioning default. "normal" is for surface-aware control.
              The output IMAGE is always 3-channel; if multiple passes are
              selected, the FIRST one is returned as the IMAGE output and
              the rest are saved alongside (the brick orchestrator fetches
              them via the output filenames in the workflow result).
      framing: tightness of camera framing. 1.5 (default) ≈ body fills 65%
               of frame; 1.12 ≈ body fills 88% (tight portrait, the value
               used by anny_depth for ControlNet conditioning). No validation
               — 0.1 produces an invisible mesh, 10.0 a pixel dot. Both
               silently succeed. Document your choice at the call site.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": (MESH_TYPE,),
                "width": ("INT", {"default": 768, "min": 32, "max": 4096}),
                "height": ("INT", {"default": 1024, "min": 32, "max": 4096}),
                "passes": (["depth", "normal", "rgb"], {"default": "depth"}),
                "background": (["black", "white"], {"default": "black"}),
                "framing": ("FLOAT", {
                    "default": 1.12,
                    "min": 0.1, "max": 10.0, "step": 0.01,
                    "tooltip": "Camera framing tightness. 1.5 = loose (65% "
                               "frame fill). 1.12 = tight portrait (88%). "
                               "anny_depth ControlNet conditioning uses 1.12.",
                }),
                "azimuth": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0}),
                "elevation": ("FLOAT", {"default": 5.0, "min": -90.0, "max": 90.0}),
                "fov": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 170.0}),
                "target_y": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Vertical offset of the camera look-at point "
                               "in NORMALIZED mesh space. SIGN CONVENTION "
                               "(empirically verified 2026-07-28): HIGHER "
                               "target_y = subject HIGHER in frame. "
                               "target_y=+0.3 → head at row 7/1024 (nearly "
                               "clipped at top). target_y=0 → centered, "
                               "full body visible (head row 64, foot row "
                               "953). target_y=-0.05 → head row 214, feet "
                               "CLIP at row 1023. Two prior commits had "
                               "this sign backwards. 0.0 is the safe default.",
                }),
                "mesh_color": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional RGB albedo override for the rgb "
                               "pass, as 'r,g,b' floats in [0,1]. Empty = "
                               "warm gray (0.78,0.75,0.72). Example: "
                               "'0.6,0.74,0.97' for Pose Studio blue "
                               "(#98bdf7). Only affects the 'rgb' pass.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render"
    CATEGORY = "TechNoir/Poser"
    OUTPUT_NODE = True

    def render(
        self,
        mesh,
        width: int,
        height: int,
        passes: str,
        background: str,
        framing: float,
        azimuth: float,
        elevation: float,
        fov: float,
        target_y: float = 0.0,
        mesh_color: str = "",
    ):
        import io as _io
        import numpy as _np
        from PIL import Image as _PILImage
        import torch as _torch
        from .rasterize import render_passes

        # ── Unwrap + validate MESH ─────────────────────────────────────
        if not isinstance(mesh, tuple) or len(mesh) != 2:
            raise TypeError(
                f"PoserRender: mesh must be a (vertices, faces) tuple from "
                f"PoserDeformSoma. Got {type(mesh).__name__}."
            )
        vertices_cm, faces = mesh
        vertices_cm = _as_float32_np(vertices_cm, "vertices")
        faces = _as_int32_np(faces, "faces")

        if vertices_cm.ndim != 2 or vertices_cm.shape[1] != 3:
            raise RuntimeError(
                f"PoserRender: vertices must have shape (V, 3). "
                f"Got {vertices_cm.shape}."
            )
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise RuntimeError(
                f"PoserRender: faces must have shape (F, 3). Got {faces.shape}."
            )
        if vertices_cm.shape[0] == 0 or faces.shape[0] == 0:
            raise RuntimeError(
                "PoserRender: empty mesh (zero vertices or zero faces). "
                "PoserDeformSoma returned a degenerate mesh — investigate "
                "upstream, do not silently render nothing."
            )

        # ── Render passes via the shared rasterizer ────────────────────
        # render_passes returns dict {pass_name: png_bytes}. We compute
        # all requested passes (saves them to output/) but only the first
        # becomes the IMAGE tensor that flows downstream.
        passes_list = [passes]  # INPUT_TYPES restricts to one choice for now.

        # Parse optional mesh_color override (e.g. "0.6,0.74,0.97")
        mesh_color_parsed = None
        if mesh_color and mesh_color.strip():
            parts = [float(x.strip()) for x in mesh_color.split(",")]
            if len(parts) == 3:
                mesh_color_parsed = tuple(parts)

        pngs = render_passes(
            vertices_cm,
            faces,
            width=int(width),
            height=int(height),
            azimuth=float(azimuth),
            elevation=float(elevation),
            fov=float(fov),
            passes=passes_list,
            background=str(background),
            framing=float(framing),
            target_y=float(target_y),
            mesh_color=mesh_color_parsed,
        )

        if passes not in pngs:
            raise RuntimeError(
                f"PoserRender: render_passes did not produce the requested "
                f"'{passes}' pass. Got passes: {sorted(pngs.keys())}."
            )

        # ── PNG bytes → IMAGE tensor (B=1, H, W, 3) float in [0, 1] ────
        png_bytes = pngs[passes]
        img = _PILImage.open(_io.BytesIO(png_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = _np.asarray(img, dtype=_np.float32) / 255.0
        # arr shape: (H, W, 3) — add batch dim for the ComfyUI IMAGE contract.
        tensor = _torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, 3)

        # ── Persist the PNG to output/ (ControlNet preview + audit) ────
        out_path = _output_path(f"poser_render_{passes}", ".png")
        with open(out_path, "wb") as f:
            f.write(png_bytes)

        log.info(
            "[PoserRender] pass=%s %dx%d framing=%.2f az=%.1f el=%.1f "
            "fov=%.1f target_y=%.2f → %s (%d bytes)",
            passes, int(width), int(height), framing,
            float(azimuth), float(elevation), float(fov), float(target_y),
            out_path, len(png_bytes),
        )

        return {
            "result": (tensor,),
            "ui": {
                "images": [_ui_entry(out_path)],
                "poser": [{
                    "op": "render",
                    "pass": passes,
                    "width": int(width),
                    "height": int(height),
                    "framing": float(framing),
                    "azimuth": float(azimuth),
                    "elevation": float(elevation),
                    "fov": float(fov),
                    "target_y": float(target_y),
                    "background": background,
                }],
            },
        }


class PoserRenderOpenPose:
    """Project SOMA-77 joints → OpenPose skeleton image (zero estimation error).

    Bypasses DWPose entirely. Instead of rendering the mesh to RGB and
    letting a neural network ESTIMATE the skeleton from pixels, this node
    projects the 77 known 3D joint positions through the EXACT same camera
    pipeline as ``PoserRender`` (same view-projection matrix, same framing,
    same bounding-sphere normalization) and draws the OpenPose-format
    skeleton directly at the mathematically exact 2D coordinates.

    This eliminates the DWPose estimation step that introduces pixel drift
    on CG renders — the ControlNet sees a skeleton that is sub-pixel-
    aligned with the mesh render and the depth map.

    CAMERA MATCHING CONTRACT:
      This node MUST receive the same ``width``, ``height``, ``framing``,
      ``azimuth``, ``elevation``, ``fov``, and ``target_y`` values as the
      ``PoserRender`` it runs alongside. The bounding-sphere normalization
      (center + radius) is computed from the MESH input, so the mesh and
      joint projections share the same world→screen transform. Any
      mismatch in camera params produces a skeleton that is offset from
      the rendered mesh — silent failure with no error.

    INPUT CONTRACT:
      joints: JOINTS from PoserTemplateBind or PoserBindPose — (positions,
        rotations). positions: float32 (T, 77, 3), units = CENTIMETERS.
        OPTIONAL — when omitted, the node loads the 1:1 template bind
        pose from skin_standard.npz (the ACTUAL canonical SOMAX rest
        pose: ~51° arm drop, ~25° elbow bend). This bypasses the engine
        entirely and matches the weight-transfer template exactly.
        See docs/CANONICAL-SOMAX-POSE.md.
      mesh: MESH from PoserTemplateBind or PoserDeformSoma — (vertices,
        faces). Used ONLY for bounding-sphere normalization (center +
        radius). The vertices must be in the same cm coordinate system
        as the joint positions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 768, "min": 32, "max": 4096}),
                "height": ("INT", {"default": 1024, "min": 32, "max": 4096}),
                "framing": ("FLOAT", {
                    "default": 1.05,
                    "min": 0.1, "max": 10.0, "step": 0.01,
                    "tooltip": "Camera framing — MUST match PoserRender's "
                               "value for the skeleton to align with the "
                               "mesh render.",
                }),
                "azimuth": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0}),
                "elevation": ("FLOAT", {"default": 5.0, "min": -90.0, "max": 90.0}),
                "fov": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 170.0}),
                "target_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "joints": (JOINTS_TYPE, {"tooltip": "OPTIONAL. When provided, "
                               "uses those joint positions (from "
                               "PoserTemplateBind or PoserBindPose). When "
                               "OMITTED, loads the 1:1 template bind pose "
                               "from skin_standard.npz (51° arm drop, 25° "
                               "elbow bend) — the ACTUAL canonical SOMAX "
                               "rest pose, not a synthetic A-pose."}),
                "mesh": (MESH_TYPE, {"tooltip": "OPTIONAL. When provided, bounding-"
                               "sphere normalization uses the mesh vertices "
                               "(same as PoserRender). When omitted, the "
                               "bounding sphere is computed from the 77 joints "
                               "directly — needed for the 'soma' identity_model "
                               "which can't deform (no PoserDeformSoma)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render"
    CATEGORY = "TechNoir/Poser"
    OUTPUT_NODE = True

    def render(
        self,
        width: int,
        height: int,
        framing: float,
        azimuth: float,
        elevation: float,
        fov: float,
        target_y: float = 0.0,
        joints=None,
        mesh=None,
    ):
        import math as _math
        import numpy as _np
        from PIL import Image as _PILImage
        import torch as _torch

        # ── Resolve joint positions ────────────────────────────────────
        # When joints input is provided (from PoserTemplateBind or
        # PoserBindPose), unwrap it. When joints is None, load the 1:1
        # template bind pose from skin_standard.npz — the ACTUAL canonical
        # SOMAX rest pose (51° arm drop, 25° elbow bend), not the synthetic
        # SOMA77_APOSE (which was a zero-bend 45° A-pose).
        _tpl_verts_cm = None  # template mesh vertices for bounding sphere
        if joints is not None:
            if not isinstance(joints, tuple) or len(joints) != 2:
                raise TypeError(
                    f"PoserRenderOpenPose: joints must be a (positions, rotations) "
                    f"tuple from PoserBindPose. Got {type(joints).__name__}."
                )
            joint_pos_cm = _as_float32_np(joints[0], "joint positions")
            # Collapse time dimension if present: (T, 77, 3) → (77, 3)
            if joint_pos_cm.ndim == 3:
                joint_pos_cm = joint_pos_cm[0]
            if joint_pos_cm.shape != (77, 3):
                raise RuntimeError(
                    f"PoserRenderOpenPose: expected joint positions shape "
                    f"(77, 3), got {joint_pos_cm.shape}. SOMA-77 joint count "
                    f"is fixed."
                )
        else:
            # ── 1:1 template bind pose from skin_standard.npz ───────────
            # This is the EXACT same file the weight transfer uses. The pose
            # has ~51° arm drop with ~25° elbow bend — the ACTUAL canonical
            # SOMAX rest pose. NOT the synthetic SOMA77_APOSE (which was a
            # zero-bend 45° computed A-pose that didn't match the template).
            #
 # EXACTLY WHAT THE IT EXPECTS.")
            joint_pos_cm, _, _tpl_verts_cm, _ = _load_template_bind()
            log.info(
                "[PoserRenderOpenPose] using 1:1 template bind pose "
                "from skin_standard.npz (~51° arm drop, ~25° elbow bend)"
            )

        # ── Bounding-sphere normalization ────────────────────────────────
        # When mesh is provided (from PoserDeformSoma), use mesh vertices for
        # the bounding sphere — matches PoserRender exactly.
        # When mesh is None (soma identity_model — no deform), compute the
        # bounding sphere from the 77 joints directly. The joints cover wrist
        # and ankle but not fingertips/toes, so apply a 1.15× margin to
        # approximate the full body extent.
        if mesh is not None:
            if not isinstance(mesh, tuple) or len(mesh) != 2:
                raise TypeError(
                    f"PoserRenderOpenPose: mesh must be a (vertices, faces) "
                    f"tuple. Got {type(mesh).__name__}."
                )
            verts_cm = _as_float32_np(mesh[0], "vertices")
            # Shared bounding-sphere helper (suggestion #2, 2026-07-11).
            # Same function as rasterize.py / PoserRender — guarantees
            # pixel alignment by construction.
            center, radius = _bounding_sphere(verts_cm)
        elif _tpl_verts_cm is not None:
            # Template mesh bounding sphere — pixel-aligns with PoserRender
            # (depth) which renders the same template mesh vertices. This
            # is critical: when PoserRenderOpenPose loads the template
            # internally (no mesh input), it MUST use the same vertex-based
            # bounding sphere as PoserRender, otherwise the skeleton and
            # depth are at different scales and the ControlNets fight.
            # Same helper as PoserRender — drift structurally impossible.
            center, radius = _bounding_sphere(_tpl_verts_cm)
        else:
            # Joint-only bounding sphere (deformed pose, no mesh available).
            center, radius = _bounding_sphere(joint_pos_cm)
            # Joints don't reach fingertips/toes/skull-top — add margin so
            # the skeleton framing matches what the mesh would produce.
            radius *= 1.15
        if radius < 1e-6:
            raise ValueError(
                "PoserRenderOpenPose: degenerate — zero bounding-sphere "
                "radius. Cannot normalize joints."
            )
        joints_c = (joint_pos_cm - center) / radius

        # ── Camera pipeline (pure numpy — avoids device mismatch) ────────
        # The math is IDENTICAL to rasterize.py's _look_at + _perspective,
        # but computed in pure numpy. We can't call those helpers directly
        # because _perspective() uses device=_dev() (cuda), while our eye/
        # target/up tensors are on cpu — cross-device mm crashes.
        az = _math.radians(azimuth)
        el = _math.radians(elevation)
        cam_dist = 1.0 / _math.tan(_math.radians(fov) / 2.0) * framing
        eye = _np.array([
            cam_dist * _math.cos(el) * _math.sin(az),
            cam_dist * _math.sin(el),
            cam_dist * _math.cos(el) * _math.cos(az),
        ], dtype=_np.float32)
        target = _np.array([0.0, target_y, 0.0], dtype=_np.float32)
        up = _np.array([0.0, 1.0, 0.0], dtype=_np.float32)

        # View matrix (look-at) — mirrors _look_at in rasterize.py:86-100
        f = target - eye
        f /= _np.linalg.norm(f)
        r = _np.cross(f, up)
        r /= _np.linalg.norm(r)
        u = _np.cross(r, f)
        view = _np.eye(4, dtype=_np.float32)
        view[0, :3] = r
        view[1, :3] = u
        view[2, :3] = -f
        view[0, 3] = -_np.dot(r, eye)
        view[1, 3] = -_np.dot(u, eye)
        view[2, 3] = _np.dot(f, eye)

        # Perspective projection — mirrors _perspective in rasterize.py:103-114
        t = _math.tan(_math.radians(fov) / 2.0)
        aspect = width / height
        near, far = max(cam_dist - 2.0, 0.1), cam_dist + 2.0
        proj = _np.zeros((4, 4), dtype=_np.float32)
        proj[0, 0] = 1.0 / (t * aspect)
        proj[1, 1] = 1.0 / t
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = 2.0 * far * near / (near - far)
        proj[3, 2] = -1.0

        view_proj = proj @ view

        # ── Project 77 joints → 2D screen coordinates ───────────────────
        joints_homo = _np.hstack(
            [joints_c, _np.ones((77, 1), dtype=_np.float32)]
        )  # (77, 4)
        joints_clip = joints_homo @ view_proj.T  # (77, 4)
        # Perspective divide — points behind the camera (w <= 0) are invalid.
        w_clip = joints_clip[:, 3]
        valid = w_clip > 1e-6
        joints_ndc = _np.full((77, 2), _np.nan, dtype=_np.float32)
        joints_ndc[valid] = (
            joints_clip[valid, :2] / w_clip[valid, None]
        )  # (77, 2) — NDC x, y in [-1, 1]

        # NDC → screen pixels.
        # render_passes uses nvdiffrast (OpenGL: row 0 = bottom), then
        # _tensor_to_png flips vertically (row 0 = top). The net mapping
        # from NDC to the final PNG row is:
        #   x_px = (ndc_x + 1) * 0.5 * width
        #   y_px = (1 - ndc_y) * 0.5 * height
        joints_x = (joints_ndc[:, 0] + 1.0) * 0.5 * width
        joints_y = (1.0 - joints_ndc[:, 1]) * 0.5 * height
        joints_2d = _np.stack([joints_x, joints_y], axis=1)  # (77, 2)
        # NaN joints (behind camera) are skipped during drawing.

        # ── Authoritative OpenPose render (no guessing) ────────────────────
        # We delegate drawing to the EXACT renderer that produced the
        # training images for the OpenPose ControlNet: the
        # ``comfyui_controlnet_aux`` port of lllyasviel/ControlNet's
        # annotator (the ``OpenposePreprocessor`` node's own code). This
        # guarantees the skeleton is pixel-compatible with what the CN was
        # trained on — the COCO-18 body topology, the fixed rainbow limb
        # colors, the Neck→Nose / Nose→Eye head wiring (eyes branch off the
        # NOSE, never the neck; NO mid-hip — each hip connects to the neck),
        # and the HSV-rainbow hands with red dots.
        #
        # Source of truth (verbatim) installed in the ComfyUI container:
        #   custom_controlnet_aux/open_pose/util.py :: draw_bodypose, draw_handpose
        #   custom_controlnet_aux/open_pose/body.py :: Keypoint(x, y)  # x,y ∈ [0,1]
        # memory; use the authoritative renderer.
        from custom_controlnet_aux.open_pose.util import (
            draw_bodypose as _draw_bodypose,
            draw_handpose as _draw_handpose,
        )
        from custom_controlnet_aux.open_pose.body import Keypoint as _Keypoint
        from .library.skeleton import COCO_TO_SOMA_IDX as _COCO_TO_SOMA

        # SOMA-77 hand-joint indices for the 21-point COCO hand
        # [wrist, thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4].
        # SOMA fingers are a numbered chain (1..4) plus a terminal "End";
        # COCO uses 4 per finger. For the index/middle/ring/pinky we keep
        # the four numbered joints and drop "End"; the thumb has only three
        # numbered joints + End, so its 4th COCO joint IS the End.
        _SOMA_LEFT_HAND = [14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 29, 30, 31, 32, 34, 35, 36, 37]
        _SOMA_RIGHT_HAND = [42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 57, 58, 59, 60, 62, 63, 64, 65]

        def _kp(soma_idx: int) -> "_Keypoint | None":
            """Build a normalized [0,1] Keypoint from a SOMA-77 joint, or
            None when the joint projected behind the camera (NaN)."""
            x, y = joints_2d[soma_idx]
            if _np.isnan(x) or _np.isnan(y):
                return None
            return _Keypoint(
                x=float(x) / float(width),
                y=float(y) / float(height),
            )

        def _hand(soma_idxs: "list[int]") -> "list[_Keypoint] | None":
            """Build the 21-point COCO hand, or None to drop it.

            ``draw_handpose``'s dot loop does NOT tolerate None entries
            (only its edge loop guards with ``if k1 is None``) — it
            dereferences ``keypoint.x`` unconditionally. DWPose itself
            always emits all 21 hand joints or omits the hand entirely.
            We match that contract: if any SOMA joint is missing
            (projected behind the camera), drop the whole hand rather
            than crash the authoritative renderer.
            """
            kps = [_kp(i) for i in soma_idxs]
            return kps if all(k is not None for k in kps) else None

        # 18 COCO-18 body joints. Indices 16, 17 (ears) have no SOMA joint
        # (SOMA-77 has no ears) → None, so the ear dots/limbs are skipped by
        # the renderer exactly as they are when DWPose fails to find ears.
        # ``draw_bodypose`` handles None gracefully in BOTH its limb and dot
        # loops, so partial body data is safe (unlike hands).
        body_keypoints: "list[_Keypoint | None]" = [
            (_kp(_COCO_TO_SOMA[i]) if i in _COCO_TO_SOMA else None)
            for i in range(18)
        ]
        left_hand = _hand(_SOMA_LEFT_HAND)
        right_hand = _hand(_SOMA_RIGHT_HAND)

        # Black canvas — identical setup to draw_poses() in the
        # authoritative renderer. We pass the canvas through the SAME draw
        # functions, so the pixel output matches the CN training data.
        canvas = _np.zeros((int(height), int(width), 3), dtype=_np.uint8)
        canvas = _draw_bodypose(canvas, body_keypoints)
        canvas = _draw_handpose(canvas, left_hand)
        canvas = _draw_handpose(canvas, right_hand)

        # ── canvas → ComfyUI IMAGE tensor ───────────────────────────────
        # No channel swap: the CN was trained on the raw draw_poses() output
        # converted to a tensor the same way (see common_annotator_call:
        # torch.from_numpy(result / 255.0)). Replicating that exactly is what
        # makes the conditioning image valid.
        tensor = _torch.from_numpy(
            canvas.astype(_np.float32) / 255.0
        ).unsqueeze(0)  # (1, H, W, 3)

        # Persist PNG (audit trail — byte-identical to the CN input).
        # PIL fromarray preserves the array values verbatim (no BGR/RGB
        # reinterpretation), so the file matches what the CN receives.
        out_path = _output_path("poser_openpose", ".png")
        _PILImage.fromarray(canvas).save(out_path, format="PNG")
        with open(out_path, "rb") as _f:
            png_bytes = _f.read()

        log.info(
            "[PoserRenderOpenPose] %dx%d framing=%.2f az=%.1f el=%.1f "
            "fov=%.1f target_y=%.2f → %s (%d bytes) — 77 SOMA joints → "
            "COCO-18 via authoritative custom_controlnet_aux renderer",
            int(width), int(height), framing,
            float(azimuth), float(elevation), float(fov), float(target_y),
            out_path, len(png_bytes),
        )

        return {
            "result": (tensor,),
            "ui": {
                "images": [_ui_entry(out_path)],
            },
        }


# ════════════════════════════════════════════════════════════════════════
# Node: RayRenderGLBViews — Blender EEVEE_NEXT multi-angle GLB renderer
# ════════════════════════════════════════════════════════════════════════
# Zero-deviation ComfyUI node port of the render_glb_views pipeline family
# (media/comfyui/families/render_glb_views.py). Calls the SAME
# blender_render.render_glb() that the /poser/render_glb HTTP route uses —
# identical renderer (EEVEE_NEXT, 3-point studio lighting), identical
# output. Lets the one-flow character_vnccs workflow render the TRELLIS
# mesh at 8 angles with NO HTTP side-channel and NO LoadImage placeholder.
# The 8 views are FIXED to match render_glb_views.DEFAULT_VIEWS exactly
# (same azimuths/elevations/names) — a true 1:1 port.

_MELITE_RENDER_VIEWS = [
    {"azimuth": 0,   "elevation": 5, "name": "front_mesh"},
    {"azimuth": 45,  "elevation": 5, "name": "front_right_mesh"},
    {"azimuth": 90,  "elevation": 5, "name": "right_mesh"},
    {"azimuth": 135, "elevation": 5, "name": "back_right_mesh"},
    {"azimuth": 180, "elevation": 5, "name": "back_mesh"},
    {"azimuth": 225, "elevation": 5, "name": "back_left_mesh"},
    {"azimuth": 270, "elevation": 5, "name": "left_mesh"},
    {"azimuth": 315, "elevation": 5, "name": "front_left_mesh"},
]
_MELITE_RENDER_NAMES = tuple(v["name"] for v in _MELITE_RENDER_VIEWS)


class RayRenderGLBViews:
    """Render a GLB at 8 angles via Blender EEVEE_NEXT headless.

    ComfyUI node port of the render_glb_views family — calls the same
    ``blender_render.render_glb`` as the ``/poser/render_glb`` HTTP route.
    Produces the ``*_mesh`` reference images consumed as Picture-2 by the
    VNCCS qwen-image-edit rotation steps.

    Returns 8 IMAGE outputs (front/front_right/right/back_right/back/
    back_left/left/front_left) for direct wiring into the 8 VNCCS rotation
    edit nodes — no disk round-trip, no LoadImage placeholder, no HTTP.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_path": ("STRING", {"multiline": False,
                    "tooltip": "GLB to render (filename in input/ or output/, or absolute path).",
                    "forceInput": True,
                    }),
                "resolution": ("INT", {"default": 768, "min": 64, "max": 2048,
                    "tooltip": "Square render resolution (matches Qwen-Image-Edit input)."}),
                "samples": ("INT", {"default": 64, "min": 1, "max": 4096,
                    "tooltip": "EEVEE_NEXT TAA samples (higher = cleaner)."}),
                "framing": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 3.0,
                    "tooltip": "Camera margin multiplier (1.0=exact fit, 1.2=20% headroom)."}),
                "focal_length": ("INT", {"default": 50, "min": 10, "max": 400}),
                "transparent": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",) * len(_MELITE_RENDER_VIEWS)
    RETURN_NAMES = _MELITE_RENDER_NAMES
    FUNCTION = "render"
    CATEGORY = "TechNoir/Mesh"
    OUTPUT_NODE = True

    def render(self, glb_path: str, resolution: int = 768, samples: int = 64,
               framing: float = 1.2, focal_length: int = 50,
               transparent: bool = False):
        import folder_paths
        import numpy as np
        import torch
        from PIL import Image
        from io import BytesIO
        from .blender_render import render_glb

        def _resolve(filename):
            filename = (filename or "").strip()
            if os.path.isabs(filename) and os.path.isfile(filename):
                return filename
            for d in (folder_paths.get_input_directory, folder_paths.get_output_directory):
                p = os.path.join(d(), filename)
                if os.path.isfile(p):
                    return p
            raise FileNotFoundError(
                f"RayRenderGLBViews: glb {filename!r} not found (abs/input/output)")

        src = _resolve(glb_path)
        with open(src, "rb") as f:
            glb_bytes = f.read()

        t0 = time.perf_counter()
        renders = render_glb(
            glb_bytes,
            views=_MELITE_RENDER_VIEWS,
            resolution=resolution,
            samples=samples,
            transparent=transparent,
            focal_length=focal_length,
            framing=framing,
            width=resolution,
            height=resolution,
        )
        log.info("[RayRenderGLBViews] %.1fs → %d views",
                 time.perf_counter() - t0, len(renders))

        out_dir = folder_paths.get_output_directory()
        by_name = {r["name"]: r for r in renders}
        tensors, ui_images = [], []
        for view in _MELITE_RENDER_VIEWS:
            name = view["name"]
            r = by_name.get(name)
            if r is None:
                raise RuntimeError(f"RayRenderGLBViews: render {name!r} missing")
            png = r["png_bytes"]
            dest = os.path.join(out_dir, f"{name}_{int(time.time() * 1000)}.png")
            with open(dest, "wb") as f:
                f.write(png)
            arr = np.array(Image.open(BytesIO(png)).convert("RGB")).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).unsqueeze(0))  # (1,H,W,C)
            ui_images.append({"filename": os.path.basename(dest), "subfolder": "", "type": "output"})

        return {"result": tuple(tensors), "ui": {"images": ui_images}}


# ============================================================
# Node: RayCreatureRigBridge - bind a static textured mesh onto an
# anyCreature species skeleton (inherits its idle/move animations).
# ============================================================
# Path (a) of docs/asset-pipelines/GAME-KIT-PROGRESS.md (2026-08-25):
# TRELLIS meshes have zero animation paths (no rig; somax/kimodo are
# humanoid-only by W12 law). This node binds such a mesh onto an
# anyCreature GLB skeleton - bounds-fit + nearest-bone-segment skinning -
# producing ONE GLB with skins + species animations + original textures.
# CPU-only Blender subprocess (creature_rig_bridge.bridge_creature).

class RayCreatureRigBridge:
    """Bind a static textured mesh onto an anyCreature skeleton."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "creature_glb": ("STRING", {"multiline": False,
                    "tooltip": "anyCreature GLB (armature + idle/move): filename in input/ or output/, or absolute path.",
                    "forceInput": True,
                    }),
                "mesh_glb": ("STRING", {"multiline": False,
                    "tooltip": "Static textured mesh GLB (e.g. TRELLIS asset) to bind onto the skeleton.",
                    "forceInput": True,
                    }),
                "height_ratio": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 10.0,
                    "tooltip": "Mesh height as a fraction of the creature bind-pose height."}),
                "max_influences": ("INT", {"default": 3, "min": 1, "max": 8,
                    "tooltip": "Max bone influences per vertex (nearest segments win)."}),
                "strip_body": ("BOOLEAN", {"default": True,
                    "tooltip": "Delete the creature body meshes; export only the bound mesh."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "execute"
    CATEGORY = "melite/creature"

    @staticmethod
    def _resolve(filename):
        filename = (filename or "").strip()
        if os.path.isabs(filename) and os.path.isfile(filename):
            return filename
        for d in (folder_paths.get_input_directory, folder_paths.get_output_directory):
            p = os.path.join(d(), filename)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(
            f"RayCreatureRigBridge: glb {filename!r} not found (abs/input/output)")

    def execute(self, creature_glb, mesh_glb, height_ratio, max_influences, strip_body):
        import time
        from .creature_rig_bridge import bridge_creature
        with open(self._resolve(creature_glb), "rb") as f:
            creature_bytes = f.read()
        with open(self._resolve(mesh_glb), "rb") as f:
            mesh_bytes = f.read()
        out_dir = folder_paths.get_output_directory()
        out_path = os.path.join(out_dir, f"bridged_{int(time.time() * 1000)}.glb")
        t0 = time.perf_counter()
        stats = bridge_creature(
            creature_bytes, mesh_bytes, out_path,
            height_ratio=float(height_ratio),
            max_influences=int(max_influences),
            strip_body=bool(strip_body),
        )
        log.info("[RayCreatureRigBridge] %.1fs -> %s (%s bones)",
                 time.perf_counter() - t0, out_path, stats.get("bones"))
        return {"result": (out_path,)}


NODE_CLASS_MAPPINGS = {
    "MeshFixWinding": MeshFixWinding,
    "MeshDedup": MeshDedup,
    "CleanMesh": CleanMesh,
    "CompositeCharacter": CompositeCharacter,
# "RayProjectViewsToTexture": RayProjectViewsToTexture,  # REMOVED 2026-07-28 (upstream Comfy3D)
    # "RayXatlasUnwrap": RayXatlasUnwrap,  # REMOVED 2026-07-27 — SOMAX template
    #                                  # UVs make this redundant (35-min CPU
    #                                  # work whose output was discarded).
    "RaySAMSegmentHairClothing": RaySAMSegmentHairClothing,
    "RaySomaxBake": RaySomaxBake,
    # Poser nodes — ComfyScript path for SOMA mesh → depth/normal ControlNet.
    # See the long docstring above PoserBindPose for why these exist alongside
    # the /poser/skin/* HTTP routes (webview uses HTTP; pipelines use nodes).
    "PoserBindPose": PoserBindPose,
    "PoserBindMesh": PoserBindMesh,
    "PoserTemplateBind": PoserTemplateBind,
    "PoserDeformSoma": PoserDeformSoma,
    "PoserRender": PoserRender,
    "PoserRenderOpenPose": PoserRenderOpenPose,
    "RayRenderGLBViews": RayRenderGLBViews,
    "RayCreatureRigBridge": RayCreatureRigBridge,
    "RaySwapBaseColorTexture": RaySwapBaseColorTexture,
    "FitProp": FitProp,
    "FitRow": FitRow,
    "SkinPack": SkinPack,
    "SkinApply": SkinApply,
    "SkinSidecar": SkinSidecar,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshFixWinding": "🔧 Fix Mesh Winding (TRELLIS Cleanup)",
    "MeshDedup": "🔧 Dedup Vertices (Coincident Merge)",
    "CleanMesh": "🔧 Clean Mesh §4.42 (weld→repair→QEM decimate)",
    "CompositeCharacter": "🔧 Composite Character (body+hair+outfit)",
# "RayProjectViewsToTexture": "...",  # REMOVED 2026-07-28 (upstream Comfy3D)
    # "RayXatlasUnwrap": "🔧 Xatlas UV Unwrap (clean non-overlapping atlas)",  # REMOVED 2026-07-27
    "RaySAMSegmentHairClothing": "🎨 SAM Segment Hair + Outfit (ComfyUI-tracked)",
    "RaySomaxBake": "🔧 Somax Bake (motion NPZ → animated GLB)",
    "PoserBindPose": "🦴 Poser Bind Pose (SOMA-77 joints)",
    "PoserBindMesh": "🦴 Poser Bind Mesh (bind-pose vertices+faces)",
    "PoserTemplateBind": "🦴 Poser Template Bind (1:1 skin_standard.npz — joints+mesh)",
    "PoserDeformSoma": "🦴 Poser Deform SOMA (joints → mesh)",
    "PoserRender": "🦴 Poser Render (mesh → ControlNet image)",
    "PoserRenderOpenPose": "🦴 Poser Render OpenPose (joints → perfect skeleton, no DWPose)",
    "RayRenderGLBViews": "🎬 Render GLB Views (Blender EEVEE_NEXT ×8 angles)",
    "RayCreatureRigBridge": "🦾 Creature Rig Bridge (TRELLIS mesh -> anyCreature anims)",
    "RaySwapBaseColorTexture": "🎨 Swap BaseColor Texture (projected→rigged, preserve rig)",
    "FitProp": "Fit Prop (seat a static prop onto a SOMAX body)",
    "FitRow": "Fit Row (assemble the meld attachment row from knobs)",
    "SkinPack": "Skin Pack (deterministic detail maps, CPU)",
    "SkinApply": "Skin Apply (attach detail maps to a SOMAX body)",
    "SkinSidecar": "Skin Sidecar (controller sidecar from kind knobs)",
}
