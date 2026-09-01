"""SAM-based hair/clothing segmentation engine.

Wraps Meta's Segment-Anything ViT-B in a PipelinePatcher so ComfyUI's
``current_loaded_models`` tracks it and VRAM accounting works natively.
NO module-level ``_sam_predictor`` global — the patcher is the single
source of truth, held alive by ``melite_model_base``'s cache. ComfyUI can
evict it via ``free_memory`` when VRAM is needed for another model.

Replaces the deleted HTTP route ``/poser/segment/hair_clothing`` per the
comfyui-script-only architecture policy (2026-07-26).

Upstream audit (2026-07-26)
---------------------------
This module IS upstream: it imports Meta's official ``segment_anything``
package (``sam_model_registry["vit_b"]`` + ``SamPredictor``) and loads
Meta's official ``sam_vit_b_01ec64.pth`` weights. It is NOT a re-implementation
of SAM. The previous ``server.py:_get_sam()`` (now deleted) used the exact
same library calls.

What's custom here is the *wrapping* — and only because ComfyUI ships no
SAM ViT-B loader. Verified 2026-07-26:

  • ``comfy_extras/nodes_sam3.py`` provides ONLY ``SAM3_Detect`` /
    ``SAM3_VideoTrack`` — these expect ``model.model.diffusion_model``
    from a Wan video model (the SCAIL-2 ecosystem). Wrong tool for
    static-image hair/clothing segmentation.
  • ``grep -rln 'SAMLoader|LoadSAM|SAM_Loader' /root/ComfyUI`` → 0 hits.
  • ``find /root/ComfyUI -name '*.py' | xargs grep -l segment_anything``
    matches only this file.

The hair-vs-clothing mask selection is necessarily application-specific
(no upstream node knows what "hair" or "outfit" means). If a true
upstream ComfyUI SAM loader is later desired, install ComfyUI-Impact-Pack
or ComfyUI-Segment-Anything and replace this module with a thin
ComfyScript wrapper around their SAMLoader + SAMSegment nodes.
"""
from __future__ import annotations

import io
import logging
import os
import threading

import numpy as np

log = logging.getLogger(__name__)


# SAM checkpoint candidates — first existing wins. Order:
#   1. $SAM_VIT_B_CHECKPOINT (explicit override)
#   2. /mnt/data/models/image-gen/comfyui/sams/sam_vit_b_01ec64.pth (canonical host mount)
#   3. /root/ComfyUI/models/sams/sam_vit_b_01ec64.pth (ComfyUI model dir)
#   4. /tmp/sam_vit_b.pth (legacy path, kept for back-compat)
_SAM_CKPT_CANDIDATES = (
    os.environ.get("SAM_VIT_B_CHECKPOINT", ""),
    "/mnt/data/models/image-gen/comfyui/sams/sam_vit_b_01ec64.pth",
    "/root/ComfyUI/models/sams/sam_vit_b_01ec64.pth",
    "/tmp/sam_vit_b.pth",
)


def _sam_checkpoint_path() -> str:
    for p in _SAM_CKPT_CANDIDATES:
        if p and os.path.isfile(p):
            return p
    raise RuntimeError(
        "SAM ViT-B checkpoint not found in any of: "
        + ", ".join([p for p in _SAM_CKPT_CANDIDATES if p])
        + ". Set SAM_VIT_B_CHECKPOINT or mount the checkpoint."
    )


_PIPELINE_KEY_PREFIX = "sam:"
_pipeline_lock = threading.Lock()


class _SAMAdapter:
    """Adapter exposing the nn.Module interface PipelinePatcher needs.

    ``SamPredictor`` wraps ``sam`` (a torch nn.Module) but isn't itself an
    nn.Module — it has no ``.to()``, ``parameters()``, or ``buffers()``.
    ``PipelinePatcher.partially_load`` / ``detach`` call ``self.model.to(dev)``
    and ``_measure_size`` iterates ``vars(model).values()`` looking for
    nn.Modules. This adapter:

      • Stores the predictor (so callers still get ``set_image`` / ``predict``).
      • Forwards ``.to()`` to the underlying ``sam`` nn.Module.
      • Forwards ``parameters()`` / ``buffers()`` / ``children()`` so size
        accounting walks the actual model.

    The predictor itself owns no parameters — it's a thin Python class that
    caches image embeddings produced by ``sam``. So moving ``sam`` moves
    100% of the VRAM footprint; the predictor follows transparently.
    """

    def __init__(self, predictor):
        # vars(self) MUST contain an nn.Module for _iter_sub_modules.
        self.predictor = predictor
        self.sam = predictor.model  # the actual nn.Module

    def to(self, device):
        self.sam.to(device)
        return self

    def parameters(self, recurse=True):
        return self.sam.parameters(recurse=recurse)

    def buffers(self, recurse=True):
        return self.sam.buffers(recurse=recurse)

    def children(self):
        return self.sam.children()

    def __getattr__(self, name):
        # Forward anything else (training/eval/state_dict/etc) to sam.
        return getattr(self.sam, name)


def _get_sam_predictor():
    """Return a cached SamPredictor, wrapped so ComfyUI tracks its VRAM.

    The predictor's underlying ``sam`` model is wrapped in a
    ``PipelinePatcher`` (via the ``_SAMAdapter`` shim) and registered with
    ComfyUI's ``current_loaded_models`` via ``melite_model_base.register_pipeline``.
    ComfyUI then auto-evicts it under VRAM pressure — no manual cleanup.
    """
    try:
        from melite_model_base import (
            PipelinePatcher, register_pipeline, get_pipeline,
        )
    except ImportError:
        import os as _os
        import sys as _sys

        _vendor_dir = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)), "_vendor"
        )
        if _vendor_dir not in _sys.path:
            _sys.path.insert(0, _vendor_dir)
        try:
            from melite_model_base import (
                PipelinePatcher, register_pipeline, get_pipeline,
            )
        except ImportError as _e:
            raise RuntimeError(
                "melite_model_base unavailable: neither installed nor vendored "
                "(./_vendor/melite_model_base missing or corrupt — reinstall "
                "this node pack). Original error: " + str(_e)
            ) from _e

    ckpt = _sam_checkpoint_path()
    cache_key = f"{_PIPELINE_KEY_PREFIX}{ckpt}"

    patcher = get_pipeline(cache_key)
    if patcher is not None:
        return patcher.model.predictor  # unwrap _SAMAdapter → SamPredictor

    with _pipeline_lock:
        patcher = get_pipeline(cache_key)
        if patcher is not None:
            return patcher.model.predictor

        import torch
        from segment_anything import sam_model_registry, SamPredictor

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_b"](checkpoint=ckpt)
        sam.to(dev)
        predictor = SamPredictor(sam)

        # Wrap sam (the actual nn.Module) in _SAMAdapter so PipelinePatcher
        # gets a .to()-able object, then register so ComfyUI tracks VRAM.
        adapter = _SAMAdapter(predictor)
        patcher = PipelinePatcher(
            adapter,
            load_device=torch.device(dev),
            offload_device=torch.device("cpu"),
            name=f"sam-vit-b:{os.path.basename(ckpt)}",
        )
        patcher.pipeline_type = "sam"
        register_pipeline(cache_key, patcher)
        log.info("[sam] loaded ViT-B from %s on %s (ComfyUI-tracked)", ckpt, dev)
        return predictor


def segment_hair_clothing(image_bytes: bytes) -> tuple[bytes, bytes]:
    """Segment a character image into hair + outfit sprites via SAM.

    Args:
        image_bytes: source character image (any PIL-readable format).

    Returns:
        (hair_png_bytes, outfit_png_bytes) — both composited on white.

    Raises:
        RuntimeError: on any failure (no silent fallbacks).
    """
    import cv2
    from PIL import Image

    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_rgb = np.array(pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_rgb.shape[:2]

    # Background removal (find fg bbox for prompt points)
    corners = [
        img_bgr[2:8, 2:8].mean(axis=(0, 1)),
        img_bgr[2:8, w-8:w-2].mean(axis=(0, 1)),
        img_bgr[h-8:h-2, 2:8].mean(axis=(0, 1)),
        img_bgr[h-8:h-2, w-8:w-2].mean(axis=(0, 1)),
    ]
    bg = np.mean(corners, axis=0)
    diff = np.abs(img_bgr.astype(np.float32) - bg).sum(axis=2)
    fg = diff > 40
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    ys, xs = np.where(fg)
    if len(ys) == 0:
        raise RuntimeError(
            "segment_hair_clothing: no foreground detected — the source image "
            "may be uniform-colored or all-background."
        )
    top, bot = ys.min(), ys.max()
    cx = (xs.min() + xs.max()) // 2

    # SAM (PipelinePatcher-wrapped → ComfyUI-tracked)
    predictor = _get_sam_predictor()
    predictor.set_image(img_rgb)

    # Hair: positive point at TOPMOST foreground pixel (definitely hair),
    # negative point at face center (exclude face from hair mask)
    hair_y = top + max(2, (bot - top) * 3 // 100)
    face_y = top + (bot - top) * 28 // 100
    hm, hs, _ = predictor.predict(
        point_coords=np.array([[cx, hair_y], [cx, face_y]]),
        point_labels=np.array([1, 0]),  # 1=include hair top, 0=exclude face
        multimask_output=True,
    )
    hair_mask = hm[np.argmax(hs)] & fg

    # Clothing: point at torso (55% down)
    torso_y = top + (bot - top) * 55 // 100
    cm, cs, _ = predictor.predict(
        point_coords=np.array([[cx, torso_y]]), point_labels=np.array([1]),
        multimask_output=True,
    )
    cloth_mask = cm[np.argmax(cs)] & fg

    log.info("[sam] hair=%dpx cloth=%dpx", int(hair_mask.sum()), int(cloth_mask.sum()))

    # Composite on white
    def _comp(mask):
        r = np.full_like(img_rgb, 255)
        r[mask] = img_rgb[mask]
        return Image.fromarray(r)

    hair_buf, cloth_buf = io.BytesIO(), io.BytesIO()
    _comp(hair_mask).save(hair_buf, format="PNG")
    _comp(cloth_mask).save(cloth_buf, format="PNG")
    return hair_buf.getvalue(), cloth_buf.getvalue()
