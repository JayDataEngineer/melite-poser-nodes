"""SkinPack + SkinApply nodes — deterministic detail maps onto a SOMAX body (skin-card stroke 2).

Thin IO wrappers (the pack pattern): SkinPack is pure CPU (size, seed ->
albedo/normal IMAGE tensors, the skin_surgery twin of tools/skin_pack.py);
SkinApply eats the body path + map tensors and writes the skinned GLB
(bytes surgery, numpy/PIL only). Kinds/tints never enter here — the
sidecar rides a SaveText node fed by the card compose (estate SSOT).
"""

from __future__ import annotations

import logging
import os

from .skin_surgery import apply_skin_bytes as _apply_bytes, build_sidecar as _build_sidecar, encode_png as _encode_png, generate_detail_maps as _generate

log = logging.getLogger(__name__)


class SkinPack:
    """Deterministic skin detail maps (CPU — claims no GPU, runs anywhere).

    Inputs: size (map edge px — the closed 256/512/1024/2048 vocabulary,
    refused estate-side by the card schema AND here), seed (non-negative).
    Outputs: albedo IMAGE + normal IMAGE (float32 B,H,W,C) for SaveImage
    harvesting and SkinApply.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 1}),
                "seed": ("INT", {"default": 20260906, "min": 0, "max": 2**31 - 1, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo", "normal")
    FUNCTION = "pack"
    CATEGORY = "TechNoir/Mesh"

    def pack(self, size: int, seed: int):
        import torch
        if int(size) not in (256, 512, 1024, 2048):
            raise RuntimeError("SkinPack refused: size %r not in [256, 512, 1024, 2048]" % (size,))
        if int(seed) < 0:
            raise RuntimeError("SkinPack refused: seed %r is negative — the -1 RANDOM sentinel is minted compose-side (SkinParams admits it, the skin compose mints it); a raw -1 here means the flow bypassed compose" % (seed,))
        albedo, normal = _generate(int(size), int(seed))
        to_t = lambda a: torch.from_numpy(a.astype("float32") / 255.0).unsqueeze(0)
        log.info("[SkinPack] size=%d seed=%d", int(size), int(seed))
        return (to_t(albedo), to_t(normal))


class SkinApply:
    """Attach detail maps to a SOMAX body GLB (bytes surgery, no rig touched).

    Inputs: body_glb_path (forceInput STRING — absolute mount path from
    the card door) + albedo/normal IMAGEs (SkinPack). Output: glb_path
    (the skinned GLB, three_model-surfaced) + skin_record JSON string.
    Refuses non-GLB bytes and UV-less bodies (ValueError -> RuntimeError).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "body_glb_path": ("STRING", {"multiline": False, "forceInput": True}),
                "albedo": ("IMAGE", {}),
                "normal": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("glb_path", "skin_record")
    FUNCTION = "apply"
    CATEGORY = "TechNoir/Mesh"
    OUTPUT_NODE = True

    def apply(self, body_glb_path: str, albedo, normal):
        import numpy as np
        from .nodes import _input_path, _output_path, _ui_entry
        body_src = _input_path(body_glb_path)
        if not body_src or not os.path.isfile(body_src):
            raise RuntimeError("SkinApply: body not found in ComfyUI input/ (%s)." % (body_glb_path,))
        with open(body_src, "rb") as f:
            body_bytes = f.read()
        to_png = lambda t: _encode_png((np.clip(t[0].detach().cpu().numpy(), 0, 1) * 255).astype("uint8"))
        try:
            out_bytes, record = _apply_bytes(body_bytes, to_png(albedo), to_png(normal))
        except ValueError as e:
            raise RuntimeError("SkinApply refused: %s" % e)
        out_path = _output_path("skin_body")
        with open(out_path, "wb") as f:
            f.write(out_bytes)
        import json as _json
        log.info("[SkinApply] output=%s (%d bytes) materials=%s", out_path, len(out_bytes), record.get("materials_touched"))
        return {
            "result": (out_path, _json.dumps(record, sort_keys=True)),
            "ui": {"three_model": [_ui_entry(out_path)]},
        }

class SkinSidecar:
    """The controller sidecar from knob initials (the FitRow precedent).

    Inputs: kind (combo — the 20 meld kinds, a typo refuses at prompt
    validation) + size + seed. Output: sidecar STRING (twin of
    skin_spec.sidecar — the parity wall pins all 20 rows). Pure CPU.
    """

    @classmethod
    def INPUT_TYPES(cls):
        from .skin_surgery import KIND_TINTS as _KINDS
        return {
            "required": {
                "kind": (sorted(_KINDS.keys()), {"default": "keeper"}),
                "size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 1}),
                "seed": ("INT", {"default": 20260906, "min": 0, "max": 2**31 - 1, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("sidecar",)
    FUNCTION = "sidecar"
    CATEGORY = "TechNoir/Mesh"

    def sidecar(self, kind: str, size: int, seed: int):
        try:
            return (_build_sidecar(str(kind), int(size), int(seed)),)
        except (KeyError, ValueError) as e:
            raise RuntimeError("SkinSidecar refused: %s" % e)
