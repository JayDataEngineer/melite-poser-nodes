"""FitProp + FitRow nodes — seat a static prop onto a SOMAX body inside a queue run.

Thin IO wrappers (the pack pattern): files in from ComfyUI input/, bytes
through fit_surgery.fit_prop_bytes, fitted GLB out to ComfyUI output/ with
a ui.three_model entry. The math lives in fit_surgery.py (the estate twin
is tools/fit_prop.py); these classes own NO geometry.
"""

from __future__ import annotations

import json
import logging
import os
import time

from .fit_surgery import build_row_json as _build_row, fit_prop_bytes as _fit_bytes

log = logging.getLogger(__name__)


class FitProp:
    """Seat a static prop (hat / sword / coat) onto a SOMAX body GLB.

    Inputs (all forceInput STRINGs — absolute mount paths resolved at
    the card validation door, the CompositeCharacter pattern):
      body_glb_path — the SOMAX body (.glb, mixamorig bones)
      prop_glb_path — the static prop (.glb, unskinned, unanimated)
      row_json      — the meld attachment row VERBATIM (Model, Bone,
                      Align, FitRatio, FitRefBone(s), SeatRatio,
                      FingerBone, ThumbBone, TargetWidth, ...)

    Outputs: glb_path (the fitted GLB) + fit_record (the measurement
    record as JSON — seat math the verify loop audits).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "body_glb_path": ("STRING", {"multiline": False, "forceInput": True}),
                "prop_glb_path": ("STRING", {"multiline": False, "forceInput": True}),
                "row_json": ("STRING", {"multiline": False, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("glb_path", "fit_record")
    FUNCTION = "fit"
    CATEGORY = "TechNoir/Mesh"
    OUTPUT_NODE = True

    def fit(self, body_glb_path: str, prop_glb_path: str, row_json: str):
        from .nodes import _input_path, _output_path, _ui_entry
        body_src = _input_path(body_glb_path)
        if not body_src or not os.path.isfile(body_src):
            raise RuntimeError(
                'FitProp: body not found in ComfyUI input/ (%s).' % body_glb_path)
        prop_src = _input_path(prop_glb_path)
        if not prop_src or not os.path.isfile(prop_src):
            raise RuntimeError(
                'FitProp: prop not found in ComfyUI input/ (%s).' % prop_glb_path)
        try:
            row = json.loads(row_json)
        except ValueError as e:
            raise RuntimeError("FitProp: row_json is not JSON: %s" % e)
        with open(body_src, "rb") as f:
            body_bytes = f.read()
        with open(prop_src, "rb") as f:
            prop_bytes = f.read()
        log.info(
            "[FitProp] body=%s (%d bytes) prop=%s (%d bytes)",
            body_src, len(body_bytes), prop_src, len(prop_bytes))
        t0 = time.perf_counter()
        try:
            out_bytes, record = _fit_bytes(body_bytes, prop_bytes, row)
        except ValueError as e:
            raise RuntimeError("FitProp refused: %s" % e)
        out_path = _output_path("fit_prop")
        with open(out_path, "wb") as f:
            f.write(out_bytes)
        log.info(
            "[FitProp] output=%s (%d bytes) in %.2fs measured=%s",
            out_path, len(out_bytes), time.perf_counter() - t0, record.get("measured"))
        return {
            "result": (out_path, json.dumps(record, sort_keys=True)),
            "ui": {"three_model": [_ui_entry(out_path)]},
        }


class FitRow:
    """Assemble the meld attachment row from knob initials (verb/flow door).

    Static flow templates cannot build the row_json string FitProp eats,
    and hardcoded rows are banned — so the knobs (the verb-knob == flow-
    initial handshake names) assemble HERE through fit_surgery.build_row_json
    (the _Row gate refuses conflicts at prompt time). The fit card renderer
    wires this node too: ONE runtime derivation (rung 6), never a baked
    row beside it. Pure function — no side effects, not an output node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"multiline": False, "default": "prop"}),
                "bone": ("STRING", {"multiline": False, "default": ""}),
                "align": (["socket", "fit", "grip"], {"default": "socket"}),
                "position": ("STRING", {"multiline": False, "default": "0,0,0"}),
                "rotation": ("STRING", {"multiline": False, "default": "0,0,0"}),
                "fit_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
                "fit_ref_bone": ("STRING", {"multiline": False, "default": ""}),
                "fit_ref_bone2": ("STRING", {"multiline": False, "default": ""}),
                "seat_ratio": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01}),
                "finger_bone": ("STRING", {"multiline": False, "default": ""}),
                "thumb_bone": ("STRING", {"multiline": False, "default": ""}),
                "target_width": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("row_json",)
    FUNCTION = "build"
    CATEGORY = "TechNoir/Mesh"

    def build(self, **knobs):
        try:
            return (_build_row(dict(knobs)),)
        except ValueError as e:
            raise RuntimeError("FitRow refused: %s" % e)
