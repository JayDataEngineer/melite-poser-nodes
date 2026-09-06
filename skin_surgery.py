"""Skin surgery — deterministic detail maps + GLB texture attach (skin-card stroke 2).

Date: 2026-09-06. The fit_surgery twin pattern: this module owns the
BYTES (numpy/PIL only — no trimesh, no torch at import); skin_nodes.py
owns the ComfyUI IO; the estate twin is tools/skin_pack.py (the
parity wall pins generate_detail_maps byte-identical).

Kinds/tints live estate-side (py/skin_spec.py KIND_TABLE) — the sidecar
rides INTO SkinApply as a string the card compose assembles. This module
never names a kind: maps are pure (size, seed).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .fit_surgery import build_glb as _build_glb, parse_glb as _parse_glb


def _periodic_value_noise(size: int, cells: int, rng) -> np.ndarray:
    gx = rng.random((cells, cells))
    xs = np.linspace(0, cells, size, endpoint=False)
    fx = xs - xs.astype(int) if cells > 1 else xs * 0
    xi = xs.astype(int) % cells
    xw = fx * fx * (3 - 2 * fx)
    i0 = xi
    i1 = (xi + 1) % cells
    ys = np.linspace(0, cells, size, endpoint=False)
    yi = ys.astype(int) % cells
    yw = ys - ys.astype(int)
    yw = yw * yw * (3 - 2 * yw)
    j0 = yi
    j1 = (yi + 1) % cells
    v00 = gx[np.ix_(j0, i0)]
    v01 = gx[np.ix_(j0, i1)]
    v10 = gx[np.ix_(j1, i0)]
    v11 = gx[np.ix_(j1, i1)]
    wx = xw[None, :]
    wy = yw[:, None]
    top = v00 * (1 - wx) + v01 * wx
    bot = v10 * (1 - wx) + v11 * wx
    return top * (1 - wy) + bot * wy


def _fbm(size: int, base_cells: int, octaves: int, rng, persistence: float = 0.5) -> np.ndarray:
    acc = np.zeros((size, size))
    amp = 1.0
    total = 0.0
    for o in range(octaves):
        cells = base_cells * (2 ** o)
        if cells > size:
            break
        acc += amp * _periodic_value_noise(size, cells, rng)
        total += amp
        amp *= persistence
    return acc / total


def generate_detail_maps(size: int, seed: int) -> tuple:
    if size not in (256, 512, 1024, 2048):
        raise ValueError("size %r not in [256, 512, 1024, 2048]" % (size,))
    rng = np.random.default_rng(seed)
    pores = _fbm(size, 96, 2, rng)
    undulation = _fbm(size, 12, 3, rng)
    fine_lines = _fbm(size, 48, 1, rng)
    height = 0.45 * pores + 0.35 * undulation + 0.20 * fine_lines
    base = np.array([1.00, 0.930, 0.865])
    shade = 0.92 + 0.16 * height
    albedo = base[None, None, :] * shade[:, :, None]
    albedo = np.clip(albedo, 0, 1)
    gy = np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)
    gx = np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)
    strength = 2.2
    nx = -gx * strength * size / 96.0
    ny = -gy * strength * size / 96.0
    nz = np.ones_like(height)
    nlen = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx /= nlen
    ny /= nlen
    nz /= nlen
    normal = np.stack([nx, ny, nz], axis=-1) * 0.5 + 0.5
    return (albedo * 255).astype(np.uint8), (normal * 255).astype(np.uint8)


def encode_png(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def apply_skin_bytes(body: bytes, albedo_png: bytes, normal_png: bytes) -> tuple:
    gltf, bindata = _parse_glb(body)
    prims = [p for m in gltf.get("meshes", []) for p in m.get("primitives", [])]
    if not prims:
        raise ValueError("body has no mesh primitives")
    bald = [i for i, p in enumerate(prims) if "TEXCOORD_0" not in (p.get("attributes") or {})]
    if bald:
        raise ValueError("primitives %s have no TEXCOORD_0 — detail maps have nowhere to land (non-somax UVs)" % bald)
    data = bytearray(bindata)
    bvs = gltf.setdefault("bufferViews", [])
    imgs = gltf.setdefault("images", [])
    samplers = gltf.setdefault("samplers", [])
    textures = gltf.setdefault("textures", [])

    def _append_blob(blob: bytes) -> int:
        off = len(data)
        data.extend(blob)
        while len(data) % 4:
            data.append(0)
        bvs.append({"buffer": 0, "byteOffset": off, "byteLength": len(blob)})
        return len(bvs) - 1

    sampler = {"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497, "name": "skin_detail_repeat"}
    samplers.append(sampler)
    smp = len(samplers) - 1
    imgs.append({"bufferView": _append_blob(albedo_png), "mimeType": "image/png", "name": "skin_detail_albedo"})
    alb_img = len(imgs) - 1
    imgs.append({"bufferView": _append_blob(normal_png), "mimeType": "image/png", "name": "skin_detail_normal"})
    nrm_img = len(imgs) - 1
    textures.append({"sampler": smp, "source": alb_img, "name": "skin_albedo"})
    alb_tex = len(textures) - 1
    textures.append({"sampler": smp, "source": nrm_img, "name": "skin_normal"})
    nrm_tex = len(textures) - 1
    gltf["buffers"] = [{"byteLength": len(data)}]
    mats = gltf.setdefault("materials", [])
    if not mats:
        mats.append({"name": "skin", "pbrMetallicRoughness": {}})
    touched = 0
    for m in mats:
        pbr = m.setdefault("pbrMetallicRoughness", {})
        pbr["baseColorTexture"] = {"index": alb_tex, "texCoord": 0}
        m["normalTexture"] = {"index": nrm_tex, "texCoord": 0, "scale": 1.0}
        pbr.setdefault("metallicFactor", 0.0)
        pbr.setdefault("roughnessFactor", 0.65)
        touched += 1
    out = _build_glb(gltf, bytes(data))
    record = {"maps": {"albedo": [len(albedo_png)], "normal": [len(normal_png)]}, "materials_touched": touched, "primitives": len(prims), "sampler": "repeat"}
    return out, record



# The kind table TWIN (estate SSOT: py/skin_spec.py KIND_TABLE).
# SkinSidecar assembles the controller sidecar in-pack (the FitRow
# precedent: knob initials -> contract JSON at run time); the parity
# wall pins every row equal. kind -> (tint, texture, category).
KIND_TINTS = {
    'keeper': ((0.85, 0.7, 0.55), 'young_caucasian_male', 'humanoid'),
    'ash_nomad': ((0.65, 0.5, 0.4), 'middleage_caucasian_male', 'humanoid'),
    'villager_elder': ((0.75, 0.6, 0.5), 'old_caucasian_male', 'humanoid'),
    'villager': ((0.8, 0.65, 0.5), 'young_caucasian_female', 'humanoid'),
    'caravan_trader': ((0.78, 0.62, 0.48), 'middleage_caucasian_female', 'humanoid'),
    'amf_warden': ((0.72, 0.58, 0.46), 'middleage_caucasian_male', 'humanoid'),
    'cherusci_scrivener': ((0.82, 0.68, 0.54), 'young_caucasian_male', 'humanoid'),
    'marrow_raider': ((0.6, 0.48, 0.38), 'old_caucasian_male', 'humanoid'),
    'amalgam_horror': ((0.45, 0.35, 0.3), 'flat_placeholder', 'creature'),
    'arena_factor': ((0.7, 0.55, 0.45), 'flat_placeholder', 'creature'),
    'bone_crane': ((0.85, 0.82, 0.78), 'flat_placeholder', 'creature'),
    'dune_hound': ((0.8, 0.7, 0.5), 'flat_placeholder', 'creature'),
    'emberwing': ((0.85, 0.45, 0.3), 'flat_placeholder', 'creature'),
    'gloom_tortoise': ((0.5, 0.55, 0.45), 'flat_placeholder', 'creature'),
    'husk_stalker': ((0.55, 0.45, 0.35), 'flat_placeholder', 'creature'),
    'ironvein_construct': ((0.5, 0.5, 0.52), 'flat_placeholder', 'creature'),
    'meld_creature': ((0.6, 0.5, 0.42), 'flat_placeholder', 'creature'),
    'moss_elk': ((0.55, 0.6, 0.45), 'flat_placeholder', 'creature'),
    'ridge_lizard': ((0.65, 0.58, 0.45), 'flat_placeholder', 'creature'),
    'sand_skimmer': ((0.75, 0.68, 0.55), 'flat_placeholder', 'creature'),
}


def build_sidecar(kind, size, seed):
    import json as _json
    if size not in (256, 512, 1024, 2048):
        raise ValueError("size %r not in [256, 512, 1024, 2048]" % (size,))
    try:
        tint, texture, category = KIND_TINTS[kind]
    except KeyError:
        raise KeyError("unknown skin kind %r" % (kind,))
    return _json.dumps({
        'kind': kind,
        'tint': [tint[0], tint[1], tint[2]],
        'texture': texture,
        'category': category,
        'uv': 'somax',
        'size': size,
        'seed': seed,
    }, sort_keys=True)
