"""Self-contained GLB baseColorTexture swap — preserves skin/joints/weights.

ComfyUI-node port of ``media.comfyui.glb_surgical``. Comfy3D's
``[Comfy3D] Save 3D Mesh`` does a trimesh round-trip that drops the ENTIRE
rig (skins, joints, WEIGHTS_0/JOINTS_0 accessors, node hierarchy, animation
samplers). This module extracts the projected texture from the derigged
Comfy3D output and swaps it into the ORIGINAL rigged GLB's baseColorTexture
slot — preserving every other byte.

NO COMFYUI DEPENDENCIES. Only stdlib (json, struct, typing). Imported by
the ``RaySwapBaseColorTexture`` node (custom_nodes/melite-poser-nodes/nodes.py).
"""
from __future__ import annotations

import json
import struct
from typing import Any

__all__ = ["extract_base_color_texture", "swap_base_color_texture"]


def _pad4(b: bytes, pad: bytes = b"\x00") -> bytes:
    """Pad bytes to 4-byte alignment (GLB chunk alignment requirement)."""
    return b + pad * ((4 - len(b) % 4) % 4)


def _parse_glb_chunks(glb_bytes: bytes) -> tuple[dict[str, Any], bytearray]:
    """Parse a GLB into (json_dict, bin_bytearray). Hard-errors if malformed."""
    if glb_bytes[:4] != b"glTF":
        raise ValueError(
            f"swap_base_color_texture: not a GLB (bad magic {glb_bytes[:4]!r}). "
            f"Expected b'glTF'."
        )
    version, total_len = struct.unpack_from("<II", glb_bytes, 4)
    if version != 2:
        raise ValueError(
            f"swap_base_color_texture: unsupported GLB version {version}.")
    off = 12
    j_len = int.from_bytes(glb_bytes[off:off + 4], "little")
    js = json.loads(glb_bytes[off + 8:off + 8 + j_len].decode("utf-8"))
    off += 8 + j_len
    b_len = int.from_bytes(glb_bytes[off:off + 4], "little")
    bin0 = bytearray(glb_bytes[off + 8:off + 8 + b_len])
    return js, bin0


def _rebuild_glb(js: dict[str, Any], bin_bytes: bytearray) -> bytes:
    """Rebuild a GLB from (json_dict, bin_bytes) with proper chunk alignment."""
    js_bytes = _pad4(json.dumps(js).encode("utf-8"), pad=b" ")
    bin_padded = _pad4(bytes(bin_bytes))
    total = 12 + 8 + len(js_bytes) + 8 + len(bin_padded)
    out = bytearray()
    out += b"glTF" + (2).to_bytes(4, "little") + total.to_bytes(4, "little")
    out += len(js_bytes).to_bytes(4, "little") + b"JSON" + js_bytes
    out += len(bin_padded).to_bytes(4, "little") + b"BIN\x00" + bin_padded
    return bytes(out)


def _resolve_base_color_image(js: dict[str, Any]) -> tuple[int, int]:
    """Find (texture_index, image_index) for materials[0].baseColorTexture.

    Handles KHR_texture_basisu (KTX2) and EXT_texture_webp extensions.
    """
    if not js.get("materials"):
        raise ValueError(
            "swap_base_color_texture: GLB has no materials[]. Cannot swap "
            "baseColorTexture on a material-less mesh.")
    mat = js["materials"][0]
    pbr = mat.get("pbrMetallicRoughness", {})
    if "baseColorTexture" not in pbr:
        raise ValueError(
            "swap_base_color_texture: materials[0] has no "
            "pbrMetallicRoughness.baseColorTexture.")
    bct_idx = pbr["baseColorTexture"]["index"]
    tex = js["textures"][bct_idx]
    img_idx = tex.get("source")
    if img_idx is None:
        for ext in ("KHR_texture_basisu", "EXT_texture_webp"):
            ext_src = tex.get("extensions", {}).get(ext, {}).get("source")
            if ext_src is not None:
                img_idx = ext_src
                break
    if img_idx is None:
        raise ValueError(
            "swap_base_color_texture: cannot resolve baseColorTexture image "
            "source (no tex.source and no KHR_texture_basisu/EXT_texture_webp "
            "extension source). The GLB is malformed.")
    return bct_idx, img_idx


def extract_base_color_texture(glb_bytes: bytes) -> tuple[bytes, str]:
    """Extract the baseColorTexture image bytes + mime type from a GLB.

    Used after upstream projection: pull the projected atlas PNG out of the
    saved (derigged) GLB so we can swap it into the original rigged GLB.

    Returns:
        (image_bytes, mime_type) — mime is "image/png", "image/jpeg", etc.
    """
    js, bin0 = _parse_glb_chunks(glb_bytes)
    _, img_idx = _resolve_base_color_image(js)
    img = js["images"][img_idx]
    if "bufferView" not in img:
        raise ValueError(
            "extract_base_color_texture: baseColorTexture image has no "
            "bufferView (not embedded). Only embedded GLB textures supported.")
    bv = js["bufferViews"][img["bufferView"]]
    byte_off = bv.get("byteOffset", 0)
    img_bytes = bytes(bin0[byte_off:byte_off + bv["byteLength"]])
    if len(img_bytes) != bv["byteLength"]:
        raise ValueError(
            f"extract_base_color_texture: bufferView byteLength "
            f"{bv['byteLength']} exceeds BIN chunk bounds (offset {byte_off}, "
            f"BIN len {len(bin0)}). GLB is corrupt.")
    mime = img.get("mimeType", "image/png")
    return img_bytes, mime


def swap_base_color_texture(
    glb_bytes: bytes,
    new_image_bytes: bytes,
    mime: str = "image/png",
) -> bytes:
    """Replace materials[0].baseColorTexture image, preserving everything else.

    Appends the new image bytes to the BIN buffer and repoints the image's
    bufferView at it (orphaning the old image bytes — no offset shifting of
    other bufferViews, which would break the skin/animation accessors that we
    MUST preserve).

    Strips KHR_texture_basisu / EXT_texture_webp extensions on the texture.

    Args:
        glb_bytes: the ORIGINAL GLB (with rigging) — its baseColorTexture
            will be replaced.
        new_image_bytes: PNG/JPEG/etc bytes of the new texture.
        mime: mime type matching new_image_bytes. Default "image/png".

    Returns:
        New GLB bytes with the swapped texture. Every byte outside
        baseColorTexture's image payload is preserved (mesh, UVs, skin,
        joints, weights, nodes, animations, samplers).
    """
    js, bin0 = _parse_glb_chunks(glb_bytes)

    has_existing = (
        js.get("textures")
        and js.get("materials")
        and "baseColorTexture" in js["materials"][0].get("pbrMetallicRoughness", {})
    )

    if has_existing:
        bct_idx, img_idx = _resolve_base_color_image(js)
        tex = js["textures"][bct_idx]
        img = js["images"][img_idx]
        if "bufferView" not in img:
            raise ValueError(
                "swap_base_color_texture: original GLB's baseColorTexture "
                "image has no bufferView (not embedded). Cannot swap.")
        bv = js["bufferViews"][img["bufferView"]]
    else:
        # Create texture infrastructure from scratch.
        new_off = len(bin0)
        bin0 += _pad4(new_image_bytes)
        js.setdefault("bufferViews", [])
        new_bv = {"buffer": 0, "byteOffset": new_off,
                  "byteLength": len(new_image_bytes)}
        js["bufferViews"].append(new_bv)
        new_bv_idx = len(js["bufferViews"]) - 1
        js.setdefault("images", [])
        js["images"].append({"bufferView": new_bv_idx, "mimeType": mime})
        new_img_idx = len(js["images"]) - 1
        js.setdefault("textures", [])
        js["textures"].append({"source": new_img_idx})
        new_tex_idx = len(js["textures"]) - 1
        if not js.get("materials"):
            js["materials"] = [{}]
        js["materials"][0].setdefault("pbrMetallicRoughness", {})
        js["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {
            "index": new_tex_idx}
        js.setdefault("buffers", [{"byteLength": len(bin0)}])
        js["buffers"][0]["byteLength"] = len(bin0)
        return _rebuild_glb(js, bin0)

    # Append new image bytes to BIN, repoint the bufferView.
    new_off = len(bin0)
    bin0 += _pad4(new_image_bytes)
    bv["byteOffset"] = new_off
    bv["byteLength"] = len(new_image_bytes)
    tex.pop("extensions", None)
    tex["source"] = img_idx
    img["mimeType"] = mime
    js["buffers"][0]["byteLength"] = len(bin0)
    return _rebuild_glb(js, bin0)
