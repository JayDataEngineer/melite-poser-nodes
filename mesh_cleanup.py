"""Mesh cleanup operations — pure bytes-in/bytes-out, no file I/O.

These were previously in-process in melite-head's ``scripts/character_generation.py`` (now ``core/pipelines/character.py``)
which violated the architecture (melite-head = HTTP orchestrator only; ALL mesh
work happens in ComfyUI). They were moved here so melite-head calls them via
ComfyScript-submitted workflows that hit the ``MeshFixWinding`` and
``MeshDedup`` ComfyUI nodes registered in ``nodes.py``.

Each function takes raw GLB bytes and returns raw GLB bytes — no file I/O.
The nodes handle file upload (input/) and download (output/); melite-head
orchestrates via ``ComfyUIClient.upload_file`` + ``submit_comfy_workflow``.
"""
from __future__ import annotations

import io
import json
import logging
import struct

import numpy as np

log = logging.getLogger(__name__)

# glTF componentType → numpy dtype
_DTYPE_MAP = {5126: 'f4', 5123: 'u2', 5121: 'u1', 5122: 'i2', 5125: 'u4'}
_NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def _parse_glb(raw: bytes) -> tuple[dict, bytearray]:
    """Split GLB into (gltf-json, bin-chunk). Returns (json_dict, bin_bytearray)."""
    magic, version, total_len = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67:
        raise ValueError(f"not a GLB (magic=0x{magic:08x})")
    json_len, json_type = struct.unpack_from("<II", raw, 12)
    if json_type != 0x4E4F534A:  # 'JSON'
        raise ValueError(f"chunk 0 not JSON (type=0x{json_type:08x})")
    gltf = json.loads(raw[20:20 + json_len].decode())
    off = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<II", raw, off)
    if bin_type != 0x004E4942:  # 'BIN\0'
        raise ValueError(f"chunk 1 not BIN (type=0x{bin_type:08x})")
    bindata = bytearray(raw[off + 8:off + 8 + bin_len])
    return gltf, bindata


def _build_glb(gltf: dict, bindata: bytearray) -> bytes:
    """Rebuild a GLB from (gltf-json, bin-chunk). Pads to 4-byte alignment."""
    json_str = json.dumps(gltf, separators=(',', ':'))
    while len(json_str) % 4 != 0:
        json_str += ' '
    json_bytes = json_str.encode('utf-8')
    while len(bindata) % 4 != 0:
        bindata += b'\x00'
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bindata)
    out = bytearray()
    out += struct.pack('<III', 0x46546C67, 2, total_len)
    out += struct.pack('<II', len(json_bytes), 0x4E4F534A)
    out += json_bytes
    out += struct.pack('<II', len(bindata), 0x004E4942)
    out += bytes(bindata)
    return bytes(out)


def _read_acc(bindata: bytearray, gltf: dict, acc_idx: int) -> np.ndarray:
    acc = gltf['accessors'][acc_idx]
    bv = gltf['bufferViews'][acc['bufferView']]
    offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    fmt = _DTYPE_MAP[acc['componentType']]
    ncomp = _NCOMP[acc['type']]
    arr = np.frombuffer(bytes(bindata), dtype=np.dtype(fmt),
                        count=acc['count'] * ncomp, offset=offset)
    return arr.reshape(acc['count'], ncomp) if ncomp > 1 else arr


def _convert_glb_textures_to_png(glb_bytes: bytes) -> bytes:
    """Replace GLB textures (WebP/KTX2) with valid PNG via manual GLB rebuild.

    Loads via trimesh (reads WebP), re-encodes textures via PIL (valid PNG),
    and builds the GLB from scratch. Same logic as the old in-process version,
    now operating on bytes (no file I/O).
    """
    import trimesh
    from PIL import Image

    mesh = trimesh.load(io.BytesIO(glb_bytes), force='mesh', process=False, file_type='glb')
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    uvs = None
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uvs = np.asarray(mesh.visual.uv, dtype=np.float32)
    mat = getattr(mesh.visual, 'material', None)

    tex_pngs = []  # list of (png_bytes, is_basecolor)
    for attr, is_bc in [('baseColorTexture', True), ('metallicRoughnessTexture', False)]:
        tex = getattr(mat, attr, None) if mat else None
        if tex is not None:
            arr = np.asarray(tex)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format='PNG')
            tex_pngs.append((buf.getvalue(), is_bc))

    bin_data = bytearray()

    def _add(data):
        while len(bin_data) % 4:
            bin_data.append(0)
        off = len(bin_data)
        bin_data.extend(data)
        return off, len(data)

    pos_o, pos_l = _add(verts.tobytes())
    nrm_o, nrm_l = _add(normals.tobytes())
    uv_o, uv_l = _add(uvs.tobytes()) if uvs is not None else (0, 0)
    idx_o, idx_l = _add(faces.tobytes())
    tex_offs = [_add(png) for png, _ in tex_pngs]

    V, F = len(verts), len(faces)
    bvs = [
        {"buffer": 0, "byteOffset": pos_o, "byteLength": pos_l},
        {"buffer": 0, "byteOffset": nrm_o, "byteLength": nrm_l},
    ]
    acc = [
        {"bufferView": 0, "componentType": 5126, "count": V, "type": "VEC3"},
        {"bufferView": 1, "componentType": 5126, "count": V, "type": "VEC3"},
    ]
    nbv = 2
    if uvs is not None:
        bvs.append({"buffer": 0, "byteOffset": uv_o, "byteLength": uv_l})
        acc.append({"bufferView": nbv, "componentType": 5126, "count": V, "type": "VEC2"})
        nbv += 1
    bvs.append({"buffer": 0, "byteOffset": idx_o, "byteLength": idx_l})
    idx_acc = nbv
    acc.append({"bufferView": nbv, "componentType": 5125, "count": F * 3, "type": "SCALAR"})
    nbv += 1
    images = []
    for i, (to, tl) in enumerate(tex_offs):
        bvs.append({"buffer": 0, "byteOffset": to, "byteLength": tl})
        images.append({"bufferView": nbv + i, "mimeType": "image/png"})
    textures_g = [{"source": i} for i in range(len(tex_pngs))]
    mat_g = {"pbrMetallicRoughness": {}}
    for i, (_, is_bc) in enumerate(tex_pngs):
        if is_bc:
            mat_g["pbrMetallicRoughness"]["baseColorTexture"] = {"index": i}
        else:
            mat_g["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": i}
    prim_attrs = {"POSITION": 0, "NORMAL": 1}
    if uvs is not None:
        prim_attrs["TEXCOORD_0"] = 2
    prim = {"attributes": prim_attrs, "indices": idx_acc, "material": 0}

    while len(bin_data) % 4:
        bin_data.append(0)
    gltf = {
        "asset": {"version": "2.0", "generator": "melite-pipeline"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [prim]}], "materials": [mat_g],
        "accessors": acc, "bufferViews": bvs, "images": images,
        "textures": textures_g, "buffers": [{"byteLength": len(bin_data)}],
    }
    out = _build_glb(gltf, bytearray(bin_data))
    log.info("[mesh_cleanup] convert_glb_textures_to_png: %d textures, %d bytes",
             len(tex_pngs), len(out))
    return out


def fix_winding(glb_bytes: bytes, preserve_rig: bool = False) -> bytes:
    """Fix inconsistent face winding (inward normals) in a GLB.

    For each face, if its geometric normal points toward the mesh centroid
    (inward), swap v1<->v2 to flip it outward. Optionally converts WebP
    textures to PNG (skipped when ``preserve_rig`` is True to keep skin/anim).

    Returns: new GLB bytes (always a fresh copy).
    """
    gltf, bindata = _parse_glb(glb_bytes)
    prim = gltf['meshes'][0]['primitives'][0]

    if 'POSITION' not in prim.get('attributes', {}):
        log.warning("[mesh_cleanup] fix_winding: no POSITION attribute — returning input")
        return glb_bytes
    verts = _read_acc(bindata, gltf, prim['attributes']['POSITION']).astype(np.float64)

    if 'indices' not in prim:
        log.info("[mesh_cleanup] fix_winding: non-indexed mesh — no-op")
        return glb_bytes

    faces = _read_acc(bindata, gltf, prim['indices']).astype(np.int64)
    if faces.ndim == 1:
        faces = faces.reshape(-1, 3)

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    fn = cross / (norms + 1e-12)

    centroid = verts.mean(axis=0)
    face_centers = (v0 + v1 + v2) / 3.0
    outward = face_centers - centroid
    outward_norm = np.linalg.norm(outward, axis=1, keepdims=True)
    outward = outward / (outward_norm + 1e-12)

    dots = np.sum(fn * outward, axis=1)
    flipped_mask = dots < 0
    n_flipped = int(flipped_mask.sum())
    log.info("[mesh_cleanup] fix_winding: %d/%d faces (%.1f%%) had inward normals",
             n_flipped, len(faces), n_flipped / len(faces) * 100)

    if n_flipped == 0:
        # Still do convert_to_png so textures are valid for downstream stages.
        return _convert_glb_textures_to_png(glb_bytes) if not preserve_rig else glb_bytes

    fixed_faces = faces.copy()
    fixed_faces[flipped_mask, 1], fixed_faces[flipped_mask, 2] = \
        faces[flipped_mask, 2].copy(), faces[flipped_mask, 1].copy()

    indices_acc_idx = prim['indices']
    acc = gltf['accessors'][indices_acc_idx]
    bv = gltf['bufferViews'][acc['bufferView']]
    offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    orig_dtype = _DTYPE_MAP[acc['componentType']]
    fixed_bytes = fixed_faces.astype(orig_dtype).tobytes()
    if len(fixed_bytes) > bv['byteLength']:
        log.error("[mesh_cleanup] fixed indices larger than bufferView — skipping")
        return glb_bytes
    bindata[offset:offset + len(fixed_bytes)] = fixed_bytes

    new_glb = _build_glb(gltf, bindata)
    if not preserve_rig:
        new_glb = _convert_glb_textures_to_png(new_glb)
    return new_glb


def dedup(glb_bytes: bytes, tol: float = 5e-4) -> bytes:
    """Merge coincident vertices (within ``tol`` distance) to eliminate Z-fighting.

    Uses scipy.spatial.cKDTree to find pairs, union-find to merge, then
    rebuilds the GLB with averaged positions + remapped faces/attributes.
    """
    from scipy.spatial import cKDTree

    gltf, bindata = _parse_glb(glb_bytes)
    prim = gltf['meshes'][0]['primitives'][0]
    verts = _read_acc(bindata, gltf, prim['attributes']['POSITION']).astype(np.float64)
    faces = _read_acc(bindata, gltf, prim['indices']).reshape(-1, 3).astype(np.int64)
    V, F = len(verts), len(faces)

    tree = cKDTree(verts)
    pairs = tree.query_pairs(r=tol, output_type='ndarray')
    if len(pairs) == 0:
        log.info("[mesh_cleanup] dedup: no coincident verts (tol=%.1e) — no-op", tol)
        return glb_bytes

    parent_uf = list(range(V))

    def find(x):
        while parent_uf[x] != x:
            parent_uf[x] = parent_uf[parent_uf[x]]
            x = parent_uf[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent_uf[ra] = rb

    for v0, v1 in pairs:
        union(int(v0), int(v1))

    new_id_map = {}
    next_id = 0
    old_to_new = np.zeros(V, dtype=np.int64)
    for v in range(V):
        r = find(v)
        if r not in new_id_map:
            new_id_map[r] = next_id
            next_id += 1
        old_to_new[v] = new_id_map[r]

    new_V = next_id
    n_merged = V - new_V
    log.info("[mesh_cleanup] dedup: merged %d/%d verts → %d (%d pairs, tol=%.1e)",
             n_merged, V, new_V, len(pairs), tol)

    new_verts = np.zeros((new_V, 3), dtype=np.float64)
    counts = np.bincount(old_to_new)
    np.add.at(new_verts, old_to_new, verts)
    new_verts /= counts[:, None]

    new_faces = old_to_new[faces]
    degenerate = ((new_faces[:, 0] == new_faces[:, 1]) |
                  (new_faces[:, 1] == new_faces[:, 2]) |
                  (new_faces[:, 0] == new_faces[:, 2]))
    new_faces = new_faces[~degenerate]
    n_degen = F - len(new_faces)
    if n_degen > 0:
        log.info("[mesh_cleanup] dedup: removed %d degenerate faces", n_degen)

    new_faces_flat = new_faces.reshape(-1).astype(np.uint32)

    pos_acc = gltf['accessors'][prim['attributes']['POSITION']]
    pos_acc['count'] = new_V
    if 'min' in pos_acc:
        pos_acc['min'] = new_verts.min(axis=0).tolist()
    if 'max' in pos_acc:
        pos_acc['max'] = new_verts.max(axis=0).tolist()

    idx_acc = gltf['accessors'][prim['indices']]
    idx_acc['count'] = len(new_faces_flat)

    # Read all attributes, rebuild arrays with new_v count
    attr_data = {}
    for attr_name, attr_idx in prim['attributes'].items():
        attr_data[attr_name] = _read_acc(bindata, gltf, attr_idx)

    for attr_name in attr_data:
        if attr_name == 'POSITION':
            continue
        old_arr = attr_data[attr_name]
        new_arr = np.zeros((new_V,) + old_arr.shape[1:], dtype=old_arr.dtype)
        np.add.at(new_arr, old_to_new, old_arr)
        if old_arr.dtype.kind == 'f':
            if attr_name == 'WEIGHTS_0':
                new_arr /= counts[:, None]
                row_sums = new_arr.sum(axis=1, keepdims=True)
                new_arr = new_arr / np.maximum(row_sums, 1e-10)
            else:
                new_arr /= counts[:, None]
        else:
            for nv in range(new_V):
                mask = old_to_new == nv
                first_old = np.argmax(mask)
                new_arr[nv] = old_arr[first_old]
        attr_data[attr_name] = new_arr

    # Rebuild binary buffer
    new_bindata = bytearray()

    def _align4(o):
        return (o + 3) & ~3

    new_bindata.extend(new_verts.astype(np.float32).tobytes())
    gltf['bufferViews'][pos_acc['bufferView']] = {
        'buffer': 0, 'byteOffset': 0,
        'byteLength': new_V * 12, 'target': 34962,
    }
    pos_acc['byteOffset'] = 0
    bv_offset = len(new_bindata)

    for attr_name, attr_idx in prim['attributes'].items():
        if attr_name == 'POSITION':
            continue
        bv_offset = _align4(bv_offset)
        arr = attr_data[attr_name]
        new_bindata.extend(arr.tobytes())
        acc = gltf['accessors'][attr_idx]
        acc['count'] = new_V
        acc['byteOffset'] = 0
        gltf['bufferViews'][acc['bufferView']] = {
            'buffer': 0, 'byteOffset': bv_offset,
            'byteLength': arr.nbytes, 'target': 34962,
        }
        bv_offset = len(new_bindata)

    bv_offset = _align4(bv_offset)
    new_bindata.extend(new_faces_flat.astype(np.uint32).tobytes())
    gltf['bufferViews'][idx_acc['bufferView']] = {
        'buffer': 0, 'byteOffset': bv_offset,
        'byteLength': len(new_faces_flat) * 4, 'target': 34963,
    }
    idx_acc['byteOffset'] = 0

    if gltf.get('buffers'):
        gltf['buffers'][0]['byteLength'] = len(new_bindata)

    out = _build_glb(gltf, bytearray(new_bindata))
    log.info("[mesh_cleanup] dedup: wrote %d verts, %d faces (was %d/%d), %d bytes",
             new_V, len(new_faces), V, F, len(out))
    return out
