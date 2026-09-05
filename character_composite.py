"""Composite a finished character: body + hair + outfit → one GLB.

Two merge strategies, both bytes-in/bytes-out (no file I/O, no trimesh):

  HAIR (helmet method)
    Hair is a static GLB from TRELLIS (no skeleton). We parent it as a
    child of the body's head bone (``mixamorig:Head`` for SOMAX rigs).
    Setting the hair node's local matrix to ``inverse(head_world_at_bind)``
    means: at bind pose the hair sits in its original model-space location
    (because head_world * inverse(head_world) = identity), and during
    animation the hair follows the head bone's animated transform.

  OUTFIT (skinned sibling)
    ``transfer_skin_weights`` produces a fully-rigged standalone outfit
    GLB whose JOINTS_0 indices reference a COPY of the body's skeleton
    (same joint order). We append the outfit's mesh primitive to the
    body's mesh and append its accessor data to the body's buffer. The
    joint indices already match the body's skin.joints list because
    transfer_skin_weights preserves order. No skeleton duplication.

Both ops raise ``RuntimeError`` with a concrete message on every failure
mode — silent crashes were the original sin of this pipeline.
"""
from __future__ import annotations

import json
import logging
import struct

import numpy as np

log = logging.getLogger(__name__)

_DTYPE_MAP = {5126: 'f4', 5123: 'u2', 5121: 'u1', 5122: 'i2', 5125: 'u4'}
_NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}

# Joint names we accept as "the head bone" — first match wins.
# SOMAX produces Mixamo rigs, so ``mixamorig:Head`` is the canonical name.
HEAD_JOINT_CANDIDATES = (
    "mixamorig:Head",
    "Head",
    "head",
    "Bip01 Head",
    "head_bone",
)


# ════════════════════════════════════════════════════════════════════════
# GLB I/O
# ════════════════════════════════════════════════════════════════════════

def _parse_glb(raw: bytes) -> tuple[dict, bytearray]:
    """Split GLB into (gltf-json, bin-chunk). Raises ValueError on malformed."""
    if len(raw) < 20:
        raise ValueError(f"GLB too small ({len(raw)} bytes)")
    magic, version, total_len = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67:
        raise ValueError(f"not a GLB (magic=0x{magic:08x})")
    if version != 2:
        raise ValueError(f"unsupported GLB version {version}")
    if total_len > len(raw):
        raise ValueError(f"GLB total_len {total_len} > buffer {len(raw)}")
    json_len, json_type = struct.unpack_from("<II", raw, 12)
    if json_type != 0x4E4F534A:
        raise ValueError(f"chunk 0 not JSON (type=0x{json_type:08x})")
    gltf = json.loads(raw[20:20 + json_len].decode())
    off = 20 + json_len
    if off + 8 > len(raw):
        # Single-chunk GLB (JSON only, no BIN) — treat as empty bin.
        return gltf, bytearray()
    bin_len, bin_type = struct.unpack_from("<II", raw, off)
    if bin_type != 0x004E4942:
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


def _collect_mesh_vertices(gltf: dict, bindata: bytearray) -> np.ndarray:
    """Collect all POSITION accessor vertices from all mesh primitives."""
    chunks = []
    for mesh in gltf.get('meshes', []):
        for prim in mesh['primitives']:
            pos_idx = prim.get('attributes', {}).get('POSITION')
            if pos_idx is None or pos_idx >= len(gltf.get('accessors', [])):
                continue
            verts = _read_acc(bindata, gltf, pos_idx)
            if verts.ndim == 2 and verts.shape[1] == 3:
                chunks.append(verts)
    if not chunks:
        return np.zeros((1, 3), dtype=np.float64)
    return np.vstack(chunks).astype(np.float64)


def _write_acc_inplace(bindata: bytearray, gltf: dict, acc_idx: int,
                       arr: np.ndarray) -> None:
    """Overwrite accessor binary data in-place (same dtype/count)."""
    acc = gltf['accessors'][acc_idx]
    bv = gltf['bufferViews'][acc['bufferView']]
    offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    fmt = np.dtype(_DTYPE_MAP[acc['componentType']])
    raw = np.ascontiguousarray(arr.astype(fmt)).tobytes()
    end = offset + len(raw)
    if end > len(bindata):
        raise RuntimeError(
            f"_write_acc_inplace: accessor {acc_idx} write [{offset}:{end}] "
            f"exceeds buffer ({len(bindata)} bytes)"
        )
    bindata[offset:end] = raw


def _align_hair_to_head(
    body_gltf: dict, body_bin: bytearray,
    hair_gltf: dict, hair_bin: bytearray,
    head_world: np.ndarray,
) -> None:
    """Transform hair vertices from Trellis model-space to body model-space.

    Without this, the helmet-method parenting (matrix = inverse(head_world))
    leaves the hair at the Trellis origin — the body's pelvis/feet area —
    producing the "hair flies in the air" bug.

    Transform:  v' = (v - hair_center) * scale + target_center
      hair_center  = centroid of hair bbox (Trellis-space)
      scale        = head_size / hair_max_dim  (fit hair to head)
      target_center = head_pos + up * head_size * 0.5  (centered over skull)

    After this transform, vertices are in body-space at the head. The
    existing matrix=inverse(head_world) then correctly keeps them there
    at bind and follows the head during animation.
    """
    # 1. Hair bbox in Trellis-space
    hair_verts = _collect_mesh_vertices(hair_gltf, hair_bin)
    hair_min = hair_verts.min(axis=0)
    hair_max = hair_verts.max(axis=0)
    hair_center = (hair_min + hair_max) / 2.0
    hair_dim = hair_max - hair_min
    hair_max_dim = float(hair_dim.max())
    if hair_max_dim < 1e-8:
        log.warning("[composite] hair mesh has zero extent — skipping alignment")
        return

    # 2. Body bbox to estimate proportions
    body_verts = _collect_mesh_vertices(body_gltf, body_bin)
    body_min = body_verts.min(axis=0)
    body_max = body_verts.max(axis=0)
    body_dim = body_max - body_min
    body_height = float(body_dim.max())
    if body_height < 1e-8:
        log.warning("[composite] body mesh has zero extent — skipping alignment")
        return

    # 3. Head size ≈ 1/8 of body height (standard human proportions)
    head_size = body_height / 8.0
    scale = head_size / hair_max_dim

    # 4. Head bone world position
    head_pos = head_world[:3, 3].astype(np.float64)

    # 5. "Up" axis = the body's tallest dimension
    up_axis = int(np.argmax(body_dim))

    # 6. Target center: skull center (head bone + half head size upward)
    target_center = head_pos.copy()
    target_center[up_axis] += head_size * 0.5

    log.info(
        "[composite] hair align: hair_extent=%.4f head_size=%.4f scale=%.4f "
        "head_pos=[%.3f,%.3f,%.3f] up=%d",
        hair_max_dim, head_size, scale,
        head_pos[0], head_pos[1], head_pos[2], up_axis,
    )

    # 7. Transform each POSITION accessor in-place
    for mesh in hair_gltf.get('meshes', []):
        for prim in mesh['primitives']:
            pos_idx = prim.get('attributes', {}).get('POSITION')
            if pos_idx is None or pos_idx >= len(hair_gltf.get('accessors', [])):
                continue
            verts = _read_acc(hair_bin, hair_gltf, pos_idx).astype(np.float64)
            verts_new = (verts - hair_center) * scale + target_center
            _write_acc_inplace(hair_bin, hair_gltf, pos_idx, verts_new)
            acc = hair_gltf['accessors'][pos_idx]
            acc['min'] = verts_new.min(axis=0).tolist()
            acc['max'] = verts_new.max(axis=0).tolist()


# ════════════════════════════════════════════════════════════════════════
# Matrix math for helmet parenting
# ════════════════════════════════════════════════════════════════════════

def _local_matrix(node: dict) -> np.ndarray:
    """4×4 local transform of a glTF node (handles TRS or matrix)."""
    M = np.eye(4, dtype=np.float64)
    if 'matrix' in node:
        m = node['matrix']
        if len(m) != 16:
            raise ValueError(f"node matrix has {len(m)} elements, expected 16")
        return np.array(m, dtype=np.float64).reshape(4, 4).T
    if 'translation' in node:
        t = node['translation']
        M[0, 3], M[1, 3], M[2, 3] = t[0], t[1], t[2]
    if 'rotation' in node:
        x, y, z, w = node['rotation']
        # Normalize to be safe (mixed-pose rigs sometimes have |q| != 1)
        n = (x * x + y * y + z * z + w * w) ** 0.5
        if n < 1e-12:
            raise ValueError("zero-length rotation quaternion")
        x, y, z, w = x / n, y / n, z / n, w / n
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)
        M[:3, :3] = R
    if 'scale' in node:
        s = node['scale']
        S = np.diag([s[0], s[1], s[2], 1.0])
        M = M @ S
    return M


def _world_matrix(node_idx: int, nodes: list[dict],
                  parent_of: dict[int, int | None]) -> np.ndarray:
    """Compute the world transform of a node by walking up to the scene root."""
    chain = []
    cur = node_idx
    seen = set()
    while cur is not None and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        cur = parent_of.get(cur)
    if cur is not None:
        raise ValueError(f"cycle in node hierarchy at node {cur}")
    M = np.eye(4, dtype=np.float64)
    for idx in reversed(chain):
        M = M @ _local_matrix(nodes[idx])
    return M


def _build_parent_map(nodes: list[dict], scene_roots: list[int]) -> dict[int, int | None]:
    """parent_of[node_idx] = parent_idx or None for roots."""
    parent_of: dict[int, int | None] = {}
    for root in scene_roots:
        parent_of[root] = None
    changed = True
    # Iterate to fixed point — node order isn't guaranteed parent-first.
    while changed:
        changed = False
        for i, node in enumerate(nodes):
            if i in parent_of and parent_of[i] is not None:
                continue
            if i not in parent_of:
                continue
            for child in node.get('children', []):
                if child not in parent_of:
                    parent_of[child] = i
                    changed = True
    return parent_of


def _find_head_joint(gltf: dict) -> int:
    """Return the node index of the head bone. Raises RuntimeError if absent."""
    skin = (gltf.get('skins') or [{}])[0]
    joint_nodes = skin.get('joints') or []
    if not joint_nodes:
        raise RuntimeError(
            "composite: body GLB has no skin.joints — cannot parent hair to head"
        )
    nodes = gltf.get('nodes', [])
    # First pass: exact match against canonical names.
    for candidate in HEAD_JOINT_CANDIDATES:
        for jn in joint_nodes:
            if jn >= len(nodes):
                continue
            name = nodes[jn].get('name', '')
            if name == candidate:
                log.info("[composite] head joint: %s (node %d)", name, jn)
                return jn
    # Second pass: case-insensitive substring match.
    for jn in joint_nodes:
        if jn >= len(nodes):
            continue
        name = nodes[jn].get('name', '').lower()
        if 'head' in name and 'end' not in name:
            log.info("[composite] head joint (substring): %s (node %d)", name, jn)
            return jn
    # Last resort: the joint whose position is highest on Y (top of skeleton).
    best_jn, best_y = None, -np.inf
    for jn in joint_nodes:
        if jn >= len(nodes):
            continue
        node = nodes[jn]
        if 'translation' in node:
            y = node['translation'][1]
        elif 'matrix' in node:
            y = node['matrix'][13]  # col-major translation
        else:
            continue
        if y > best_y:
            best_y, best_jn = y, jn
    if best_jn is None:
        raise RuntimeError(
            "composite: no head joint found (tried exact + substring + Y-max); "
            f"joints={[nodes[j].get('name', f'node{j}') for j in joint_nodes[:10]]}"
        )
    log.info("[composite] head joint (Y-max fallback): node %d", best_jn)
    return best_jn


# ════════════════════════════════════════════════════════════════════════
# Composite — hair + outfit into body
# ════════════════════════════════════════════════════════════════════════

def _append_accessor_data(gltf: dict, bindata: bytearray,
                          src_gltf: dict, src_bindata: bytes,
                          src_acc_idx: int) -> int:
    """Append a source accessor's binary data to the destination buffer.

    Copies the data, adds a bufferView, adds an accessor with the same
    properties as the source. Returns the new accessor index in dest.
    """
    src_acc = src_gltf['accessors'][src_acc_idx]
    src_bv = src_gltf['bufferViews'][src_acc['bufferView']]
    src_offset = src_bv.get('byteOffset', 0) + src_acc.get('byteOffset', 0)
    # Compute the actual byte length of this accessor's data.
    ncomp = _NCOMP[src_acc['type']]
    fmt_size = np.dtype(_DTYPE_MAP[src_acc['componentType']]).itemsize
    byte_len = src_acc['count'] * ncomp * fmt_size
    if src_offset + byte_len > len(src_bindata):
        raise RuntimeError(
            f"composite: accessor {src_acc_idx} bytes [{src_offset}:{src_offset + byte_len}] "
            f"exceed source buffer ({len(src_bindata)} bytes)"
        )
    chunk = bytes(src_bindata[src_offset:src_offset + byte_len])

    # Pad dest buffer to 4-byte alignment before appending.
    while len(bindata) % 4:
        bindata.append(0)
    new_offset = len(bindata)
    bindata.extend(chunk)

    new_bv_idx = len(gltf['bufferViews'])
    gltf['bufferViews'].append({
        'buffer': 0,
        'byteOffset': new_offset,
        'byteLength': byte_len,
        'target': src_bv.get('target', 34962),
    })
    new_acc_idx = len(gltf['accessors'])
    new_acc = {
        'bufferView': new_bv_idx,
        'componentType': src_acc['componentType'],
        'count': src_acc['count'],
        'type': src_acc['type'],
    }
    if 'min' in src_acc:
        new_acc['min'] = src_acc['min']
    if 'max' in src_acc:
        new_acc['max'] = src_acc['max']
    gltf['accessors'].append(new_acc)
    return new_acc_idx


def _merge_mesh_primitive(
    dst_gltf: dict, dst_bin: bytearray,
    src_gltf: dict, src_bin: bytes,
    src_prim: dict, material_remap: dict[int, int],
) -> dict:
    """Copy src_prim into dst, remapping accessor indices + material.

    Returns the new primitive dict (attributes remapped, indices remapped,
    material remapped). Does NOT append to dst_gltf['meshes'] — caller does
    that so they can choose which mesh to add it to.
    """
    new_attrs = {}
    for attr_name, src_acc_idx in src_prim.get('attributes', {}).items():
        if src_acc_idx >= len(src_gltf['accessors']):
            raise RuntimeError(
                f"composite: src primitive attribute {attr_name} → accessor "
                f"{src_acc_idx} is out of range (src has {len(src_gltf['accessors'])})"
            )
        new_attrs[attr_name] = _append_accessor_data(
            dst_gltf, dst_bin, src_gltf, src_bin, src_acc_idx)
    new_prim: dict = {'attributes': new_attrs}
    if 'indices' in src_prim:
        new_prim['indices'] = _append_accessor_data(
            dst_gltf, dst_bin, src_gltf, src_bin, src_prim['indices'])
    if 'material' in src_prim:
        m = src_prim['material']
        if m not in material_remap:
            raise RuntimeError(
                f"composite: src primitive references material {m} not in remap "
                f"(src has {len(src_gltf.get('materials', []))} materials)"
            )
        new_prim['material'] = material_remap[m]
    if 'mode' in src_prim:
        new_prim['mode'] = src_prim['mode']
    return new_prim


def _merge_materials(dst_gltf: dict, src_gltf: dict) -> dict[int, int]:
    """Append src materials to dst; return {src_material_idx: dst_material_idx}.

    Also remaps texture/image indices.
    """
    remap: dict[int, int] = {}
    if not src_gltf.get('materials'):
        return remap
    n_dst_tex = len(dst_gltf.get('textures', []))
    n_dst_img = len(dst_gltf.get('images', []))
    # Append src textures + images (no dedup — they belong to src meshes).
    # Textures get their `source` (image index) bumped by n_dst_img so they
    # point at our appended image copies, not dst's pre-existing images.
    for tex in src_gltf.get('textures', []):
        new_tex = dict(tex)
        if 'source' in new_tex:
            new_tex['source'] += n_dst_img
        dst_gltf.setdefault('textures', []).append(new_tex)
    for img in src_gltf.get('images', []):
        new_img = dict(img)
        # If image uses a bufferView, that bufferView lives in src —
        # _remap_images (called after us) copies the bytes and updates the
        # dst copy's bufferView. Standalone data URIs copy fine.
        dst_gltf.setdefault('images', []).append(new_img)
    for i, mat in enumerate(src_gltf['materials']):
        new_idx = len(dst_gltf.get('materials', []))
        # Bump texture indices by n_dst_tex so they point at our appended copies.
        new_mat = json.loads(json.dumps(mat))  # deep copy
        _bump_texture_indices(new_mat, n_dst_tex)
        dst_gltf.setdefault('materials', []).append(new_mat)
        remap[i] = new_idx
    return remap


def _bump_texture_indices(mat: dict, delta: int) -> None:
    """Increment every texture index inside a material by delta."""
    pbr = mat.get('pbrMetallicRoughness', {})
    for key in ('baseColorTexture', 'metallicRoughnessTexture'):
        if key in pbr and 'index' in pbr[key]:
            pbr[key]['index'] += delta
    for key in ('normalTexture', 'occlusionTexture', 'emissiveTexture'):
        if key in mat and 'index' in mat[key]:
            mat[key]['index'] += delta


def merge_hair(body_bytes: bytes, hair_bytes: bytes) -> bytes:
    """Parent a static hair GLB to the body's head bone (helmet method).

    Raises RuntimeError if:
      - the body has no skeleton
      - no head joint can be located
      - the hair has no mesh data
    """
    if not body_bytes:
        raise RuntimeError("composite.merge_hair: body_bytes is empty")
    if not hair_bytes:
        raise RuntimeError("composite.merge_hair: hair_bytes is empty")

    body_gltf, body_bin = _parse_glb(body_bytes)
    hair_gltf, hair_bin = _parse_glb(hair_bytes)

    if not body_gltf.get('skins'):
        raise RuntimeError("composite.merge_hair: body has no skin — cannot find head bone")
    if not body_gltf.get('scenes'):
        raise RuntimeError("composite.merge_hair: body has no scene")
    if not hair_gltf.get('meshes'):
        raise RuntimeError("composite.merge_hair: hair has no mesh")

    head_node_idx = _find_head_joint(body_gltf)
    scene_roots = body_gltf['scenes'][body_gltf.get('scene', 0)]['nodes']
    parent_of = _build_parent_map(body_gltf.get('nodes', []), scene_roots)
    head_world = _world_matrix(head_node_idx, body_gltf.get('nodes', []), parent_of)
    try:
        head_world_inv = np.linalg.inv(head_world)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"composite.merge_hair: head_world non-invertible: {e}")

    # Align hair vertices from Trellis-space to body-space at the head bone.
    # MUST happen before merge so the copied vertex data is already aligned.
    _align_hair_to_head(body_gltf, body_bin, hair_gltf, hair_bin, head_world)

    # Append hair's mesh data (geometry + textures + materials) to body.
    mat_remap = _merge_materials(body_gltf, hair_gltf)
    # Copy any image bufferViews (images reference bufferViews directly, not
    # through accessors). Must happen AFTER _merge_materials appends images.
    _remap_images(body_gltf, body_bin, hair_gltf, hair_bin)

    # Build hair mesh primitives in body's coordinate space.
    hair_mesh_idx = len(body_gltf['meshes'])
    new_prims = []
    for prim in hair_gltf['meshes'][0]['primitives']:
        new_prims.append(_merge_mesh_primitive(
            body_gltf, body_bin, hair_gltf, hair_bin, prim, mat_remap))
    body_gltf['meshes'].append({'primitives': new_prims})

    # Add hair mesh node parented to head bone. Local transform =
    # inverse(head_world_at_bind) so at bind pose the hair stays in model
    # space; during animation it follows the head bone's animated transform.
    flat_inv = head_world_inv.T.reshape(-1).tolist()
    hair_node_idx = len(body_gltf['nodes'])
    body_gltf['nodes'].append({
        'name': 'hair_helmet',
        'mesh': hair_mesh_idx,
        'matrix': flat_inv,
    })
    # Parent hair_node to head_node (it's a child in the hierarchy).
    head_node = body_gltf['nodes'][head_node_idx]
    head_node.setdefault('children', []).append(hair_node_idx)

    # Update buffer length.
    if body_gltf.get('buffers'):
        body_gltf['buffers'][0]['byteLength'] = len(body_bin)
    else:
        body_gltf['buffers'] = [{'byteLength': len(body_bin)}]

    out = _build_glb(body_gltf, body_bin)
    log.info("[composite] merge_hair: body+hair = %d bytes (head joint node %d)",
             len(out), head_node_idx)
    return out


def _append_buffer_view_data(dst_gltf: dict, dst_bin: bytearray,
                             src_gltf: dict, src_bin: bytes,
                             src_bv_idx: int) -> int:
    """Copy a source bufferView's raw bytes into dst; return new bv index.

    Use this for bufferViews that are NOT referenced by an accessor (e.g.
    images reference bufferViews directly). Accessor-backed bufferViews
    should go through ``_append_accessor_data`` instead.
    """
    src_bv = src_gltf['bufferViews'][src_bv_idx]
    src_off = src_bv.get('byteOffset', 0)
    byte_len = src_bv['byteLength']
    if src_off + byte_len > len(src_bin):
        raise RuntimeError(
            f"composite: bufferView {src_bv_idx} bytes [{src_off}:{src_off + byte_len}] "
            f"exceed source buffer ({len(src_bin)} bytes)"
        )
    chunk = bytes(src_bin[src_off:src_off + byte_len])
    while len(dst_bin) % 4:
        dst_bin.append(0)
    new_offset = len(dst_bin)
    dst_bin.extend(chunk)
    new_bv_idx = len(dst_gltf['bufferViews'])
    dst_gltf['bufferViews'].append({
        'buffer': 0,
        'byteOffset': new_offset,
        'byteLength': byte_len,
    })
    return new_bv_idx


def _remap_images(dst_gltf: dict, dst_bin: bytearray,
                 src_gltf: dict, src_bin: bytes) -> None:
    """Copy src image bufferView bytes into dst and fix up the dst copies.

    Images that use ``uri`` (data URIs) are copied as-is by ``_merge_materials``
    via deep copy. Only bufferView-backed images need byte copies.

    MUST update the dst copies (appended by _merge_materials), not the src
    originals — the src dicts are never read again and modifying them has no
    effect on the output GLB.
    """
    n_src_img = len(src_gltf.get('images', []))
    if n_src_img == 0:
        return
    # The dst copies start at this index (dst had this many images before
    # _merge_materials appended the src copies).
    dst_start = len(dst_gltf.get('images', [])) - n_src_img
    for i, src_img in enumerate(src_gltf.get('images', [])):
        if 'bufferView' in src_img:
            old_bv = src_img['bufferView']
            new_bv = _append_buffer_view_data(dst_gltf, dst_bin, src_gltf, src_bin, old_bv)
            dst_gltf['images'][dst_start + i]['bufferView'] = new_bv


def _align_outfit_to_body(
    body_gltf: dict, body_bin: bytearray,
    out_gltf: dict, out_bin: bytearray,
) -> None:
    """Scale + translate outfit vertices to match the body's coordinate frame.

    Problem (2026-08-12, user: "Clothing is much bigger than the somax
    character, drowning them"): the body goes through Trellis + SOMAX;
    the outfit goes through Trellis only. SOMAX normalizes to a different
    unit scale than Trellis's default, so the outfit's raw vertices are in
    a different coordinate frame than the body's — even though
    transfer_rig copied the body's skeleton into the outfit GLB.

    Without alignment, merge_outfit appends the outfit primitives as-is,
    and the outfit dwarfs the body.

    Transform:  v' = (v - out_center) * scale + body_center
      out_center   = centroid of outfit bbox
      body_center  = centroid of body bbox  (both share the up-axis offset)
      scale        = body_width / out_width
      width        = max(X-extent, Z-extent) — shoulder/torso width,
                     independent of outfit length (no head/feet).

    Only the horizontal centering is adjusted — the Y (up) offset is
    preserved from the outfit's own bbox, since the outfit and body share
    the same ground plane (both generated from the same pose).
    """
    body_verts = _collect_mesh_vertices(body_gltf, body_bin)
    out_verts = _collect_mesh_vertices(out_gltf, out_bin)

    body_min, body_max = body_verts.min(axis=0), body_verts.max(axis=0)
    out_min, out_max = out_verts.min(axis=0), out_verts.max(axis=0)
    body_dim = body_max - body_min
    out_dim = out_max - out_min

    # Width = the LARGER horizontal dimension (X or Z), robust to pose
    # asymmetry (e.g., A-pose has wider X than Z). Y is up.
    body_width = float(max(body_dim[0], body_dim[2]))
    out_width = float(max(out_dim[0], out_dim[2]))
    if out_width < 1e-8 or body_width < 1e-8:
        log.warning(
            "[composite] outfit or body has zero width — skipping alignment")
        return

    scale = body_width / out_width

    # Center the outfit on the body in XZ (horizontal), keep outfit Y.
    out_center = (out_min + out_max) / 2.0
    body_center = (body_min + body_max) / 2.0

    log.info(
        "[composite] outfit align: out_width=%.4f body_width=%.4f "
        "scale=%.4f out_center=[%.3f,%.3f,%.3f] body_center=[%.3f,%.3f,%.3f]",
        out_width, body_width, scale,
        out_center[0], out_center[1], out_center[2],
        body_center[0], body_center[1], body_center[2],
    )

    # Apply uniform scale + XZ recenter, Y preserved from outfit.
    target_center = out_center.copy()
    target_center[0] = body_center[0]
    target_center[2] = body_center[2]

    for mesh in out_gltf.get('meshes', []):
        for prim in mesh['primitives']:
            pos_idx = prim.get('attributes', {}).get('POSITION')
            if pos_idx is None or pos_idx >= len(out_gltf.get('accessors', [])):
                continue
            verts = _read_acc(out_bin, out_gltf, pos_idx).astype(np.float64)
            verts_new = (verts - out_center) * scale + target_center
            _write_acc_inplace(out_bin, out_gltf, pos_idx, verts_new)
            acc = out_gltf['accessors'][pos_idx]
            acc['min'] = verts_new.min(axis=0).tolist()
            acc['max'] = verts_new.max(axis=0).tolist()


def merge_outfit(body_bytes: bytes, outfit_bytes: bytes) -> bytes:
    """Append a rigged outfit's primitive to the body's mesh.

    The outfit GLB (from transfer_skin_weights) carries a COPY of the body's
    skeleton. We verify the joint order matches, then append the outfit's
    primitive to the body's first mesh. Joint indices in JOINTS_0 already
    point at the same logical joints (same order), so no remap is needed.

    Raises RuntimeError if:
      - body has no skin
      - outfit has no skin
      - joint counts differ
      - outfit has no mesh primitive
    """
    if not body_bytes:
        raise RuntimeError("composite.merge_outfit: body_bytes is empty")
    if not outfit_bytes:
        raise RuntimeError("composite.merge_outfit: outfit_bytes is empty")

    body_gltf, body_bin = _parse_glb(body_bytes)
    out_gltf, out_bin = _parse_glb(outfit_bytes)

    if not body_gltf.get('skins'):
        raise RuntimeError("composite.merge_outfit: body has no skin")
    if not out_gltf.get('skins'):
        raise RuntimeError(
            "composite.merge_outfit: outfit has no skin — did transfer_skin_weights run?"
        )
    if not out_gltf.get('meshes'):
        raise RuntimeError("composite.merge_outfit: outfit has no mesh")

    body_skin = body_gltf['skins'][0]
    out_skin = out_gltf['skins'][0]
    body_joints = body_skin.get('joints', [])
    out_joints = out_skin.get('joints', [])
    if len(body_joints) != len(out_joints):
        raise RuntimeError(
            f"composite.merge_outfit: joint count mismatch "
            f"(body={len(body_joints)}, outfit={len(out_joints)}). "
            f"transfer_skin_weights must copy the SAME joints as the body."
        )
    # Verify joint-node names match (order-sensitive). If they don't, the
    # JOINTS_0 indices in the outfit won't map to the right body bones.
    body_nodes = body_gltf.get('nodes', [])
    out_nodes = out_gltf.get('nodes', [])
    mismatches = []
    for i, (bj, oj) in enumerate(zip(body_joints, out_joints)):
        bn = body_nodes[bj].get('name', f'node{bj}') if bj < len(body_nodes) else '?'
        on = out_nodes[oj].get('name', f'node{oj}') if oj < len(out_nodes) else '?'
        if bn != on:
            mismatches.append(f"joint {i}: body={bn} outfit={on}")
    if mismatches:
        raise RuntimeError(
            "composite.merge_outfit: joint-name mismatch — outfit skeleton is not "
            "the body's. Re-run transfer_skin_weights against the current body. "
            "First mismatches: " + "; ".join(mismatches[:5])
        )

    # Scale + recenter outfit to match body's coordinate frame.
    # MUST happen before merging so vertices are in body-space.
    _align_outfit_to_body(body_gltf, body_bin, out_gltf, out_bin)

    # Append outfit primitive(s) to body's first mesh.
    mat_remap = _merge_materials(body_gltf, out_gltf)
    _remap_images(body_gltf, body_bin, out_gltf, out_bin)

    body_mesh = body_gltf['meshes'][0]
    for prim in out_gltf['meshes'][0]['primitives']:
        if 'JOINTS_0' not in prim.get('attributes', {}):
            raise RuntimeError(
                "composite.merge_outfit: outfit primitive has no JOINTS_0 — "
                "transfer_skin_weights didn't run. Cannot skin to body skeleton."
            )
        if 'WEIGHTS_0' not in prim.get('attributes', {}):
            raise RuntimeError(
                "composite.merge_outfit: outfit primitive has no WEIGHTS_0 — "
                "transfer_skin_weights didn't run. Cannot skin to body skeleton."
            )
        new_prim = _merge_mesh_primitive(
            body_gltf, body_bin, out_gltf, out_bin, prim, mat_remap)
        body_mesh['primitives'].append(new_prim)

    # The outfit's primitive gets the body's skin via the mesh node's skin
    # reference — it's already set on the body's mesh node. No skin duplication.
    if body_gltf.get('buffers'):
        body_gltf['buffers'][0]['byteLength'] = len(body_bin)
    else:
        body_gltf['buffers'] = [{'byteLength': len(body_bin)}]

    out = _build_glb(body_gltf, body_bin)
    log.info("[composite] merge_outfit: body+outfit = %d bytes (%d joints verified)",
             len(out), len(body_joints))
    return out


def composite_character(
    body_bytes: bytes,
    hair_bytes: bytes | None = None,
    outfit_bytes: bytes | None = None,
) -> bytes:
    """Compose the final character GLB: body (+ hair) (+ outfit).

    Sequential: body → +hair → +outfit. Each step returns a fresh GLB.
    Hard-fails on any error — no partial output, no silent crash.

    (2026-09-05) BODY-ONLY is legal: the anny card's descope made the
    body alone the card (hair/outfit move to a future trellis-clothing
    card). With no parts, the body is still PARSE-VALIDATED (a
    malformed GLB refuses loudly) and passed through — a composite of
    one, never a blind copy.
    """
    if not hair_bytes and not outfit_bytes:
        _parse_glb(body_bytes)          # validation only — loud if malformed
        return body_bytes
    result = body_bytes
    if hair_bytes:
        result = merge_hair(result, hair_bytes)
    if outfit_bytes:
        result = merge_outfit(result, outfit_bytes)
    return result
