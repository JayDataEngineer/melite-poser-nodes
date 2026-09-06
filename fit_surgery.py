"""fit_surgery.py — the three meld fit laws as bytes-in / bytes-out GLB surgery.

Pack-side twin of estate tools/fit_prop.py (same function names, same math,
same refusal strings): the ComfyUI FitProp node calls fit_prop_bytes here so
a queue run seats props with the identical solver the e2e wall pins. This
file stands alone upstream (stdlib + numpy only): the closed row contract
lives ONCE in estate py/fit_spec.py AttachmentRow; the _Row gate below
mirrors its inference + conflict refusals, and tests/walls/suites/
test_fit_parity.py proves the twins agree bit-for-bit on real keeper assets.

Part of melite-poser-nodes (own repo, vendored as a submodule).
"""

from __future__ import annotations

import json
import math
import struct

import numpy as np


class FitMode:
    """Mode constants (NOT the estate enum — this pack imports nothing estate-side)."""

    socket = 'socket'
    fit = 'fit'
    grip = 'grip'


def _num(v: object, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ValueError('fit row %s = %r: a finite number is required' % (name, v))
    return float(v)


def _nonneg(v: object, name: str, default: float) -> float:
    if v is None or v == '':
        return default
    f = _num(v, name)
    if f < 0.0:
        raise ValueError('fit row %s = %r: negative is nonsense (never silently clamped)' % (name, v))
    return f


def _vec3(v: object, name: str) -> tuple:
    if v is None or v == '' or v == []:
        return (0.0, 0.0, 0.0)
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ValueError('fit row %s = %r: want [x, y, z]' % (name, v))
    return (_num(v[0], name), _num(v[1], name), _num(v[2], name))


def _str(v: object) -> str:
    return str(v or '')


class _Row:
    """The gated attachment row (mirrors py/fit_spec.py AttachmentRow inference + conflicts)."""

    def __init__(self, d: dict) -> None:
        if not isinstance(d, dict):
            raise ValueError('fit row must be a meld-JSON object (got %s)' % type(d).__name__)
        self.model = _str(d.get('Model', '')).strip()
        self.bone = _str(d.get('Bone', '')).strip()
        if not self.model:
            raise ValueError('fit row needs Model (which prop to seat)')
        if not self.bone:
            raise ValueError('fit row needs Bone (which bone carries it)')
        self.position = _vec3(d.get('Position', []), 'Position')
        self.rotation = _vec3(d.get('Rotation', []), 'Rotation')
        self.target_width = _nonneg(d.get('TargetWidth', 0.0), 'TargetWidth', 0.0)
        self.fit_ratio = _nonneg(d.get('FitRatio', 0.0), 'FitRatio', 0.0)
        self.fit_ref_bone = _str(d.get('FitRefBone', '')).strip()
        self.fit_ref_bone2 = _str(d.get('FitRefBone2', '')).strip()
        self.seat_ratio = _nonneg(d.get('SeatRatio', 0.12), 'SeatRatio', 0.12)
        self.seat_offset_y = _num(d.get('SeatOffsetY', 0.0) or 0.0, 'SeatOffsetY')
        self.finger_bone = _str(d.get('FingerBone', '')).strip()
        self.thumb_bone = _str(d.get('ThumbBone', '')).strip()
        align = _str(d.get('Align', ''))
        if align == 'grip':
            self.mode = FitMode.grip
        elif self.fit_ratio > 0.0 or self.fit_ref_bone:
            self.mode = FitMode.fit
        else:
            self.mode = FitMode.socket
        if self.mode is FitMode.fit:
            if not self.fit_ref_bone:
                raise ValueError('fit needs FitRefBone (nothing to measure)')
            if self.fit_ratio <= 0.0:
                raise ValueError('fit needs FitRatio > 0 (got %r)' % (self.fit_ratio,))
            if self.target_width != 0.0:
                raise ValueError('fit + TargetWidth=%r conflict: FitRatio owns width' % (self.target_width,))
        elif self.mode is FitMode.grip:
            if not self.finger_bone:
                raise ValueError('grip needs FingerBone (no blade axis)')
            if self.position != (0.0, 0.0, 0.0) or self.rotation != (0.0, 0.0, 0.0):
                raise ValueError('grip + Position/Rotation conflict: the grip law owns orientation')
        else:
            if self.fit_ratio != 0.0 or self.fit_ref_bone:
                raise ValueError('socket + fit knobs conflict: socket rows never measure (use FitRatio)')
            if self.finger_bone or self.thumb_bone:
                raise ValueError('socket + finger/thumb conflict: hand refs belong to mode=grip')





def _parse_xyz(v: object, name: str) -> list:
    """Chip-friendly x,y,z strings (or triples) -> [x, y, z] floats."""
    if v is None or v == '':
        return [0.0, 0.0, 0.0]
    if isinstance(v, str):
        parts = [p.strip() for p in v.split(',')]
        if len(parts) != 3:
            raise ValueError('fit row %s = %r: want x,y,z' % (name, v))
        try:
            return [float(p) for p in parts]
        except ValueError:
            raise ValueError('fit row %s = %r: want numbers' % (name, v))
    return list(_vec3(v, name))


def build_row_json(knobs: dict) -> str:
    """Knob initials (lowercase, the flow/verb handshake) -> the meld row JSON.

    THE runtime derivation (rung 6): the FitRow node, the fit-seat flow
    and the fit card renderer all assemble here — never a second
    implementation beside it. The estate twin (schema.meld_row) exists
    ONLY for phase-1 validation; the parity wall pins them equal.
    Raises ValueError naming the knob on every refusal (the _Row gate).
    """
    if not isinstance(knobs, dict):
        raise ValueError('fit knobs must be an object (got %s)' % type(knobs).__name__)
    align = str(knobs.get('align', 'socket') or 'socket').strip()
    if align not in ('socket', 'fit', 'grip'):
        raise ValueError("fit row align = %r: want socket | fit | grip" % (knobs.get('align'),))
    meld = {
        'Model': str(knobs.get('model_name', '') or '').strip() or 'prop',
        'Bone': str(knobs.get('bone', '') or '').strip(),
        'SheatheBone': '',
        'Motion': '',
        'Position': _parse_xyz(knobs.get('position', '0,0,0'), 'position'),
        'Rotation': _parse_xyz(knobs.get('rotation', '0,0,0'), 'rotation'),
        'TargetWidth': _nonneg(knobs.get('target_width', 0.0), 'target_width', 0.0),
        'FitRatio': _nonneg(knobs.get('fit_ratio', 0.0), 'fit_ratio', 0.0),
        'FitRefBone': str(knobs.get('fit_ref_bone', '') or '').strip(),
        'FitRefBone2': str(knobs.get('fit_ref_bone2', '') or '').strip(),
        'SeatRatio': _nonneg(knobs.get('seat_ratio', 0.22), 'seat_ratio', 0.22),
        'SeatOffsetY': 0.0,
        'Align': 'grip' if align == 'grip' else '',
        'FingerBone': str(knobs.get('finger_bone', '') or '').strip(),
        'ThumbBone': str(knobs.get('thumb_bone', '') or '').strip(),
    }
    _Row(meld)  # the gate — conflicts refuse here, never downstream
    return json.dumps(meld, sort_keys=True)


_DTYPE = {5120: 'i1', 5121: 'u1', 5122: 'i2', 5123: 'u2', 5125: 'u4', 5126: 'f4'}
_NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def parse_glb(raw: bytes) -> tuple:
    if len(raw) < 20:
        raise ValueError('GLB too small (%d bytes)' % len(raw))
    magic, version, _total = struct.unpack_from('<III', raw, 0)
    if magic != 0x46546C67:
        raise ValueError('not a GLB (magic=0x%08x)' % magic)
    if version != 2:
        raise ValueError('unsupported GLB version %d' % version)
    json_len, json_type = struct.unpack_from('<II', raw, 12)
    if json_type != 0x4E4F534A:
        raise ValueError('chunk 0 not JSON')
    gltf = json.loads(raw[20:20 + json_len].decode())
    off = 20 + json_len
    if off + 8 > len(raw):
        return gltf, b''
    bin_len, bin_type = struct.unpack_from('<II', raw, off)
    if bin_type != 0x004E4942:
        raise ValueError('chunk 1 not BIN')
    return gltf, raw[off + 8:off + 8 + bin_len]


def build_glb(gltf: dict, bindata: bytes) -> bytes:
    js = json.dumps(gltf, separators=(',', ':')).encode()
    while len(js) % 4:
        js += b' '
    pad = b'\x00' * ((4 - len(bindata) % 4) % 4)
    bindata = bindata + pad
    total = 12 + 8 + len(js) + (8 + len(bindata) if bindata else 0)
    head = struct.pack('<III', 0x46546C67, 2, total)
    out = head + struct.pack('<II', len(js), 0x4E4F534A) + js
    if bindata:
        out += struct.pack('<II', len(bindata), 0x004E4942) + bindata
    return out


def read_accessor(gltf: dict, bindata: bytes, idx: int) -> np.ndarray:
    acc = gltf['accessors'][idx]
    comp = _DTYPE.get(acc['componentType'])
    if comp is None:
        raise ValueError('accessor %d: bad componentType %r' % (idx, acc['componentType']))
    ncomp = _NCOMP[acc['type']]
    bv = gltf['bufferViews'][acc['bufferView']]
    start = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    count = acc['count']
    dt = np.dtype(comp)
    unit = ncomp * dt.itemsize
    stride = bv.get('byteStride', unit)
    buf = np.frombuffer(bindata, dtype=np.uint8)
    if stride == unit:
        flat = np.frombuffer(buf[start:start + count * unit].tobytes(), dtype=dt)
        return flat.reshape(count, ncomp) if ncomp > 1 else flat
    out = np.empty((count, ncomp), dtype=dt)
    for i in range(count):
        chunk = buf[start + i * stride:start + i * stride + unit].tobytes()
        out[i] = np.frombuffer(chunk, dtype=dt)
    return out


def node_local(node: dict) -> np.ndarray:
    M = np.eye(4)
    if 'matrix' in node:
        return np.array(node['matrix'], dtype=float).reshape(4, 4).T
    if 'scale' in node:
        M = np.diag([node['scale'][0], node['scale'][1], node['scale'][2], 1.0]) @ M
    if 'rotation' in node:
        x, y, z, w = (float(v) for v in node['rotation'])
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
            [0, 0, 0, 1]], dtype=float)
        M = R @ M
    if 'translation' in node:
        T = np.eye(4)
        T[0, 3], T[1, 3], T[2, 3] = (float(v) for v in node['translation'])
        M = T @ M
    return M


def unity_euler_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    ax, ay, az = (math.radians(v) for v in (rx, ry, rz))
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def quat_from_mat(R: np.ndarray) -> tuple:
    t = float(R[0, 0] + R[1, 1] + R[2, 2])
    if t > 0.0:
        s = 2.0 * math.sqrt(t + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (x, y, z, w)


def bind_worlds(gltf: dict) -> tuple:
    nodes = gltf.get('nodes', [])
    name_to_idx: dict = {}
    for i, n in enumerate(nodes):
        nm = n.get('name', '')
        if nm:
            if nm in name_to_idx:
                raise ValueError('duplicate node name %r (Unity lookup ambiguous)' % nm)
            name_to_idx[nm] = i
    parent: dict = {}
    for i, n in enumerate(nodes):
        for c in n.get('children', []):
            parent[c] = i
    worlds: dict = {}
    def world(i: int) -> np.ndarray:
        if i not in worlds:
            local = node_local(nodes[i])
            worlds[i] = local if i not in parent else world(parent[i]) @ local
        return worlds[i]
    for i in range(len(nodes)):
        world(i)
    return name_to_idx, worlds


def prop_positions(gltf: dict, bindata: bytes) -> np.ndarray:
    verts = []
    seen: set = set()
    for mesh in gltf.get('meshes', []):
        for prim in mesh.get('primitives', []):
            ai = prim.get('attributes', {}).get('POSITION')
            if ai is None or ai in seen:
                continue
            seen.add(ai)
            verts.append(np.asarray(read_accessor(gltf, bindata, ai), dtype=float).reshape(-1, 3))
    if not verts:
        raise ValueError('prop meshes carry no POSITION attributes')
    return np.vstack(verts)


def _bone_frame(bone_world: np.ndarray, bone: str) -> tuple:
    Rb = bone_world[:3, :3]
    cols = [float(np.linalg.norm(Rb[:, i])) for i in range(3)]
    if max(cols) - min(cols) > 1e-4 * max(1.0, max(cols)):
        raise ValueError('bone %r bind world non-uniformly scaled %r' % (bone, cols))
    if cols[0] < 1e-9:
        raise ValueError('bone %r bind world degenerate' % bone)
    return Rb / cols[0], cols[0]


def law_socket(verts: np.ndarray, bone_world: np.ndarray, row) -> tuple:
    Rb_n, bs = _bone_frame(bone_world, row.bone)
    R = Rb_n @ unity_euler_deg(row.rotation[0], row.rotation[1], row.rotation[2])
    width = float(np.ptp(verts, axis=0).max())
    s = (row.target_width / width) if row.target_width > 0.0 else 1.0
    if width < 1e-9:
        raise ValueError('prop %r degenerate (zero width)' % row.model)
    off = np.array([row.position[0], row.position[1], row.position[2]], dtype=float)
    pos = bone_world[:3, 3] + Rb_n @ (off * bs)
    M = np.eye(4)
    M[:3, :3] = R * s
    M[:3, 3] = pos
    return M, {'scale': s, 'prop_width_m': width * s}


def law_fit(verts: np.ndarray, bone_world: np.ndarray, span_m: float, row, ref_pos: np.ndarray) -> tuple:
    width = float(np.ptp(verts, axis=0).max())
    if width < 1e-9:
        raise ValueError('prop %r degenerate (zero width)' % row.model)
    s = fit_width(span_m, row) / width
    Rb_n, _bs = _bone_frame(bone_world, row.bone)
    u = Rb_n[:, 1].copy()
    Rv = (Rb_n @ verts.T).T
    proj = Rv @ u
    Rc = Rb_n @ verts.mean(axis=0)
    drop = seat_drop(span_m, row)
    pos = ref_pos - u * (drop + s * float(proj.min()))
    for i in range(3):
        axis = Rb_n[:, i]
        if abs(float(axis @ u)) > 0.99:
            continue
        pos = pos + axis * float((ref_pos - s * Rc) @ axis - pos @ axis)
    M = np.eye(4)
    M[:3, :3] = Rb_n * s
    M[:3, 3] = pos
    return M, {'scale': s, 'span_m': span_m, 'seat_drop_m': drop, 'prop_width_m': width * s}


def law_grip(verts: np.ndarray, bone_world: np.ndarray, finger_pos: np.ndarray,
             thumb_pos: np.ndarray | None, row) -> tuple:
    ext = verts.max(axis=0) - verts.min(axis=0)
    longest = int(np.argmax(ext))
    others_all = [i for i in range(3) if i != longest]
    flat_order = sorted(range(3), key=lambda i: float(ext[i]))
    flattest = flat_order[0]
    mid = [i for i in range(3) if i not in (longest, flattest)][0]
    width = float(ext.max())
    s = (row.target_width / width) if row.target_width > 0.0 else 1.0
    bone_pos = bone_world[:3, 3]
    f = finger_pos - bone_pos
    if float(np.linalg.norm(f)) < 1e-9:
        raise ValueError('finger bone %r coincides with %r' % (row.finger_bone, row.bone))
    f = f / float(np.linalg.norm(f))
    t = None
    if thumb_pos is not None:
        tt = thumb_pos - bone_pos
        tt = tt - f * float(tt @ f)
        if float(np.linalg.norm(tt)) > 1e-6:
            t = tt / float(np.linalg.norm(tt))
    if t is None:
        arb = np.array([1.0, 0.0, 0.0]) if abs(float(f[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
        t = arb - f * float(arb @ f)
        t = t / float(np.linalg.norm(t))
    m = np.cross(t, f)
    m = m / float(np.linalg.norm(m))
    t2 = np.cross(f, m)
    basis = np.eye(3)
    basis[:, longest] = f
    basis[:, flattest] = t2
    basis[:, mid] = m
    if float(np.linalg.det(basis)) < 0.0:
        basis[:, mid] = -m
        t2 = np.cross(f, basis[:, mid])
        basis[:, flattest] = t2
    axis_vals = verts[:, longest]
    a_min, a_max = float(axis_vals.min()), float(axis_vals.max())
    nsl = 24
    edges = np.linspace(a_min, a_max, nsl + 1)
    widths = np.zeros(nsl)
    for k in range(nsl):
        sel = verts[(axis_vals >= edges[k]) & (axis_vals <= edges[k + 1])]
        widths[k] = float(np.ptp(sel[:, others_all].reshape(-1, len(others_all)), axis=0).max()) if len(sel) else 0.0
    w = int(np.argmax(widths))
    tip_end = 0 if widths[0] <= widths[-1] else nsl - 1
    nontip = (nsl - 1) - tip_end
    grip_t = 0.5 * (w / (nsl - 1) + nontip / (nsl - 1))
    grip_local = verts.mean(axis=0)
    grip_local[longest] = a_min + grip_t * (a_max - a_min)
    pos = bone_pos - (basis @ grip_local) * s
    M = np.eye(4)
    M[:3, :3] = basis * s
    M[:3, 3] = pos
    return M, {'scale': s, 'prop_width_m': width * s,
               'grip_local': [float(v) for v in grip_local],
               'longest_axis': longest, 'tip_end': tip_end, 'slices': nsl}


def graft(body: dict, body_bin: bytes, prop: dict, prop_bin: bytes,
          bone_idx: int, local: np.ndarray, model: str) -> tuple:
    for key in ('buffers', 'bufferViews', 'accessors', 'images', 'textures',
                'samplers', 'materials', 'meshes', 'nodes', 'scenes'):
        body.setdefault(key, [])
        prop.setdefault(key, [])
    if len(prop.get('buffers', [])) > 1:
        raise ValueError('prop %r has many buffers (single-buffer only)' % model)
    if prop.get('skins'):
        raise ValueError('prop %r skinned (socket law: static only)' % model)
    if prop.get('animations'):
        raise ValueError('prop %r animated (static props only)' % model)
    base = (len(body_bin) + 3) & ~3
    new_bin = body_bin + b'\x00' * (base - len(body_bin)) + prop_bin
    sh = {
        'bv': len(body['bufferViews']), 'acc': len(body['accessors']),
        'img': len(body['images']), 'smp': len(body['samplers']),
        'tex': len(body['textures']), 'mat': len(body['materials']),
        'mesh': len(body['meshes']), 'node': len(body['nodes']),
    }
    for bv in prop['bufferViews']:
        nb = dict(bv)
        nb['byteOffset'] = nb.get('byteOffset', 0) + base
        body['bufferViews'].append(nb)
    for acc in prop['accessors']:
        na = dict(acc)
        na['bufferView'] = na['bufferView'] + sh['bv']
        body['accessors'].append(na)
    for im in prop['images']:
        ni = dict(im)
        if 'bufferView' in ni:
            ni['bufferView'] = ni['bufferView'] + sh['bv']
        body['images'].append(ni)
    for sm in prop['samplers']:
        body['samplers'].append(dict(sm))
    for tx in prop['textures']:
        nt = dict(tx)
        if 'source' in nt:
            nt['source'] = nt['source'] + sh['img']
        if 'sampler' in nt:
            nt['sampler'] = nt['sampler'] + sh['smp']
        body['textures'].append(nt)
    for mt in prop['materials']:
        nm = json.loads(json.dumps(mt))
        _shift_tex_ref(nm, sh['tex'])
        body['materials'].append(nm)
    for me in prop['meshes']:
        nm = json.loads(json.dumps(me))
        for prim in nm.get('primitives', []):
            for k in list(prim.get('attributes', {}).keys()):
                prim['attributes'][k] = prim['attributes'][k] + sh['acc']
            if 'indices' in prim:
                prim['indices'] = prim['indices'] + sh['acc']
            if 'material' in prim:
                prim['material'] = prim['material'] + sh['mat']
        body['meshes'].append(nm)
    prop_scenes = prop.get('scenes', [])
    if prop_scenes and prop_scenes[0].get('nodes'):
        roots = list(prop_scenes[0]['nodes'])
    else:
        roots = list(range(len(prop['nodes'])))
    t = [float(local[0, 3]), float(local[1, 3]), float(local[2, 3])]
    sc = float(np.linalg.norm(local[:3, 0]))
    if sc < 1e-9:
        raise ValueError('fitted local matrix degenerate (scale ~0)')
    q = quat_from_mat(local[:3, :3] / sc)
    grafted = []
    for r in roots:
        nn = json.loads(json.dumps(prop['nodes'][r]))
        nn.pop('matrix', None)
        nn['translation'] = t
        nn['rotation'] = [q[0], q[1], q[2], q[3]]
        nn['scale'] = [sc, sc, sc]
        nn['name'] = model
        if 'mesh' in nn:
            nn['mesh'] = nn['mesh'] + sh['mesh']
        nn['children'] = [c + sh['node'] for c in nn.get('children', [])]
        body['nodes'].append(nn)
        grafted.append(len(body['nodes']) - 1)
    for idx, n in enumerate(prop['nodes']):
        if idx in roots:
            continue
        nn = json.loads(json.dumps(n))
        nn['children'] = [c + sh['node'] for c in nn.get('children', [])]
        if 'mesh' in nn:
            nn['mesh'] = nn['mesh'] + sh['mesh']
        body['nodes'].append(nn)
    body['nodes'][bone_idx].setdefault('children', []).extend(grafted)
    return body, new_bin


def _shift_tex_ref(o: object, add: int) -> None:
    if isinstance(o, dict):
        if 'index' in o and isinstance(o['index'], int) and 'texCoord' in o:
            o['index'] = o['index'] + add
        for v in o.values():
            _shift_tex_ref(v, add)
    elif isinstance(o, list):
        for v in o:
            _shift_tex_ref(v, add)


def seat_drop(span_m: float, row) -> float:
    """Prop-bottom seat below the ref bone: span x seat_ratio + gate stamp."""
    if span_m < 0.0:
        raise ValueError('span %r is negative' % (span_m,))
    return span_m * row.seat_ratio + row.seat_offset_y


def fit_width(span_m: float, row) -> float:
    """Prop width from a measured bone span (mode fit only — other modes never measure)."""
    if row.mode is not FitMode.fit:
        raise ValueError('fit_width on mode=%r: only fit measures' % (row.mode,))
    if span_m < 0.0:
        raise ValueError('span %r is negative' % (span_m,))
    return span_m * row.fit_ratio


def fit_prop_bytes(body_bytes: bytes, prop_bytes: bytes, row_dict: dict) -> tuple:
    """Bytes in / bytes out (the FitProp node door): fitted GLB + measurement record.

    Raises ValueError naming the cause on every refusal; nothing partially returns.
    """
    row = _Row(row_dict)
    body, body_bin = parse_glb(body_bytes)
    prop, prop_bin = parse_glb(prop_bytes)
    names, worlds = bind_worlds(body)
    if row.bone not in names:
        raise ValueError('bone %r not in body (%d named nodes)' % (row.bone, len(names)))
    bone_idx = names[row.bone]
    bone_world = worlds[bone_idx]
    verts = prop_positions(prop, prop_bin)
    if row.mode is FitMode.socket:
        desired, meas = law_socket(verts, bone_world, row)
    elif row.mode is FitMode.fit:
        if row.fit_ref_bone not in names:
            raise ValueError('fit_ref_bone %r not in body' % row.fit_ref_bone)
        second = row.fit_ref_bone2 or row.bone
        if second not in names:
            raise ValueError('fit_ref_bone2 %r not in body' % second)
        a = worlds[names[second]][:3, 3]
        b = worlds[names[row.fit_ref_bone]][:3, 3]
        span = float(np.linalg.norm(b - a))
        if span < 1e-9:
            raise ValueError('fit span ~0 (%r coincides)' % row.fit_ref_bone)
        desired, meas = law_fit(verts, bone_world, span, row, b)
    else:
        if row.finger_bone not in names:
            raise ValueError('finger_bone %r not in body' % row.finger_bone)
        thumb = None
        if row.thumb_bone:
            if row.thumb_bone not in names:
                raise ValueError('thumb_bone %r not in body' % row.thumb_bone)
            thumb = worlds[names[row.thumb_bone]][:3, 3]
        desired, meas = law_grip(verts, bone_world, worlds[names[row.finger_bone]][:3, 3], thumb, row)
    local = np.linalg.inv(bone_world) @ desired
    out_gltf, out_bin = graft(body, body_bin, prop, prop_bin, bone_idx, local, row.model)
    record = {
        'model': row.model,
        'measured': meas,
        'local_matrix': [[float(v) for v in r] for r in local],
    }
    return build_glb(out_gltf, out_bin), record
