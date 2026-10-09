"""Direct SOMA motion → glTF animation baking for SOMAX-77 rigged GLBs.

ONE PATH. There is no Blender retarget anymore.

Pipeline:
    TRELLIS → SOMAX → bake_motion() → animated GLB

When the SOMAX output GLB (skin.name === "SOMA-Mixamo", 77 mixamorig joints)
is paired with a SOMA-77 motion NPZ (posed_joints + global_rot_mats), the
joint hierarchies match one-to-one. We can convert the motion to local TRS
quaternions and write them as glTF animation channels targeting the existing
joint nodes — no Blender subprocess, no scene-graph mutation, no retargeting
heuristics.

Used by POST /poser/somax/bake, which Pose Studio's TextToPoseTab calls
exclusively. There is no fallback path.

Translations: intentionally NOT animated. The NPZ's posed_joints are in
SOMA-template space (~170 cm adult); the GLB's mesh is at source scale
(~100 cm child). Baking template-space translations stretches the mesh
into adult proportions = body horror. The GLB's bind-pose translations
(already in mesh space) are correct and remain static. Rotations alone
produce all visible motion for in-place and arm/leg/torso animations.
"""
from __future__ import annotations

import io
import json
import struct
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from .skin import _global_rots_to_local_rots


# ── GLB chunk constants ──────────────────────────────────────────────────
_GLB_MAGIC = 0x46546C67   # 'glTF'
_CHUNK_JSON = 0x4E4F534A  # 'JSON'
_CHUNK_BIN = 0x004E4942   # 'BIN\0'


def read_glb_bytes(buf: bytes) -> tuple[dict[str, Any], bytearray]:
    """Parse a GLB into (json_dict, bin_bytearray). Raises on bad magic."""
    if len(buf) < 12:
        raise ValueError(f"GLB too short: {len(buf)} bytes")
    magic, version, total = struct.unpack("<III", buf[:12])
    if magic != _GLB_MAGIC:
        raise ValueError(f"bad GLB magic {magic:#x}")
    if total > len(buf):
        raise ValueError(f"GLB truncated: header says {total}, got {len(buf)}")
    i, jb_raw, bin_raw = 12, None, None
    while i + 8 <= len(buf):
        cl, ct = struct.unpack("<II", buf[i:i + 8])
        chunk = buf[i + 8:i + 8 + cl]
        i += 8 + cl
        if ct == _CHUNK_JSON:
            jb_raw = chunk
        elif ct == _CHUNK_BIN:
            bin_raw = chunk
    if jb_raw is None:
        raise ValueError("GLB has no JSON chunk")
    return json.loads(jb_raw), bytearray(bin_raw or b"")


def write_glb_bytes(jb: dict[str, Any], bin_data: bytearray) -> bytes:
    """Serialize (json_dict, bin_bytearray) → GLB bytes. Pads to 4-byte alignment."""
    while len(bin_data) % 4 != 0:
        bin_data.append(0)
    jb_raw = json.dumps(jb, separators=(",", ":")).encode("utf-8")
    while len(jb_raw) % 4 != 0:
        jb_raw += b" "
    total = 12 + 8 + len(jb_raw) + 8 + len(bin_data)
    out = struct.pack("<III", _GLB_MAGIC, 2, total)
    out += struct.pack("<II", len(jb_raw), _CHUNK_JSON) + jb_raw
    out += struct.pack("<II", len(bin_data), _CHUNK_BIN) + bytes(bin_data)
    return out


def is_somax_glb(jb: dict[str, Any]) -> bool:
    """A SOMAX GLB has skin.name === 'SOMA-Mixamo' with 77 joints.

    Pose Studio's only input path is TRELLIS → SOMAX, so every loaded
    character GLB MUST pass this check. If it doesn't, the user has
    bypassed the pipeline (e.g. dropped a random Mixamo download), which
    is an error — not a fallback-to-retarget case.
    """
    skins = jb.get("skins") or []
    for skin in skins:
        if skin.get("name") == "SOMA-Mixamo" and len(skin.get("joints") or []) == 77:
            return True
    return False


def describe_somax_mismatch(jb: dict[str, Any]) -> str:
    """Human-readable diagnosis of why a GLB isn't a SOMAX output.

    Returned to the frontend as the HTTP 400 error body so the user can
    tell whether they dropped a legacy Blender-retargeted GLB (skin.name
    === 'Armature'), a raw T-pose, or something else entirely.
    """
    skins = jb.get("skins") or []
    if not skins:
        return ("Not a SOMAX GLB: file has no skin (no skeleton + weights). "
                "Pose Studio accepts TRELLIS → SOMAX outputs only — drop the "
                "GLB from Assets → SOMAX 3D, not a raw mesh export.")
    names = ", ".join(repr(s.get("name")) for s in skins)
    joints = [len(s.get("joints") or []) for s in skins]
    if any(s.get("name") == "Armature" for s in skins):
        return (f"Not a SOMAX GLB: skin.name is {names} (legacy Blender-retargeted "
                f"output, {joints[0]} joints). Pose Studio accepts SOMAX outputs only "
                f"(skin.name === 'SOMA-Mixamo'). Re-run the character through "
                f"TRELLIS → SOMAX 3D and drop that output instead.")
    if any(s.get("name") == "SOMA-Mixamo" and len(s.get("joints") or []) != 77 for s in skins):
        return (f"Not a SOMAX GLB: skin.name is 'SOMA-Mixamo' but joints={joints} "
                f"(expected 77). File is malformed.")
    return (f"Not a SOMAX GLB: skin.name is {names}, joints={joints}. "
            f"Pose Studio accepts SOMAX outputs only (skin.name === 'SOMA-Mixamo', "
            f"77 joints). Drop the GLB from Assets → SOMAX 3D.")


def normalize_scene_graph(jb: dict[str, Any]) -> dict[str, Any]:
    """Defensive: ensure (armature, mesh) are the only scene roots.

    Some pre-fix SOMAX outputs (commit 155cd4e) and any TRELLIS-wrapped
    source meshes leave the mesh node nested inside a 'world' wrapper
    AND listed as a scene root — a node in two positions. Three.js r184
    creates a duplicate SkinnedMesh whose skeleton never binds, then
    Box3.setFromObject crashes with 'Cannot read properties of undefined
    (reading matrixWorld)'.

    This step finds the skin-bearing mesh node, detaches it from any
    parent's children array, and sets scenes[0].nodes to exactly
    [armature, mesh]. Idempotent — baking a baked GLB is a no-op here.
    """
    skins = jb.get("skins") or []
    if not skins:
        return jb  # nothing to normalize; is_somax_glb() will reject later

    skin0 = skins[0]
    armature_idx = skin0.get("skeleton")
    skin0.get("joints") or []

    # Find the skin-bearing mesh node.
    mesh_idx = None
    for i, node in enumerate(jb.get("nodes", [])):
        if "skin" in node and node["skin"] == 0:
            mesh_idx = i
            break
    if mesh_idx is None or armature_idx is None:
        return jb  # malformed; let downstream fail loudly

    # Detach mesh from any parent's children.
    for parent_node in jb.get("nodes", []):
        kids = parent_node.get("children")
        if kids and mesh_idx in kids:
            parent_node["children"] = [c for c in kids if c != mesh_idx]
            if not parent_node["children"]:
                del parent_node["children"]

    # Build the canonical scene: armature + mesh, in that order.
    # Armature first so its world transform propagates to the joints
    # before the mesh evaluates its skin.
    jb.setdefault("scenes", [{"nodes": []}])[0].setdefault("nodes", [])
    new_roots = [armature_idx]
    if mesh_idx != armature_idx:
        new_roots.append(mesh_idx)
    jb["scenes"][0]["nodes"] = new_roots
    return jb


def bake_motion(
    glb_bytes: bytes,
    npz_bytes: bytes,
    anim_name: str = "motion",
    fps: float = 30.0,
) -> tuple[bytes, dict[str, Any]]:
    """Bake SOMA-77 motion NPZ onto a SOMAX-77 rigged GLB.

    Args:
        glb_bytes: SOMAX output GLB (must have skin.name === "SOMA-Mixamo").
        npz_bytes: SOMA motion NPZ with posed_joints (T,77,3) and
                   global_rot_mats (T,77,3,3).
        anim_name: glTF animation name.
        fps: Target frames per second. Time stamps are written as t/fps.

    Returns:
        (animated_glb_bytes, diagnostics) where diagnostics contains
        frame_count, fps, joint_count, bind_rotation_max_diff_deg.

    Raises:
        ValueError: GLB is not a SOMAX output, or NPZ is malformed.
    """
    jb, bin_data = read_glb_bytes(glb_bytes)

    if not is_somax_glb(jb):
        raise ValueError(describe_somax_mismatch(jb))

    # Idempotent: detach mesh from 'world' wrappers, set scene roots to
    # [armature, mesh]. Prevents the matrixWorld crash on pre-fix inputs.
    normalize_scene_graph(jb)

    motion = np.load(io.BytesIO(npz_bytes), allow_pickle=False)
    required = {"posed_joints", "global_rot_mats"}
    if not required.issubset(set(motion.keys())):
        raise ValueError(
            f"NPZ missing required keys: {required - set(motion.keys())}"
        )

    posed_joints = np.asarray(motion["posed_joints"], dtype=np.float32)        # (T, 77, 3)
    global_rot_mats = np.asarray(motion["global_rot_mats"], dtype=np.float32)  # (T, 77, 3, 3)
    T, J, _ = posed_joints.shape
    if J != 77:
        raise ValueError(f"NPZ has {J} joints; expected 77 (SOMA-77).")
    if global_rot_mats.shape[:2] != (T, J):
        raise ValueError(
            f"NPZ shape mismatch: posed_joints T={T},J={J} but "
            f"global_rot_mats shape={global_rot_mats.shape}"
        )

    # ── Local rotations via parent-chain inversion ──────────────────────
    # SOMA outputs world-space rotations per joint. glTF animation channels
    # target LOCAL rotations (parent-relative). _global_rots_to_local_rots
    # does the chain inversion using SOMA-77 parent indices. The SOMAX GLB's
    # joint hierarchy mirrors SOMA-77 exactly (soma_weight_transfer builds
    # it that way), so local rots write directly into the matching nodes.
    local_rot_mats = _global_rots_to_local_rots(global_rot_mats)               # (T, 77, 3, 3)
    local_quats = R.from_matrix(local_rot_mats.reshape(-1, 3, 3)).as_quat()    # xyzw, (T*77, 4)
    local_quats = local_quats.reshape(T, J, 4).astype(np.float32)
    # glTF quaternion component order is xyzw — scipy already returns xyzw.

    # ── Diagnostic: how far is frame 0 from the bind pose? ──────────────
    # Large values indicate the GLB's bind pose differs from SOMA's T-pose,
    # which is normal (SOMAX keeps the source mesh's bind pose). This is
    # informational, not a defect.
    joint_nodes = jb["skins"][0]["joints"]
    bind_rot_max_diff = 0.0
    for i in range(77):
        node = jb["nodes"][joint_nodes[i]]
        bind_q = np.asarray(node.get("rotation", [0, 0, 0, 1]), dtype=np.float32)
        dot = abs(float(np.dot(bind_q, local_quats[0, i])))
        ang = 2 * np.arccos(min(dot, 1.0))
        bind_rot_max_diff = max(bind_rot_max_diff, ang)

    # ── FK GATE (2026-08-29 stale-module incident) ──────────────────────
    # The packed channels must reproduce the NPZ's own posed_joints when
    # forward-kinematic'd against the rig's bind translations (root-
    # relative). A stale/wrong conversion module (e.g. globals written as
    # locals) passes every downstream render gate that doesn't measure
    # limb ratios — this comparison is numeric and cannot be fooled.
    # Hard-fail above 15cm max joint error.
    #
    # THE SCALE MATCH (2026-10-09, run-9641b18af83e): the truth is in
    # SOMA-template space (~170 cm adult) while the FK is in the ASSET's
    # mesh space (the docstring's own ~100 cm child case) — comparing
    # them raw fires on every proportion difference (the commission's
    # 0.87 m Kimodo asset died at 50 cm with a mathematically perfect
    # conversion). The truth is uniform-scaled into asset space first:
    # one factor from the bind-pose reach (identity-rotation FK, root-
    # relative max radius) over the template's frame-0 reach, clamped to
    # a sane band so a genuinely wrong rig (wrong skeleton, unit mess)
    # still trips the gate. Real conversion bugs survive the scaling —
    # globals-as-locals twists break directions, not just lengths.
    _children_of = {}
    for _i, _n in enumerate(jb["nodes"]):
        for _c in _n.get("children", []):
            _children_of[_c] = _i
    _jspace = {ji: k for k, ji in enumerate(joint_nodes)}
    _parent = [_jspace.get(_children_of.get(ji)) for ji in joint_nodes]
    _bind_t = [np.asarray(jb["nodes"][ji].get("translation", [0, 0, 0]),
                          dtype=np.float64) for ji in joint_nodes]

    # The scale factor: bind-pose reach (identity-rotation FK — the
    # asset's skeleton extent in GLOBAL mesh space, root-relative max
    # joint position) over the template's frame-0 reach (posed_joints'
    # own global extent). Both sides are global positions: a max
    # single-bone metric here would compare a femur against a wingspan.
    def _bind_world(k):
        p = _parent[k]
        return _bind_t[k] if p is None else _bind_world(p) + _bind_t[k]

    _bind_pts = np.array([_bind_world(k) for k in range(J)])
    _bind_pts -= _bind_pts[0]
    _t0 = posed_joints[0].astype(np.float64)
    _t0 = _t0 - _t0[0]
    _template_reach = float(np.abs(_t0).max())
    _asset_reach = float(np.abs(_bind_pts).max())
    _scale = _asset_reach / _template_reach if _template_reach > 1e-9 else 1.0
    if not (0.2 <= _scale <= 5.0):
        raise ValueError(
            f"FK GATE FAILED: the asset skeleton's reach ({_asset_reach:.3f}m) "
            f"is {int(_scale*100)}% of the motion template's — the rig is not "
            f"the SOMA-77 template this NPZ was posed against (wrong GLB or "
            f"unit mess), refusing to bake."
        )

    def _fk_err(t):
        world = {}
        def W(k):
            if k in world:
                return world[k]
            M = np.eye(4)
            M[:3, :3] = R.from_quat(local_quats[t, k].astype(np.float64)).as_matrix()
            M[:3, 3] = _bind_t[k]
            p = _parent[k]
            out = M if p is None else W(p) @ M
            world[k] = out
            return out
        pts = np.array([W(k)[:3, 3] for k in range(J)])
        pts -= pts[0]
        truth = posed_joints[t].astype(np.float64)
        truth = (truth - truth[0]) * _scale
        return float(np.abs(pts - truth).max())

    _fk_max = max(_fk_err(t) for t in {0, T // 4, T // 2, T - 1})
    if _fk_max > 0.15:
        raise ValueError(
            f"FK GATE FAILED: packed rotations deviate {int(_fk_max*100)}cm "
            f"from posed_joints ground truth scaled into asset space "
            f"(factor {_scale:.3f}, limit 15cm). The conversion math is "
            f"wrong (stale module? wrong locals?) — refusing to ship "
            f"garbage bytes."
        )

    # ── Pack animation binary (joint-major for byteOffset slicing) ──────
    while len(bin_data) % 4 != 0:
        bin_data.append(0)

    # Time accessor: T float32s, one per frame.
    time_data = (np.arange(T, dtype=np.float32) * (1.0 / float(fps))).tobytes()
    time_off = len(bin_data)
    bin_data.extend(time_data)

    # Rotation data, joint-major: (J, T, 4) flattened → joint i occupies
    # bytes [i*T*16 .. (i+1)*T*16]. Each rotation accessor slices into
    # this buffer with byteOffset = i * T * 16. This avoids one accessor
    # per (joint, frame) and keeps the buffer contiguous.
    rot_data = local_quats.transpose(1, 0, 2).reshape(-1, 4).astype(np.float32).tobytes()
    rot_off = len(bin_data)
    bin_data.extend(rot_data)
    while len(bin_data) % 4 != 0:
        bin_data.append(0)

    n_bv = len(jb.setdefault("bufferViews", []))
    n_acc = len(jb.setdefault("accessors", []))
    time_bv, rot_bv = n_bv, n_bv + 1
    jb["bufferViews"].append({"buffer": 0, "byteOffset": time_off,  "byteLength": len(time_data)})
    jb["bufferViews"].append({"buffer": 0, "byteOffset": rot_off,   "byteLength": len(rot_data)})

    # glTF componentType 5126 = FLOAT, type SCALAR = time, VEC4 = rotation.
    time_acc = n_acc
    jb["accessors"].append({
        "bufferView": time_bv, "componentType": 5126, "count": T,
        "type": "SCALAR", "min": [0.0], "max": [(T - 1) / float(fps)],
    })

    # 77 rotation accessors, each slicing into rot_bv at offset i*T*16.
    rot_acc_base = n_acc + 1
    for i in range(77):
        jb["accessors"].append({
            "bufferView": rot_bv, "byteOffset": i * T * 16,
            "componentType": 5126, "count": T, "type": "VEC4",
        })

    # One sampler + one rotation channel per joint.
    samplers: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    for i in range(77):
        s = len(samplers)
        samplers.append({
            "input": time_acc,
            "output": rot_acc_base + i,
            "interpolation": "LINEAR",
        })
        channels.append({
            "sampler": s,
            "target": {"node": joint_nodes[i], "path": "rotation"},
        })

    jb.setdefault("animations", []).append({
        "name": anim_name,
        "samplers": samplers,
        "channels": channels,
    })
    jb.setdefault("buffers", [{"byteLength": 0}])[0]["byteLength"] = len(bin_data)

    out_bytes = write_glb_bytes(jb, bin_data)
    return out_bytes, {
        "frame_count": int(T),
        "fps": float(fps),
        "joint_count": int(J),
        "channel_count": len(channels),
        "bind_rotation_max_diff_deg": float(np.degrees(bind_rot_max_diff)),
    }
