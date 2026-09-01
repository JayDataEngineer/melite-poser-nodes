"""BVH → SOMA-77 retargeter for the Pose Studio external pose library.

Self-contained: parses any BVH file (Mixamo, CMU, Blender, or custom),
forward-kinematics each frame to world-space joint positions, maps the
source joints onto the 77-joint SOMA skeleton by name + hierarchy, scales
to SOMA proportions, and fills unmatched joints (fingers, eyes, jaw, toe
ends) from the SOMA neutral pose template.

The output is a (T, 77, 3) float32 array of SOMA-77 joint positions — the
exact shape Pose Studio's `editableJoints` / `KeyframeState.joints` /
`deformSomaJoints` consume. No rotations needed; the backend derives
rotations from bone directions when re-skinning.

Name mapping covers the three major BVH conventions:
  • Mixamo (Hips, Spine, Spine1, Spine2, LeftShoulder, LeftArm, …)
  • CMU/ASF (root, lhipjoint, lfemur, ltibia, lowerback, …)
  • Blender/Standard (Hip, Spine, Chest, Head, Shoulder.L, …)

Usage:
    from media.motion.bvh_retarget import retarget_bvh_text
    frames = retarget_bvh_text(bvh_text)  # (T, 77, 3) float32
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

# SOMA-77 skeleton topology — vendored into this ComfyUI pack so the
# retargeter is fully self-contained (no melite-head media/ dependency).
# This keeps the corpus + BVH import + skinning all inside inference-comfyui,
# per the 3-container architecture.
from .skeleton import (
    SOMA77_JOINT_NAMES,
    SOMA77_NEUTRAL,
    SOMA77_PARENTS,
)


# ──────────────────────────────────────────────────────────────────────────
# BVH parser — self-contained, handles HIERARCHY + MOTION sections.
# ──────────────────────────────────────────────────────────────────────────

class BVHJoint:
    """One joint in the parsed BVH hierarchy."""
    __slots__ = ("name", "offset", "channels", "parent", "children",
                 "depth", "rest_pos")
    def __init__(self, name: str, offset: np.ndarray, channels: List[str]):
        self.name = name
        self.offset = offset  # local offset from parent (cm in typical BVH)
        self.channels = channels  # e.g. ["Xposition","Yposition","Zposition","Zrotation",...]
        self.parent: Optional[int] = None  # index into joint list
        self.children: List[int] = []
        self.depth = 0
        self.rest_pos = np.zeros(3, dtype=np.float64)  # computed after parse


class BVHData:
    """Parsed BVH: joint hierarchy + per-frame channel values."""
    def __init__(self):
        self.joints: List[BVHJoint] = []
        self.name_to_idx: Dict[str, int] = {}
        self.frames: np.ndarray = np.zeros((0, 0), dtype=np.float32)  # (T, C)
        self.fps: float = 30.0


def _tokenize_bvh(text: str) -> Tuple[List[List[str]], ...]:
    """Split BVH text into hierarchy tokens and frame token lists."""
    hierarchy_lines: List[List[str]] = []
    motion_lines: List[List[str]] = []
    in_motion = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tokens = re.split(r"\s+", line)
        if tokens[0] == "MOTION":
            in_motion = True
            continue
        if in_motion:
            motion_lines.append(tokens)
        else:
            hierarchy_lines.append(tokens)
    return hierarchy_lines, motion_lines


def parse_bvh(text: str) -> BVHData:
    """Parse a BVH file into joint hierarchy + motion frames."""
    if not text or not text.strip():
        raise ValueError("BVH text is empty")
    if "HIERARCHY" not in text:
        raise ValueError("BVH missing HIERARCHY section — not a valid BVH file")
    data = BVHData()
    hierarchy_lines, motion_lines = _tokenize_bvh(text)

    # ── Parse hierarchy ──
    joint_stack: List[int] = []  # stack of joint indices
    frame_time = 1.0 / 30.0

    for tokens in hierarchy_lines:
        if not tokens:
            continue
        key = tokens[0]
        if key in ("ROOT", "JOINT"):
            name = tokens[1] if len(tokens) > 1 else f"joint_{len(data.joints)}"
            # De-duplicate names (some BVH editors produce duplicates)
            if name in data.name_to_idx:
                name = f"{name}_{len(data.joints)}"
            j = BVHJoint(name, np.zeros(3), [])
            idx = len(data.joints)
            data.joints.append(j)
            data.name_to_idx[name] = idx
            if joint_stack:
                j.parent = joint_stack[-1]
                j.depth = data.joints[j.parent].depth + 1
                data.joints[j.parent].children.append(idx)
        elif key == "End":
            # End Site — create a synthetic leaf joint named "<parent>_EndSite"
            pidx = joint_stack[-1] if joint_stack else None
            pname = data.joints[pidx].name if pidx is not None else "root"
            name = f"{pname}_EndSite"
            if name in data.name_to_idx:
                name = f"{name}_{len(data.joints)}"
            j = BVHJoint(name, np.zeros(3), [])
            idx = len(data.joints)
            data.joints.append(j)
            data.name_to_idx[name] = idx
            if pidx is not None:
                j.parent = pidx
                j.depth = data.joints[pidx].depth + 1
                data.joints[pidx].children.append(idx)
        elif key == "OFFSET":
            if joint_stack:
                off = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])],
                               dtype=np.float64)
                # The last-added joint is the current leaf
                data.joints[-1].offset = off
        elif key == "CHANNELS":
            n = int(tokens[1])
            chans = tokens[2:2 + n]
            if data.joints:
                data.joints[-1].channels = chans
        elif key == "{":
            # The joint that was just declared becomes the current scope
            if data.joints:
                joint_stack.append(len(data.joints) - 1)
        elif key == "}":
            if joint_stack:
                joint_stack.pop()
        elif key == "Frames:":
            int(tokens[1])
        elif key == "Frame" and len(tokens) >= 3 and tokens[1] == "Time:":
            frame_time = float(tokens[2])

    # Compute rest-pose world positions (accumulate offsets down the tree)
    def _walk_rest(idx: int, accum: np.ndarray):
        data.joints[idx].rest_pos = accum + data.joints[idx].offset
        for c in data.joints[idx].children:
            _walk_rest(c, data.joints[idx].rest_pos)

    roots = [i for i, j in enumerate(data.joints) if j.parent is None]
    for r in roots:
        _walk_rest(r, np.zeros(3, dtype=np.float64))

    # ── Parse motion frames ──
    frame_rows = [row for row in motion_lines
                  if row and row[0] not in ("Frames:", "Frame")
                  and len(row) > 1]
    if frame_rows:
        # Validate against channel count from the hierarchy so downstream FK
        # doesn't IndexError on malformed BVHs.
        total_channels = sum(len(j.channels) for j in data.joints)
        first_width = len(frame_rows[0])
        if total_channels > 0 and first_width < total_channels:
            raise ValueError(
                f"BVH motion frame has {first_width} values but hierarchy "
                f"declares {total_channels} channels — file is truncated or "
                f"malformed."
            )
        # Pad any short rows with zeros (some BVH writers drop trailing zeros).
        normalized = []
        for row in frame_rows:
            if len(row) < total_channels:
                row = row + ["0.0"] * (total_channels - len(row))
            normalized.append([float(v) for v in row[:total_channels]])
        data.frames = np.array(normalized, dtype=np.float32)
    data.fps = 1.0 / frame_time if frame_time > 0 else 30.0
    return data


# ──────────────────────────────────────────────────────────────────────────
# Forward kinematics — compute world positions for each frame.
# ──────────────────────────────────────────────────────────────────────────

def _euler_to_matrix(angles_rad: np.ndarray, order: str) -> np.ndarray:
    """Convert Euler angles (3,) to a 3×3 rotation matrix.
    order = e.g. 'ZYX' (intrinsic) matching BVH channel order."""
    c, s = np.cos(angles_rad), np.sin(angles_rad)
    cx, cy, cz = c
    sx, sy, sz = s
    if order.upper() == "ZYX":
        # Rz @ Ry @ Rx
        return np.array([
            [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
            [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
            [-sy,     sx * cy,                cx * cy],
        ], dtype=np.float64)
    elif order.upper() == "XYZ":
        return np.array([
            [cy * cz, -cy * sz, sy],
            [cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy],
            [sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy],
        ], dtype=np.float64)
    elif order.upper() == "YZX":
        return (np.array([
            [cz, -sz, 0], [0, 1, 0], [sz, 0, cz]
        ]) @ np.array([
            [1, 0, 0], [0, cy, -sy], [0, sy, cy]
        ]) @ np.array([
            [cx, -sx, 0], [sx, cx, 0], [0, 0, 1]
        ]))
    elif order.upper() == "ZXY":
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        return Rz @ Rx @ Ry
    else:
        # Generic fallback: compose in the given order (intrinsic)
        R = np.eye(3)
        axis_map = {"X": np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]),
                    "Y": np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]),
                    "Z": np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])}
        for ax in reversed(order.upper()):
            R = R @ axis_map[ax]
        return R


def forward_kinematics_all_frames(bvh: BVHData, max_frames: int = 240) -> np.ndarray:
    """Compute world-space positions for every joint at every frame.
    Returns (T, J, 3) float32 in the BVH's native units (typically cm).

    Handles position channels on any joint (not just the root). When a joint
    has position channels, they REPLACE the rest offset for that frame — this
    is how the SOMA BVH encodes root motion on the Hips joint while keeping a
    static Root wrapper at the origin.
    """
    T = min(bvh.frames.shape[0], max_frames)
    J = len(bvh.joints)
    out = np.zeros((T, J, 3), dtype=np.float64)

    # Pre-extract per-joint channel indices and rotation orders
    chan_layout: List[Tuple[List[int], List[int], str]] = []
    for j in bvh.joints:
        pos_idx: List[int] = []
        rot_idx: List[int] = []
        rot_order = ""
        for ci, ch in enumerate(j.channels):
            if ch.endswith("position"):
                pos_idx.append(ci)
            elif ch.endswith("rotation"):
                rot_idx.append(ci)
                rot_order += ch[0]
        chan_layout.append((pos_idx, rot_idx, rot_order))

    # Per-joint channel column offset (where this joint's channels start in
    # the flattened frame array)
    col_offsets: List[int] = []
    col = 0
    for j in bvh.joints:
        col_offsets.append(col)
        col += len(j.channels)

    # Process each frame
    for t in range(T):
        frame = bvh.frames[t]
        world_pos: List[np.ndarray] = [np.zeros(3)] * J
        world_rot: List[np.ndarray] = [np.eye(3)] * J

        for idx in range(J):
            j = bvh.joints[idx]
            pos_idx, rot_idx, rot_order = chan_layout[idx]
            base = col_offsets[idx]

            # Local rotation from Euler angles
            if rot_idx and rot_order:
                angles = np.array([frame[base + i] for i in rot_idx], dtype=np.float64)
                angles_rad = np.deg2rad(angles)
                local_R = _euler_to_matrix(angles_rad, rot_order)
            else:
                local_R = np.eye(3, dtype=np.float64)

            # Local translation: position channels REPLACE the rest offset
            # when present (BVH convention for joints with position channels).
            # Otherwise use the hierarchy OFFSET (the rest-pose bone vector).
            if pos_idx:
                local_trans = np.array(
                    [frame[base + i] for i in pos_idx], dtype=np.float64,
                )
            else:
                local_trans = j.offset.astype(np.float64)

            if j.parent is None:
                world_pos[idx] = local_trans
                world_rot[idx] = local_R
            else:
                p = j.parent
                world_pos[idx] = world_pos[p] + world_rot[p] @ local_trans
                world_rot[idx] = world_rot[p] @ local_R

        for idx in range(J):
            out[t, idx] = world_pos[idx]

    return out.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────
# Name mapping — source joint name → SOMA-77 joint index.
# ──────────────────────────────────────────────────────────────────────────

# Mixamo / standard BVH naming (Mixamo uses this exact convention).
# Maps source joint name → SOMA-77 index.
# NOTE: "Root" is deliberately NOT mapped — many BVHs (including SOMA's own
# export) wrap the real skeleton in a static Root joint at the origin. Mapping
# it to SOMA index 0 (Hips) would corrupt the scale + centering.
MIXAMO_ALIAS: Dict[str, int] = {
    "Hips": 0, "hip": 0, "pelvis": 0, "Pelvis": 0,
    "Spine": 1, "spine1": 1,
    "Spine1": 2, "spine2": 2,
    "Spine2": 3, "chest": 3, "Chest": 3, "upperChest": 3,
    "Neck": 4, "neck": 4, "neck1": 4,
    "Neck1": 5, "neck2": 5,
    "Head": 6, "head": 6,
    "HeadEnd": 7, "head_end": 7,
    # Left arm (Mixamo convention)
    "LeftShoulder": 11, "LeftShoulderEnd": 11,
    "LeftArm": 12, "leftarm": 12,
    "LeftForeArm": 13, "LeftForeArmEnd": 13,
    "LeftHand": 14, "lefthand": 14,
    # Right arm
    "RightShoulder": 39, "RightShoulderEnd": 39,
    "RightArm": 40,
    "RightForeArm": 41, "RightForeArmEnd": 41,
    "RightHand": 42,
    # Left leg (Mixamo: LeftUpLeg = hip/thigh, LeftLeg = knee/shin)
    "LeftUpLeg": 67, "LeftUpLegEnd": 67,
    "LeftLeg": 68, "LeftLegEnd": 68,
    "LeftFoot": 69, "LeftFootEnd": 69,
    "LeftToeBase": 70, "LeftToe": 70,
    # Right leg
    "RightUpLeg": 72, "RightUpLegEnd": 72,
    "RightLeg": 73, "RightLegEnd": 73,
    "RightFoot": 74, "RightFootEnd": 74,
    "RightToeBase": 75, "RightToe": 75,
}

# CMU / ASF naming convention (mocap). CMU hierarchy has an extra level vs
# SOMA: hipjoint (ball joint at hip socket, offset from root) → femur (zero
# offset from hipjoint — just a rotational joint) → tibia (knee, offset =
# thigh length) → foot → toes.
# So hipjoint and femur are co-located and both map to SOMA LeftLeg (67).
# Whoever appears first in the BVH wins SOMA 67; the other is skipped by the
# `if soma_idx not in mapping.values()` guard in build_name_mapping().
CMU_ALIAS: Dict[str, int] = {
    "lowerback": 1,
    "upperback": 2,
    "thorax": 3,
    "lowerneck": 4,
    "upperneck": 5,
    "head": 6,
    "lclavicle": 11,
    "lhumerus": 12,
    "lradius": 13,
    "lwrist": 14,
    "lhand": 14,
    "rclavicle": 39,
    "rhumerus": 40,
    "rradius": 41,
    "rwrist": 42,
    "rhand": 42,
    "lhipjoint": 67,
    "lfemur": 67,      # co-located with lhipjoint (zero offset in CMU ASF)
    "ltibia": 68,      # knee (SOMA LeftShin)
    "lfoot": 69,       # ankle
    "ltoes": 70,
    "rhipjoint": 72,
    "rfemur": 72,      # co-located with rhipjoint
    "rtibia": 73,      # knee
    "rfoot": 74,
    "rtoes": 75,
}

# Blender naming convention (e.g. "Shoulder.L", "Arm.R")
BLENDER_ALIAS: Dict[str, int] = {
    "hip": 0, "hips": 0,
    "spine": 1, "spine.001": 2, "spine.002": 3,
    "chest": 3,
    "neck": 4, "neck.001": 5,
    "head": 6,
    "shoulder.l": 11, "shoulder.r": 39,
    "arm.l": 12, "upper_arm.l": 12,
    "forearm.l": 13, "fore_arm.l": 13,
    "hand.l": 14, "wrist.l": 14,
    "arm.r": 40, "upper_arm.r": 40,
    "forearm.r": 41, "fore_arm.r": 41,
    "hand.r": 42, "wrist.r": 42,
    "thigh.l": 67, "upper_leg.l": 67,
    "shin.l": 68, "lower_leg.l": 68,
    "foot.l": 69,
    "toe.l": 70,
    "thigh.r": 72, "upper_leg.r": 72,
    "shin.r": 73, "lower_leg.r": 73,
    "foot.r": 74,
    "toe.r": 75,
}

# Combine all aliases. Lowercased keys for case-insensitive lookup.
_ALL_ALIASES: Dict[str, int] = {}
for _alias_table in (MIXAMO_ALIAS, CMU_ALIAS, BLENDER_ALIAS):
    for _k, _v in _alias_table.items():
        _ALL_ALIASES.setdefault(_k.lower(), _v)


def _resolve_name(src_name: str) -> Optional[int]:
    """Resolve a source joint name to a SOMA-77 index via alias tables.

    ALIAS TABLES ARE CONSULTED BEFORE the SOMA-77 exact-name match. This is
    deliberate: SOMA reuses some joint names with DIFFERENT semantics than
    Mixamo/Blender. The catastrophic case is the legs — SOMA's thigh joints
    are named ``LeftLeg``(67)/``RightLeg``(72), but Mixamo's *shin* joints
    are also named ``LeftLeg``/``RightLeg`` (Mixamo's thighs are
    ``LeftUpLeg``/``RightUpLeg``). If the SOMA exact match wins, the source
    shins grab SOMA's thigh indices, the real shins (SOMA 68/73) never map,
    and the legs can never bend — every clip renders as a stick standing
    straight regardless of the motion. Checking aliases first lets
    convention-specific semantics (``LeftLeg``→68, ``RightLeg``→73) win.
    SOMA-native BVHs still resolve: their core-joint names (Hips, Spine,
    Neck, Head, arms) are all present in MIXAMO_ALIAS, and SOMA-only finger
    names (LeftHandThumb1…) hit the exact-match fallback below without
    colliding with anything.
    """
    # Exact alias match (case-sensitive) — convention-specific semantics.
    if src_name in _ALL_ALIASES:
        return _ALL_ALIASES[src_name]
    # Case-insensitive alias match
    low = src_name.lower()
    if low in _ALL_ALIASES:
        return _ALL_ALIASES[low]
    # SOMA-native exact match (case-sensitive) — only reached if no alias
    # matched. Covers SOMA finger/toe names not in the alias tables.
    if src_name in SOMA77_JOINT_NAMES:
        return SOMA77_JOINT_NAMES.index(src_name)
    # Strip common suffixes/prefixes and retry
    cleaned = re.sub(r"[_\-.]\s*$", "", low)
    cleaned = re.sub(r"^\s*[_\-.]", "", cleaned)
    cleaned = re.sub(r"\.001$", "", cleaned)
    if cleaned in _ALL_ALIASES:
        return _ALL_ALIASES[cleaned]
    # Try matching without side suffixes (.L/.R/_L/_R)
    m = re.match(r"^(.+?)[._]([lr])$", low)
    if m:
        base, side = m.group(1), m.group(2)
        side_suffix = "left" if side == "l" else "right"
        for cand in (f"{base}{side_suffix}", f"{side_suffix}{base}"):
            if cand in _ALL_ALIASES:
                return _ALL_ALIASES[cand]
    return None


def build_name_mapping(bvh: BVHData) -> Dict[int, int]:
    """Map source joint indices → SOMA-77 indices.
    Returns {src_joint_idx: soma77_idx}.

    Special cases:
      • "Root" wrapper (SOMA export convention): a static Root joint at the
        origin that wraps the real Hips joint. Skipped when Hips is found.
      • CMU "root": the actual pelvis joint (no wrapper). Mapped to SOMA 0
        when no Hips/Pelvis match exists.
    """
    mapping: Dict[int, int] = {}
    for src_idx, joint in enumerate(bvh.joints):
        soma_idx = _resolve_name(joint.name)
        if soma_idx is not None:
            if soma_idx not in mapping.values():
                mapping[src_idx] = soma_idx

    # If no hip (SOMA 0) was mapped, look for a "root" joint (CMU convention
    # where the root IS the pelvis). Skip joints literally named "Root" when
    # they're static wrappers (offset == 0 AND at the top of the hierarchy).
    if 0 not in mapping.values():
        for src_idx, joint in enumerate(bvh.joints):
            if joint.name.lower() in ("root", "pelvis", "hip") and joint.parent is None:
                # Heuristic: a static wrapper has near-zero offset; a real
                # pelvis has a non-zero Y offset (height from ground).
                if abs(joint.offset[1]) > 1.0 or joint.name.lower() != "root":
                    mapping[src_idx] = 0
                    break
                # CMU root: has position channels carrying motion
                if any(c.endswith("position") for c in joint.channels):
                    mapping[src_idx] = 0
                    break
    else:
        # A hip was mapped — if a "Root" wrapper also resolved to 0 (via the
        # alias table), drop it so it doesn't corrupt centering / scale.
        hip_indices = [idx for idx, sidx in mapping.items() if sidx == 0]
        if len(hip_indices) > 1:
            for idx in hip_indices:
                if bvh.joints[idx].name.lower() == "root":
                    del mapping[idx]

    return mapping


# ──────────────────────────────────────────────────────────────────────────
# Retargeter — the main entry point.
# ──────────────────────────────────────────────────────────────────────────

def _place_soma_frame(
    src_rest: np.ndarray,
    src_pose: np.ndarray,
    soma_to_src: Dict[int, int],
    root_offset: np.ndarray,
) -> np.ndarray:
    """Place one frame of SOMA-77 joints via ROTATION TRANSFER.

    Unlike the old "copy source positions + uniform scale" approach (which
    preserved the source skeleton's proportions/units and produced 7×-long
    legs and 3 m spines), this transfers each bone's *rotation* from the
    source onto the SOMA-77 bind skeleton — so SOMA-77 bone lengths hold by
    construction and only the joint angles come from the mocap.

    Per bone (SOMA parent p → child i):
      • If both p and i are mapped to source joints (sp, si): the bone's world
        rotation is R = rotation(src_rest[sp]→src_rest[si] ,
                                 src_pose[sp]→src_pose[si]).
        Then out[i] = out[p] + R · (SOMA77_NEUTRAL[i] − SOMA77_NEUTRAL[p]).
      • If i is unmapped (fingers, jaw, eyes, secondary spine levels missing
        from the source): it inherits its parent's transferred rotation, so
        the unmapped chain keeps its SOMA-77 bind pose (relaxed hands) while
        following the parent limb — e.g. fingers stay curled-at-rest and ride
        the retargeted wrist. This is what stops jaw/eyes/fingers collapsing
        to a single garbage point.

    `root_offset` is the world position of SOMA joint 0 (hips); the caller
    chooses hip-centered (zeros) or the animated hip height.
    """
    out = np.zeros((77, 3), dtype=np.float64)
    world_R: List[np.ndarray] = [np.eye(3, dtype=np.float64) for _ in range(77)]
    placed = np.zeros(77, dtype=bool)

    out[0] = root_offset
    placed[0] = True
    # world_R[0] stays identity; SOMA-77's only root children are Spine1(1),
    # LeftLeg(67), RightLeg(72) — all mapped for CMU/Mixamo — so each gets its
    # own bone rotation directly and never needs world_R[0].

    # Iterate to convergence (SOMA77_PARENTS is topologically parent<child, so
    # one pass suffices, but loop for safety on any out-of-order edge case).
    for _ in range(6):
        if placed.all():
            break
        for i in range(1, 77):
            if placed[i]:
                continue
            p = SOMA77_PARENTS[i]
            if p < 0 or not placed[p]:
                continue
            si = soma_to_src.get(i)
            sp = soma_to_src.get(p)
            if si is not None and sp is not None:
                d_rest = src_rest[si] - src_rest[sp]
                d_pose = src_pose[si] - src_pose[sp]
                nr = float(np.linalg.norm(d_rest))
                npr = float(np.linalg.norm(d_pose))
                if nr > 1e-6 and npr > 1e-6:
                    R = _rotation_between_vectors(d_rest / nr, d_pose / npr)
                else:
                    R = world_R[p]
            else:
                # Unmapped joint (or mapped joint with unmapped parent, rare):
                # inherit the parent's world rotation → bind local pose rides
                # the retargeted limb. This is the fix for fingers/jaw/eyes.
                R = world_R[p]
            bind = SOMA77_NEUTRAL[i] - SOMA77_NEUTRAL[p]
            out[i] = out[p] + R @ bind
            world_R[i] = R
            placed[i] = True

    # Any still-unplaced joint (disconnected): fall back to absolute neutral.
    for i in range(77):
        if not placed[i]:
            out[i] = SOMA77_NEUTRAL[i]
    return out


def _rotation_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix that maps unit vector a → unit vector b."""
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-9:
        return np.eye(3)
    if c < -1.0 + 1e-9:
        # 180° rotation around any perpendicular axis
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1, 0])
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        return np.eye(3) + 2 * (K @ K)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.eye(3)
    K = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + K + (K @ K) * (1.0 / (1.0 + c))


def retarget_bvh_text(
    bvh_text: str,
    max_frames: int = 240,
    start_at_hips: bool = True,
) -> np.ndarray:
    """Retarget a BVH file's motion to SOMA-77 joint positions.

    Args:
        bvh_text: Raw BVH file contents (UTF-8).
        max_frames: Cap on frames returned (subsamples uniformly if exceeded).
        start_at_hips: If True, translate every frame so the hips sit at the
            SOMA neutral origin (0,0,0) — Pose Studio expects hip-centered poses.

    Returns:
        (T, 77, 3) float32 array of SOMA-77 joint positions in meters.
    """
    bvh = parse_bvh(bvh_text)
    if not bvh.joints:
        raise ValueError("BVH parse produced no joints")

    mapping = build_name_mapping(bvh)
    if len(mapping) < 5:
        raise ValueError(
            f"Could not map BVH joints to SOMA-77 (only {len(mapping)} matched). "
            "Ensure the BVH uses a recognized naming convention "
            "(Mixamo, CMU/ASF, Blender, or SOMA)."
        )

    soma_to_src: Dict[int, int] = {v: k for k, v in mapping.items()}

    # Source rest-pose world positions (bind pose, from hierarchy offsets).
    src_rest = np.array([j.rest_pos for j in bvh.joints])

    # Source animated world positions for all frames (source proportions/units).
    src_world = forward_kinematics_all_frames(bvh, max_frames=max_frames)
    T = src_world.shape[0]

    if T > max_frames:
        idx = np.linspace(0, T - 1, max_frames).astype(int)
        src_world = src_world[idx]
        T = max_frames

    out = np.zeros((T, 77, 3), dtype=np.float32)

    # Hip source index for root placement.
    hip_src = soma_to_src.get(0)

    for t in range(T):
        if start_at_hips:
            root_offset = np.zeros(3, dtype=np.float64)
        elif hip_src is not None:
            # Preserve animated hip height so sitting/jumping reads correctly;
            # scaled by hip→neck ratio to bring source units into metres.
            soma_hip_neck = float(np.linalg.norm(SOMA77_NEUTRAL[4] - SOMA77_NEUTRAL[0]))
            src_hip_neck = (float(np.linalg.norm(src_rest[soma_to_src[4]] - src_rest[hip_src]))
                            if 4 in soma_to_src else 0.0)
            sc = soma_hip_neck / src_hip_neck if src_hip_neck > 1e-6 else 1.0
            root_offset = src_world[t, hip_src] * sc
        else:
            root_offset = np.zeros(3, dtype=np.float64)

        out[t] = _place_soma_frame(src_rest, src_world[t], soma_to_src, root_offset)

    return out


def retarget_single_frame(
    bvh_text: str,
    frame_idx: int = 0,
    start_at_hips: bool = True,
) -> np.ndarray:
    """Retarget a single frame from a BVH to SOMA-77.
    Returns (77, 3) float32."""
    frames = retarget_bvh_text(bvh_text, max_frames=frame_idx + 1,
                                start_at_hips=start_at_hips)
    idx = min(frame_idx, frames.shape[0] - 1)
    return frames[idx]
