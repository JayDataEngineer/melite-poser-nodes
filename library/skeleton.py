"""Canonical SOMA-77 / COCO-18 skeleton topology — the single source of truth.

Every other module that needs joint names, parent indices, neutral joint
positions, or the SOMA↔COCO joint mapping must import from here. Do NOT
re-declare these constants elsewhere — the cross-reference test
(``tests/test_skeleton_consistency.py``) enforces this.

Sources:
  - SOMA-77 ordering: ``vendor/kimodo/kimodo/skeleton/definitions.py``
    (``SOMASkeleton77.bone_order_names_with_parents`` — verified 77 entries).
  - Neutral joint positions: meters, T-pose, hips at origin. Mirrors the
    bind pose baked into the SOMA skin mesh npz.
  - COCO-18 ordering: matches ``poser.py::_KP_ORDER`` (the DWPose / vnccs
    contract). 16 of the 18 COCO joints have direct SOMA counterparts; the
    two ear joints are extrapolated from eye + head geometry (see
    ``gateway/poser_soma.py``).

Coordinate conventions:
  SOMA frame: +X = subject's left, +Y = up, +Z = forward (subject faces +Z)
  Image frame: 0,0 = top-left, +X = right, +Y = down
  Subject's right = image LEFT (COCO convention, matches ``poser.py``).
"""
from __future__ import annotations

from typing import Final

import numpy as np

__all__ = [
    "SOMA77_JOINT_NAMES",
    "SOMA77_IDX",
    "SOMA77_PARENTS",
    "SOMA77_NEUTRAL",
    "SOMA77_APOSE",
    "compute_apose",
    "COCO18_JOINT_NAMES",
    "SOMA_TO_COCO_DIRECT",
    "COCO_TO_SOMA_IDX",
    "DIRECT_COCO_TO_SOMA_INDICES",
]


# ── SOMA-77 joint names (canonical ordering) ───────────────────────────────
SOMA77_JOINT_NAMES: Final[tuple[str, ...]] = (
    "Hips",                                                                          # 0
    "Spine1", "Spine2", "Chest",                                                     # 1-3
    "Neck1", "Neck2", "Head", "HeadEnd",                                            # 4-7
    "Jaw", "LeftEye", "RightEye",                                                    # 8-10
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",                           # 11-14
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",       # 15-18
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",     # 19-23
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd", # 24-28
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",         # 29-33
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",    # 34-38
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",                       # 39-42
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",   # 43-46
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",     # 47-51
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd", # 52-56
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",         # 57-61
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",    # 62-66
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",                # 67-71
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",           # 72-76
)

SOMA77_IDX: Final[dict[str, int]] = {name: i for i, name in enumerate(SOMA77_JOINT_NAMES)}


# ── SOMA-77 parent topology (-1 = root) ────────────────────────────────────
SOMA77_PARENTS: Final[tuple[int, ...]] = (
    -1,   # 0  Hips
    0,    # 1  Spine1
    1,    # 2  Spine2
    2,    # 3  Chest
    3,    # 4  Neck1
    4,    # 5  Neck2
    5,    # 6  Head
    6,    # 7  HeadEnd
    6,    # 8  Jaw
    6,    # 9  LeftEye
    6,    # 10 RightEye
    3,    # 11 LeftShoulder
    11,   # 12 LeftArm
    12,   # 13 LeftForeArm
    13,   # 14 LeftHand
    14,   # 15 LeftHandThumb1
    15,   # 16 LeftHandThumb2
    16,   # 17 LeftHandThumb3
    17,   # 18 LeftHandThumbEnd
    14,   # 19 LeftHandIndex1
    19,   # 20 LeftHandIndex2
    20,   # 21 LeftHandIndex3
    21,   # 22 LeftHandIndex4
    22,   # 23 LeftHandIndexEnd
    14,   # 24 LeftHandMiddle1
    24,   # 25 LeftHandMiddle2
    25,   # 26 LeftHandMiddle3
    26,   # 27 LeftHandMiddle4
    27,   # 28 LeftHandMiddleEnd
    14,   # 29 LeftHandRing1
    29,   # 30 LeftHandRing2
    30,   # 31 LeftHandRing3
    31,   # 32 LeftHandRing4
    32,   # 33 LeftHandRingEnd
    14,   # 34 LeftHandPinky1
    34,   # 35 LeftHandPinky2
    35,   # 36 LeftHandPinky3
    36,   # 37 LeftHandPinky4
    37,   # 38 LeftHandPinkyEnd
    3,    # 39 RightShoulder
    39,   # 40 RightArm
    40,   # 41 RightForeArm
    41,   # 42 RightHand
    42,   # 43 RightHandThumb1
    43,   # 44 RightHandThumb2
    44,   # 45 LeftHandThumb3
    45,   # 46 LeftHandThumbEnd
    42,   # 47 RightHandIndex1
    47,   # 48 RightHandIndex2
    48,   # 49 RightHandIndex3
    49,   # 50 RightHandIndex4
    50,   # 51 RightHandIndexEnd
    42,   # 52 RightHandMiddle1
    52,   # 53 RightHandMiddle2
    53,   # 54 RightHandMiddle3
    54,   # 55 RightHandMiddle4
    55,   # 56 RightHandMiddleEnd
    42,   # 57 RightHandRing1
    57,   # 58 RightHandRing2
    58,   # 59 RightHandRing3
    59,   # 60 RightHandRing4
    60,   # 61 RightHandRingEnd
    42,   # 62 RightHandPinky1
    62,   # 63 RightHandPinky2
    63,   # 64 RightHandPinky3
    64,   # 65 RightHandPinky4
    65,   # 66 RightHandPinkyEnd
    0,    # 67 LeftLeg
    67,   # 68 LeftShin
    68,   # 69 LeftFoot
    69,   # 70 LeftToeBase
    70,   # 71 LeftToeEnd
    0,    # 72 RightLeg
    72,   # 73 RightShin
    73,   # 74 RightFoot
    74,   # 75 RightToeBase
    75,   # 76 RightToeEnd
)


# ── SOMA-77 neutral joint positions (T-pose, meters, hips at origin) ───────
SOMA77_NEUTRAL: Final[np.ndarray] = np.array([
    [0.000000, 0.000000, 0.000000],          # 0  Hips
    [-0.000137, 0.050038, -0.000537],        # 1  Spine1
    [-0.000137, 0.121291, -0.000836],        # 2  Spine2
    [-0.000137, 0.196791, -0.008995],        # 3  Chest
    [-0.001954, 0.459904, -0.014529],        # 4  Neck1
    [-0.001954, 0.536998, 0.008497],         # 5  Neck2
    [-0.001954, 0.598287, 0.028034],         # 6  Head
    [-0.001918, 0.758941, 0.009680],         # 7  HeadEnd
    [-0.001928, 0.603043, 0.058984],         # 8  Jaw
    [0.030110, 0.652089, 0.103903],          # 9  LeftEye
    [-0.034179, 0.651906, 0.103617],         # 10 RightEye
    [0.016079, 0.429163, 0.042139],          # 11 LeftShoulder
    [0.165278, 0.429163, -0.012884],         # 12 LeftArm
    [0.452671, 0.429163, -0.012910],         # 13 LeftForeArm
    [0.723611, 0.429163, -0.012884],         # 14 LeftHand
    [0.746375, 0.415242, 0.019030],          # 15 LeftHandThumb1
    [0.786504, 0.396961, 0.035447],          # 16 LeftHandThumb2
    [0.814489, 0.396961, 0.035447],          # 17 LeftHandThumb3
    [0.846297, 0.396961, 0.035447],          # 18 LeftHandThumbEnd
    [0.756086, 0.423843, 0.010078],          # 19 LeftHandIndex1
    [0.819732, 0.423964, 0.011864],          # 20 LeftHandIndex2
    [0.856356, 0.423964, 0.011864],          # 21 LeftHandIndex3
    [0.879648, 0.423964, 0.011864],          # 22 LeftHandIndex4
    [0.907244, 0.422158, 0.010733],          # 23 LeftHandIndexEnd
    [0.755246, 0.431573, -0.002881],         # 24 LeftHandMiddle1
    [0.817153, 0.428980, -0.012906],         # 25 LeftHandMiddle2
    [0.860719, 0.428980, -0.012906],         # 26 LeftHandMiddle3
    [0.890687, 0.428980, -0.012906],         # 27 LeftHandMiddle4
    [0.913730, 0.426034, -0.013224],         # 28 LeftHandMiddleEnd
    [0.752437, 0.428626, -0.016110],         # 29 LeftHandRing1
    [0.810982, 0.423764, -0.029848],         # 30 LeftHandRing2
    [0.854488, 0.423764, -0.029848],         # 31 LeftHandRing3
    [0.881001, 0.423764, -0.029848],         # 32 LeftHandRing4
    [0.900362, 0.424541, -0.029849],         # 33 LeftHandRingEnd
    [0.752266, 0.426063, -0.028888],         # 34 LeftHandPinky1
    [0.803144, 0.412751, -0.046600],         # 35 LeftHandPinky2
    [0.833854, 0.412752, -0.046600],         # 36 LeftHandPinky3
    [0.849351, 0.412752, -0.046600],         # 37 LeftHandPinky4
    [0.868799, 0.411173, -0.046028],         # 38 LeftHandPinkyEnd
    [-0.013938, 0.428594, 0.043146],         # 39 RightShoulder
    [-0.164310, 0.428594, -0.012310],        # 40 RightArm
    [-0.451677, 0.428594, -0.012336],        # 41 RightForeArm
    [-0.723013, 0.428594, -0.012310],        # 42 RightHand
    [-0.745753, 0.414755, 0.019322],         # 43 RightHandThumb1
    [-0.785868, 0.396480, 0.035731],         # 44 RightHandThumb2
    [-0.813817, 0.396480, 0.035731],         # 45 RightHandThumb3
    [-0.845655, 0.396480, 0.035731],         # 46 RightHandThumbEnd
    [-0.755546, 0.423394, 0.010519],         # 47 RightHandIndex1
    [-0.818965, 0.423519, 0.012302],         # 48 RightHandIndex2
    [-0.855514, 0.423519, 0.012302],         # 49 RightHandIndex3
    [-0.878789, 0.423519, 0.012302],         # 50 RightHandIndex4
    [-0.906407, 0.421712, 0.011171],         # 51 RightHandIndexEnd
    [-0.754694, 0.431060, -0.002299],        # 52 RightHandMiddle1
    [-0.816502, 0.428472, -0.012308],        # 53 RightHandMiddle2
    [-0.859991, 0.428472, -0.012308],        # 54 RightHandMiddle3
    [-0.889994, 0.428472, -0.012308],        # 55 RightHandMiddle4
    [-0.913019, 0.425528, -0.012625],        # 56 RightHandMiddleEnd
    [-0.751870, 0.427915, -0.015398],        # 57 RightHandRing1
    [-0.810412, 0.423054, -0.029135],        # 58 RightHandRing2
    [-0.853800, 0.423054, -0.029135],        # 59 RightHandRing3
    [-0.880349, 0.423054, -0.029135],        # 60 RightHandRing4
    [-0.899685, 0.423829, -0.029136],        # 61 RightHandRingEnd
    [-0.751677, 0.425167, -0.028151],        # 62 RightHandPinky1
    [-0.802591, 0.411846, -0.045875],        # 63 RightHandPinky2
    [-0.833218, 0.411846, -0.045875],        # 64 RightHandPinky3
    [-0.848683, 0.411846, -0.045875],        # 65 RightHandPinky4
    [-0.868134, 0.410269, -0.045303],        # 66 RightHandPinkyEnd
    [0.100432, -0.084345, 0.025957],         # 67 LeftLeg
    [0.100432, -0.516563, 0.017927],         # 68 LeftShin
    [0.100432, -0.938114, -0.016888],        # 69 LeftFoot
    [0.100432, -0.988708, 0.115427],         # 70 LeftToeBase
    [0.100336, -1.005185, 0.180558],         # 71 LeftToeEnd
    [-0.100473, -0.082953, 0.026203],        # 72 RightLeg
    [-0.100473, -0.516575, 0.018148],        # 73 RightShin
    [-0.100473, -0.937749, -0.016636],       # 74 RightFoot
    [-0.100473, -0.988545, 0.116206],        # 75 RightToeBase
    [-0.100377, -1.004888, 0.180812],        # 76 RightToeEnd
], dtype=np.float32)


# ── A-pose: T-pose neutral with arms rotated down from the shoulder sockets ─
# The SOMA NPZ engine's rest pose has ~43° elbow bend (a statistical mean
# from MoCap data). SOMA77_NEUTRAL above is a textbook T-pose (zero elbow
# bend, all arm joints at the same Y). But a T-pose has arms straight out
# horizontally — unusual for character generation. The A-pose rotates each
# arm chain down from the shoulder socket, keeping the elbows STRAIGHT (zero
# bend) while putting the arms in a natural resting position.
#
# Joint indices rotated:
#   Left arm:  12 (LeftArm) through 38 (LeftHandPinkyEnd) — pivot at joint 12
#   Right arm: 40 (RightArm) through 66 (RightHandPinkyEnd) — pivot at joint 40
# The clavicle joints (11, 39) are NOT rotated — they stay at shoulder height.

def compute_apose(
    neutral: np.ndarray,
    arm_drop_deg: float = 45.0,
) -> np.ndarray:
    """Convert the T-pose neutral into an A-pose by rotating arms down.

    Pivots at the shoulder sockets (joint 12 = LeftArm, joint 40 = RightArm)
    and rotates each entire arm chain (elbow, wrist, fingers) around the Z
    axis. The clavicle and torso are untouched.

    Args:
        neutral: (77, 3) T-pose joint positions in meters (SOMA77_NEUTRAL).
        arm_drop_deg: Degrees to lower each arm from horizontal.
            0 = T-pose (arms straight out), 90 = arms hanging straight down.
            Default 45 — a natural A-pose with arms at 45° below horizontal.

    Returns:
        (77, 3) A-pose joint positions in meters. Arms are perfectly straight
        (zero elbow bend), just rotated down from the shoulder.
    """
    apose = neutral.copy()
    theta = np.radians(arm_drop_deg)
    c, s = np.cos(theta), np.sin(theta)

    # Left arm: extends in +X. Rotate CLOCKWISE around Z (−theta) to drop it.
    # R_z(-theta) = [[c,  s, 0], [-s, c, 0], [0, 0, 1]]
    pivot_L = neutral[12].copy()  # LeftArm = shoulder socket
    for i in range(12, 39):       # LeftArm → LeftHandPinkyEnd
        local = neutral[i] - pivot_L
        apose[i, 0] = pivot_L[0] + c * local[0] + s * local[1]
        apose[i, 1] = pivot_L[1] - s * local[0] + c * local[1]
        # z unchanged (rotation is in XY plane)

    # Right arm: extends in -X. Rotate COUNTER-CLOCKWISE around Z (+theta).
    # R_z(+theta) = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    pivot_R = neutral[40].copy()  # RightArm = shoulder socket
    for i in range(40, 67):       # RightArm → RightHandPinkyEnd
        local = neutral[i] - pivot_R
        apose[i, 0] = pivot_R[0] + c * local[0] - s * local[1]
        apose[i, 1] = pivot_R[1] + s * local[0] + c * local[1]

    return apose


# ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
# ⚠️  DEPRECATED — DO NOT USE FOR REFERENCE GENERATION  (2026-07-30)
# ⚠️
# ⚠️  SOMA77_APOSE is a SYNTHETIC 45° arm-drop with STRAIGHT ELBOWS. It does
# ⚠️  NOT match the real SOMAX template bind pose in skin_standard.npz.
# ⚠️
# ⚠️  Real template pose (measured from bind_rig_transform):
# ⚠️     Upper arm drop: 51.2°   (this A-pose: 45.0°)   → 6.2° ERROR
# ⚠️     Forearm drop:   35.4°   (this A-pose: 45.0°)   → 9.6° ERROR
# ⚠️     Elbow bend:     25.2°   (this A-pose:  0.0°)   → 25.2° ERROR ← huge
# ⚠️
# ⚠️  Using this pose for openpose/depth/weight-transfer references causes
# ⚠️  body horror: armpits static, hands mangled, face tilts up.
# ⚠️
# ⚠️  The canonical pose lives in:
# ⚠️     /opt/kimodo/kimodo/assets/skeletons/somaskel77/skin_standard.npz
# ⚠️  Load it via PoserTemplateBind / _load_template_bind() in nodes.py.
# ⚠️
# ⚠️  Full reference: docs/CANONICAL-SOMAX-POSE.md
# ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
SOMA77_APOSE: Final[np.ndarray] = compute_apose(SOMA77_NEUTRAL, arm_drop_deg=45.0)


# ── COCO-18 joint ordering (must match poser.py::_KP_ORDER) ────────────────
COCO18_JOINT_NAMES: Final[list[str]] = [
    "nose", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
]


# ── Direct SOMA→COCO joint mapping (16 of 18 joints) ───────────────────────
# Ears are NOT in this table — they're extrapolated separately by
# ``poser_soma.extrapolate_ears`` (SOMA-77 has no ear joints).
SOMA_TO_COCO_DIRECT: Final[dict[str, str]] = {
    "nose":        "Head",
    "neck":        "Neck1",
    # COCO "shoulder" = the glenohumeral joint (where the humerus meets the
    # scapula) = SOMA LeftArm/RightArm — the actual shoulder SOCKET. It must
    # NOT be mapped to LeftShoulder/RightShoulder, which are the CLAVICLE
    # joints sitting beside the neck (X≈1.5cm vs the socket at X≈16cm).
    # Using the clavicle makes the upper-arm line (clavicle→elbow) land at
    # ~37° below horizontal — almost identical to the forearm's ~35° — so the
    # 25° elbow bend is geometrically invisible and the arm renders STRAIGHT
    # (A-pose). The socket gives the true 51° upper-arm drop + 16° visible
    # elbow kink that defines the SOMAX canon rest pose. See
    # docs/CANONICAL-SOMAX-POSE.md §2.1.
    "r_shoulder":  "RightArm",        # subject's right = SOMA -X side (socket)
    "r_elbow":     "RightForeArm",
    "r_wrist":     "RightHand",
    "l_shoulder":  "LeftArm",         # subject's left = SOMA +X side (socket)
    "l_elbow":     "LeftForeArm",
    "l_wrist":     "LeftHand",
    "r_hip":       "RightLeg",        # SOMA "Leg" = COCO "hip" (upper leg)
    "r_knee":      "RightShin",
    "r_ankle":     "RightFoot",
    "l_hip":       "LeftLeg",
    "l_knee":      "LeftShin",
    "l_ankle":     "LeftFoot",
    "r_eye":       "RightEye",
    "l_eye":       "LeftEye",
}


def _build_coco_to_soma_idx() -> dict[int, int]:
    """COCO-18 index → SOMA-77 index, for the 16 directly-mapped joints.

    Inverse of ``SOMA_TO_COCO_DIRECT``. Indices 16 and 17 (ears) are absent
    — callers fill them via ear extrapolation.
    """
    return {
        coco_i: SOMA77_IDX[soma_name]
        for coco_i, coco_name in enumerate(COCO18_JOINT_NAMES[:16])
        for soma_name in (SOMA_TO_COCO_DIRECT[coco_name],)
    }


# COCO-18 index → SOMA-77 index (ears excluded). Replaces the previously
# hardcoded ``_COCO_TO_SOMA`` table in ``gateway/poser_skin.py``.
COCO_TO_SOMA_IDX: Final[dict[int, int]] = _build_coco_to_soma_idx()

# COCO-18 index → SOMA-77 index, as a positional list (ears excluded).
# Convenience for arithmetic where a list subscript is cheaper than a dict
# lookup. Matches the legacy ``_DIRECT_SOMA_INDICES`` in ``poser_soma.py``.
DIRECT_COCO_TO_SOMA_INDICES: Final[list[int]] = [
    COCO_TO_SOMA_IDX[i] for i in range(16)
]


# ── Integrity asserts — fail fast at import if the topology is edited wrong ─
assert len(SOMA77_JOINT_NAMES) == 77, (
    f"Expected 77 SOMA joints, got {len(SOMA77_JOINT_NAMES)}"
)
assert len(set(SOMA77_JOINT_NAMES)) == 77, "SOMA-77 joint names must be unique"
assert len(SOMA77_PARENTS) == 77, (
    f"Expected 77 parent entries, got {len(SOMA77_PARENTS)}"
)
assert SOMA77_NEUTRAL.shape == (77, 3), (
    f"Expected SOMA77_NEUTRAL shape (77, 3), got {SOMA77_NEUTRAL.shape}"
)
assert SOMA77_APOSE.shape == (77, 3), (
    f"Expected SOMA77_APOSE shape (77, 3), got {SOMA77_APOSE.shape}"
)
# A-pose wrists must be BELOW the T-pose wrists (arms dropped).
# T-pose wrist Y = 0.429 (same as shoulder). A-pose wrist Y should be
# significantly lower. This catches accidental sign errors in the rotation.
assert SOMA77_APOSE[42, 1] < SOMA77_NEUTRAL[42, 1] - 0.1, (
    "A-pose right wrist must be >10cm below T-pose wrist — rotation sign error?"
)
assert all(-1 <= p < 77 for p in SOMA77_PARENTS), "Parent index out of range"
assert sum(1 for p in SOMA77_PARENTS if p == -1) == 1, "Exactly one root (Hips)"
assert len(COCO18_JOINT_NAMES) == 18, (
    f"Expected 18 COCO joints, got {len(COCO18_JOINT_NAMES)}"
)
assert len(SOMA_TO_COCO_DIRECT) == 16, "16 direct SOMA→COCO mappings expected"
assert set(SOMA_TO_COCO_DIRECT.values()) <= set(SOMA77_JOINT_NAMES), (
    "SOMA_TO_COCO_DIRECT references unknown SOMA joint"
)
assert set(SOMA_TO_COCO_DIRECT) == set(COCO18_JOINT_NAMES[:16]), (
    "SOMA_TO_COCO_DIRECT keys must cover COCO-18 excluding ears"
)
assert set(COCO_TO_SOMA_IDX.values()) == {
    SOMA77_IDX[n] for n in SOMA_TO_COCO_DIRECT.values()
}, "COCO_TO_SOMA_IDX must be the exact inverse of SOMA_TO_COCO_DIRECT"
