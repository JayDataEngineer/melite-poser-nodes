"""Runtime pose-metric measurements from skin_standard.npz.

Suggestion #3 (2026-07-31): replaces the hardcoded ``"arm_drop_deg": 51.2,
"elbow_bend_deg": 25.2`` constants in the family-function metrics dicts
with values MEASURED from the actual template bind pose. If anyone swaps
in a different npz, the metrics will reflect that — no more "trust me"
numbers that can silently drift from the rendered geometry.

MIRRORED CALL SITES:
  * media/comfyui/families/poser_template_views.py — runtime, loads
    this module via importlib (melite-head can't import the hyphenated
    nodepack dir as a package).
  * tests/unit/test_somax_pose_alignment.py — TestCanonPose previously
    inlined this math; now calls these helpers so the test and the
    runtime CANNOT drift apart.

The functions take a (77, 3) positions-cm array + a list of 77 SOMA
joint names (both straight from the npz). They return Python floats so
the metrics dict serializes cleanly to JSON.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _side_names(side: str) -> tuple[str, str, str]:
    """Return (arm_joint, forearm_joint, hand_joint) names for the side.

    SOMA joint naming convention: ``Left``/``Right`` prefix + segment
    name. ``side`` accepts "left", "right", "Left", "Right" (case-insensitive).
    """
    s = side.lower()
    if s == "left":
        return "LeftArm", "LeftForeArm", "LeftHand"
    if s == "right":
        return "RightArm", "RightForeArm", "RightHand"
    raise ValueError(
        f"side must be 'left' or 'right', got {side!r}. SOMA joints are "
        f"named LeftArm/LeftForeArm/LeftHand + Right*/... — no other sides."
    )


def _joint_index(joint_names: Iterable[str], name: str) -> int:
    """Find the index of ``name`` in ``joint_names``; raise clearly if absent.

    A missing joint name usually means the npz was swapped for one with a
    different naming convention (e.g. ``L_Arm`` instead of ``LeftArm``).
    Fail loud — silent fallback to index 0 would report garbage metrics.
    """
    names = list(joint_names)
    try:
        return names.index(name)
    except ValueError:
        raise KeyError(
            f"joint {name!r} not found in joint_names (have "
            f"{names[:5]!r}...). The npz naming convention may have changed."
        )


def upper_arm_drop_deg(
    positions_cm: np.ndarray, joint_names: Iterable[str], side: str = "left"
) -> float:
    """3D angle of the upper-arm segment from horizontal. Negative = down.

    SOMAX canon ≈ -51.2° (arm drops below horizontal). T-pose = 0°.
    A-pose = -90° (straight down).

    The angle is computed in the (X, Y, Z) world frame:
      v = elbow - shoulder
      horizontal_extent = sqrt(v.x² + v.z²)
      angle = atan2(v.y, horizontal_extent)

    Positive Y is up in the SOMA convention, so a downward-dropping arm
    yields a negative angle.
    """
    arm_name, forearm_name, _ = _side_names(side)
    idx = {n: i for i, n in enumerate(joint_names)}
    shoulder = positions_cm[idx[arm_name]]
    elbow = positions_cm[idx[forearm_name]]
    v = elbow - shoulder
    horizontal = math.sqrt(float(v[0]) ** 2 + float(v[2]) ** 2)
    return math.degrees(math.atan2(float(v[1]), horizontal))


def elbow_bend_deg(
    positions_cm: np.ndarray, joint_names: Iterable[str], side: str = "left"
) -> float:
    """3D elbow bend angle (unsigned). 0° = straight, SOMAX canon ≈ 25°.

    Angle between the upper-arm segment (shoulder→elbow) and the forearm
    segment (elbow→hand), measured AT the elbow joint. ``abs(dot)`` is
    NOT used — this is the true 3D joint angle, not the 2D projection
    kink. The 2D front-view kink is smaller (~12°) because the canon
    bend is partly out of plane.
    """
    arm_name, forearm_name, hand_name = _side_names(side)
    idx = {n: i for i, n in enumerate(joint_names)}
    shoulder = positions_cm[idx[arm_name]]
    elbow = positions_cm[idx[forearm_name]]
    hand = positions_cm[idx[hand_name]]
    v1 = elbow - shoulder
    v2 = hand - elbow
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = float(np.dot(v1, v2)) / (n1 * n2)
    return math.degrees(math.acos(min(1.0, max(-1.0, cos_a))))


def load_template_metrics(skin_path) -> dict[str, float]:
    """Load skin_standard.npz and measure the canon pose metrics.

    This is the ONE call site family functions use to replace the old
    hardcoded ``"arm_drop_deg": 51.2, "elbow_bend_deg": 25.2`` values.
    Loads the npz, extracts joint positions in cm, measures left + right
    sides, and returns a flat metrics dict.

    Args:
        skin_path: path-like to skin_standard.npz.

    Returns:
        Dict with four keys: ``arm_drop_deg_left``, ``arm_drop_deg_right``,
        ``elbow_bend_deg_left``, ``elbow_bend_deg_right``. All Python floats.
        The family function picks the left-side values for the top-level
        ``arm_drop_deg`` / ``elbow_bend_deg`` metrics (the SOMAX template
        is symmetric to within 1e-3°, so side doesn't matter — but
        reporting both makes asymmetry visible if a future npz breaks it).
    """
    d = np.load(skin_path, allow_pickle=True)
    names = list(d["rig_joint_names"])
    # bind_rig_transform is (77, 4, 4); the joint position is the
    # translation column [:3, 3]. npz stores METERS — convert to cm to
    # match the renderer's convention.
    pos_cm = d["bind_rig_transform"][:, :3, 3].astype(np.float32) * 100.0
    return {
        "arm_drop_deg_left": upper_arm_drop_deg(pos_cm, names, "left"),
        "arm_drop_deg_right": upper_arm_drop_deg(pos_cm, names, "right"),
        "elbow_bend_deg_left": elbow_bend_deg(pos_cm, names, "left"),
        "elbow_bend_deg_right": elbow_bend_deg(pos_cm, names, "right"),
    }
