"""Fetch a curated set of REAL CMU Motion Capture Database clips into the
Pose Studio corpus, retargeted to SOMA-77.

This is a BUILD-TIME tool. It downloads BVH files from the CMU mocap database
(via the Shriinivas/cmubvh GitHub mirror — every file is genuine CMU capture
data, the canonical free academic mocap catalog at mocap.cs.cmu.edu), retargets
each to SOMA-77 via the pack's vendored retargeter
(``custom_nodes/melite-poser-nodes/library/bvh_retarget.py``), and writes them
into ``pose_corpus_cache/`` alongside the Kimodo cache entries.

Run (from anywhere; the script resolves paths from its own location):
    python custom_nodes/melite-poser-nodes/tools/fetch_cmu_corpus.py

Provenance for every entry is recorded in the manifest's ``source`` field, so
the UI can show exactly where each motion came from. Nothing here is
synthesized — every frame is real human motion capture.
"""
from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from library.bvh_retarget import retarget_bvh_text  # noqa: E402

CACHE = PACK / "library" / "pose_corpus_cache"
CACHE.mkdir(parents=True, exist_ok=True)
MANIFEST = CACHE / "manifest.json"

# SOMA-77 reference convention: the Kimodo-lifted motions place the hip at
# y≈1.0 with a hip→head bone-chain distance of ~0.60m (head ≈ 1.6m, feet ≈ 0).
# CMU BVH data ships at a different unit scale, so each clip is uniformly
# rescaled to match the SOMA-77 hip→head length and re-rooted so the hip
# sits at y=1.0 — same convention as the rest of the corpus.
TARGET_HIP_Y = 1.0
TARGET_HIP_TO_HEAD = 0.60
HIP_IDX = 0
HEAD_IDX = 6


def _normalize_to_soma_scale(frames: np.ndarray) -> np.ndarray:
    """Rescale a (T, 77, 3) clip to the SOMA-77 convention.

    CMU BVH units differ from SOMA-77 meters; without this, head sits at ~0.8m
    instead of ~1.6m and the plausibility checks fail. Uniform scale by the
    hip→head ratio, then translate so the hip is at y=1.0 (root offset matches
    the Kimodo-lifted clips so the two sources are interchangeable in the UI).
    """
    hip = frames[:, HIP_IDX, :]                       # (T, 3)
    head = frames[:, HEAD_IDX, :]
    dist = float(np.linalg.norm(head - hip, axis=-1).mean())
    if dist < 1e-6:
        return frames
    scale = TARGET_HIP_TO_HEAD / dist
    centered = (frames - hip[:, None, :]) * scale     # re-root at origin
    return centered + np.array([0.0, TARGET_HIP_Y, 0.0])  # hip → y=1.0

RAW_BASE = "https://github.com/Shriinivas/cmubvh/raw/main"

# Map subject number → Sequence directory name in the cmubvh repo.
def _seq_dir(subject: int) -> str:
    ranges = [
        (9, "Sequence-001-009"), (14, "Sequence-010-014"), (19, "Sequence-015-019"),
        (29, "Sequence-020-029"), (34, "Sequence-030-034"), (39, "Sequence-035-039"),
        (45, "Sequence-040-045"), (56, "Sequence-046-056"), (75, "Sequence-060-075"),
        (85, "Sequence-081-085"), (94, "Sequence-086-094"), (111, "Sequence-102-111"),
        (128, "Sequence-113-128"),
    ]
    for hi, name in ranges:
        if subject <= hi:
            return name
    raise ValueError(f"No sequence directory for subject {subject}")

# Curated CMU selection — (subject, trial, name, category, description).
# Drawn from the official CMU index (cmu-mocap-index-text.txt). Diverse coverage
# across locomotion, dance, combat, sport, and daily activity.
SELECTED: list[tuple[int, int, str, str, str]] = [
    (2,  1, "Walk",                 "locomotion", "CMU subject 02, trial 01 — walk"),
    (2,  3, "Run / Jog",            "locomotion", "CMU subject 02, trial 03 — run/jog"),
    (2,  4, "Jump & Balance",       "locomotion", "CMU subject 02, trial 04 — jump, balance"),
    (2,  5, "Punch / Strike",       "combat",     "CMU subject 02, trial 05 — punch/strike"),
    (2,  7, "Swordplay",            "combat",     "CMU subject 02, trial 07 — swordplay"),
    (5,  2, "Dance — Pirouette",    "dance",      "CMU subject 05, trial 02 — dance, expressive arms, pirouette"),
    (5,  6, "Dance — Cartwheel & Jete", "dance",  "CMU subject 05, trial 06 — cartwheel-like start, pirouettes, jete"),
    (6,  2, "Basketball Dribble",   "sport",      "CMU subject 06, trial 02 — basketball, forward dribble"),
    (13, 4, "Sitting (chin in hand)", "daily",    "CMU subject 13, trial 04 — sit on stepstool, chin in hand"),
    (13, 11, "Forward Jump",        "locomotion", "CMU subject 13, trial 11 — forward jump"),
    (13, 17, "Boxing",              "combat",     "CMU subject 13, trial 17 — boxing"),
    (13, 26, "Wave / Direct Traffic", "gesture",  "CMU subject 13, trial 26 — direct traffic, wave"),
    (13, 29, "Jumping Jacks & Squats", "exercise","CMU subject 13, trial 29 — jumping jacks, side twists, bend over, squats"),
    (16, 1, "Loose Walk",           "locomotion", "CMU subject 16, trial 01 — walk"),  # placeholder until verified
    (49, 6, "Cartwheel",            "acrobatic",  "CMU subject 49, trial 06 — cartwheel"),
    (74, 3, "Kick",                 "combat",     "CMU subject 74, trial 03 — kick"),
]


def _download_bvh(subject: int, trial: int) -> str:
    """Download + extract one CMU BVH file, return its text content."""
    seq = _seq_dir(subject)
    s, t = f"{subject:02d}", f"{trial:02d}"
    zip_name = f"{s}_{t}.zip"
    # The Data dir lives under Sequence-XXX/SS/Data/SS_TT.zip
    url = f"{RAW_BASE}/{seq}/{s}/Data/{zip_name}"
    import urllib.request
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tf:
        urllib.request.urlretrieve(url, tf.name)
        with zipfile.ZipFile(tf.name) as zf:
            bvh_name = next(n for n in zf.namelist() if n.endswith(".bvh"))
            return zf.read(bvh_name).decode("utf-8", errors="replace")


def _existing_entries() -> list[dict]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text()).get("entries", [])
    return []


def main() -> None:
    entries = _existing_entries()
    # Drop any prior CMU entries so re-runs replace cleanly.
    entries = [e for e in entries if not e.get("slug", "").startswith("cmu_")]
    new = []
    for subject, trial, name, category, desc in SELECTED:
        slug = f"cmu_{subject:02d}_{trial:02d}"
        try:
            bvh_text = _download_bvh(subject, trial)
            frames = retarget_bvh_text(bvh_text).astype(np.float32)
            frames = _normalize_to_soma_scale(frames)
        except Exception as e:  # pragma: no cover — network/parse resilience
            print(f"  SKIP {slug}: {e}")
            continue
        out = CACHE / f"{slug}.npy"
        np.save(out, frames)
        entry = {
            "slug": slug,
            "name": name,
            "description": desc,
            "category": category,
            "fps": 30,
            "frame_count": int(frames.shape[0]),
            "is_motion": True,
            "tags": ["cmu", "mocap", "soma-77", "upstream"],
            "source": f"CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu) — subject {subject}, trial {trial}",
            "file": out.name,
        }
        new.append(entry)
        print(f"  {slug}: {frames.shape}  «{name}»")
    entries.extend(new)
    manifest = {
        "version": 3,
        "source": "real",
        "notes": "Curated real upstream mocap (CMU) + Kimodo model output + canonical SOMA-77 reference poses.",
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(new)} CMU clips ({len(entries)} total entries) to {MANIFEST}")


if __name__ == "__main__":
    main()
