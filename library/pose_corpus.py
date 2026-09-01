"""Built-in pose corpus for the Pose Studio external library.

Loads REAL motion data — actual Kimodo model output lifted to SOMA-77 via the
model's own skeleton classes, plus the canonical SOMA-77 T-pose / rest pose.
The data is precomputed by ``tools/build_pose_corpus.py`` (which imports torch
+ the Kimodo skeleton classes) and cached as ``.npy`` files alongside this
module. This module is **numpy-only** so the slim melite-head gateway (no torch)
can import it without pulling in torch or kimodo.

Every frame in the corpus is real model output or canonical skeleton data —
nothing hand-synthesized. The old procedural sine-wave builders were removed
because procedurally faking human motion produces garbage; see
``tools/build_pose_corpus.py`` for how the cache is regenerated.

Output format matches what Pose Studio's ``editableJoints`` consumes:
hip-centered world positions in meters, shape (T, 77, 3).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

_CACHE_DIR = Path(__file__).resolve().parent / "pose_corpus_cache"
_MANIFEST = _CACHE_DIR / "manifest.json"


@dataclass
class CorpusPose:
    slug: str                   # "tpose"
    name: str                   # "T-Pose (canonical SOMA-77)"
    description: str            # one-line summary
    category: str               # "stance" | "motion"
    frames: np.ndarray          # (T, 77, 3) float32, hip-centered, meters
    fps: float                  # native frame rate (0 for static)
    tags: List[str] = field(default_factory=list)
    source: str = ""            # provenance — where the frames came from

    @property
    def is_motion(self) -> bool:
        return self.frames.shape[0] > 1

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])

    def to_dict(self, include_frames: bool = True) -> dict:
        d = {
            "slug": self.slug,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "is_motion": self.is_motion,
            "tags": list(self.tags),
            "source": self.source,
        }
        if include_frames:
            d["frames"] = self.frames.tolist()
        return d


# ── Cache loading ─────────────────────────────────────────────────────────

def _load_manifest() -> list[dict]:
    if not _MANIFEST.exists():
        raise FileNotFoundError(
            f"Pose corpus cache missing at {_MANIFEST}. "
            f"Run `python tools/build_pose_corpus.py` to regenerate it from "
            f"the bundled Kimodo demo motions."
        )
    return json.loads(_MANIFEST.read_text()).get("entries", [])


def _entry_to_pose(entry: dict) -> CorpusPose:
    """Build a CorpusPose from a manifest entry, lazy-loading its .npy frames."""
    frames = np.load(_CACHE_DIR / entry["file"]).astype(np.float32, copy=False)
    return CorpusPose(
        slug=entry["slug"],
        name=entry["name"],
        description=entry.get("description", ""),
        category=entry.get("category", "motion"),
        frames=frames,
        fps=float(entry.get("fps", 0)),
        tags=list(entry.get("tags", [])),
        source=entry.get("source", ""),
    )


_REGISTRY: Optional[dict[str, CorpusPose]] = None


def _ensure_registry() -> dict[str, CorpusPose]:
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    reg: dict[str, CorpusPose] = {}
    for entry in _load_manifest():
        try:
            pose = _entry_to_pose(entry)
            reg[pose.slug] = pose
        except Exception as e:  # pragma: no cover — defensive
            import sys
            print(f"[pose_corpus] WARN: failed to load {entry.get('slug')}: {e}",
                  file=sys.stderr)
    _REGISTRY = reg
    return reg


def list_corpus() -> List[CorpusPose]:
    """Return all corpus poses (static + motion). Stable ordering by manifest."""
    reg = _ensure_registry()
    return [reg[k] for k in _manifest_order() if k in reg]


def _manifest_order() -> list[str]:
    return [e["slug"] for e in _load_manifest()]


def get_pose(slug: str) -> Optional[CorpusPose]:
    """Fetch a single pose by slug. Returns None if not found."""
    return _ensure_registry().get(slug)


def corpus_summary() -> List[dict]:
    """Lightweight listing (no frame data) for UI panels."""
    return [
        {
            "slug": p.slug,
            "name": p.name,
            "description": p.description,
            "category": p.category,
            "fps": p.fps,
            "frame_count": p.frame_count,
            "is_motion": p.is_motion,
            "tags": list(p.tags),
            "source": p.source,
        }
        for p in list_corpus()
    ]


# ── Full upstream CMU catalog (browseable metadata + on-demand fetch) ──────

_CMU_CATALOG = _CACHE_DIR.parent / "cmu_catalog.json"
_CMU_INDEX = _CACHE_DIR.parent / "cmu_mocap_index.txt"
_RAW_BASE = "https://github.com/Shriinivas/cmubvh/raw/main"
TARGET_HIP_Y = 1.0
TARGET_HIP_TO_HEAD = 0.60
_HIP_IDX = 0
_HEAD_IDX = 6


def _catalog_doc() -> dict:
    if not _CMU_CATALOG.exists():
        raise FileNotFoundError(
            f"CMU catalog missing at {_CMU_CATALOG}. Run tools/build_cmu_catalog.py."
        )
    return json.loads(_CMU_CATALOG.read_text())


def list_cmu_catalog(
    query: Optional[str] = None,
    category: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> dict:
    """Search/browse the full upstream CMU mocap catalog (metadata only).

    Returns {version, count, total, by_category, entries:[{subject,trial,slug,
    description,subject_theme,category,source}, ...]}. Entries are filtered by
    case-insensitive substring `query` (against description + theme) and/or
    exact `category`, then offset/limited for paging.
    """
    doc = _catalog_doc()
    entries = doc["entries"]
    q = (query or "").strip().lower()
    cat = (category or "").strip().lower()
    if q:
        entries = [e for e in entries
                   if q in e["description"].lower() or q in e["subject_theme"].lower()]
    if cat and cat != "all":
        entries = [e for e in entries if e["category"] == cat]
    total = len(entries)
    if offset:
        entries = entries[offset:]
    if limit is not None and limit > 0:
        entries = entries[:limit]
    return {
        "version": doc.get("version", 1),
        "source": doc.get("source", ""),
        "total": total,
        "catalog_count": doc.get("count", 0),
        "by_category": doc.get("by_category", {}),
        "entries": entries,
    }


def cmu_catalog_categories() -> list[dict]:
    """Category counts for UI faceting."""
    doc = _catalog_doc()
    bc = doc.get("by_category", {})
    return [{"category": k, "count": v} for k, v in sorted(bc.items(), key=lambda kv: -kv[1])]


def _seq_dir(subject: int) -> str:
    # Mirrors the directory layout of the Shriinivas/cmubvh GitHub mirror.
    # Each CMU batch lives in a "Sequence-LO-HI" folder; subject numbers have
    # gaps (e.g. 57-59, 95-101, 129-130 were never released), so the catalog
    # only indexes subjects that exist and every indexed subject must resolve
    # here. Keep this in sync with the repo's top-level directory listing.
    ranges = [
        (9, "Sequence-001-009"), (14, "Sequence-010-014"), (19, "Sequence-015-019"),
        (29, "Sequence-020-029"), (34, "Sequence-030-034"), (39, "Sequence-035-039"),
        (45, "Sequence-040-045"), (56, "Sequence-046-056"), (75, "Sequence-060-075"),
        (80, "Sequence-076-080"), (85, "Sequence-081-085"), (94, "Sequence-086-094"),
        (111, "Sequence-102-111"), (128, "Sequence-113-128"),
        (135, "Sequence-131-135"), (140, "Sequence-136-140"), (144, "Sequence-141-144"),
    ]
    for hi, name in ranges:
        if subject <= hi:
            return name
    raise ValueError(f"No sequence directory for CMU subject {subject}")


def _normalize_to_soma_scale(frames: np.ndarray) -> np.ndarray:
    """Re-root hip to y=1.0 with hip→head ≈ 0.60 m (SOMA-77 convention)."""
    hip = frames[:, _HIP_IDX, :]
    head = frames[:, _HEAD_IDX, :]
    dist = float(np.linalg.norm(head - hip, axis=-1).mean())
    if dist < 1e-6:
        return frames + np.array([0.0, TARGET_HIP_Y, 0.0])
    scale = TARGET_HIP_TO_HEAD / dist
    centered = (frames - hip[:, None, :]) * scale
    return centered + np.array([0.0, TARGET_HIP_Y, 0.0])


def fetch_cmu_motion(subject: int, trial: int, max_frames: int = 240) -> np.ndarray:
    """Download one CMU BVH on demand and retarget it to SOMA-77 positions.

    Pure numpy + stdlib (urllib/zipfile) — ComfyUI-safe (no torch needed).
    The BVH comes from the Shriinivas/cmubvh mirror (genuine CMU capture
    data); retargeting uses the rotation-transfer retargeter vendored into
    this pack at ``.bvh_retarget``.

    Returns (T, 77, 3) float32 hip-rooted at y=1.0, in metres.
    """
    import tempfile
    import zipfile
    import urllib.request
    from .bvh_retarget import retarget_bvh_text

    seq = _seq_dir(subject)
    zip_name = f"{subject:02d}_{trial:02d}.zip"
    url = f"{_RAW_BASE}/{seq}/{subject:02d}/Data/{zip_name}"
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tf:
        urllib.request.urlretrieve(url, tf.name)
        with zipfile.ZipFile(tf.name) as zf:
            bvh_name = next(n for n in zf.namelist() if n.endswith(".bvh"))
            bvh_text = zf.read(bvh_name).decode("utf-8", errors="replace")
    frames = retarget_bvh_text(bvh_text, max_frames=max_frames).astype(np.float32)
    return _normalize_to_soma_scale(frames).astype(np.float32)
