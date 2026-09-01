"""Build the Pose Studio corpus cache from REAL Kimodo model output.

This is a BUILD-TIME tool (run manually / in CI), NOT imported at runtime.
It uses torch + the Kimodo skeleton classes to lift the bundled
``kimodo-soma-rp`` demo motions (SOMA-30) into SOMA-77 (the skeleton the
Pose Studio renders), and emits the canonical T-pose/rest pose from
``SOMASkeleton77``'s own rest-pose data. Output is plain ``.npy`` files +
a ``manifest.json`` that the runtime corpus loader
(``custom_nodes/melite-poser-nodes/library/pose_corpus.py``) serves from
inside inference-comfyui.

Run (from inside the inference-comfyui container, which has torch + kimodo):
    python /root/ComfyUI/custom_nodes/melite-poser-nodes/tools/build_pose_corpus.py

Replaces the old procedural sine-wave pose corpus entirely. Every frame in
the cache is real model output / canonical skeleton data — nothing hand-synthesized.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PACK = Path(__file__).resolve().parents[1]
KIMODO = REPO / "vendor" / "kimodo"
if str(KIMODO) not in sys.path:
    sys.path.insert(0, str(KIMODO))

import torch  # noqa: E402
from kimodo.assets import SKELETONS_ROOT  # noqa: E402
from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77  # noqa: E402

DEMO_ROOT = KIMODO / "kimodo" / "assets" / "demo" / "examples" / "kimodo-soma-rp"
OUT = PACK / "library" / "pose_corpus_cache"
OUT.mkdir(parents=True, exist_ok=True)


def _short_name(text: str) -> str:
    """Turn a prompt into a concise label.

    "A person runs forward and then leaps..." -> "Runs forward and leaps"
    """
    t = text.strip().rstrip(".")
    # Strip leading subject + helper verbs: "A person is walking" -> "walking"
    for prefix in ("A person is ", "The person is ", "A person was ",
                   "A person ", "The person ", "A character ", "Someone "):
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix):]
            break
    # Lowercase the first letter so it reads as an action phrase, then cap.
    if t:
        t = t[0].upper() + t[1:]
    if len(t) > 42:
        t = t[:39].rstrip() + "…"
    return t


def _demo_label(meta: dict) -> tuple[str, str]:
    """Return (name, description) from a demo meta.json, using the real prompt."""
    if "text" in meta:
        t = meta["text"]
        return _short_name(t), t
    if "texts" in meta:  # multi-prompt: name from the joined actions
        parts = [_short_name(x) for x in meta["texts"]]
        name = "  ·  ".join(parts)
        if len(name) > 42:
            name = name[:39].rstrip() + "…"
        desc = f"Multi-prompt: {' | '.join(meta['texts'])}"
        return name, desc
    return "Untitled", ""


def _lift_30_to_77(skel30: SOMASkeleton30, soma77: SOMASkeleton77,
                   npz_path: Path) -> np.ndarray:
    """Convert a SOMA-30 demo motion to SOMA-77 (T, 77, 3) world positions.

    Uses the model's own skeleton classes:
      global_rot_30 → local_rot_30 (global_rots_to_local_rots)
      → local_rot_77 (to_SOMASkeleton77, fills relaxed hands)
      → posed_77 (soma77.fk)
    """
    d = np.load(npz_path, allow_pickle=True)
    g30 = torch.from_numpy(np.ascontiguousarray(d["global_rot_mats"])).float()
    p30 = torch.from_numpy(np.ascontiguousarray(d["posed_joints"])).float()
    local30 = skel30.global_rots_to_local_rots(g30.unsqueeze(0))   # (1,T,30,3,3)
    local77 = skel30.to_SOMASkeleton77(local30.squeeze(0))         # (T,77,3,3)
    root_pos = p30[:, 0, :].unsqueeze(0)                            # (1,T,3)
    _, posed77, _ = soma77.fk(local77.unsqueeze(0), root_pos)
    return posed77.squeeze(0).detach().cpu().numpy().astype(np.float32)


def build_motions(skel30, soma77) -> list[dict]:
    entries = []
    for demo_dir in sorted(DEMO_ROOT.iterdir()):
        npz = demo_dir / "motion.npz"
        meta_f = demo_dir / "meta.json"
        if not npz.exists() or not meta_f.exists():
            continue
        meta = json.loads(meta_f.read_text())
        name, desc = _demo_label(meta)
        frames = _lift_30_to_77(skel30, soma77, npz)
        duration = meta.get("duration") or meta.get("durations", [5.0])
        if isinstance(duration, list):
            duration = sum(duration)
        fps = round(frames.shape[0] / float(duration)) if duration else 30
        slug = demo_dir.name
        out = OUT / f"{slug}.npy"
        np.save(out, frames)
        entries.append({
            "slug": slug,
            "name": name,
            "description": desc,
            "category": "motion",
            "fps": fps,
            "frame_count": int(frames.shape[0]),
            "is_motion": True,
            "tags": ["kimodo", "model-output", "soma-77"],
            "source": "vendor/kimodo/.../kimodo-soma-rp (real model generation)",
            "file": out.name,
        })
        print(f"  motion {slug}: {frames.shape} fps={fps}  «{name}»")
    return entries


def _rest_local_rots() -> torch.Tensor:
    """Load the SOMA-77 relaxed-hands rest local rotations (77, 3, 3).

    The buffer isn't auto-registered for soma77 (no rest_pose_local_rot.p), so
    load the bundled relaxed_hands_rest_pose.npy directly.
    """
    npy = SKELETONS_ROOT / "somaskel77" / "relaxed_hands_rest_pose.npy"
    r = np.load(npy).astype(np.float32)
    return torch.from_numpy(r)


def build_canonical_tpose(soma77) -> dict:
    """Canonical SOMA-77 T-pose from the skeleton's own standard_tpose transform."""
    rest = _rest_local_rots()                                  # (77,3,3)
    tp_local, _ = soma77.to_standard_tpose(rest.unsqueeze(0))  # (1,77,3,3)
    local = tp_local.reshape(1, 1, 77, 3, 3)
    # Hip at y≈1.0 so the figure stands at the same height as live generations
    # (feet land near y=0). Matches the demo-motion root convention.
    root = torch.tensor([[[0.0, 1.0, 0.0]]])
    _, posed, _ = soma77.fk(local, root)
    frames = posed.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (1,77,3)
    out = OUT / "tpose.npy"
    np.save(out, frames)
    print(f"  t-pose: {frames.shape}")
    return {
        "slug": "tpose",
        "name": "T-Pose (canonical SOMA-77)",
        "description": "Canonical T-pose from SOMASkeleton77.to_standard_tpose — the skeleton's own reference pose.",
        "category": "stance",
        "fps": 0,
        "frame_count": 1,
        "is_motion": False,
        "tags": ["reference", "tpose", "canonical"],
        "source": "SOMASkeleton77.to_standard_tpose",
        "file": out.name,
    }


def build_rest_pose(soma77) -> dict:
    """SOMA-77 relaxed rest pose (arms down, relaxed hands) — the bind pose."""
    rest = _rest_local_rots()
    local = rest.reshape(1, 1, 77, 3, 3)
    root = torch.tensor([[[0.0, 1.0, 0.0]]])
    _, posed, _ = soma77.fk(local, root)
    frames = posed.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (1,77,3)
    out = OUT / "rest.npy"
    np.save(out, frames)
    print(f"  rest: {frames.shape}")
    return {
        "slug": "rest",
        "name": "Rest Pose (relaxed hands)",
        "description": "SOMA-77 bind pose with relaxed hands — the model's relaxed rest local rotations.",
        "category": "stance",
        "fps": 0,
        "frame_count": 1,
        "is_motion": False,
        "tags": ["reference", "rest", "bind"],
        "source": "SOMASkeleton77 relaxed_hands_rest_pose.npy",
        "file": out.name,
    }


def main():
    skel30 = SOMASkeleton30(SKELETONS_ROOT / "somaskel30")
    soma77 = SOMASkeleton77(SKELETONS_ROOT / "somaskel77")
    print(f"Building corpus cache into {OUT}")
    entries = []
    entries.append(build_canonical_tpose(soma77))
    entries.append(build_rest_pose(soma77))
    entries.extend(build_motions(skel30, soma77))
    manifest = {"version": 2, "source": "real", "entries": entries}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(entries)} poses to {OUT/'manifest.json'}")


if __name__ == "__main__":
    main()
