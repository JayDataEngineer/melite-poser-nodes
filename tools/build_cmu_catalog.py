"""Build the FULL CMU Graphics Lab mocap catalog as browseable metadata.

Parses the canonical CMU index (cmu-mocap-index-text.txt, compiled by
B. Hahne from mocap.cs.cmu.edu) into a structured JSON catalog: one entry
per (subject, trial) with description + keyword-derived category. This is
the UPSTREAM CATALOG — every motion CMU ever captured (~2600 clips), not a
hand-curated 16.

Pose Studio browses this catalog; clicking an entry fetches + retargets that
single BVH on demand (the cmu_fetch endpoint), so the full catalog ships
as ~80 KB of metadata, not gigabytes of frames.

Run (the SRC index file ships at the repo root as
``media/motion/cmu_mocap_index.txt`` — restore it from git history if
it's been deleted, or download a fresh copy from the pycmo/cmubvh project):
    python custom_nodes/melite-poser-nodes/tools/build_cmu_catalog.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PACK = Path(__file__).resolve().parents[1]
SRC = REPO / "services" / "motion" / "cmu_mocap_index.txt"
OUT = PACK / "library" / "cmu_catalog.json"

# Keyword → category. First match wins (order matters: most-specific first).
CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("acrobatic", ("cartwheel", "flip", "somersault", "handspring", "tumble", "handstand", "roundoff")),
    ("combat",    ("punch", "kick", "box", "fight", "strike", "martial", "sword", "fence", "karate",
                   "kung fu", "hit", "slap", "wrestle", "spar")),
    ("dance",     ("dance", "pirouette", "jete", "ballet", "flamenco", "break", "salsa", "tango",
                   "waltz", "choreograph", "hop _in place")),
    ("sport",     ("basketball", "soccer", "football", "tennis", "baseball", "golf", "swim", "bowl",
                   "dribble", "volleyball", "hockey", "row", "climb", "ski", "skate", "bicycle", "bike")),
    ("jump",      ("jump", "leap", "hop", "vault", "bound", "hurdle")),
    ("locomotion",("walk", "run", "jog", "stride", "lope", "march", "sneak", "tiptoe", "stagger",
                   "stumble", "limp", "crawl", "climb")),
    ("exercise",  ("stretch", "squat", "jumping jack", "lunge", "yoga", "sit-up", "push-up",
                   "bend over", "touching toes", "range of motion")),
    ("gesture",   ("wave", "point", "gesture", "direct traffic", "reach", "grab", "throw", "catch",
                   "push", "pull", "lift", "carry", "flag down")),
    ("seat",      ("sit", "chair", "stool", "seat", "cross-legged", "kneel", "squat down")),
    ("daily",     ("drink", "eat", "wash", "sweep", "rake", "phone", "door", "window", "cook",
                   "clean", "dress", "undress", "shoe", "wipe", "pour", "smoke", "sleep", "yawn",
                   "sneeze", "read", "write", "type", "shovel", "mop")),
    ("performance",("mime", "juggle", "magic", "circus", "puppet", "act", "performance", "posing")),
]

SUBJECT_RE = re.compile(r"^Subject\s*#\s*(\d+)\s*\((.*)\)\s*$")
# Subject numbers go up to 144 (3 digits). The old \d{1,2} silently dropped
# every subject >= 100 — basketball (102), stylized walks (142), punching (144), etc.
MOTION_RE  = re.compile(r"^(\d{1,3})_(\d{2})\s+(.*)$")

def categorize(text: str) -> str:
    low = text.lower()
    for cat, kws in CATEGORY_RULES:
        if any(kw in low for kw in kws):
            return cat
    return "misc"

def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Index not found at {SRC}. Download cmu-mocap-index-text.txt "
                         f"from https://github.com/una-dinosauria/cmu-mocap first.")
    entries: list[dict] = []
    cur_subject: int | None = None
    cur_theme = ""
    for raw in SRC.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        ms = SUBJECT_RE.match(line)
        if ms:
            cur_subject = int(ms.group(1))
            cur_theme = ms.group(2).strip()
            continue
        mm = MOTION_RE.match(line)
        if mm and cur_subject is not None:
            s = cur_subject
            t = int(mm.group(2))
            desc = mm.group(3).strip()
            # Some subjects (e.g. 94 "indian dance") have every trial labeled
            # "Unknown". Dropping those loses real captures whose THEME is known.
            # Fall back to the subject theme so the entry stays browseable; only
            # skip if we have no label at all.
            if desc.lower() in ("unknown", ""):
                desc = cur_theme
            if not desc:
                continue
            cat = categorize(f"{cur_theme} {desc}")
            entries.append({
                "subject": s,
                "trial": t,
                "slug": f"cmu_{s:02d}_{t:02d}",
                "description": desc,
                "subject_theme": cur_theme,
                "category": cat,
                "source": f"CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu) — subject {s:02d}, trial {t:02d}",
            })
    # de-dup by slug (keep first)
    seen = set(); uniq = []
    for e in entries:
        if e["slug"] in seen: continue
        seen.add(e["slug"]); uniq.append(e)
    from collections import Counter
    by_cat = Counter(e["category"] for e in uniq)
    doc = {
        "version": 1,
        "source": "CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu). "
                  "Index compiled by B. Hahne. BVH via github.com/Shriinivas/cmubvh.",
        "count": len(uniq),
        "by_category": dict(by_cat),
        "entries": uniq,
    }
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"Wrote {len(uniq)} CMU motions to {OUT}")
    print("By category:", dict(by_cat.most_common()))

if __name__ == "__main__":
    main()
