"""Pose corpus library — vendored into melite-poser-nodes so the corpus,
BVH retargeter, and SOMA-77 skeleton topology all live inside the
inference-comfyui container (per the 3-container architecture:
editor → melite-head → ComfyUI). No melite-head media/ dependency.

Modules:
  • skeleton       — SOMA-77 joint names, parents, neutral positions, COCO-18 mapping
  • bvh_retarget   — parse any BVH → (T, 77, 3) SOMA-77 positions
  • pose_corpus    — built-in pose library + CMU catalog browse/fetch
"""
