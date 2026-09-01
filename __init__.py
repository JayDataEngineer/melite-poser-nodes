"""Tech Noir Poser custom nodes for ComfyUI.

Serves the /poser/* HTTP routes Pose Studio + Body Studio need, directly
on ComfyUI's aiohttp server. NO separate poser-skin container, NO gpu-all
image — all mesh/skin work happens inside inference-comfyui using the
py-soma-x library (declared via the plugin lane / image base env).

Routes registered on PromptServer.instance.routes:
  GET  /poser/skin/mesh             — bind mesh {vertices, faces}
  POST /poser/skin/mesh             — SHAPED bind mesh (identity_coeffs in body)
  GET  /poser/skin/identity-space   — identity coefficient metadata
  GET  /poser/skin/bind_joints      — neutral bind-pose SOMA-77 joints
  POST /poser/skin/bind_joints      — shaped bind-pose joints
  POST /poser/skin/deform_soma      — SOMA-77 positions → deformed vertices
  GET  /poser/text-to-pose/registry — static dataset/skeleton catalog
  GET  /poser/text-to-pose/models   — static Kimodo model list
  POST /poser/text-to-pose          — submit text-to-pose generation
  GET  /poser/text-to-pose/jobs/{id} — poll generation job status
  DEL  /poser/text-to-pose/jobs/{id} — cancel a running job
  GET  /poser/library/corpus        — list built-in pose library (26 entries)
  GET  /poser/library/corpus/{slug} — full pose with (T, 77, 3) frames
  POST /poser/library/import_bvh    — retarget BVH → SOMA-77 positions
  GET  /poser/library/cmu_catalog   — browse 2435-entry CMU mocap catalog
  POST /poser/library/cmu_fetch     — fetch + retarget one CMU clip on demand
  POST /poser/render                — render COCO-18 keypoints → PNG stick figure
  GET  /poser/health                — health probe

Subpackages:
  library/  — pose corpus data + BVH retargeter + SOMA↔COCO projection
              (skeleton.py, bvh_retarget.py, pose_corpus.py, soma_to_coco.py,
               pose_corpus_cache/*.npy, cmu_catalog.json)
  tools/    — build-time scripts to regenerate the corpus cache from Kimodo
              model output + the CMU mocap database (run inside the container)

The mesh engine is a singleton (cached per identity_model) — built once on
first request, reused for every subsequent slider drag / preset load.
"""
from . import server  # noqa: F401  — registers /poser/* routes on import

# ComfyUI workflow nodes — invoked via ComfyScript from melite-head.
# These are the ComfyScript-compliant path for mesh cleanup ops: melite-head
# uploads a GLB via ComfyUIClient.upload_file, builds a single-node workflow
# via media.comfyui.builders.build_mesh_*, and submits via
# submit_comfy_workflow. NO /poser/mesh/* HTTP side-routes.
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
