"""SOMA skin engine for the melite-poser-nodes ComfyUI pack.

Loads the SOMA mesh + identity model (Anny/MHR/SOMA) once, then serves bind
meshes, shaped meshes, and SOMA-77 joint positions from the cache. All CPU,
no GPU contention — runs in a thread off the aiohttp handler so the
ComfyUI event loop never blocks on the ~30 ms set_identity() call.

The py-soma-x library (``from soma import SOMALayer``) unifies Anny / MHR /
SMPL / SMPL-X onto SOMA's canonical 18056-vertex topology, so they all answer
to the same SOMA-77 joint poses Pose Studio produces.

Model data lives under /mnt/data/models/cache/huggingface/models--nvidia--soma-x/
(snapshots/<hash>/...), which is the HF cache. The ComfyUI container
already mounts /mnt/data/models, so no extra volume wiring is needed.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)

# ── SOMA-77 joint reference (neutral T-pose, Y-up, meters) ──────────────────
# Mirror of media/motion/skeleton.py's neutral_joints — the canonical
# SOMA-77 joint positions in T-pose. Kept as a documentation reference of
# the SOMA-77 topology; NO LONGER USED as a runtime fallback anywhere in
# this file (all such fallbacks were removed 2026-07-19 per the project's
# "NO SILENT FALLBACKS" invariant — bad data must raise, not paper over).
# fmt: off
NEUTRAL_JOINTS: np.ndarray = np.array([
    [0.0, 0.000, 0.000],   # 0 pelvis
    [0.0, 0.182, 0.000],   # 1 spine_1
    [0.0, 0.364, 0.000],   # 2 spine_2
    [0.0, 0.546, 0.000],   # 3 spine_3
    [0.0, 0.728, 0.000],   # 4 neck
    [0.0, 0.793, 0.000],   # 5 head
    [0.0, 0.828, 0.000],   # 6 nose
    [0.0, 0.810, 0.000],   # 7 left_eye
    [0.0, 0.810, 0.000],   # 8 right_eye
    [0.0, 0.820, 0.000],   # 9 left_ear
    [0.0, 0.820, 0.000],   # 10 right_ear
    [0.0, 0.700, 0.180],   # 11 left_collar
    [0.0, 0.700, -0.180],  # 12 right_collar
    [0.0, 0.700, 0.450],   # 13 left_shoulder
    [0.0, 0.700, -0.450],  # 14 right_shoulder
    [0.0, 0.560, 0.450],   # 15 left_elbow
    [0.0, 0.560, -0.450],  # 16 right_elbow
    [0.0, 0.430, 0.450],   # 17 left_wrist
    [0.0, 0.430, -0.450],  # 18 right_wrist
    [0.0, 0.230, 0.120],   # 19 left_hip
    [0.0, 0.230, -0.120],  # 20 right_hip
    [0.0, 0.230, 0.120],   # 21 left_knee (length pelvis→knee same as pelvis→hip)
    [0.0, 0.230, -0.120],  # 22 right_knee
    [0.0, 0.000, 0.120],   # 23 left_ankle
    [0.0, 0.000, -0.120],  # 24 right_ankle
] + [[0.0, 0.0, 0.0]] * 52, dtype=np.float32)  # 25-76: hand joints (zero — filled by engine)
# fmt: on

# ── Phenotype dims per identity model ───────────────────────────────────────
# These are the USER-FACING identity parameter counts — what prepare_identity
# actually accepts as ``identity_coeffs``. The 128 ``num_shape_components``
# are internal PCA basis vectors, NOT the phenotype tensor.
#
# Determined empirically by probing the TorchScript models inside SOMALayer:
#   anny: prepare_identity rejects ≠11 with
#           "phenotype_kwargs tensor must have shape [bs, 11]"
#   mhr:  TorchScript shape_vectors has n=45; einsum fails at any other dim.
#
# MHR additionally requires ``scale_params`` of shape (B, 68) — it has
# ``num_scale_params = 68``. SOMA has 56 (unused here); anny has 0.
_PHENOTYPE_DIMS: dict[str, int] = {"anny": 11, "mhr": 45}

# scale_params per identity model (from SOMALayer.num_scale_params).
#   mhr:  requires (B, 68) tensor — num_scale_params = 68
#   anny: requires an empty dict {} — anny's get_phenotype_blendshape_coefficients
#         subscriptes scale_params as local_changes; None → TypeError.
#         An empty dict avoids the issue while passing no local changes.
#   soma: unused (npz path, no SOMALayer)
_SCALE_PARAM_DIMS: dict[str, int] = {"mhr": 68}
_ANNY_SCALE_PARAMS: dict = {}  # empty dict, not None


# ── SOMA-77 kinematic tree (parent indices) ────────────────────────────────
# Mirrors kimodo.skeleton.SOMASkeleton77.joint_parents. Used by
# _global_rots_to_local_rots() to convert Kimodo's global rotation
# matrices to the local (parent-relative) rotations that pose() expects.
# Root (joint 0) has parent -1 (no parent).
_SOMA77_JOINT_PARENTS = np.array([
    -1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6,         # 0-10: spine chain + head/eyes/ears
    3, 11, 12, 13, 14, 15, 16, 17,             # 11-18: left arm (collar→wrist)
    14, 19, 20, 21, 22, 14, 24, 25, 26, 27,    # 19-28: left fingers
    14, 29, 30, 31, 32, 14, 34, 35, 36, 37,    # 29-38: left fingers cont.
    3, 39, 40, 41, 42, 43, 44, 45,             # 39-46: right arm (collar→wrist)
    42, 47, 48, 49, 50, 42, 52, 53, 54, 55,    # 47-56: right fingers
    42, 57, 58, 59, 60, 42, 62, 63, 64, 65,    # 57-66: right fingers cont.
    0, 67, 68, 69, 70,                          # 66-70: left leg (hip→ankle)
    0, 72, 73, 74, 75,                          # 71-75: right leg (hip→ankle)
], dtype=np.int64)


def _global_rots_to_local_rots(global_rots: np.ndarray) -> np.ndarray:
    """Convert global (world-space) rotation matrices to local (parent-relative).

    Mirrors kimodo.skeleton.global_rots_to_local_rots.
    For joint i with parent p: local_i = parent_global_inv @ global_i.
    Root (parent=-1) keeps its global rotation as-is.

    Args:
        global_rots: (T, 77, 3, 3) global rotation matrices.
    Returns:
        (T, 77, 3, 3) local rotation matrices.
    """
    T, J = global_rots.shape[:2]
    parents = _SOMA77_JOINT_PARENTS[:J]

    # Gather parent rotations: (T, J, 3, 3) where entry [t, i] = global_rots[t, parents[i]]
    # For root (parent=-1), use identity.
    parent_rots = np.zeros((T, J, 3, 3), dtype=np.float32)
    for i in range(J):
        p = int(parents[i])
        if p < 0:
            parent_rots[:, i] = np.eye(3, dtype=np.float32)
        else:
            parent_rots[:, i] = global_rots[:, p]

    # local = parent_inv @ global
    parent_inv = np.transpose(parent_rots, axes=(0, 1, 3, 2))  # transpose = inverse for rot mats
    local_rots = np.einsum('tNmn,tNno->tNmo', parent_inv, global_rots)
    return local_rots.astype(np.float32)


# ── Engine cache ────────────────────────────────────────────────────────────
_cache: dict[str, "SkinEngine"] = {}
_lock = threading.Lock()


def get_engine(identity_model: str = "soma") -> "SkinEngine":
    """Get (creating+cached) a SkinEngine for the requested identity model."""
    identity_model = (identity_model or "soma").lower()
    with _lock:
        if identity_model not in _cache:
            device = _pick_device()
            _cache[identity_model] = SkinEngine(identity_model, device=device)
            log.info(
                "SkinEngine(%s) built — %d vertices, device=%s",
                identity_model,
                _cache[identity_model].vertices.shape[0],
                device,
            )
    return _cache[identity_model]


def _pick_device() -> str:
    """Prefer CUDA — the anny identity forward pass is ~10x faster on GPU."""
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ── Engine ──────────────────────────────────────────────────────────────────
class SkinEngine:
    """Cached SOMA mesh + identity model.

    For SOMA: loads the neutral npz (vertices + faces + LBS data).
    For Anny/MHR: uses py-soma-x's SOMALayer for parametric reshaping.

    Public attrs:
        vertices   : np.ndarray (V, 3) float32 — current bind-pose vertices
        faces      : np.ndarray (F, 3) int32   — triangle indices
        bind_joints: np.ndarray (77, 3) float32 — current bind-pose joints
        default_vertices   : np.ndarray (V, 3) — neutral-identity snapshot
        default_bind_joints: np.ndarray (77, 3) — neutral-identity joints
    """

    SUPPORTED = {"soma", "anny", "mhr"}

    def __init__(self, identity_model: str, device: str = "cpu"):
        if identity_model not in self.SUPPORTED:
            raise ValueError(
                f"identity_model must be one of {sorted(self.SUPPORTED)}, "
                f"got {identity_model!r}"
            )
        self.identity_model = identity_model
        self.device = device
        self._model: Any = None  # SOMALayer for anny/mhr; None for soma
        # SOMA-77 skeleton for global→local rotation conversion (anny/mhr).
        # Populated in _load_soma_x(); stays None for the npz SOMA engine.
        self._skel: Any = None
        self._root_idx: int = 0
        self._neutral_root: Any = None

        if identity_model == "soma":
            self._load_soma_npz()
        else:
            self._load_soma_x()

        # Snapshot the neutral bind pose BEFORE any set_identity call —
        # GET endpoints serve this; POST endpoints mutate the live state.
        self.default_vertices = self.vertices.copy()
        self.default_bind_joints = self.bind_joints.copy()

    # ── Loaders ────────────────────────────────────────────────────────────
    def _load_soma_npz(self) -> None:
        """SOMA neutral mesh from the cached HF snapshot.

        SOMA_neutral.npz key map (see nvidia/soma-x on HF):
            mean            (18056, 3)  float32 — bind-pose vertices (canonical)
            bind_shape      (18056, 3)  float32 — bind-pose shape (same topology)
            triangles       (36108, 3)  int32   — triangle vertex indices
            shapedirs       (128, 54168) float32 — identity blend basis
            joint_names     (78,)       str     — joint labels (root + 77)
            bind_pose_local (78, 4, 4)  float32 — local bind-pose joint matrices
            bind_pose_world (78, 4, 4)  float32 — world-space bind pose matrices
            t_pose_local    (78, 4, 4)  float32 — local T-pose joint matrices
            t_pose_world    (78, 4, 4)  float32 — world-space T-pose matrices
            skinning_weights_*                    — sparse LBS weights
        """
        npz_path = self._find_soma_npz()
        data = np.load(npz_path, allow_pickle=False)

        # `mean` is the canonical bind mesh; `bind_shape` is the same
        # topology under the neutral identity. Use `mean` for GET /mesh.
        if "mean" in data.files:
            verts_key = "mean"
        elif "bind_shape" in data.files:
            verts_key = "bind_shape"
        else:
            raise KeyError(
                f"SOMA npz has neither 'mean' nor 'bind_shape' — keys: {data.files!r}"
            )
        self.vertices = np.asarray(data[verts_key], dtype=np.float32)

        # Faces: 'triangles' is the canonical key in nvidia/soma-x SOMA_neutral.npz.
        # 'faces' was a legacy alternate never actually seen in the wild — the
        # previous "try triangles, fallback to faces" silently masked a corrupt
        # or wrong npz as "compatible". Now we accept ONLY 'triangles' and
        # raise on anything else, so a wrong file fails loud at load time.
        if "triangles" not in data.files:
            raise KeyError(
                f"SOMA npz has no 'triangles' key — found {data.files!r}. "
                f"The canonical nvidia/soma-x SOMA_neutral.npz MUST have "
                f"'triangles' (36108, 3) int32. The file is either corrupt "
                f"or an incompatible version. Re-download with "
                f"huggingface-cli download nvidia/soma-x."
            )
        self.faces = np.asarray(data["triangles"], dtype=np.int32)

        # Bind joints: extract translations from bind_pose_world.
        # npz has 78 joints (root + 77 body joints). Drop root (index 0)
        # so we return SOMA-77 to match Pose Studio's skeleton overlay.
        self.bind_joints = self._extract_joints(data, "bind_pose_world")

        # Cache the full SOMA-77 world bind transforms (4×4 per joint) for
        # POST /poser/skin/bind_pose. anny/mhr derive these from
        # `SOMALayer.public_bind_transforms_world()` after `prepare_identity`;
        # SOMA reads them directly from the npz. We drop root (index 0) to
        # match the anny/mhr 77-joint contract used by callers downstream.
        if "bind_pose_world" not in data.files:
            raise KeyError(
                f"SOMA npz missing 'bind_pose_world' — found {data.files!r}. "
                f"Needed for full 4×4 bind transforms (POST /bind_pose)."
            )
        bpw = np.asarray(data["bind_pose_world"], dtype=np.float32)  # (78,4,4)
        if bpw.shape[0] != 78 or bpw.shape[1:] != (4, 4):
            raise ValueError(
                f"SOMA npz 'bind_pose_world' has shape {bpw.shape} — "
                f"expected (78, 4, 4)."
            )
        # SOMA npz translations are in CENTIMETERS (matches vertex units:
        # body height 168.4 = 1.68m). anny/mhr's SOMALayer returns METERS,
        # which the server multiplies by 100 to reach cm. To keep the
        # contract uniform ("transforms are always in meters"), divide
        # the SOMA translations by 100 here. Rotations stay unitless.
        bpw_m = bpw.copy()
        bpw_m[:, :3, 3] = bpw_m[:, :3, 3] / 100.0
        self._soma_bind_world_77 = bpw_m[1:78]  # drop root → (77, 4, 4) meters
        log.info(
            "SOMA npz loaded: V=%d F=%d J=%d (verts_key=%s)",
            self.vertices.shape[0],
            self.faces.shape[0],
            self.bind_joints.shape[0],
            verts_key,
        )

    def _extract_joints(self, data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
        """Pull SOMA-77 joint positions from a 4×4 world-pose matrix array.

        Handles the (78, 4, 4) layout in the npz by taking the translation
        column ([row 0:3, col 3]) and dropping the root joint (index 0).

        NO FALLBACK: previously this method silently returned NEUTRAL_JOINTS
        when the npz was missing the key, had an unexpected shape, or had
        the wrong joint count. That masked real corruption — the renderer
        would draw "a body" but in the wrong pose, with no error to trace.
        Now it raises KeyError/ValueError so corruption is loud.
        """
        if key not in data.files:
            raise KeyError(
                f"SOMA npz is missing required key {key!r} — found {data.files!r}. "
                f"Cannot extract SOMA-77 joint positions. The npz is either "
                f"corrupt or an incompatible version. Re-download from "
                f"huggingface-cli download nvidia/soma-x."
            )
        world = np.asarray(data[key], dtype=np.float32)  # (78, 4, 4)
        if world.ndim != 3 or world.shape[1:] != (4, 4):
            raise ValueError(
                f"SOMA npz key {key!r} has unexpected shape {world.shape} — "
                f"expected (N, 4, 4). The npz is corrupt or incompatible."
            )
        joints_all = world[:, :3, 3]  # translation column
        # Drop root joint (index 0) → 77 joints matching Pose Studio.
        if joints_all.shape[0] == 78:
            joints = joints_all[1:78]
        elif joints_all.shape[0] == 77:
            joints = joints_all
        else:
            raise ValueError(
                f"SOMA npz {key!r} has {joints_all.shape[0]} joints — expected "
                f"77 or 78. The npz is corrupt or an incompatible version."
            )
        if joints.shape[0] != 77:
            raise ValueError(
                f"SOMA npz {key!r} yielded {joints.shape[0]} joints after "
                f"dropping root — expected exactly 77. Refusing to pad/truncate."
            )
        return np.asarray(joints[:77], dtype=np.float32)

    def _load_soma_x(self) -> None:
        """Anny/MHR via py-soma-x's SOMALayer (parametric identity)."""
        try:
            from soma import SOMALayer  # py-soma-x
        except ImportError as e:
            raise ImportError(
                f"py-soma-x is required for {self.identity_model} meshes. "
                "Install with: pip install 'py-soma-x[anny]'"
            ) from e

        log.info(
            "SkinEngine: building SOMALayer(identity_model_type=%s, device=%s)",
            self.identity_model,
            self.device,
        )
        # enable_procedural_transforms=False — the procedural JSON isn't
        # shipped in this env. LBS still works; we only lose pose-dependent
        # corrective blends (an elbow-smoothing nicety).
        #
        # data_root: the local HF snapshot (SOMA_neutral.npz + Anny/ +
        # MHR/ ...) — REQUIRED, not an optimization. py-soma-x 0.3.0's
        # no-data_root branch lazy-imports ``soma.body.assets``, a
        # module the 0.3.0 wheel/tag simply does not ship (upstream
        # packaging bug — the real module is soma/assets.py, one level
        # up); passing the already-downloaded snapshot dodges the
        # broken branch entirely and pins the assets to the cache.
        data_root = self._find_soma_npz().parent
        log.info("SkinEngine: data_root=%s", data_root)
        self._model = SOMALayer(
            identity_model_type=self.identity_model,
            device=self.device,
            enable_procedural_transforms=False,
            data_root=data_root,
        )
        # Bind-pose vertices: py-soma-x exposes this as `bind_shape`
        # (18056, 3), NOT `bind_vertices`. `shape_mean` is the same data
        # for the neutral identity. Use bind_shape for the canonical bind.
        verts = getattr(self._model, "bind_shape", None)
        if verts is None:
            verts = getattr(self._model, "shape_mean", None)
        if verts is None:
            raise AttributeError(
                "SOMALayer has neither 'bind_shape' nor 'shape_mean' — "
                f"attrs: {[a for a in dir(self._model) if not a.startswith('_')][:20]}..."
            )
        if hasattr(verts, "detach"):
            verts = verts.detach().cpu().numpy()
        self.vertices = np.asarray(verts, dtype=np.float32)

        faces = self._model.faces
        if hasattr(faces, "detach"):
            faces = faces.detach().cpu().numpy()
        self.faces = np.asarray(faces, dtype=np.int32)

        # Cache a neutral identity BEFORE extracting joints.
        # ``_extract_soma_x_joints()`` calls ``public_bind_transforms_world()``
        # which requires a cached identity — calling it first avoids the
        # "No cached identity" warning at startup.
        #
        # IMPORTANT: prepare_identity accepts PHENOTYPE dims (not PCA dims).
        # anny wants 11 phenotype values; mhr wants 45 AND scale_params (1,68).
        # Passing the full 128 PCA dims triggers
        # "phenotype_kwargs tensor must have shape [bs, 11]" (anny) or
        # "einsum subscript n has size 128 ... does not broadcast with 45" (mhr).
        #
        # CRITICAL: anny's 11 phenotype dims are gender/age/muscle/weight/
        # height/proportions/cupsize/firmness/african/asian/caucasian, each
        # in [0,1] with 0.5 = neutral adult midpoint. Using torch.zeros
        # produces an INFANT/TINY body (age+height at bottom of range) —
        # _cached_rest_shape collapses to ~0.44m instead of ~1.62m. This was
        # the root cause of the "distorted and broken" generate mesh.
        # Verified: prepare_identity([0.5]*11) → height 1.6229m (correct).
        # bind_joints is set AFTER the identity cache is prepared (below).
        # NO FALLBACK: previously this assigned NEUTRAL_JOINTS as a "provisional"
        # value, then tried prepare_identity in a try/except that swallowed the
        # error and left bind_joints as NEUTRAL_JOINTS silently. That masked
        # every load-time failure — bad phenotype dims, missing model graph,
        # broken kimodo install — as "neutral pose" output. Now we let the
        # exception propagate so load failures are LOUD.
        n_phen = _PHENOTYPE_DIMS.get(self.identity_model, 128)
        if self.identity_model == "anny":
            # 0.5 = neutral adult midpoint for every phenotype axis.
            neutral = torch.full(
                (1, n_phen), 0.5, dtype=torch.float32, device=self.device
            )
        else:
            neutral = torch.zeros(
                (1, n_phen), dtype=torch.float32, device=self.device
            )
        kw: dict[str, Any] = {}
        sp_dims = _SCALE_PARAM_DIMS.get(self.identity_model)
        if sp_dims is not None:
            kw["scale_params"] = torch.ones(
                (1, sp_dims), dtype=torch.float32, device=self.device
            )
        elif self.identity_model == "anny":
            # anny's identity chain passes scale_params as local_changes_kwargs
            # to get_phenotype_blendshape_coefficients; None → TypeError.
            # An empty dict is the correct no-op.
            kw["scale_params"] = _ANNY_SCALE_PARAMS
        self._model.prepare_identity(neutral, **kw)
        log.info("SOMALayer neutral identity cached for %s", self.identity_model)

        # ── Update self.vertices from _cached_rest_shape ──
        # bind_shape is the SHARED canonical SOMA reference mesh (same for
        # every identity). _cached_rest_shape is the identity-SPECIFIC
        # shaped mesh that pose() skins from. Serving bind_shape in GET
        # /mesh caused the "SOMA mesh appears, then pops to anny on
        # animation" bug — the bind mesh and the animated mesh were 15cm
        # apart per vertex. Now we serve the SAME mesh pose() uses.
        rest = getattr(self._model, "_cached_rest_shape", None)
        if rest is None:
            raise RuntimeError(
                f"SOMALayer.prepare_identity() for {self.identity_model} did "
                f"not populate _cached_rest_shape — cannot extract shaped "
                f"vertices. The py-soma-x version is likely incompatible."
            )
        if hasattr(rest, "detach"):
            rest = rest.detach().cpu().numpy()
        rest = np.asarray(rest, dtype=np.float32)
        if rest.ndim == 3 and rest.shape[0] == 1:
            rest = rest[0]
        # _cached_rest_shape is in METERS; convert → centimeters to
        # match the frontend convention (CM_TO_M = 0.01).
        self.vertices = (rest * 100.0).astype(np.float32)
        log.info(
            "self.vertices updated from _cached_rest_shape (%s): "
            "V=%d, height=%.1f cm",
            self.identity_model,
            self.vertices.shape[0],
            float(self.vertices[:, 1].max() - self.vertices[:, 1].min()),
        )

        # ── Cache the SOMA-77 skeleton for global→local rotation ──
        # conversion in deform(). Uses kimodo's proven SOMASkeleton77
        # (the same class the original SomaXSkin adapter relied on).
        # root_idx=0 (pelvis), neutral_root=[0,0,0] for SOMA77.
        from kimodo.skeleton import SOMASkeleton77
        self._skel = SOMASkeleton77(load=True).to(self.device)
        self._root_idx = int(self._skel.root_idx)
        self._neutral_root = (
            self._skel.neutral_joints[self._root_idx:self._root_idx + 1]
            .clone()
            .to(self.device)
        )

        # NOW extract joints — the identity cache is populated, so this
        # returns the true per-identity joint positions.
        self.bind_joints = self._extract_soma_x_joints()

    def _extract_soma_x_transforms(self) -> np.ndarray:
        """Pull SOMA-77 world bind transforms (4×4 per joint) from the SOMALayer.

        Returns shape ``(77, 4, 4)`` — float32, numpy. Each row is the
        world-space bind transform for the corresponding SOMA-77 public joint
        (Hips at index 0, Spine1 at index 1, ..., RightToeEnd at index 76).

        ``public_bind_transforms_world()`` returns a *batched* tensor
        ``(1, 78, 4, 4)`` from the upstream SOMA layer — the 78 entries are
        ``[Root, Hips, Spine1, ..., RightToeEnd]`` (Root at index 0). We drop
        Root and keep the 77 SOMA joints via ``public_joint_indices`` (which
        is ``[1, 2, ..., 77]``). This is body-type-aware: after
        ``set_identity(coeffs)``, both joint POSITIONS (from
        ``skeleton_transfer.fit``) and canonical ROTATIONS (from
        ``bind_pose_local``) are reflected here.

        Used by ``POST /poser/skin/bind_pose`` to ship the full pose
        (positions + rotations) to the ControlNet rendering pipeline.

        SOMA path: returns ``self._soma_bind_world_77`` directly — these
        are the canonical bind transforms cached from ``bind_pose_world``
        in the npz at load time. SOMA has no parametric identity model,
        so there is no dynamic recomputation step.

        Regression note (2026-07-19): previously this sliced
        ``transforms[:77]`` which kept ``[Root, Hips, ..., RightToeBase]`` —
        i.e. Root (identity) was labeled as "Hips", every other joint was
        fed its parent's rotation, and RightToeEnd was dropped. The
        resulting deformation had every joint rotated by its parent's
        bind rotation, producing a severely mangled mesh that the
        ``(x,y,z) → (x,z,-y)`` frame-fix in ``render_mesh_and_skeleton``
        merely reoriented to look vertical. Fixed by indexing via
        ``public_joint_indices`` (the canonical drop-Root selector).
        """
        if self.identity_model == "soma":
            # Cached at npz load time — see _load_soma_npz.
            if not hasattr(self, "_soma_bind_world_77") or \
                    self._soma_bind_world_77 is None:
                raise RuntimeError(
                    "SOMA engine did not cache _soma_bind_world_77 at load "
                    "time — npz load is incomplete. Cannot serve bind_pose."
                )
            return np.asarray(self._soma_bind_world_77, dtype=np.float32)
        # anny / mhr path: extract from the live SOMALayer model.
        assert self._model is not None
        method = getattr(self._model, "public_bind_transforms_world", None)
        if not callable(method):
            raise AttributeError(
                f"SOMALayer for {self.identity_model} has no callable "
                f"'public_bind_transforms_world' method — py-soma-x version "
                f"is incompatible with this engine. Cannot extract SOMA-77 "
                f"world bind transforms."
            )
        transforms = method()
        if hasattr(transforms, "detach"):
            transforms = transforms.detach().cpu().numpy()
        transforms = np.asarray(transforms, dtype=np.float32)
        # Squeeze leading batch dims: (1, 78, 4, 4) → (78, 4, 4).
        while transforms.ndim > 3:
            transforms = transforms[0]
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
            raise ValueError(
                f"public_bind_transforms_world() returned shape {transforms.shape} "
                f"— expected (N, 4, 4). py-soma-x layout has changed; "
                f"engine needs updating."
            )
        # Drop Root (index 0), keep SOMA77 (indices 1..77) via
        # public_joint_indices. This is the ONLY correct selector — Root
        # is index 0, SOMA body joints are 1..77. NO SLICE FALLBACK.
        pji = getattr(self._model, "public_joint_indices", None)
        if pji is None:
            raise AttributeError(
                f"SOMALayer for {self.identity_model} has no 'public_joint_indices' "
                f"attribute. py-soma-x must expose the canonical 77-joint "
                f"selector (Root-dropped). Without it, joint indexing is "
                f"ambiguous and the previously-hardcoded [1:78] slice caused "
                f"the parent-rotation bug (commit reg 2026-07-19)."
            )
        if hasattr(pji, "detach"):
            pji = pji.detach().cpu().numpy()
        pji = np.asarray(pji, dtype=np.int64)
        if pji.shape[0] != 77 or pji.max() >= transforms.shape[0]:
            raise ValueError(
                f"public_joint_indices is malformed: shape={pji.shape}, "
                f"max={int(pji.max())}, transforms.shape[0]={transforms.shape[0]}. "
                f"Expected exactly 77 indices, all < {transforms.shape[0]}."
            )
        return np.asarray(transforms[pji], dtype=np.float32)

    def _extract_soma_x_joints(self) -> np.ndarray:
        """Pull SOMA-77 joint POSITIONS (77, 3) from the SOMALayer.

        Thin wrapper around :meth:`_extract_soma_x_transforms` that drops
        the rotation half. Kept for backward compatibility with callers
        that only need positions (e.g. ``GET /poser/skin/bind_joints``).
        """
        return self._extract_soma_x_transforms()[:, :3, 3]

    def _find_soma_npz(self) -> Path:
        """Locate SOMA_neutral.npz in the HF cache (under /mnt/data/models)."""
        candidates = list(
            Path("/mnt/data/models").rglob("models--nvidia--soma-x/snapshots/*/SOMA_neutral.npz")
        )
        if not candidates:
            raise FileNotFoundError(
                "SOMA_neutral.npz not found under /mnt/data/models/. "
                "Run: huggingface-cli download nvidia/soma-x"
            )
        return candidates[0]

    # ── Mutation ───────────────────────────────────────────────────────────
    def set_identity(
        self,
        coeffs: list[float] | np.ndarray | None = None,
        local_changes: dict | None = None,
        custom_targets: dict | None = None,
    ) -> None:
        """Apply identity morphs (phenotype + detail + stylization).

        Uses py-soma-x's ``prepare_identity(identity_coeffs, kwargs=...)``.
        SOMA ignores this — it has no parametric identity layer.

        Defensive: if the model's identity API rejects the coefficients (e.g.
        anny wants 11 phenotype dims, not the 128 PCA components), logs a
        warning and leaves the neutral bind mesh in place. The GET endpoint
        always works; this only affects interactive shaping.
        """
        if self._model is None:
            return  # SOMA npz — no identity morphing

        if coeffs is None:
            coeffs = self._default_coeffs()

        coeffs_np = np.asarray(coeffs, dtype=np.float32).flatten()
        # Truncate to phenotype dims — prepare_identity for anny accepts 11
        # phenotype values, not the 128 PCA components the frontend may send.
        n_phen = _PHENOTYPE_DIMS.get(self.identity_model, len(coeffs_np))
        if len(coeffs_np) > n_phen:
            coeffs_np = coeffs_np[:n_phen]
        elif len(coeffs_np) < n_phen:
            coeffs_np = np.pad(coeffs_np, (0, n_phen - len(coeffs_np)))
        coeffs_t = torch.as_tensor(coeffs_np, dtype=torch.float32, device=self.device)
        if coeffs_t.ndim == 1:
            coeffs_t = coeffs_t.unsqueeze(0)

        local_kwargs = self._sanitize_local_changes(local_changes or {})
        if custom_targets:
            local_kwargs.update(custom_targets)

        # scale_params routing differs by model:
        #   MHR: needs a (1,68) tensor — local_changes go through kwargs.
        #   Anny: AnnyIdentityModel.get_rest_shape passes scale_params AS
        #         local_changes_kwargs to AnnySimplified.forward, and IGNORES
        #         the kwargs parameter. So we must merge local_changes morphs
        #         directly into scale_params for anny.
        #   SOMA: not on this path (npz engine, no identity morphing).
        call_kw: dict[str, Any] = {}
        sp_dims = _SCALE_PARAM_DIMS.get(self.identity_model)
        if sp_dims is not None:
            call_kw["scale_params"] = torch.ones(
                (1, sp_dims), dtype=torch.float32, device=self.device
            )
            if local_kwargs:
                call_kw["kwargs"] = local_kwargs
        elif self.identity_model == "anny":
            # Merge local_changes morphs into scale_params — they flow
            # through as local_changes_kwargs in AnnySimplified.forward.
            merged = {**_ANNY_SCALE_PARAMS, **local_kwargs}
            call_kw["scale_params"] = merged
        elif local_kwargs:
            # NO FALLBACK: this identity model has no routing path for
            # local_changes morphs. Previously they were silently dropped
            # (with a log.error). Now we raise — caller asked for morphs,
            # they must be applied or the call is broken.
            raise NotImplementedError(
                f"set_identity({self.identity_model}): {len(local_kwargs)} "
                f"local_changes morphs supplied, but this identity model has "
                f"no routing path for them. Morphs: "
                f"{list(local_kwargs.keys())[:5]}. Either use a supported "
                f"identity_model (anny/mhr) or do not pass local_changes."
            )

        with _lock:
            # NO FALLBACK: previously this swallowed prepare_identity errors
            # and left the previous (or neutral) mesh in place. That made
            # interactive Body Studio shaping fail silently — user moves a
            # slider, nothing changes, no error visible. Now it raises so
            # the frontend HTTP layer surfaces the failure to the UI.
            try:
                self._model.prepare_identity(coeffs_t, **call_kw)
            except Exception as e:
                raise RuntimeError(
                    f"set_identity(prepare_identity) FAILED for "
                    f"{self.identity_model} — coeffs={coeffs_np.tolist()[:4]}, "
                    f"call_kw keys={list(call_kw.keys())}. "
                    f"Refusing to leave the previous mesh in place — that "
                    f"silently masks every shaping failure. Error: {e}"
                ) from e
            # Refresh bind vertices from the reshaped model.
            #
            # prepare_identity stores the shaped mesh in _cached_rest_shape
            # (NOT bind_shape — that stays as the neutral init mesh forever).
            # _cached_rest_shape is in meters; the frontend expects cm (same
            # as bind_shape). Multiply by 100 to keep the existing pipeline.
            verts = getattr(self._model, "_cached_rest_shape", None)
            if verts is None:
                raise RuntimeError(
                    f"set_identity({self.identity_model}): _cached_rest_shape "
                    f"MISSING after prepare_identity — py-soma-x did not "
                    f"populate the shaped-mesh cache. Cannot continue with "
                    f"stale vertices; that would silently serve the wrong mesh."
                )
            if hasattr(verts, "detach"):
                verts = verts.detach().cpu().numpy()
            verts = np.asarray(verts, dtype=np.float32)
            # Squeeze batch dimension: (1, V, 3) → (V, 3)
            if verts.ndim == 3 and verts.shape[0] == 1:
                verts = verts[0]
            # Convert meters → centimeters to match bind_shape convention
            # used throughout the frontend (CM_TO_M = 0.01 in bodyMesh.ts).
            verts = verts * 100.0
            self.vertices = verts
            # Refresh joints.
            self.bind_joints = self._extract_soma_x_joints()
            # deform() now uses model.pose() directly (no LBS cache to
            # invalidate). The SOMALayer's internal _cached_rest_shape was
            # already refreshed by prepare_identity() above.

    def _default_coeffs(self) -> np.ndarray:
        """Neutral identity coefficients — sized to phenotype dims.

        anny: 0.5 across all 11 dims (neutral adult midpoint — zeros produce
              an infant/tiny body, the root cause of the crushed-mesh bug).
        mhr:  zeros (the neutral-fit npz is loaded separately when available).
        """
        n = _PHENOTYPE_DIMS.get(self.identity_model, 0)
        if self.identity_model == "anny":
            return np.full(n, 0.5, dtype=np.float32)
        return np.zeros(n, dtype=np.float32)

    def _sanitize_local_changes(self, local_changes: dict) -> dict:
        """Coerce local_changes values to float; drop anything weird."""
        out = {}
        for k, v in local_changes.items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
        return out

    # ── Skinning ───────────────────────────────────────────────────────────
    def deform(
        self,
        joints_pos: np.ndarray,
        joints_rot: np.ndarray | None = None,
    ) -> np.ndarray:
        """Deform the bind mesh for a motion sequence via py-soma-x's pose().

        This is the PROVEN anny-to-kimodo transformation, recovered from the
        git-history ``media/motion/soma_x_skin.py`` (commit d389227) and
        ``vendor/kimodo/kimodo/viz/soma_layer_skin.py``:

          1. Convert Kimodo's GLOBAL rotation matrices to LOCAL (parent-
             relative) rotations via kimodo's global_rots_to_local_rots().
          2. Compute root translation = joint_pos[:, root_idx] - neutral_root.
             For SOMA77, neutral_root = [0,0,0], so transl = HIPS position.
          3. Call model.pose(local_rots, transl, pose2rot=False,
             apply_correctives=False) — py-soma-x's native LBS + correctives.

        The previous manual-LBS implementation was distorted because it used
        bind rotations from public_bind_transforms_world() which live in a
        different rotational space than the posed global rotations. py-soma-x's
        pose() handles the bind↔pose rotation conversion internally and
        correctly — that's what it was built for.

        Args:
            joints_pos: (T, 77, 3) or (77, 3) — joint positions in METERS
                        (Kimodo's native output unit; pelvis Y ≈ 1.0).
            joints_rot: (T, 77, 3, 3) or (77, 3, 3) or None — GLOBAL rotation
                        matrices (from Kimodo's global_rot_mats output). When
                        None, identity rotations are used (bind-pose pose).
        Returns:
            (T, V, 3) deformed vertices in METERS, in the world frame (root
            at joint_pos[:, root_idx], feet near floor for a standing pose).
        """
        if self._model is None:
            raise RuntimeError("SOMA npz engine doesn't support deform — use anny/mhr")
        if self._skel is None or self._neutral_root is None:
            raise RuntimeError(
                f"SOMASkeleton77 not initialised for {self.identity_model} — "
                f"deform() unavailable. Check load-time logs for "
                f"prepare_identity/skeleton init errors."
            )

        from kimodo.skeleton import global_rots_to_local_rots

        pos_np = np.asarray(joints_pos, dtype=np.float32)
        if pos_np.ndim == 2:
            pos_np = pos_np[None, ...]
        if pos_np.shape[1] != 77:
            raise ValueError(f"Expected 77 SOMA joints, got {pos_np.shape[1]}")
        T = pos_np.shape[0]

        # ── Normalise rotation matrices to (T, 77, 3, 3) ──
        if joints_rot is not None:
            global_rots = np.asarray(joints_rot, dtype=np.float32)
            if global_rots.ndim == 3:
                global_rots = global_rots[None, ...]
            if global_rots.shape[1] != 77 or global_rots.shape[2:] != (3, 3):
                raise ValueError(
                    f"joints_rot must be (T, 77, 3, 3), got {global_rots.shape}"
                )
        else:
            # NO FALLBACK: identity rotations silently produce T-pose,
            # which masks every upstream pose-regression bug. The render
            # looks like a body but is in the wrong pose — undetectable
            # without comparing joint positions to expected values.
            # Caller MUST pass frames_soma77_rotmat explicitly.
            raise ValueError(
                "deform() requires joints_rot (frames_soma77_rotmat). "
                "Refusing to fall back to identity rotations — that "
                "silently renders T-pose regardless of the joint "
                "positions passed in, masking bugs in pose regression. "
                "Pass the canonical A-pose rotations from "
                "/poser/skin/bind_pose or the framed pose from a pose "
                "library fetch."
            )

        # ── Step 1: global → local rotations (kimodo's proven conversion) ──
        global_rots_t = torch.from_numpy(global_rots).to(self.device)
        local_rots = global_rots_to_local_rots(global_rots_t, self._skel)

        # ── Step 2: root translation = HIPS position − neutral root ──
        # SOMA77's neutral_root is [0,0,0], so transl = joint_pos[:, 0].
        # joint_pos is in METERS (Kimodo's native unit).
        root_pos = torch.from_numpy(pos_np[:, self._root_idx]).to(self.device)
        transl = root_pos - self._neutral_root.to(self.device)

        # ── Step 3: py-soma-x's native pose() — correct LBS + correctives ──
        # absolute_pose=True is mandatory here: global_rots_to_local_rots()
        # produces ABSOLUTE local rotations (parent-relative full orientation),
        # not T-pose-relative deltas. Passing them to pose() with the default
        # absolute_pose=False would double-apply the T-pose joint orient via
        # apply_joint_orient_local (R_out = orient_parent_T @ R @ orient),
        # producing a mangled body lying along the Z-axis instead of standing
        # in Y-up. Verified empirically 2026-07-19:
        #   pose(bind_local, absolute_pose=True)  → matches bind pose exactly
        #   pose(bind_local, absolute_pose=False) → Z-up mangled (body tilted)
        # See soma.geometry.rig_utils.remove_joint_orient_local docstring:
        #   "absolute local rotations (with orient baked in)" require
        #   absolute_pose=True (or removal via remove_joint_orient_local).
        try:
            with torch.inference_mode():
                out = self._model.pose(
                    local_rots.to(dtype=torch.float32),
                    transl=transl.to(dtype=torch.float32),
                    pose2rot=False,
                    apply_correctives=False,
                    absolute_pose=True,
                )
        except Exception as e:
            raise RuntimeError(
                f"model.pose() FAILED for {self.identity_model} — "
                f"T={T}, local_rots={local_rots.shape}, transl={transl.shape}. "
                f"This is the core skinning call; if it fails, check that "
                f"prepare_identity was called with the correct phenotype "
                f"(anny wants [0.5]*11). Error: {e}"
            ) from e

        verts = out["vertices"] if isinstance(out, dict) else out.vertices
        verts_np = verts.detach().cpu().numpy().astype(np.float32)
        # verts_np: (T, V, 3) in METERS, world frame.
        return verts_np


# ── Identity metadata ───────────────────────────────────────────────────────

_DEFAULT_NUM_COMPS = 128

# Anny's 11 phenotype dimensions in tensor order (matches the raw anny
# model's phenotype_labels when all_phenotypes=True). Values are 0..1 with
# 0.5 = neutral adult.
_ANNY_PHENOTYPE_LABELS = [
    "gender", "age", "muscle", "weight", "height", "proportions",
    "cupsize", "firmness", "african", "asian", "caucasian",
]

# Body-part keywords that map to the "face" category for local_changes
# grouping. Everything else is "body".
_FACE_PARTS = frozenset({
    "head", "eye", "eyes", "eyebrow", "eyelid", "nose", "nostril",
    "mouth", "lip", "lips", "teeth", "jaw", "chin", "cheek", "cheeks",
    "ear", "ears", "forehead", "scalp", "smile",
})

# Mapping from individual body-part prefixes to display group names.
# Morph labels are kebab-case like "upperarm-fat-incr"; the first segment
# (after optional "measure-" prefix) is the body part.
_PART_GROUP_MAP = {
    # Arms
    "upperarm": "Arms", "lowerarm": "Arms", "forearm": "Arms",
    "elbow": "Arms", "hand": "Hands", "hands": "Hands",
    "finger": "Hands", "fingers": "Hands", "wrist": "Hands",
    # Legs
    "thigh": "Legs", "calf": "Legs", "shin": "Legs", "knee": "Legs",
    "leg": "Legs", "legs": "Legs", "foot": "Feet", "feet": "Feet",
    "toe": "Feet", "toes": "Feet",
    # Torso
    "torso": "Torso", "back": "Torso", "spine": "Torso",
    "stomach": "Torso", "belly": "Torso", "waist": "Torso",
    "abdomen": "Torso", "rib": "Torso", "ribs": "Torso",
    "chest": "Chest", "breast": "Chest", "breasts": "Chest",
    "nipple": "Chest", "pectoral": "Chest",
    # Hips
    "hip": "Hips", "hips": "Hips", "pelvis": "Hips",
    "buttock": "Hips", "buttocks": "Hips", "glute": "Hips",
    # Shoulders
    "shoulder": "Shoulders", "shoulders": "Shoulders",
    "clavicle": "Shoulders", "scapula": "Shoulders",
    # Neck
    "neck": "Neck", "throat": "Neck",
    # Face — grouped per feature
    "head": "Head", "skull": "Head", "cranium": "Head", "scalp": "Head",
    "forehead": "Forehead",
    "eye": "Eyes", "eyes": "Eyes", "eyebrow": "Eyes", "eyelid": "Eyes",
    "nose": "Nose", "nostril": "Nose",
    "mouth": "Mouth", "lip": "Mouth", "lips": "Mouth", "teeth": "Mouth",
    "jaw": "Jaw",
    "chin": "Chin",
    "cheek": "Cheeks", "cheeks": "Cheeks",
    "ear": "Ears", "ears": "Ears",
}


def _group_local_change_labels(labels: list[str]) -> list[dict]:
    """Organize flat morph-label list into categorized groups for the frontend.

    Returns a list of ``{label, indices, category}`` dicts sorted body-first
    then face, alphabetically within each category.
    """
    part_indices: dict[str, list[int]] = {}
    for i, raw in enumerate(labels):
        parts = raw.split("-")
        # Strip "measure-" prefix (just flags it as a measurement morph).
        if parts[0] == "measure" and len(parts) > 1:
            parts = parts[1:]
        prefix = parts[0] if parts else raw
        group_name = _PART_GROUP_MAP.get(prefix)
        if group_name is None:
            # Try singular/plural variants.
            group_name = _PART_GROUP_MAP.get(prefix.rstrip("s"), prefix.title())
        part_indices.setdefault(group_name, []).append(i)

    # Sort: body groups first, then face, alphabetical within each.
    def _sort_key(item):
        name = item[0]
        is_face = any(name.lower().startswith(fp) for fp in _FACE_PARTS)
        return (1 if is_face else 0, name.lower())

    groups = []
    for name, indices in sorted(part_indices.items(), key=_sort_key):
        is_face = any(name.lower().startswith(fp) for fp in _FACE_PARTS)
        groups.append({
            "label": name,
            "indices": indices,
            "category": "face" if is_face else "body",
        })
    return groups


def _scan_makehuman_targets() -> dict | None:
    """Scan references/makehuman_targets/*.target for community blendshapes."""
    # Search multiple candidate paths — the repo root differs between local
    # dev and Docker (references mounted at /app/references in the container).
    candidates = [
        Path("/app/references/makehuman_targets"),                      # Docker mount
        Path(__file__).resolve().parents[2] / "references" / "makehuman_targets",  # local dev
        Path.cwd() / "references" / "makehuman_targets",                # CWD fallback
    ]
    targets_dir = next((p for p in candidates if p.is_dir()), None)
    if targets_dir is None:
        return None

    items = []
    for tf in sorted(targets_dir.glob("*.target")):
        stem = tf.stem
        target_id = stem.lower().replace(" ", "_").replace("-", "_")
        display = stem.replace("_", " ").replace("-", " ").title()
        items.append({
            "id": target_id,
            "name": display,
            "description": "Community MakeHuman .target blendshape",
            "glyph": "✨",
        })
    if not items:
        return None
    return {
        "items": items,
        "defaults": [0.0] * len(items),
        "range": [0.0, 1.5],
    }


def identity_space(model: str) -> dict:
    """Identity coefficient metadata for the requested model.

    For anny: returns 11 labeled phenotype dims (gender, age, muscle…),
    full local_changes morph metadata (189+ detail morphs organized into
    body/face groups), and community MakeHuman .target blendshapes.
    For mhr: generic pc_NN PCA labels.
    For soma: no phenotype sliders — fixed neutral mesh.
    """
    model = (model or "soma").lower()
    if model == "soma":
        return {
            "identity_model": "soma",
            "dims": 0,
            "labels": [],
            "defaults": [],
            "range": [0.0, 0.0],
            "groups": None,
        }

    n = _PHENOTYPE_DIMS.get(model, _DEFAULT_NUM_COMPS)

    # ── Anny: rich metadata via engine introspection ───────────────────
    # NO FALLBACK: previously, on introspection failure, identity_space()
    # silently returned generic pc_NN labels with _degraded=True. The
    # frontend would then render "pc_00 / pc_01 / ..." sliders for an Anny
    # body — completely non-functional but no error surfaced to the user.
    # Now we let _anny_identity_space raise and propagate up to the HTTP
    # layer so the failure is visible in the API response and docker logs.
    if model == "anny":
        return _anny_identity_space(n)

    # ── Generic labels (mhr only — pc_NN PCA components) ───────────────
    labels = [f"pc_{i:02d}" for i in range(n)]
    return {
        "identity_model": model,
        "dims": n,
        "labels": labels,
        "defaults": [0.0] * n,
        "range": [-3.0, 3.0],
        "groups": None,
    }


def _anny_identity_space(n_phen: int) -> dict:
    """Build rich identity-space metadata for Anny by introspecting the engine.

    Navigates SOMALayer → AnnyIdentityModel → AnnySimplified → raw anny_model
    to pull the real phenotype_labels and local_change_labels.

    NO FALLBACK: raises on any introspection failure. The previous
    ``return None`` pattern silently downgraded the entire Body Studio
    UI to generic pc_NN labels — non-functional but no visible error.
    """
    try:
        engine = get_engine("anny")
    except Exception as e:
        raise RuntimeError(
            f"anny identity_space: engine not available ({e}). The Anny "
            f"SOMALayer must be loaded before identity_space() is called. "
            f"This is a deployment ordering bug, not a transient condition."
        ) from e

    try:
        soma_layer = engine._model  # SOMALayer
        # SOMALayer.identity_model → AnnyIdentityModel
        anny_identity = soma_layer.identity_model
        # AnnyIdentityModel.identity_model → AnnySimplified
        anny_simplified = anny_identity.identity_model
        # AnnySimplified.anny_model → raw anny model with full labels
        raw_anny = anny_simplified.anny_model
        local_labels = list(raw_anny.local_change_labels)
    except (AttributeError, TypeError) as e:
        raise RuntimeError(
            f"anny identity_space: introspection failed ({e}). The py-soma-x "
            f"version exposes a different attribute graph than expected — "
            f"SOMALayer → AnnyIdentityModel → AnnySimplified → anny_model. "
            f"Check py-soma-x version compatibility."
        ) from e

    if not local_labels:
        raise RuntimeError(
            "anny identity_space: raw_anny.local_change_labels is empty. "
            "The Anny model graph is incomplete or corrupt — no detail morph "
            "labels available. The Body Studio cannot function without these."
        )

    phen_labels = _ANNY_PHENOTYPE_LABELS[:n_phen]
    groups = _group_local_change_labels(local_labels)
    custom_targets = _scan_makehuman_targets()

    return {
        "identity_model": "anny",
        "dims": n_phen,
        "labels": phen_labels,
        "defaults": [0.5] * n_phen,
        "range": [0.0, 1.0],
        "groups": None,
        "local_changes": {
            "dims": len(local_labels),
            "labels": local_labels,
            "defaults": [0.0] * len(local_labels),
            "range": [-1.0, 2.0],
            "groups": groups,
        },
        "custom_targets": custom_targets,
    }
