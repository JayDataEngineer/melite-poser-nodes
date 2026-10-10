"""Blender headless GLB renderer — multi-angle previews for quality verification.

Runs ``blender --background --python`` as a subprocess inside the inference
container (Blender 4.2.5 LTS, engine ``BLENDER_EEVEE_NEXT``). Produces clean,
well-lit PNG renders from specified camera angles. Used by the
``/poser/render_glb`` endpoint for the harsh 9-10/10 visual critique of every
pipeline stage.

Design decisions:
  • Subprocess (not ``import bpy``) — avoids polluting ComfyUI's Python env.
  • EEVEE_NEXT — fast (GPU), good enough for preview. Cycles optional.
  • 3-point lighting + gray studio background — consistent, repeatable.
  • Track-To constraint on camera — no Euler-sign guesswork (Discovery 5).
  • Transparent film option — for compositing / clean inspection.

All file paths are container-internal (``/tmp/blender_render_*``); the server
endpoint handles base64 transport across the HTTP boundary.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import logging

log = logging.getLogger(__name__)

def _resolve_blender_bin() -> str:
    """BLENDER_BIN env → PATH → the ray repo's portable Blender
    (host-clone topology: the workstation has no system blender and
    no /usr/local write access; the workspace-tools copy is the
    documented fallback) → LOUD FileNotFoundError. The ONE law,
    shared verbatim with melite-autorig-nodes/nodes.py
    (features/008 M13, cured 2026-10-21): the old silent
    fall-through returned a bare "blender" that died at subprocess
    spawn with a worse error than this refusal."""
    env = os.environ.get("BLENDER_BIN")
    if env and os.path.isfile(env):
        return env
    import shutil
    on_path = shutil.which("blender")
    if on_path:
        return on_path
    from pathlib import Path
    portable = Path.home() / "Documents/programs/ray/scratch/blender/blender-4.2.5-linux-x64/blender"
    if portable.is_file():
        return str(portable)
    raise FileNotFoundError(
        "Blender binary not found. Set BLENDER_BIN, put 'blender' on "
        "PATH, or provide the portable copy at "
        f"{portable}"
    )


# ── The Blender-side script (run via --python) ─────────────────────────────

_RENDER_SCRIPT = r'''
import bpy, sys, json, math, os

# Parse args after "--" separator (Blender passes them through)
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []
glb_path = argv[0]
output_dir = argv[1]
config = json.loads(argv[2])

# --- Clear scene ---
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# --- Import GLB ---
bpy.ops.import_scene.gltf(filepath=glb_path)

# --- Material fix: glTF pipelines (SOMAX/TRELLIS) emit a metallic-roughness
# map as a WebP that Blender 4.2 cannot load from packed data ("unknown file-
# format"). The glTF importer still links Metallic+Roughness to a Separate
# Color node reading that broken image, so the sockets get driven to 1.0
# (dark mirror-metal that reflects only the world background, hiding the
# baseColorTexture entirely). Links override default_value, so we must
# unlink the Metallic + Roughness sockets, THEN set sane fixed values.
# (Mirror of MultiAngleGLBViewer's client-side material fix.)
for mat in bpy.data.materials:
    if not mat.use_nodes:
        continue
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if not bsdf:
        continue
    for sock_name in ("Metallic", "Roughness"):
        sock = bsdf.inputs.get(sock_name)
        if sock and sock.is_linked:
            for link in list(sock.links):
                nt.links.remove(link)
        if sock:
            sock.default_value = 0.0 if sock_name == "Metallic" else 0.55

    # Base-color texture: force-reload every image datablock and set sRGB color
    # space. TRELLIS/SOMAX GLBs pack textures that the importer sometimes leaves
    # in Non-Color (linear) space or fails to fully load → flat gray renders.
    # Ensure the Image Texture → Principled Base Color link is intact and the
    # image is treated as sRGB color data.
    base_sock = bsdf.inputs.get("Base Color")
    for node in nt.nodes:
        if node.bl_idname == 'ShaderNodeTexImage':
            img = node.image
            if img is not None:
                try:
                    img.reload()
                except Exception:
                    pass
                # Color textures use sRGB; data maps (normal/MR) stay Non-Color.
                # Heuristic: if this node feeds Base Color/Emission, it's sRGB.
                try:
                    feeds_color = any(
                        l.from_node == node and l.to_node == bsdf and
                        l.to_socket.name in ("Base Color", "Emission Color", "Emission")
                        for l in nt.links
                    )
                    if feeds_color and img.colorspace_settings.name != 'sRGB':
                        img.colorspace_settings.name = 'sRGB'
                except Exception:
                    pass

    # --- GATE-RENDER MATERIAL LAW (2026-08-25, bisect-proven) ---------------
    # This container's Blender runs on a degraded GL context (EGL_BAD_MATCH at
    # render start). Bisect evidence: ANY material whose shader SAMPLES an
    # image texture fails its draw PSO and the object is silently skipped
    # (invisible mesh, world-only pixels stdev<1). Opaque/alpha/MR fixes do
    # NOT help; deleting the image node restores rendering (stdev ~5.7).
    # Law: gate renders never sample images. Replace every Base Color image
    # link with that image's own CPU-computed mean color (species tint kept,
    # GL dependency gone); pure-black bakes fall back to neutral gray.
    mat.blend_method = 'OPAQUE'
    alpha_sock = bsdf.inputs.get("Alpha")
    if alpha_sock is not None:
        for link in list(alpha_sock.links):
            nt.links.remove(link)
        alpha_sock.default_value = 1.0
    base_sock = bsdf.inputs.get("Base Color")
    if base_sock is not None:
        for node in list(nt.nodes):
            if node.bl_idname != 'ShaderNodeTexImage':
                continue
            if not any(l.from_node == node and l.to_socket == base_sock for l in nt.links):
                continue
            mean_rgb = None
            img = node.image
            if img is not None:
                try:
                    w, h = img.size[0], img.size[1]
                    buf = [0.0] * (w * h * 4)
                    img.pixels.foreach_get(buf)
                    step = max(1, (w * h) // 4096)
                    rs = gs = bs_ = cnt = 0.0
                    for i in range(0, w * h, step):
                        rs += buf[i*4]; gs += buf[i*4+1]; bs_ += buf[i*4+2]; cnt += 1
                    mean_rgb = (rs / cnt, gs / cnt, bs_ / cnt)
                except Exception:
                    mean_rgb = None
            for link in list(base_sock.links):
                if link.from_node == node:
                    nt.links.remove(link)
            col = (0.62, 0.62, 0.66, 1.0)
            if mean_rgb is not None and sum(mean_rgb) > 0.09:
                col = (min(1.0, mean_rgb[0] * 1.4), min(1.0, mean_rgb[1] * 1.4), min(1.0, mean_rgb[2] * 1.4), 1.0)
            base_sock.default_value = col

# --- Collect mesh objects + compute bounding box ---
# HELPER-MESH EXCLUSION (2026-08-25): SOMAX/transfer_rig GLBs carry parentless
# material-less rig helpers (an Icosphere 2x2x2) that poison the framing box
# AND render as gray junk. Hide them; frame only materialized meshes.
mesh_objs = [o for o in bpy.data.objects if o.type == 'MESH']
if not mesh_objs:
    raise RuntimeError("No mesh objects found in GLB")
_frame_objs = []
for obj in mesh_objs:
    if obj.data.materials:
        _frame_objs.append(obj)
    else:
        obj.hide_render = True
        obj.hide_viewport = True
if _frame_objs:
    mesh_objs = _frame_objs

all_coords = []
for obj in mesh_objs:
    for v in obj.bound_box:
        all_coords.append(obj.matrix_world @ type(obj.location)(v))
if not all_coords:
    raise RuntimeError("No vertices found")

xs = [c.x for c in all_coords]
ys = [c.y for c in all_coords]
zs = [c.z for c in all_coords]
center = type(all_coords[0])((sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)))
height = max(zs) - min(zs)
radius = max(max(xs)-min(xs), max(ys)-min(ys), height) / 2.0

# --- World setup ---
scene.world = bpy.data.worlds.new("StudioWorld")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.15, 0.15, 0.18, 1.0)  # dark gray
bg.inputs[1].default_value = 0.8

# --- Render settings ---
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.resolution_x = config.get("width", 768)
scene.render.resolution_y = config.get("height", 1024)
scene.render.resolution_percentage = 100
scene.render.film_transparent = config.get("transparent", False)
scene.render.image_settings.file_format = 'PNG'
# EEVEE_NEXT quality
scene.eevee.taa_render_samples = config.get("samples", 64)

# --- Camera ---
cam_data = bpy.data.cameras.new("Cam")
cam_data.lens = config.get("focal_length", 50)  # mm
cam_obj = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

# --- FOV-aware framing math ───────────────────────────────────────────
# BUG FIX (2026-07-30): the old ``cam_dist = radius * framing`` heuristic
# ignored the camera's field of view, so a 50mm lens at framing=2.0 put the
# camera at ~1.8m for a 1.8m-tall character — but the visible height at
# that distance was only ~1.3m, CLIPPING the head and feet.
#
# Correct approach: compute the distance from the actual FOV (focal length
# + sensor size + render aspect ratio) so the full bounding box fits in
# frame, then use ``framing`` as a MARGIN multiplier (1.0 = exact fit,
# 1.2 = 20% headroom). This matches the formula already used by
# rasterize.py and texture_project.py (``cam_dist = R / tan(fov/2)``).
sensor_w = cam_data.sensor_width  # 36mm default (full frame)
focal = cam_data.lens
res_x = config.get("width", 768)
res_y = config.get("height", 768)
aspect = res_x / res_y if res_y > 0 else 1.0
# sensor_fit='AUTO': sensor_width maps to the LONGER render dimension.
if aspect >= 1.0:  # landscape or square
    fov_h = 2 * math.atan(sensor_w / (2 * focal))
    fov_v = 2 * math.atan(math.tan(fov_h / 2) / aspect)
else:  # portrait — sensor_width maps to vertical
    fov_v = 2 * math.atan(sensor_w / (2 * focal))
    fov_h = 2 * math.atan(math.tan(fov_v / 2) * aspect)
# Base distance = fit the bounding box in BOTH dimensions, take the max
# (height is the binding constraint for standing characters).
bbox_w = max(max(xs) - min(xs), max(ys) - min(ys))
_dist_v = (height / 2) / max(math.tan(fov_v / 2), 1e-6)
_dist_h = (bbox_w / 2) / max(math.tan(fov_h / 2), 1e-6)
cam_dist_base = max(_dist_v, _dist_h)

# Track-To constraint (no Euler guesswork — Discovery 5)
track_target = bpy.data.objects.new("TrackTarget", None)
scene.collection.objects.link(track_target)
track_target.location = center
constraint = cam_obj.constraints.new(type='TRACK_TO')
constraint.target = track_target
constraint.track_axis = 'TRACK_NEGATIVE_Z'
constraint.up_axis = 'UP_Y'

# --- 3-point lighting ---
def add_light(name, loc, energy, light_type='AREA', size=2.0):
    ld = bpy.data.lights.new(name, type=light_type)
    ld.energy = energy
    if light_type == 'AREA':
        ld.size = size
    lo = bpy.data.objects.new(name, ld)
    lo.location = loc
    scene.collection.objects.link(lo)
    tc = lo.constraints.new(type='TRACK_TO')
    tc.target = track_target
    tc.track_axis = 'TRACK_NEGATIVE_Z'
    return lo

dist = radius * 4
add_light("Key",  (center.x+dist*0.6, center.y-dist*0.8, center.z+dist*0.4), config.get("key_energy", 800))
add_light("Fill",  (center.x-dist*0.6, center.y-dist*0.5, center.z+dist*0.2), config.get("fill_energy", 300))
add_light("Rim",   (center.x, center.y+dist*0.8, center.z+dist*0.5), config.get("rim_energy", 500))

# --- Render each angle ---
results = []
for view in config.get("views", [{"azimuth": 0, "elevation": 0}]):
    az = math.radians(view.get("azimuth", 0))
    el = math.radians(view.get("elevation", 0))
    # framing is a MARGIN multiplier on the FOV-computed base distance:
    # 1.0 = exact fit, 1.2 = 20% headroom (default), 1.5 = loose.
    cam_dist = cam_dist_base * config.get("framing", 1.2)
    # Spherical → Cartesian (azimuth around Z, elevation from XY plane)
    cam_obj.location = (
        center.x + cam_dist * math.cos(el) * math.sin(az),
        center.y - cam_dist * math.cos(el) * math.cos(az),
        center.z + cam_dist * math.sin(el),
    )
    name = view.get("name", f"az{view.get('azimuth',0)}_el{view.get('elevation',0)}")
    out_path = os.path.join(output_dir, f"{name}.png")
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    results.append({"name": name, "path": out_path})
    print(f"RENDERED:{name}:{out_path}")

print(json.dumps({"results": results}))
'''


def render_glb(
    glb_bytes: bytes,
    views: list[dict] | None = None,
    resolution: int = 800,
    samples: int = 64,
    transparent: bool = False,
    focal_length: int = 50,
    framing: float = 1.2,
    width: int | None = None,
    height: int | None = None,
) -> list[dict]:
    """Render a GLB from multiple angles using Blender headless.

    Args:
        glb_bytes: Raw GLB file bytes.
        views: List of view specs. Each may be:
              - a preset name string: "front", "back", "left", "right",
                "3q_left" (315°), "3q_right" (45°), "top"
              - a dict {azimuth, elevation, name}
              Default: 4 standard angles.
        resolution: Render resolution (square). Used as fallback when
              width/height are not provided.
        width: Render width in pixels. Overrides resolution.
        height: Render height in pixels. Overrides resolution.
        samples: EEVEE_NEXT TAA samples (higher = cleaner).
        transparent: If True, render with transparent background (for compositing).
        focal_length: Camera focal length in mm.
        framing: Camera margin multiplier on the FOV-computed fit distance.
              1.0 = bounding box exactly fills the frame, 1.2 = 20% headroom
              (default), 1.5 = loose.

    Returns:
        List of {name, png_bytes} dicts.
    """
    # Preset name → {azimuth, elevation}. Azimuth in degrees, 0=front, CW.
    _PRESETS = {
        "front": (0, 5), "back": (180, 5),
        "left": (270, 5), "right": (90, 5),
        "3q_left": (315, 5), "3q_right": (45, 5),
        "front_left": (315, 5), "front_right": (45, 5),
        "top": (0, 89),
    }

    def _normalize(v):
        if isinstance(v, dict):
            name = v.get("name") or f"az{v.get('azimuth', 0)}"
            return {"azimuth": v.get("azimuth", 0), "elevation": v.get("elevation", 5), "name": name}
        if isinstance(v, str):
            key = v.strip().lower()
            az, el = _PRESETS.get(key, (0, 5))
            return {"azimuth": az, "elevation": el, "name": key}
        if isinstance(v, (int, float)):
            return {"azimuth": int(v), "elevation": 5, "name": f"az{int(v)}"}
        return {"azimuth": 0, "elevation": 5, "name": "front"}

    if views is None:
        views = [
            {"azimuth": 0, "elevation": 5, "name": "front"},
            {"azimuth": 90, "elevation": 5, "name": "right"},
            {"azimuth": 180, "elevation": 5, "name": "back"},
            {"azimuth": 270, "elevation": 5, "name": "left"},
        ]
    else:
        views = [_normalize(v) for v in views]

    with tempfile.TemporaryDirectory(prefix="blender_render_") as tmpdir:
        glb_path = os.path.join(tmpdir, "input.glb")
        with open(glb_path, "wb") as f:
            f.write(glb_bytes)

        config = {
            "views": views,
            "resolution": resolution,
            "width": width if width is not None else resolution,
            "height": height if height is not None else resolution,
            "samples": samples,
            "transparent": transparent,
            "focal_length": focal_length,
            "framing": framing,
        }
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        cmd = [
            _resolve_blender_bin(), "--background", "--python-expr", _RENDER_SCRIPT, "--",
            glb_path, tmpdir, json.dumps(config),
        ]
        rw = width if width is not None else resolution
        rh = height if height is not None else resolution
        log.info("[blender_render] rendering %d views at %dx%d (%d samples)",
                 len(views), rw, rh, samples)
        # Blender startup + GLB import (~20s) + ~45s per view for a
        # complex mesh (245K verts) at 768px/64 samples.
        render_timeout = 60 + 60 * len(views)
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=render_timeout,
            env={**os.environ, "BLENDER_USER_CONFIG": tmpdir,
                 "HOME": tmpdir})
        if proc.returncode != 0:
            log.error("[blender_render] Blender failed (rc=%d): %s",
                      proc.returncode, proc.stderr[-500:] if proc.stderr else "?")
            raise RuntimeError(f"Blender render failed: {proc.stderr[-300:]}")

        results = []
        for view in views:
            name = view.get("name", f"az{view['azimuth']}")
            png_path = os.path.join(tmpdir, f"{name}.png")
            if os.path.exists(png_path):
                with open(png_path, "rb") as f:
                    results.append({"name": name, "png_bytes": f.read()})
            else:
                log.warning("[blender_render] missing render: %s", png_path)
        log.info("[blender_render] done → %d renders", len(results))
        return results
