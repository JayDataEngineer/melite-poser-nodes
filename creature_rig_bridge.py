"""Blender-side creature rig bridge - bind an arbitrary (e.g. TRELLIS) mesh
onto an anyCreature species skeleton so it inherits that species baked
"idle"/"move" animations.

Decision 2026-08-25 (operator): creatures are anyCreature-only (law c);
this bridge is the R&D path (a) toward textured AND animated creatures -
the only such path on this stack, since somax/transfer_rig/kimodo are
humanoid-only by W12 law.

Method:
  1. Import creature GLB (armature + skinned meshes + actions) and mesh GLB
     (static textured asset).
  2. Normalize the mesh onto the creature: uniform scale fit (param),
     XZ-center on the creature bounds, base-aligned to min Z.
  3. Skin: every mesh vertex -> distance to each deformable bone SEGMENT
     (head->tail, world space); top-K bones by inverse distance, normalized.
     Written as vertex groups + ARMATURE modifier (LBS contract).
  4. Optionally strip the creature own body meshes (mesh-only export).
  5. Export GLB with animations + skins.

Runs blender --background --python as a subprocess inside the inference
container (same discipline as blender_render.py). CPU-only.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import logging

log = logging.getLogger(__name__)

BLENDER_BIN = os.environ.get("BLENDER_BIN", "blender")

_BRIDGE_SCRIPT = r'''
import bpy, sys, json, math, os

argv = sys.argv
argv = argv[argv.index("--") + 1:]
config = json.loads(argv[0])

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.data.objects if o not in before]

def world_bbox(objs):
    import mathutils
    pts = []
    deps = bpy.context.evaluated_depsgraph_get()
    for o in objs:
        if o.type != 'MESH':
            continue
        ev = o.evaluated_get(deps)
        for corner in ev.bound_box:
            pts.append(ev.matrix_world @ mathutils.Vector(corner))
    if not pts:
        return None
    xs=[p.x for p in pts]; ys=[p.y for p in pts]; zs=[p.z for p in pts]
    return (min(xs),min(ys),min(zs)),(max(xs),max(ys),max(zs))

clear_scene()
import_glb(config["creature_glb"])
import_glb(config["mesh_glb"])

arm = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
if arm is None:
    raise RuntimeError("no armature in creature GLB")
body_meshes = [o for o in bpy.data.objects
               if o.type == 'MESH' and o.find_armature() == arm]
if not body_meshes:
    raise RuntimeError("no skinned body mesh on creature armature")
target = next((o for o in bpy.data.objects
               if o.type == 'MESH' and o.find_armature() != arm
               and o.data.materials), None)
if target is None:
    raise RuntimeError("no materialized target mesh found")

cb = world_bbox(body_meshes)
tb = world_bbox([target])
if cb is None or tb is None:
    raise RuntimeError("bbox failure")
cmin, cmax = cb
tmin, tmax = tb
c_height = max(cmax[2]-cmin[2], 1e-6)
t_height = max(tmax[2]-tmin[2], 1e-6)

scale = (c_height * config.get("height_ratio", 1.0)) / t_height
target.scale = (target.scale[0]*scale, target.scale[1]*scale, target.scale[2]*scale)
bpy.context.view_layer.update()
tb2 = world_bbox([target])
tmin2, tmax2 = tb2
cx = (cmin[0]+cmax[0])/2.0; cy = (cmin[1]+cmax[1])/2.0
dx = cx - (tmin2[0]+tmax2[0])/2.0
dy = cy - (tmin2[1]+tmax2[1])/2.0
dz = cmin[2] - tmin2[2]
target.location = (target.location[0]+dx, target.location[1]+dy, target.location[2]+dz)
bpy.context.view_layer.update()

mw = arm.matrix_world
deform_bones = [pb for pb in arm.pose.bones if pb.bone.use_deform] or list(arm.pose.bones)
bones = []
for pb in deform_bones:
    b = pb.bone
    bones.append((b.name, mw @ b.head_local, mw @ b.tail_local))
if not bones:
    raise RuntimeError("no deformable bones")

def seg_dist(p, a, b):
    abx, aby, abz = b[0]-a[0], b[1]-a[1], b[2]-a[2]
    apx, apy, apz = p[0]-a[0], p[1]-a[1], p[2]-a[2]
    denom = abx*abx + aby*aby + abz*abz
    t = 0.0 if denom == 0 else max(0.0, min(1.0, (apx*abx + apy*aby + apz*abz)/denom))
    dx = p[0]-(a[0]+t*abx); dy = p[1]-(a[1]+t*aby); dz = p[2]-(a[2]+t*abz)
    return (dx*dx+dy*dy+dz*dz) ** 0.5

# --- yaw alignment (2026-08-25 broadside finding): TRELLIS meshes bind
# ~90 deg off the rig spine (mesh long axis X, rig spine Z) because the
# bridge scaled + centered but never rotated. Align the mesh principal
# horizontal axis onto the rig's; resolve the 180 deg ambiguity by scoring
# mean nearest-bone-segment distance for both candidates.
yaw_deg = 0.0
if config.get("yaw_align", True):
    import mathutils, math as _math
    def _principal_xy(pts):
        mx = sum(p[0] for p in pts)/len(pts); my = sum(p[1] for p in pts)/len(pts)
        axx = sum((p[0]-mx)**2 for p in pts)/len(pts)
        ayy = sum((p[1]-my)**2 for p in pts)/len(pts)
        axy = sum((p[0]-mx)*(p[1]-my) for p in pts)/len(pts)
        vx, vy = 1.0, 0.0
        for _ in range(32):
            nx = axx*vx + axy*vy; ny = axy*vx + ayy*vy
            n = max((nx*nx+ny*ny)**0.5, 1e-12)
            vx, vy = nx/n, ny/n
        return vx, vy
    rig_pts = []
    for _, a, b in bones:
        rig_pts.append(a); rig_pts.append(b)
    rig_vx, rig_vy = _principal_xy(rig_pts)
    _d0 = bpy.context.evaluated_depsgraph_get()
    _e0 = target.evaluated_get(_d0)
    _m0 = _e0.to_mesh()
    mesh_pts = [tuple(target.matrix_world @ v.co) for v in list(_m0.vertices)[::4]]
    _e0.to_mesh_clear()
    if len(mesh_pts) >= 16:
        mvx, mvy = _principal_xy(mesh_pts)
        base_yaw = _math.atan2(rig_vy, rig_vx) - _math.atan2(mvy, mvx)
        def _mean_bone_dist(stride=24):
            _d = bpy.context.evaluated_depsgraph_get()
            _e = target.evaluated_get(_d)
            _m = _e.to_mesh()
            tot = 0.0; n = 0
            for v in list(_m.vertices)[::stride]:
                p = target.matrix_world @ v.co
                tot += min(seg_dist(p, a, b) for _, a, b in bones)
                n += 1
            _e.to_mesh_clear()
            return tot/max(n,1)
        bb0 = world_bbox([target])
        c0 = ((bb0[0][0]+bb0[1][0])/2.0, (bb0[0][1]+bb0[1][1])/2.0)
        def _apply(yaw):
            T = mathutils.Matrix.Translation((c0[0], c0[1], 0.0))
            R = mathutils.Matrix.Rotation(yaw, 4, 'Z')
            target.matrix_world = (T @ R @ T.inverted()) @ target.matrix_world
            bpy.context.view_layer.update()
        best_yaw, best_score = None, None
        for cand in (base_yaw, base_yaw + _math.pi):
            _apply(cand)
            s = _mean_bone_dist()
            if best_score is None or s < best_score:
                best_yaw, best_score = cand, s
        _apply(best_yaw)
        yaw_deg = _math.degrees(best_yaw) % 360.0
        # re-center X/Y + re-base Z in the NEW orientation
        tb3 = world_bbox([target]); t3min, t3max = tb3
        target.location = (
            target.location[0] + cx - (t3min[0]+t3max[0])/2.0,
            target.location[1] + cy - (t3min[1]+t3max[1])/2.0,
            target.location[2] + cmin[2] - t3min[2],
        )
        bpy.context.view_layer.update()

deps = bpy.context.evaluated_depsgraph_get()
ev = target.evaluated_get(deps)
mesh = ev.to_mesh()
K = config.get("max_influences", 3)
falloff_mode = config.get("weight_falloff", "linear")   # linear | inv_power
power = float(config.get("weight_power", 2.0))
win_mode = config.get("weight_window", "none")           # none | smoothstep
groups = {}
for name, _, _ in bones:
    groups[name] = target.vertex_groups.new(name=name)
for v in mesh.vertices:
    wp = target.matrix_world @ v.co
    dists = sorted(((seg_dist(wp, a, b), name) for name, a, b in bones))
    top = dists[:K]
    if falloff_mode == "inv_power":
        dmin, dmax = top[0][0], top[-1][0]
        span = max(dmax - dmin, 1e-9)
        raw = []
        for d, name in top:
            w = max(d, 1e-4) ** (-power)
            if win_mode == "smoothstep":
                t = min(max((d - dmin) / span, 0.0), 1.0)
                w *= 1.0 - (t * t * (3.0 - 2.0 * t))
            raw.append((w, name))
        total = sum(w for w, _ in raw) or 1e-6
        for w, name in raw:
            wn = w / total
            if wn > 0:
                groups[name].add([v.index], wn, 'REPLACE')
    else:
        total = sum(d for d, _ in top) or 1e-6
        for d, name in top:
            w = 1.0 - d/total
            if w > 0:
                groups[name].add([v.index], w, 'REPLACE')
ev.to_mesh_clear()

target.parent = arm
mod = target.modifiers.new("Armature", 'ARMATURE')
mod.object = arm

if config.get("strip_body", True):
    bpy.ops.object.select_all(action='DESELECT')
    for m in body_meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.delete()

out = config["output_glb"]
bpy.ops.export_scene.gltf(
    filepath=out,
    export_format='GLB',
    export_skins=True,
    export_animations=True,
    export_apply=False,
)
print(json.dumps({
    "ok": True,
    "out": out,
    "bytes": os.path.getsize(out),
    "scale": scale,
    "bones": len(bones),
    "yaw_deg": round(yaw_deg, 1),
}))
'''


def bridge_creature(
    creature_glb_bytes: bytes,
    mesh_glb_bytes: bytes,
    output_path: str,
    height_ratio: float = 1.0,
    max_influences: int = 3,
    strip_body: bool = True,
    weight_falloff: str = "linear",
    weight_power: float = 2.0,
    weight_window: str = "none",
    yaw_align: bool = True,
) -> dict:
    """Bind mesh_glb_bytes onto the creature skeleton; returns stats dict."""
    with tempfile.TemporaryDirectory(prefix="rig_bridge_") as td:
        cpath = os.path.join(td, "creature.glb")
        mpath = os.path.join(td, "mesh.glb")
        with open(cpath, "wb") as f:
            f.write(creature_glb_bytes)
        with open(mpath, "wb") as f:
            f.write(mesh_glb_bytes)
        cfg = {
            "creature_glb": cpath,
            "mesh_glb": mpath,
            "output_glb": output_path,
            "height_ratio": height_ratio,
            "max_influences": max_influences,
            "strip_body": strip_body,
            "weight_falloff": weight_falloff,
            "weight_power": weight_power,
            "weight_window": weight_window,
            "yaw_align": yaw_align,
        }
        script_path = os.path.join(td, "bridge_inner.py")
        with open(script_path, "w") as f:
            f.write(_BRIDGE_SCRIPT)
        cmd = [
            BLENDER_BIN, "--background",
            "--python", script_path,
            "--", json.dumps(cfg),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        tail = chr(10).join((proc.stdout + proc.stderr).splitlines()[-12:])
        for line in (proc.stdout or "").splitlines():
            if line.startswith('{"ok"'):
                return json.loads(line)
        raise RuntimeError("bridge failed (" + str(proc.returncode) + "): " + tail)
