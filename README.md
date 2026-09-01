# melite-poser-nodes

template-mesh poser pipeline: PoserRender depth/rgb/openpose, CompositeCharacter, MeshFixWinding, RaySomaxBake, RayProjectViewsToTexture, RaySAMSegmentHairClothing.

Tech Noir Poser custom nodes for ComfyUI.

## Nodes

- `BLENDER_USER_CONFIG`
- `CHANNELS`
- `CleanMesh`
- `CompositeCharacter`
- `HOME`
- `MOTION`
- `MeshDedup`
- `MeshFixWinding`
- `OFFSET`
- `PoserBindMesh`
- `PoserBindPose`
- `PoserDeformSoma`
- `PoserRender`
- `PoserRenderOpenPose`
- `PoserTemplateBind`
- `RGB`
- `RayCreatureRigBridge`
- `RayProjectViewsToTexture`
- `RayRenderGLBViews`
- `RaySAMSegmentHairClothing`
- `RaySomaxBake`
- `RaySwapBaseColorTexture`
- `RayXatlasUnwrap`
- `XYZ`
- `YZX`
- `ZXY`

## Install

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/JayDataEngineer/melite-poser-nodes.git
```

Restart ComfyUI.

## Provenance

Published from the inference estate (`inference.cpp` repo, `plugins/comfyui/custom_nodes/melite-poser-nodes`) on 2026-09-01.

## License

MIT — see LICENSE.
