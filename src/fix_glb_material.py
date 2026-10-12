#!/usr/bin/env python3
"""Fix GLB materials written by trimesh so glTF viewers render them correctly.

trimesh tints textured materials grey (baseColorFactor [102,102,102]) and leaves
them single-sided, which makes viewers show the avatar dark with black backfaces.
This rewrites every material to: white base-colour factor, matte (metallic 0,
roughness 1), double-sided. Run after bake_atlas.py.

Usage: fix_glb_material.py <file.glb> [more.glb ...]
"""
import sys
import pygltflib


def fix(path):
    g = pygltflib.GLTF2().load(path)
    for m in g.materials:
        if m.pbrMetallicRoughness is None:
            m.pbrMetallicRoughness = pygltflib.PbrMetallicRoughness()
        m.pbrMetallicRoughness.baseColorFactor = [1, 1, 1, 1]
        m.pbrMetallicRoughness.metallicFactor = 0.0
        m.pbrMetallicRoughness.roughnessFactor = 1.0
        m.doubleSided = True
    g.save(path)
    print(f"fixed {len(g.materials)} materials in {path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("usage: fix_glb_material.py <file.glb> [more.glb ...]")
    for p in sys.argv[1:]:
        fix(p)
