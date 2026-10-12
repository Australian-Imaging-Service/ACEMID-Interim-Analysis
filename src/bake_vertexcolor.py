#!/usr/bin/env python3
"""Bake a per-VERTEX-colour PLY from a Vectra .tom's geometry + texture atlas.

Each vertex colour = average of the full-res atlas pixels its corners map to.
Robust, viewer-agnostic (no materials/UVs) — looks the same in any tool, never
black. Slightly softer than the true-texture GLB (colour is per-vertex, ~160k
samples, not per-texture-pixel).

This reproduces tom_outputs/atlas/vectra_atlas_vertexcolor.ply.

Usage: bake_vertexcolor.py <tom> <out.ply>
"""
import sys, os, zlib, io
import numpy as np
import open3d as o3d
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bake_atlas import find_chunks


def main():
    tom, out = sys.argv[1], sys.argv[2]
    data = open(tom, 'rb').read()
    ch = find_chunks(data)

    # geometry
    vo, vl = ch['Vertices'][0]; nV = vl // 12
    V = np.frombuffer(data, '<f4', nV * 3, vo).reshape(-1, 3).astype(np.float64)
    z = data.find(b'\x78\x9c', vo + vl)
    faces = np.frombuffer(zlib.decompressobj().decompress(data[z:]), '<i4')[0::2].reshape(-1, 3)
    nF = len(faces)

    # texture-vertices (u, v in [0,1], page) + per-corner texvert indices
    to, tl = ch['TexVertA'][0]; _tv = zlib.decompress(data[to:to + tl])
    TV = np.frombuffer(_tv, dtype=[('u', '<f4'), ('v', '<f4'), ('p', '<i4')], count=len(_tv) // 12)
    co, cl = ch['CnrTexVs'][0]
    trip = np.frombuffer(zlib.decompress(data[co:co + cl]), '<i4')
    trip = trip[:(len(trip) // 3) * 3].reshape(-1, 3)
    uvf = np.zeros((nF, 3, 2)); pagef = np.full(nF, -1)
    uvf[trip[:, 0], trip[:, 1], 0] = TV['u'][trip[:, 2]]
    uvf[trip[:, 0], trip[:, 1], 1] = TV['v'][trip[:, 2]]
    pagef[trip[:, 0]] = TV['p'][trip[:, 2]]

    pages = ch['_pages']
    acc = np.zeros((nV, 3)); cnt = np.zeros(nV)
    for pg in range(len(pages)):
        sel = np.where(pagef == pg)[0]
        if not len(sel):
            continue
        d0, ln = pages[pg]
        im = np.asarray(Image.open(io.BytesIO(data[d0:d0 + ln])).convert('RGB'))
        H, W = im.shape[:2]
        for ci in range(3):
            vid = faces[sel, ci]
            x = np.clip((uvf[sel, ci, 0] * W).astype(int), 0, W - 1)
            y = np.clip(((1 - uvf[sel, ci, 1]) * H).astype(int), 0, H - 1)
            np.add.at(acc, vid, im[y, x].astype(np.float64))
            np.add.at(cnt, vid, 1)
    cnt[cnt == 0] = 1
    colors = (acc / cnt[:, None]) / 255.0

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(V)
    m.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    m.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    m.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out, m)
    print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)  verts={nV} faces={nF}")


if __name__ == '__main__':
    main()
