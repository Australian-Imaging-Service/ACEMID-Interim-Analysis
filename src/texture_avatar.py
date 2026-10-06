#!/usr/bin/env python3
"""Build the raw photo-textured 3D avatar (no model projection).

Same geometry + projection as paint_sundamage.py / tom_texture.py, but samples the
actual camera photo colour at each vertex (occlusion-tested, front-facing blend).
Reads the colour images from a configurable directory so it works for patients whose
session folder has no preview JPGs (we point it at converted_jpgs/<id>/).

Usage:
  python texture_avatar.py <session_dir> <image_dir> <out_dir> [tom_path]
"""
import os
import sys

import numpy as np
import cv2
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfcm import load_calib_dir, quat_to_R
from tom_texture import load_tom_mesh


def main():
    session = sys.argv[1]
    image_dir = sys.argv[2]
    outdir = sys.argv[3]
    tom = sys.argv[4] if len(sys.argv) > 4 else os.path.join(
        session, os.path.basename(session.rstrip("/")) + ".tom")
    os.makedirs(outdir, exist_ok=True)

    V, F = load_tom_mesh(tom)
    print(f"mesh: {len(V)} verts, {len(F)} faces")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F.astype(np.int32))
    mesh.compute_vertex_normals()
    Nrm = np.asarray(mesh.vertex_normals)

    cams = load_calib_dir(os.path.join(session, "calib"))
    use = [n for n in cams if n.endswith("B")
           and os.path.exists(os.path.join(image_dir, n + ".jpg"))]
    print(f"texturing from {len(use)} B-view photos in {image_dir}")

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    acc = np.zeros((len(V), 3))
    wsum = np.zeros(len(V))
    for ni, name in enumerate(sorted(use)):
        c = cams[name]
        R = quat_to_R(c.quat, "wxyz"); C = c.pos
        fx, fy = c.f_px
        img = cv2.imread(os.path.join(image_dir, name + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        Xc = (V - C) @ R.T
        depth = -Xc[:, 2]; front = depth > 1
        u = c.pp[0] + fx * Xc[:, 0] / depth
        v = c.pp[1] - fy * Xc[:, 1] / depth
        inb = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        dirc = (C - V); dirc /= (np.linalg.norm(dirc, axis=1, keepdims=True) + 1e-9)
        facing = (Nrm * dirc).sum(1)
        cand = inb & (facing > 0.1)
        if cand.sum() == 0:
            continue
        ci = np.where(cand)[0]
        d = V[ci] - C; dist = np.linalg.norm(d, axis=1)
        rays = np.concatenate([np.repeat(C[None, :], len(ci), 0), d / dist[:, None]], 1).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        ci = ci[hit > (dist - 5.0)]
        if len(ci) == 0:
            continue
        uu = np.clip(u[ci].astype(int), 0, W - 1)
        vv = np.clip(v[ci].astype(int), 0, H - 1)
        col = img[vv, uu][:, ::-1].astype(np.float64)        # BGR->RGB
        w = facing[ci] ** 2
        acc[ci] += col * w[:, None]; wsum[ci] += w
        if ni % 10 == 0:
            print(f"  cam {ni+1}/{len(use)} {name}: {len(ci)} verts")

    colored = wsum > 0
    colors = np.full((len(V), 3), 0.75)
    colors[colored] = (acc[colored] / wsum[colored, None]) / 255.0
    print(f"coloured {colored.sum()}/{len(V)} verts ({colored.mean()*100:.1f}%)")
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))

    o3d.io.write_triangle_mesh(os.path.join(outdir, "raw_avatar.glb"), mesh)
    o3d.io.write_triangle_mesh(os.path.join(outdir, "raw_avatar.ply"), mesh)

    r = o3d.visualization.rendering.OffscreenRenderer(900, 1400)
    r.scene.set_background([1, 1, 1, 1])
    mat = o3d.visualization.rendering.MaterialRecord(); mat.shader = "defaultUnlit"
    r.scene.add_geometry("a", mesh, mat)
    bb = mesh.get_axis_aligned_bounding_box(); ctr = bb.get_center()
    Rd = max(bb.get_extent()) * 1.25
    for vname, ang in {"front": np.pi, "left": np.pi/2, "back": 0.0, "right": 3*np.pi/2}.items():
        eye = [ctr[0] + Rd*np.sin(ang), ctr[1], ctr[2] + Rd*np.cos(ang)]
        r.setup_camera(45.0, ctr, eye, [0, 1, 0])
        o3d.io.write_image(os.path.join(outdir, f"raw_{vname}.png"), r.render_to_image())
    print("done ->", outdir)


if __name__ == "__main__":
    main()
