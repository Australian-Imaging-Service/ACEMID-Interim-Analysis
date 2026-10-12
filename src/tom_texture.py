#!/usr/bin/env python3
"""Extract Vectra's own mesh from a .tom and apply photo texture by projecting
the 46 texturing-camera images onto it (occlusion-tested per-vertex colour).

Output: textured colored mesh (.glb/.ply) + turntable renders.
"""
import sys, os, zlib, struct, re
import numpy as np
import cv2
import open3d as o3d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfcm import load_calib_dir, quat_to_R


def load_tom_mesh(tom_path):
    data = open(tom_path, 'rb').read()
    # locate Vertices chunk
    vpos = data.find(b'Vertices')
    vlen = struct.unpack_from('<Q', data, vpos - 8)[0]
    vstart = vpos + 16
    nV = vlen // 12
    V = np.frombuffer(data, '<f4', nV * 3, vstart).reshape(-1, 3).astype(np.float64)
    # Tris chunk: zlib stream after vertices
    z = data.find(b'\x78\x9c', vstart + vlen)
    out = zlib.decompressobj().decompress(data[z:])
    idx = np.frombuffer(out, '<i4')
    F = idx[0::2].reshape(-1, 3).astype(np.int64)   # low word = index
    F = F[(F >= 0).all(1) & (F < nV).all(1)]
    return V, F


def main():
    tom = sys.argv[1]
    session = sys.argv[2]
    outdir = sys.argv[3]
    os.makedirs(outdir, exist_ok=True)

    V, F = load_tom_mesh(tom)
    print(f"mesh: {len(V)} verts, {len(F)} faces")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F.astype(np.int32))
    mesh.compute_vertex_normals()
    N = np.asarray(mesh.vertex_normals)

    # texturing cameras (the 'B'/primary set used by Vectra) -> use all calib cams present
    cams = load_calib_dir(os.path.join(session, 'calib'))
    tex_names = [n for n in cams
                 if os.path.exists(os.path.join(session, n + '.jpg'))]
    print(f"{len(tex_names)} candidate texture cameras")

    # raycasting scene for occlusion
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    acc = np.zeros((len(V), 3))     # weighted color sum
    wsum = np.zeros(len(V))
    for ni, name in enumerate(tex_names):
        c = cams[name]
        R = quat_to_R(c.quat, 'wxyz')           # world->cam
        C = c.pos
        fx, fy = c.f_px
        img = cv2.imread(os.path.join(session, name + '.jpg'))
        if img is None:
            continue
        H, W = img.shape[:2]
        # project all verts
        Xc = (V - C) @ R.T
        depth = -Xc[:, 2]
        front = depth > 1
        u = c.pp[0] + fx * Xc[:, 0] / depth
        v = c.pp[1] - fy * Xc[:, 1] / depth
        inb = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        # front-facing weight: vertex normal vs direction to camera
        dirc = (C - V); dirc /= (np.linalg.norm(dirc, axis=1, keepdims=True) + 1e-9)
        facing = (N * dirc).sum(1)
        cand = inb & (facing > 0.1)
        if cand.sum() == 0:
            continue
        # occlusion test via raycasting: cast from camera to candidate verts
        ci = np.where(cand)[0]
        origins = np.repeat(C[None, :], len(ci), 0)
        d = V[ci] - C
        dist = np.linalg.norm(d, axis=1)
        dirs = d / dist[:, None]
        rays = np.concatenate([origins, dirs], 1).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        visible = hit > (dist - 5.0)            # first hit at/after the vertex (5mm tol)
        ci = ci[visible]
        if len(ci) == 0:
            continue
        uu = np.clip(u[ci].astype(int), 0, W - 1)
        vv = np.clip(v[ci].astype(int), 0, H - 1)
        col = img[vv, uu][:, ::-1].astype(np.float64)   # BGR->RGB
        w = facing[ci] ** 2
        acc[ci] += col * w[:, None]
        wsum[ci] += w
        if ni % 10 == 0:
            print(f"  cam {ni+1}/{len(tex_names)} {name}: {len(ci)} verts coloured")

    colored = wsum > 0
    colors = np.full((len(V), 3), 0.5)
    colors[colored] = (acc[colored] / wsum[colored, None]) / 255.0
    print(f"coloured {colored.sum()}/{len(V)} verts ({colored.mean()*100:.1f}%)")
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))

    o3d.io.write_triangle_mesh(os.path.join(outdir, 'vectra_textured.ply'), mesh)
    o3d.io.write_triangle_mesh(os.path.join(outdir, 'vectra_textured.glb'), mesh)
    o3d.io.write_triangle_mesh(os.path.join(outdir, 'vectra_textured.obj'), mesh)

    # render turntable
    r = o3d.visualization.rendering.OffscreenRenderer(1000, 1000)
    r.scene.set_background([0.05, 0.05, 0.08, 1])
    mat = o3d.visualization.rendering.MaterialRecord(); mat.shader = 'defaultLit'
    r.scene.add_geometry('a', mesh, mat)
    r.scene.scene.set_sun_light([0.3, -0.5, -0.8], [1, 1, 1], 75000)
    r.scene.scene.enable_sun_light(True)
    bb = mesh.get_axis_aligned_bounding_box(); ctr = bb.get_center()
    Rd = max(bb.get_extent()) * 1.3
    for k in range(8):
        ang = 2 * np.pi * k / 8
        eye = [ctr[0] + Rd * np.sin(ang), ctr[1], ctr[2] + Rd * np.cos(ang)]
        r.setup_camera(50.0, ctr, eye, [0, 1, 0])
        o3d.io.write_image(os.path.join(outdir, f'tex_view{k}.png'), r.render_to_image())
    print("done -> ", outdir)


if __name__ == '__main__':
    main()
