#!/usr/bin/env python3
"""Render a rotating turntable GIF of the severity-painted avatar.

Loads the vertex-coloured mesh from paint_sundamage.py and spins it 360 deg about
the vertical axis, baking in a severity colour-bar so the GIF is self-explanatory.

Usage:
  python turntable_gif.py <mesh.ply> <out.gif> [n_frames] [width height]
"""
import os
import sys

import numpy as np
import open3d as o3d
from PIL import Image


def colorbar_strip(h, w=180):
    """Vertical mild->moderate->severe colour-bar as an (h, w, 3) uint8 array."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable
    cmap = mcolors.LinearSegmentedColormap.from_list("sev", ["#2C8A4A", "#E8A33D", "#C0392B"])
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    # thinner + shorter bar, vertically centred; bigger text
    ax = fig.add_axes([0.07, 0.30, 0.10, 0.40])
    cb = fig.colorbar(ScalarMappable(norm=mcolors.Normalize(0, 2), cmap=cmap), cax=ax)
    cb.set_ticks([0, 1, 2]); cb.set_ticklabels(["Mild", "Moderate", "Severe"])
    cb.ax.tick_params(labelsize=14)
    fig.patch.set_facecolor((235 / 255, 235 / 255, 235 / 255))  # match render bg
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def main():
    mesh_path = sys.argv[1]
    out_gif = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    W = int(sys.argv[4]) if len(sys.argv) > 4 else 640
    H = int(sys.argv[5]) if len(sys.argv) > 5 else 960
    # pass "nobar" as a later arg for raw/photo meshes (no severity legend)
    show_bar = "nobar" not in sys.argv[3:]

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    print(f"mesh: {len(mesh.vertices)} verts (has colors: {mesh.has_vertex_colors()})")

    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    r.scene.set_background([1, 1, 1, 1])
    mat = o3d.visualization.rendering.MaterialRecord(); mat.shader = "defaultUnlit"
    r.scene.add_geometry("a", mesh, mat)
    bb = mesh.get_axis_aligned_bounding_box(); ctr = bb.get_center()
    Rd = max(bb.get_extent()) * 1.25

    cbar = colorbar_strip(H) if show_bar else None
    frames = []
    for k in range(n):
        ang = 2 * np.pi * k / n
        eye = [ctr[0] + Rd * np.sin(ang), ctr[1], ctr[2] + Rd * np.cos(ang)]
        r.setup_camera(45.0, ctr, eye, [0, 1, 0])
        body = np.asarray(r.render_to_image())            # (H, W, 3) uint8
        frame = body if cbar is None else np.concatenate([body, cbar[:H]], axis=1)
        frames.append(Image.fromarray(frame))
        if k % 12 == 0:
            print(f"  frame {k+1}/{n}")

    os.makedirs(os.path.dirname(out_gif) or ".", exist_ok=True)
    # shared adaptive palette across all frames -> smaller file, no colour flicker
    pal = frames[len(frames) // 4].quantize(colors=96, method=Image.FASTOCTREE)
    qframes = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    qframes[0].save(out_gif, save_all=True, append_images=qframes[1:],
                    duration=80, loop=0, optimize=True, disposal=2)
    mb = os.path.getsize(out_gif) / 1e6
    print(f"wrote {out_gif}  ({n} frames, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
