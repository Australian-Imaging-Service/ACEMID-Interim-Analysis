#!/usr/bin/env python3
"""Paint sun-damage model predictions onto the Vectra 3D avatar.

Re-uses the exact vertex->camera projection from tom_texture.py (validated .sfcm
calibration + Open3D occlusion test), but instead of sampling the photo colour at
each projected pixel, it looks up the *predicted photodamage severity* of the tile
that covers that pixel. The result is a body mesh coloured by sun-damage severity —
the 3D analogue of the spatial-distribution figure in the BJD paper.

Cameras: the overlay uses only the 46 non-polarised **B-view** cameras (a*B / f*B),
i.e. the same views the tile pipeline cropped from — not all 92 A+B cameras.

Projection: identical projective geometry to tom_texture.py. NOTE the SFCM radial
distortion terms (K1, K2) are NOT applied here, matching tom_texture.py — this is a
pinhole projection, validated visually against the RGB texture overlay rather than a
full physical camera model. (Distortion is small for these long-focal skin cams.)

Severity is the expected (probability-weighted) class value
    severity = 1 * P(moderate) + 2 * P(severe)   in [0, 2],
so the avatar is a smooth severity map, not a hard arg-max class map.

Pipeline match (must stay in sync with tiles_croping.py):
  * the camera image (W=5202 x H=3465) is rotated 90 deg or 270 deg CCW per camera
    (get_rotation), then cut into a 5-col (a..e, across width) x 9-row (1..9, down
    height) grid -> tiles named like 'a1B_c4'.
  * we therefore project a vertex to (u,v) in the ORIGINAL image, apply the same
    rotation to (u,v), then index the 5x9 grid to find its tile + severity.

Usage:
  python paint_sundamage.py <session_dir> <preds_csv> <out_dir> [tom_path]
  # session_dir holds the *B.sfcm + *B.jpg and (default) <session>.tom
"""
import os
import sys

import numpy as np
import pandas as pd
import cv2
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfcm import load_calib_dir, quat_to_R
from tom_texture import load_tom_mesh

# tiling grid (tiles_croping.im_to_tiles): 5 columns across width, 9 rows down height
N_COL, N_ROW = 5, 9
COL_LETTERS = "abcde"
ROT90 = {"a1B", "a2B", "a5B", "a10B", "a13B", "a14B",
         "f1B", "f2B", "f5B", "f10B", "f13B", "f14B"}


def get_rotation(cam_b_name):
    """Degrees CCW that tiles_croping applied before tiling (matches its table)."""
    return 90 if cam_b_name in ROT90 else 270


def orig_to_rotated(u, v, W, H, angle):
    """Map a point in the original (WxH) image to coords in the rotated-expanded
    image, matching PIL Image.rotate(angle, expand=True) (CCW). Output frame is
    (W'=H, H'=W) for both 90 and 270."""
    if angle == 90:        # (u,v) -> (v, W-1-u)
        return v, (W - 1 - u)
    else:                  # 270 CCW: (u,v) -> (H-1-v, u)
        return (H - 1 - v), u


def _normalise_columns(df):
    """Accept either this package's per_tile_predictions.csv or the study's
    04_infer_sundamage.py output. Guarantees a numeric `pred` (0/1/2) column."""
    if "pred" not in df.columns:
        if "pred_photodamage" in df.columns:      # model-package run_inference.py
            m = {"mild": 0, "moderate": 1, "severe": 2}
            df["pred"] = df["pred_photodamage"].str.lower().map(m)
        else:                                     # fall back to arg-max of probs
            df["pred"] = df[["p_mild", "p_moderate", "p_severe"]].values.argmax(1)
    return df


def build_grids(preds_csv):
    """severity[(camera, 'a1')] -> dict cell->severity (0..2). camera key has no 'B'."""
    df = _normalise_columns(pd.read_csv(preds_csv))
    df["sev"] = df.p_moderate * 1 + df.p_severe * 2
    grids = {}
    for (cam, cell), g in df.groupby(["camera", "grid_cell"]):
        grids.setdefault(cam, {})[cell] = float(g["sev"].mean())
    return grids, df


def cell_of(u_rot, v_rot, Wr, Hr):
    col = int(np.clip(u_rot // (Wr / N_COL), 0, N_COL - 1))
    row = int(np.clip(v_rot // (Hr / N_ROW), 0, N_ROW - 1))
    return f"{COL_LETTERS[col]}{row + 1}"


def severity_to_rgb(sev):
    """0 -> green, 1 -> amber, 2 -> red (matches the showcase palette)."""
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "sev", ["#2C8A4A", "#E8A33D", "#C0392B"])
    return cmap(np.clip(sev / 2.0, 0, 1))[..., :3]


def main():
    session = sys.argv[1]
    preds_csv = sys.argv[2]
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

    grids, df = build_grids(preds_csv)
    print(f"predictions: {len(df)} tiles over {df.camera.nunique()} cameras")

    cams = load_calib_dir(os.path.join(session, "calib"))
    # only B cameras that have (a) calib and (b) predictions. The session jpg is NOT
    # needed — severity is read from the prediction grid, geometry from .sfcm/.tom.
    use = [n for n in cams if n.endswith("B") and n[:-1] in grids]
    print(f"using {len(use)} B-cameras with predictions")

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    acc = np.zeros(len(V))      # weighted severity sum
    wsum = np.zeros(len(V))
    for ni, name in enumerate(sorted(use)):
        c = cams[name]
        R = quat_to_R(c.quat, "wxyz")
        C = c.pos
        fx, fy = c.f_px
        W, H = c.width, c.height
        angle = get_rotation(name)
        grid = grids[name[:-1]]

        Xc = (V - C) @ R.T
        depth = -Xc[:, 2]
        front = depth > 1
        u = c.pp[0] + fx * Xc[:, 0] / depth
        v = c.pp[1] - fy * Xc[:, 1] / depth
        inb = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        dirc = (C - V); dirc /= (np.linalg.norm(dirc, axis=1, keepdims=True) + 1e-9)
        facing = (Nrm * dirc).sum(1)
        cand = inb & (facing > 0.1)
        if cand.sum() == 0:
            continue
        ci = np.where(cand)[0]
        d = V[ci] - C
        dist = np.linalg.norm(d, axis=1)
        dirs = d / dist[:, None]
        rays = np.concatenate([np.repeat(C[None, :], len(ci), 0), dirs], 1).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        ci = ci[hit > (dist - 5.0)]              # visible (not occluded)
        if len(ci) == 0:
            continue
        Wr, Hr = H, W                            # rotated-expanded image dims
        nfilled = 0
        for idx in ci:
            ur, vr = orig_to_rotated(u[idx], v[idx], W, H, angle)
            sev = grid.get(cell_of(ur, vr, Wr, Hr))
            if sev is None:                      # tile rejected (no skin) -> no signal
                continue
            w = facing[idx] ** 2
            acc[idx] += sev * w
            wsum[idx] += w
            nfilled += 1
        if ni % 10 == 0:
            print(f"  cam {ni+1}/{len(use)} {name}: {nfilled} verts scored")

    scored = wsum > 0
    sev_vert = np.full(len(V), np.nan)
    sev_vert[scored] = acc[scored] / wsum[scored]
    print(f"scored {scored.sum()}/{len(V)} verts ({scored.mean()*100:.1f}%)  "
          f"mean severity {np.nanmean(sev_vert):.3f}")

    colors = np.full((len(V), 3), 0.75)          # grey = no skin / not scored
    colors[scored] = severity_to_rgb(sev_vert[scored])
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))

    o3d.io.write_triangle_mesh(os.path.join(outdir, "sundamage_avatar.glb"), mesh)
    o3d.io.write_triangle_mesh(os.path.join(outdir, "sundamage_avatar.ply"), mesh)
    np.save(os.path.join(outdir, "vertex_severity.npy"), sev_vert)

    # renders: front / back / two sides
    r = o3d.visualization.rendering.OffscreenRenderer(900, 1400)
    r.scene.set_background([1, 1, 1, 1])
    mat = o3d.visualization.rendering.MaterialRecord(); mat.shader = "defaultUnlit"
    r.scene.add_geometry("a", mesh, mat)
    bb = mesh.get_axis_aligned_bounding_box(); ctr = bb.get_center()
    Rd = max(bb.get_extent()) * 1.25
    # avatar faces -Z in world, so ang=pi looks at the front
    views = {"front": np.pi, "left": np.pi / 2, "back": 0.0, "right": 3 * np.pi / 2}
    for vname, ang in views.items():
        eye = [ctr[0] + Rd * np.sin(ang), ctr[1], ctr[2] + Rd * np.cos(ang)]
        r.setup_camera(45.0, ctr, eye, [0, 1, 0])
        o3d.io.write_image(os.path.join(outdir, f"sundamage_{vname}.png"),
                           r.render_to_image())
    make_composite(outdir, sev_vert, df)
    print("done ->", outdir)
    return sev_vert


def _bg_to_white(img, bg=235, tol=12):
    """Replace the flat neutral-grey render background (R=G=B≈bg) with white.
    Body colours and the darker unscored-grey (≈191) are untouched."""
    img = img.copy()
    neutral = (img.max(2).astype(int) - img.min(2).astype(int)) <= 6
    near_bg = np.abs(img.astype(int) - bg).max(2) <= tol
    img[neutral & near_bg] = 255
    return img


def _autocrop(img, bg=235, pad=12):
    """Trim the uniform background margin around the rendered body."""
    mask = np.abs(img.astype(int) - bg).max(2) > 8
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def make_composite(outdir, sev_vert, df):
    """Front+back+sides panel with a compact severity colour-bar and caption."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable
    from PIL import Image as PILImage

    bg = (1, 1, 1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "sev", ["#2C8A4A", "#E8A33D", "#C0392B"])
    order = ["front", "left", "back", "right"]
    imgs = [_autocrop(_bg_to_white(np.array(PILImage.open(
        os.path.join(outdir, f"sundamage_{v}.png")).convert("RGB"))), pad=3) for v in order]
    aspects = [im.shape[1] / im.shape[0] for im in imgs]   # w/h per cropped body

    # figure sized to the bodies so there's almost no dead space
    H_in = 6.2
    fig = plt.figure(figsize=(H_in * (sum(aspects) + 0.45), H_in), facecolor=bg)
    gs = fig.add_gridspec(1, 4, width_ratios=aspects, wspace=0.0,
                          left=0.005, right=0.86, top=0.80, bottom=0.02)
    for i, (im, v) in enumerate(zip(imgs, order)):
        ax = fig.add_subplot(gs[0, i]); ax.axis("off")
        ax.imshow(im)
        ax.set_title(v.capitalize(), fontsize=14, color="#1f3a2d")

    # compact colour-bar, vertically centred in its own slim axes
    cax = fig.add_axes([0.90, 0.30, 0.013, 0.42])
    cb = fig.colorbar(ScalarMappable(norm=mcolors.Normalize(0, 2), cmap=cmap), cax=cax)
    cb.set_ticks([0, 1, 2]); cb.set_ticklabels(["Mild", "Moderate", "Severe"])
    cb.ax.tick_params(labelsize=13)

    pct_sev = (df.pred == 2).mean() * 100
    pid = str(df.patient_id.iloc[0]) if "patient_id" in df.columns else "unknown"
    ncam = df.camera.nunique()
    fig.suptitle(f"Sun-damage prediction on the 3D avatar — patient {pid}\n"
                 f"{len(df)} skin tiles · {ncam} B-view cameras · mean (prob-weighted) "
                 f"severity {np.nanmean(sev_vert):.2f}/2 · {pct_sev:.0f}% tiles severe · "
                 f"grey = non-skin / not scored",
                 fontsize=12.5, color="#1f3a2d", y=0.97)
    fig.savefig(os.path.join(outdir, "sundamage_avatar_panel.png"),
                dpi=140, facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":
    main()
