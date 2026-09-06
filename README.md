# Sun-damage prediction on the 3D avatar

Optional visualisation add-on for the `photodamage-dermlip-mtl-v4` model. It
takes the model's **per-tile predictions** and paints them onto the patient's
**Vectra 3D body avatar**, producing:

| Output | What it is |
|---|---|
| `raw_avatar.glb` / `.ply` | the photo-textured 3D body (no model output) |
| `raw_turntable.gif` | the raw avatar rotating 360° |
| `sundamage_avatar.glb` / `.ply` | the body coloured by predicted sun-damage severity (green→amber→red) |
| `sundamage_turntable.gif` | the severity avatar rotating, with a Mild→Moderate→Severe colour-bar |
| `sundamage_avatar_panel.png` | front/left/back/right report figure with colour-bar + caption |
| `sundamage_{front,left,back,right}.png` | the four still views |
| `vertex_severity.npy` | per-vertex severity (0–2), for custom figures |

Severity is the probability-weighted class value
`severity = 1·P(moderate) + 2·P(severe) ∈ [0, 2]` — the same `severity_score`
the model uses per patient, here resolved per body location.

---

## What you need (prerequisites)

This is **not** self-contained like the tile inference — it needs the raw Vectra
capture for the patient, because the 3D geometry and camera calibration live
there. You must supply **two inputs**:

| Input | What / where it comes from |
|---|---|
| **Raw Vectra session folder** | the patient's baseline capture dir, holding `<session>.tom` (3D mesh), `calib/*.sfcm` (per-camera calibration) and `*B.jpg` (46 B-view photos). In the study these are at `exported_data/<PATIENT>/<SESSION>/`. **Not shipped in this model package.** |
| **Per-tile predictions CSV** | `per_tile_predictions.csv` written by `run_inference.py` (the main package). One CSV may hold many patients — the driver slices one out by `patient_id`. |

> If a session has no `*B.jpg` preview photos, point the raw-texture step (step 1)
> at a folder of the converted colour images for that patient instead.

Install the extra deps (in addition to the package's `requirements.txt`):
```bash
pip install -r avatar_3d_visualization/requirements.txt
```

**How the two inputs feed the pipeline:**

![Input organisation → make_avatar.sh → outputs](figures/input_layout.png)

---

## Quick start (one command)

```bash
cd avatar_3d_visualization
bash make_avatar.sh <session_dir> <per_tile_predictions.csv> <out_dir> [patient_id]
```
Example:
```bash
bash make_avatar.sh \
    "/data/exported_data/06368481-.../20221207140356" \
    ../out/per_tile_predictions.csv \
    ../out/avatar_7778-16  7778-16
```
This runs all four steps and writes every output above into `out_dir`.

---

## Example outputs (patient 7778-16)

The **report panel** — front / left / back / right, coloured by predicted
severity (green = mild → red = severe; grey = non-skin / not scored):

![Sun-damage report panel](figures/output_panel.png)

The two **turntable GIFs** (shown here as sampled frames — in the files they
rotate a full 360°): the raw photo avatar and the severity avatar with its
colour-bar.

![Raw and severity turntable frames](figures/output_turntable_strip.png)

---

## Step by step (what the driver runs)

```bash
# 1. Raw photo avatar  (image_dir = session so it uses the *B.jpg photos)
python3 texture_avatar.py   <session_dir> <image_dir> <out_dir>

# 2. Severity avatar + still views + report panel
python3 paint_sundamage.py  <session_dir> <preds_csv> <out_dir>

# 3. Raw turntable gif        (`nobar` = no severity legend)
python3 turntable_gif.py     <out_dir>/raw_avatar.ply        <out_dir>/raw_turntable.gif        48 640 960 nobar

# 4. Severity turntable gif   (severity colour-bar baked in)
python3 turntable_gif.py     <out_dir>/sundamage_avatar.ply  <out_dir>/sundamage_turntable.gif  48 640 960
```
`turntable_gif.py` args after the output path are: `n_frames width height [nobar]`.

---

## How it works

`paint_sundamage.py` reuses the study's validated camera model: each mesh vertex
is projected into every **B-view** camera via that camera's `.sfcm` calibration,
occlusion-tested with an Open3D ray cast (so hidden surfaces aren't painted), and
the visible, front-facing cameras vote. Instead of sampling the photo colour
(that's `texture_avatar.py`), it looks up the **predicted severity of the tile**
covering that pixel and blends by how squarely each camera faces the vertex.

The tile lookup mirrors the cropping pipeline: the camera image is rotated
90°/270° per camera and cut into a 5-column × 9-row grid (`a1`…`e9`), so the
prediction CSV's `camera` + `grid_cell` map straight onto the projected pixel.

**Predictions CSV** — the packaged `paint_sundamage.py` accepts either:
- this package's `per_tile_predictions.csv` (`pred_photodamage` string), or
- the study's `04_infer_sundamage.py` output (numeric `pred`).
It only needs the columns `patient_id, camera, grid_cell, p_mild, p_moderate,
p_severe`.

---

## Notes & caveats

- **Grey = not scored** — vertices no camera could see, or that fell on a tile
  rejected as non-skin, stay neutral grey.
- These meshes use **per-vertex colour** (unlit), so they render correctly in any
  viewer, including Open3D. (The separate photo-*textured* atlas GLBs render dark
  in Open3D — not used here.)
- The projection ignores lens distortion (`K1,K2`); it's negligible for these
  long-focal skin cameras and was validated visually against the photo texture.
- Runtime is ~1–2 min per patient (mostly the two turntable renders). Lower
  `n_frames` (e.g. 24) for smaller/faster gifs.
- Files carried here: `sfcm.py`, `tom_texture.py` (mesh loader), `texture_avatar.py`,
  `paint_sundamage.py`, `turntable_gif.py`, `make_avatar.sh`.

---

## Appendix — working with the `.tom` container directly

Not needed to run the pipeline above. Included for anyone re-implementing the
visualisation or reading the geometry themselves.

### The format

A `.tom` is a chunked binary container:

```
8-byte magic:  E8 'T' 'O' 'M' 0D 0A 20 0A     (the \r\n \n guard PNG also uses)
repeated:
    uint64  data_length        little-endian
    8-byte  name               ASCII, '.'/'_' padded   e.g. 'Vertices'
    8-byte  qualifier          ASCII, '.' padded       'z' = payload is zlib
    bytes   data[data_length]
    uint32  trailer            always 0 (reserved)
```

**The 4-byte trailer is the thing to get right.** Assuming 8-byte alignment
instead survives the first two chunks of a session file and then desynchronises.
The test that your walk is correct: it consumes the file *exactly* — the last
chunk is `EndTOM` and zero bytes remain.

A session file holds 58 chunks: `Header`, `ThumbPNG` (PNG preview), `Metadata`,
`Vertices`, `Tris` (zlib), `TxtrJPG` + 46 × `TxtrJPGA` (JPEG atlas pages),
`TexAtlas`, `AtlasIms`, `TexVertA` (zlib), `CnrTexVs` (zlib), `Trnsform`, `EndTOM`.

Payload types:

| Chunk | Contents |
|---|---|
| `Vertices` | float32 XYZ, world mm (length ÷ 12 = vertex count) |
| `Tris` | zlib; int32 pairs — `idx[0::2].reshape(-1,3)` gives the faces |
| `TexVertA` | zlib; `(f32 u, f32 v, i32 page)`, u,v in [0,1] |
| `CnrTexVs` | zlib; `(i32 face, i32 corner 0-2, i32 texvert)` per-corner indirection |
| `TxtrJPG*` | raw JPEG atlas pages, ~5202×2836 |

### Tools

```bash
python3 tom_parse.py <file.tom> [outdir]     # dump every chunk, auto-inflate zlib
```

Start here — it prints the inventory and writes each chunk to disk, which is the
fastest way to see the structure.

### Optional: full-resolution photo texture

The main pipeline uses per-vertex colour, which renders correctly everywhere. If
you want the sharper photo-textured avatar from Vectra's own atlas:

```bash
python3 bake_atlas.py <file.tom> <out_dir> --maxw 5202 --quality 95
python3 fix_glb_material.py <out_dir>/vectra_atlas.glb    # REQUIRED, see below
python3 bake_vertexcolor.py <file.tom> <out>.ply          # robust .ply alternative
python3 render_atlas_flat.py <file.tom> <out_dir>         # independent preview renderer
```

Two traps: `trimesh` writes the GLB with a grey tint and single-sided faces, so
`fix_glb_material.py` must be run after `bake_atlas.py`; and **Open3D renders
these atlas GLBs dark** — that's an Open3D PBR quirk, not a bad file, so verify
with `render_atlas_flat.py` or any glTF viewer instead.

Needs the two extra deps at the bottom of `requirements.txt` (`trimesh`,
`pygltflib`).

> Note: `tom_texture.py` and `paint_sundamage.py` locate the mesh chunks by
> scanning for the `Vertices` tag and the first zlib stream after it, rather than
> by walking the container. That is more fragile than `tom_parse.py`, but it is
> the code path every study result was produced with, so it is left as-is. Prefer
> `tom_parse.py` for anything new.
