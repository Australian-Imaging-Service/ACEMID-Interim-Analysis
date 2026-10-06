#!/usr/bin/env python3
"""
Standalone site-runnable inference for the Photodamage DermLIP MTL v4 model.

Point it at a directory of already-cropped tiles and it writes per-tile predictions
and a per-patient severity table. No training code required — only this folder.

Tile requirements (see MODEL_CARD.md §3):
  * 693 x 1156 px JPG, colour on the TOP half, grayscale on the BOTTOM half
  * filename: <patient_id>_<camera>B_<grid_cell>.jpg   e.g. 7775-542_a10B_b3.jpg
    (patient_id = everything before the first underscore; used for aggregation)

Usage:
  python3 run_inference.py --tiles-dir /path/to/site_tiles --out-dir results/
  python3 run_inference.py --tiles-dir /path/to/site_tiles --out-dir results/ \
      --checkpoint best.pt --batch-size 128 --gpu 0

Outputs (in --out-dir):
  per_tile_predictions.csv   one row per tile: probabilities + argmax for both heads
  per_patient_severity.csv   one row per patient: mean probs, severity_score, modal class
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import build_model  # shipped alongside this script

# CLIP / DermLIP normalisation — MUST match training.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

PHOTO_CLASSES = ["mild", "moderate", "severe"]
PIG_CLASSES = ["low", "medium", "high"]

# <patient_id>_<camera>B_<cell>.jpg  (patient may contain hyphens/letters/digits)
NAME_RE = re.compile(r"^(?P<pid>.+?)_(?P<cam>[A-Za-z]?\d+)B_(?P<cell>[A-Za-z]?\d+)\.jpg$", re.I)


def parse_name(fname: str):
    m = NAME_RE.match(fname)
    if m:
        return m.group("pid"), m.group("cam"), m.group("cell")
    # Fallback: patient = prefix before first underscore; camera/cell unknown.
    return (fname.split("_", 1)[0], None, None)


def _prep_half(img: Image.Image) -> torch.Tensor:
    img = TF.resize(img, (256, 256))
    img = TF.center_crop(img, 224)
    return TF.normalize(TF.to_tensor(img), CLIP_MEAN, CLIP_STD)


class TileDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        img = Image.open(f).convert("RGB")
        w, h = img.size
        colour = _prep_half(img.crop((0, 0, w, h // 2)))
        gray = _prep_half(img.crop((0, h // 2, w, h)))
        return colour, gray, f.name


def load_model(checkpoint: Path, device):
    """Rebuild the architecture from config.json (if present) and load fine-tuned weights."""
    cfg_path = checkpoint.parent / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    model = build_model(
        cfg.get("backbone", "dermlip"),
        mode=cfg.get("mode", "full"),
        dual_stream=(cfg.get("input_mode", "dual-stream") == "dual-stream"),
        fusion=cfg.get("fusion", "modality-embed"),
        use_pre_projection=cfg.get("use_pre_projection", False),
        head_dropout=cfg.get("head_dropout", 0.2),
        n_photodamage=3, n_pigmentation=3,
    )
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval().to(device)
    dual = (cfg.get("input_mode", "dual-stream") == "dual-stream")
    return model, dual


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tiles_pos", nargs="?", default=None,
                    help="directory of cropped tiles (positional; same as --tiles-dir)")
    ap.add_argument("--tiles-dir", default=None, help="directory of cropped tiles (searched recursively)")
    ap.add_argument("--out-dir", default="results", help="output directory for the two CSVs")
    ap.add_argument("--checkpoint", default=str(HERE / "best.pt"), help="model checkpoint (default: best.pt beside this script)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0, help="GPU index; ignored if no CUDA")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N tiles (quick smoke test)")
    args = ap.parse_args()

    tiles_arg = args.tiles_dir or args.tiles_pos
    if not tiles_arg:
        ap.error("provide a tiles directory (positional or --tiles-dir)")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    tiles_dir = Path(tiles_arg)
    files = sorted(p for p in tiles_dir.rglob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"no .jpg tiles found under {tiles_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[device] {device}")
    print(f"[tiles]  {len(files)} tiles under {tiles_dir}")
    model, dual = load_model(Path(args.checkpoint), device)
    print(f"[model]  loaded {args.checkpoint}  (dual_stream={dual})")

    loader = DataLoader(TileDataset(files), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    rows = []
    done = 0
    with torch.no_grad():
        for colour, gray, names in loader:
            colour = colour.to(device, non_blocking=True)
            gray = gray.to(device, non_blocking=True)
            logit_p, logit_g = model(colour, gray) if dual else model(colour)
            pp = logit_p.softmax(-1).cpu().numpy()
            pg = logit_g.softmax(-1).cpu().numpy()
            for name, prob_p, prob_g in zip(names, pp, pg):
                pid, cam, cell = parse_name(name)
                rows.append({
                    "tile": name, "patient_id": pid, "camera": cam, "grid_cell": cell,
                    "p_mild": prob_p[0], "p_moderate": prob_p[1], "p_severe": prob_p[2],
                    "pred_photodamage": PHOTO_CLASSES[int(prob_p.argmax())],
                    "pg_low": prob_g[0], "pg_medium": prob_g[1], "pg_high": prob_g[2],
                    "pred_pigmentation": PIG_CLASSES[int(prob_g.argmax())],
                })
            done += len(names)
            if done % (args.batch_size * 10) < args.batch_size:
                print(f"  scored {done}/{len(files)}")

    tile_df = pd.DataFrame(rows)
    tile_csv = out_dir / "per_tile_predictions.csv"
    tile_df.to_csv(tile_csv, index=False)
    print(f"[out] wrote {tile_csv}  ({len(tile_df)} tiles)")

    # ---- per-patient aggregation ----
    agg = []
    for pid, g in tile_df.groupby("patient_id"):
        n = len(g)
        mp = g[["p_mild", "p_moderate", "p_severe"]].mean()
        mg = g[["pg_low", "pg_medium", "pg_high"]].mean()
        sev = 0 * mp["p_mild"] + 1 * mp["p_moderate"] + 2 * mp["p_severe"]
        pig = 0 * mg["pg_low"] + 1 * mg["pg_medium"] + 2 * mg["pg_high"]
        pc = g["pred_photodamage"].value_counts()
        agg.append({
            "patient_id": pid, "n_tiles": n,
            "mean_p_mild": mp["p_mild"], "mean_p_moderate": mp["p_moderate"], "mean_p_severe": mp["p_severe"],
            "severity_score": sev,
            "modal_class": g["pred_photodamage"].mode().iloc[0],
            "frac_mild": pc.get("mild", 0) / n, "frac_moderate": pc.get("moderate", 0) / n,
            "frac_severe": pc.get("severe", 0) / n,
            "mean_pg_low": mg["pg_low"], "mean_pg_medium": mg["pg_medium"], "mean_pg_high": mg["pg_high"],
            "pigmentation_score": pig,
        })
    pat_df = pd.DataFrame(agg).sort_values("severity_score", ascending=False)
    pat_csv = out_dir / "per_patient_severity.csv"
    pat_df.to_csv(pat_csv, index=False)
    print(f"[out] wrote {pat_csv}  ({len(pat_df)} patients)")
    print(f"\n[summary] patient severity_score: "
          f"mean={pat_df['severity_score'].mean():.3f}  "
          f"median={pat_df['severity_score'].median():.3f}  "
          f"range {pat_df['severity_score'].min():.3f}-{pat_df['severity_score'].max():.3f}")


if __name__ == "__main__":
    main()
