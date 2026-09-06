#!/usr/bin/env bash
# Build the sun-damage 3D-avatar visualisation for ONE patient:
#   raw photo avatar + turntable gif, severity avatar + turntable gif, report panel.
#
# Usage:
#   bash make_avatar.sh <session_dir> <preds_csv> <out_dir> [patient_id] [n_frames]
#
#   session_dir : the raw Vectra session folder for this patient's baseline capture
#                 (must contain <session>.tom, calib/*.sfcm, and *B.jpg photos)
#   preds_csv   : per-tile predictions from run_inference.py (this package).
#                 If it holds several patients, pass patient_id to slice one out.
#   out_dir     : where the .glb/.ply/.png/.gif are written
#   patient_id  : optional; filter preds_csv to this patient
#   n_frames    : optional turntable frames (default 48; use 24 for smaller gifs)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SESS="$1"; PREDS="$2"; OUT="$3"; PID="${4:-}"; NF="${5:-48}"
mkdir -p "$OUT"

# optionally slice one patient out of a multi-patient predictions csv
if [ -n "$PID" ]; then
  P2="$OUT/${PID}_preds.csv"
  python3 - "$PREDS" "$PID" "$P2" <<'PY'
import sys, pandas as pd
src, pid, dst = sys.argv[1:4]
df = pd.read_csv(src)
sub = df[df["patient_id"].astype(str) == pid] if "patient_id" in df.columns else df
sub.to_csv(dst, index=False)
print(f"[slice] {len(sub)} tiles for patient {pid}")
PY
  PREDS="$P2"
fi

echo "===== 1/4  raw photo avatar ====="
python3 "$HERE/texture_avatar.py" "$SESS" "$SESS" "$OUT"          # image_dir = session (uses *B.jpg)

echo "===== 2/4  sun-damage severity avatar + report panel ====="
python3 "$HERE/paint_sundamage.py" "$SESS" "$PREDS" "$OUT"

echo "===== 3/4  raw turntable gif ====="
python3 "$HERE/turntable_gif.py" "$OUT/raw_avatar.ply" "$OUT/raw_turntable.gif" "$NF" 640 960 nobar

echo "===== 4/4  sun-damage turntable gif (with severity colour-bar) ====="
python3 "$HERE/turntable_gif.py" "$OUT/sundamage_avatar.ply" "$OUT/sundamage_turntable.gif" "$NF" 640 960

echo ""
echo "DONE -> $OUT"
ls -1 "$OUT" | sed 's/^/   /'
