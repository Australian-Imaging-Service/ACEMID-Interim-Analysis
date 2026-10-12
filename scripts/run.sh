#!/usr/bin/env bash
# One-command runner (Linux / macOS).
#   ./run.sh <tiles_dir> [out_dir] [extra run_inference.py flags...]
#
# Auto-handles the Python environment: if torch + open_clip are already importable
# it uses your current Python; otherwise it creates a local .venv and installs
# requirements.txt once. Then it scores every tile and writes the two CSVs.
#
# Examples:
#   ./run.sh /data/site_tiles
#   ./run.sh /data/site_tiles my_results
#   ./run.sh /data/site_tiles my_results --limit 50 --gpu 0
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "usage: ./run.sh <tiles_dir> [out_dir] [extra flags...]"
  exit 0
fi
TILES="$1"; shift
OUT="results"
if [ "${1:-}" != "" ] && [[ "${1:-}" != --* ]]; then OUT="$1"; shift; fi

# Pick a Python that already has the deps; else build a local venv once.
if python3 -c 'import torch, open_clip, PIL, pandas, numpy' >/dev/null 2>&1; then
  PY=python3
  echo "[env] using current Python (dependencies already present)"
else
  if [ ! -d .venv ]; then
    echo "[env] creating local .venv and installing requirements (first run only)…"
    python3 -m venv .venv
    ./.venv/bin/pip install --upgrade pip >/dev/null
    ./.venv/bin/pip install -r requirements.txt
  fi
  PY=./.venv/bin/python
  echo "[env] using ./.venv"
fi

echo "[run] scoring tiles in: $TILES  ->  $OUT/"
"$PY" run_inference.py --tiles-dir "$TILES" --out-dir "$OUT" "$@"
echo "[done] see $OUT/per_tile_predictions.csv and $OUT/per_patient_severity.csv"
