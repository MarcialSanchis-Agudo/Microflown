#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

python scripts/run_plot_source.py \
  --npz figs/paper_figs/filled_arrays_near25_1000Hz.npz \
  --metric u_mag --subsample 10 --scale 0.02
# -> figs/paper_figs/filled_arrays_near25_1000Hz_u_mag_3d.png
# -> figs/paper_figs/filled_arrays_near25_1000Hz_u_mag_projections.png

python scripts/run_nearfield_slices.py \
  --npz figs/paper_figs/filled_arrays_near25_4000Hz.npz \
  --metric u_mag --n 160 --smooth 0.2 --db
# -> figs/paper_figs/filled_arrays_near25_4000Hz_u_mag_slices.png
