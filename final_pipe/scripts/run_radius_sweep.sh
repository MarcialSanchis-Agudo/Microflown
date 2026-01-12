#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

H5="${1:?}"; TAG="${2:?}"; F="${3:?}"
R_LIST_STR="${4:?}"; RMEAS="${5:-0.025}"; C="${6:-343}"; BIN_DEG="${7:-5}"
read -r -a RL <<< "${R_LIST_STR}"

OUTDIR="$(python - <<'PY'
import os
from pathlib import Path
from mf_pipeline.mf_config import EXPORTS
print(Path(os.environ.get("MF_OUTDIR", EXPORTS)).resolve())
PY
)"
mkdir -p "${OUTDIR}"
export MF_OUTDIR="${OUTDIR}"

RAW="${OUTDIR}/filled_arrays_${TAG}_raw_${F}Hz.npz"
FIL="${OUTDIR}/filled_arrays_${TAG}_${F}Hz.npz"

pull_into_outdir () {
  local target="$1"
  if [[ -f "${target}" ]]; then return 0; fi
  local alt="exports/$(basename "${target}")"
  if [[ -f "${alt}" ]]; then
    mkdir -p "$(dirname "${target}")"
    mv -f "${alt}" "${target}"
    echo "  [moved] ${alt} -> ${target}"
    return 0
  fi
  return 1
}

echo "== Radius sweep @ ${F} Hz =="
echo "OUTDIR: ${OUTDIR}"
echo "RMEAS:  ${RMEAS} m"

echo "  -> H5 → NPZ (raw)"
python scripts/run_paraview_h5_to_npz.py --h5 "${H5}" --f0 "${F}" --tag "${TAG}_raw"
pull_into_outdir "${RAW}" || { echo "!! Missing raw NPZ ${RAW}"; exit 1; }

echo "  -> fill"
python scripts/run_fill.py --npz-in "${RAW}" --f0 "${F}" --tag "${TAG}"
pull_into_outdir "${FIL}" || true

MAPS=(); LABELS=()
for R in "${RL[@]}"; do
  LMAX=$(python - <<PY
import math; F=${F}; R=${RMEAS}; C=${C}
kr=max(1e-9,2*math.pi*F*R/C); print(max(2,int(math.floor(kr))+1))
PY
)
  LAM=$(python - <<PY
import math; F=${F}; R=${RMEAS}; C=${C}
kr=max(1e-9,2*math.pi*F*R/C); print(f"{2e-3*max(1.0,2.0/kr):.8f}")
PY
)
  echo "  rfar=${R} m  lmax=${LMAX} lam=${LAM}"
  python scripts/run_directivity.py --npz "${FIL}" --field complex_p \
    --lmax "${LMAX}" --lam "${LAM}" --c "${C}" \
    --rfar "${R}" --auto-center --db --no-norm \
    --bin-deg "${BIN_DEG}" --tag "${TAG}_at_${R//./p}m"
  MAP="${OUTDIR}/filled_arrays_${TAG}_at_${R//./p}m_${F}Hz_p_map_lonlat.npz"
  pull_into_outdir "${MAP}" || { echo "!! Missing map ${MAP}"; exit 1; }
  MAPS+=("${MAP}")
  LABELS+=("${R} m")
done

OUT="${OUTDIR}/radius_sweep_${TAG}_${F}Hz.png"
python scripts/run_directionality_sheet.py \
  --maps "${MAPS[@]}" --labels "${LABELS[@]}" \
  --elev 0 --az 0 --no-normalize --out "${OUT}"
echo "[save] ${OUT}"

SURF="${OUTDIR}/surfaces_radius_sweep_${TAG}_${F}Hz.png"
python scripts/run_surface_mesh.py \
  --maps "${MAPS[@]}" --labels "${LABELS[@]}" \
  --mode abs --vmin -30 --vmax 0 --cols 3 --out "${SURF}"
echo "[save] ${SURF}"

echo "Done."
