#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

H5="${1:?H5 path required (quote it if it has spaces)}"
TAG_BASE="${2:?TAG_BASE required (e.g., near25 or far50)}"
FREQS_STR="${3:?space-separated frequencies in quotes}"
RFAR="${4:-1.0}"
RMEAS="${5:-0.050}"
C="${6:-343}"
BIN_DEG="${7:-5}"
NORMALIZE="${8:-1}"

OUTDIR="$(python - <<'PY'
import os
from pathlib import Path
from mf_pipeline.mf_config import EXPORTS
print(Path(os.environ.get("MF_OUTDIR", EXPORTS)).resolve())
PY
)"
mkdir -p "${OUTDIR}"
export MF_OUTDIR="${OUTDIR}"

echo "=== FULL FAR-FIELD PIPELINE ==="
echo "cwd:       $(pwd)"
echo "OUTDIR:    ${OUTDIR}"
echo "H5:        ${H5}"
echo "TAG_BASE:  ${TAG_BASE}"
echo "FREQS:     ${FREQS_STR}"
echo "RFAR:      ${RFAR} m"
echo "RMEAS:     ${RMEAS} m"
echo "c:         ${C} m/s"
echo "bin-deg:   ${BIN_DEG}°"
echo "normalize: ${NORMALIZE}"
echo

read -r -a FREQS <<< "${FREQS_STR}"

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

MAPS=()
LABELS=()

for F in "${FREQS[@]}"; do
  RAW_NPZ="${OUTDIR}/filled_arrays_${TAG_BASE}_raw_${F}Hz.npz"
  FILLED_NPZ="${OUTDIR}/filled_arrays_${TAG_BASE}_${F}Hz.npz"
  MAP_NPZ="${OUTDIR}/filled_arrays_${TAG_BASE}_at_${RFAR//./p}m_${F}Hz_p_map_lonlat.npz"

  LMAX=$(python - <<PY
import math
F=${F}; R=${RMEAS}; C=${C}
kr = max(1e-9, 2*math.pi*F*R/C)
print(max(2, int(math.floor(kr)) + 1))
PY
)
  LAM=$(python - <<PY
import math
F=${F}; R=${RMEAS}; C=${C}
kr = max(1e-9, 2*math.pi*F*R/C)
print(f"{2e-3*max(1.0, 2.0/kr):.8f}")
PY
)

  echo "[${F} Hz]  lmax=${LMAX}  lam=${LAM}"

  echo "  -> H5 → NPZ (raw)"
  python scripts/run_paraview_h5_to_npz.py --h5 "${H5}" --f0 "${F}" --tag "${TAG_BASE}_raw"
  pull_into_outdir "${RAW_NPZ}" || { echo "!! Missing raw NPZ ${RAW_NPZ}"; exit 1; }

  echo "  -> fill"
  python scripts/run_fill.py --npz-in "${RAW_NPZ}" --f0 "${F}" --tag "${TAG_BASE}"
  pull_into_outdir "${FILLED_NPZ}" || true  # some run_fill writes in-place to OUTDIR already

  echo "  -> near→far @ ${RFAR} m (bin=${BIN_DEG}°)"
  if [[ "${NORMALIZE}" == "0" ]]; then
    python scripts/run_directivity.py \
      --npz "${FILLED_NPZ}" --field complex_p \
      --lmax "${LMAX}" --lam "${LAM}" --c "${C}" \
      --rfar "${RFAR}" --auto-center --db --no-norm \
      --bin-deg "${BIN_DEG}" \
      --tag "${TAG_BASE}_at_${RFAR//./p}m"
  else
    python scripts/run_directivity.py \
      --npz "${FILLED_NPZ}" --field complex_p \
      --lmax "${LMAX}" --lam "${LAM}" --c "${C}" \
      --rfar "${RFAR}" --auto-center --db \
      --bin-deg "${BIN_DEG}" \
      --tag "${TAG_BASE}_at_${RFAR//./p}m"
  fi
  pull_into_outdir "${MAP_NPZ}" || { echo "!! Missing map ${MAP_NPZ}"; exit 1; }

  MAPS+=("${MAP_NPZ}")
  LABELS+=("${F}")
done

OUT_SHEET="${OUTDIR}/directionality_response_${TAG_BASE}_at_${RFAR//./p}m.png"
echo
echo "== Build polar sheet =="
if [[ "${NORMALIZE}" == "0" ]]; then
  python scripts/run_directionality_sheet.py \
    --maps "${MAPS[@]}" --labels "${LABELS[@]}" \
    --elev 0 --az 0 --no-normalize --out "${OUT_SHEET}"
else
  python scripts/run_directionality_sheet.py \
    --maps "${MAPS[@]}" --labels "${LABELS[@]}" \
    --elev 0 --az 0 --out "${OUT_SHEET}"
fi
echo "[save] ${OUT_SHEET}"
# ---- 3D surface panel (right after building MAPS/LABELS) ----
SURF="${OUTDIR}/surfaces_${TAG_BASE}_at_${RFAR//./p}m.png"
echo "== Build 3D surfaces =="

if [[ "${NORMALIZE}" == "0" ]]; then
  # absolute dB (distance visible)
  python scripts/run_surface_mesh.py \
    --maps "${MAPS[@]}" \
    --titles "${LABELS[@]}" \
    --mode abs --vmin -30 --vmax 0 \
    --cols 3 --out "${SURF}" \
    --cb-right 0.92 --cb-width 0.02 --cb-bottom 0.18 --cb-height 0.64
else
  # normalized dB (shape only)
  python scripts/run_surface_mesh.py \
    --maps "${MAPS[@]}" \
    --titles "${LABELS[@]}" \
    --mode norm --vmin -30 --vmax 0 \
    --cols 3 --out "${SURF}" \
    --cb-right 0.92 --cb-width 0.02 --cb-bottom 0.18 --cb-height 0.64
fi

echo "[save] ${SURF}"
echo
echo "Done."
