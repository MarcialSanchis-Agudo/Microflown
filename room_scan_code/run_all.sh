#!/bin/sh
# Reproduce every number and figure of the paper from the raw Scan&Paint 3D export.
# Usage (repository root):  PYTHON=$(which python) sh run_all.sh     (about 2 h)
# Each script's output is kept in results/<script>.log; the run stops at the first error.
PY=${PYTHON:-python3}
"$PY" -c "import numpy, matplotlib, PIL" || { echo "numpy/matplotlib/pillow not found for $PY: set PYTHON=/path/to/python"; exit 1; }
echo "using $PY"
mkdir -p results paperfigs
run() {
  name=$(basename "$1" .py)$3; echo "== $1 $2"
  "$PY" -u "$1" $2 > "results/${name}.log" 2>&1 || { cat "results/${name}.log"; echo "FAILED: $1 $2"; exit 1; }
  cat "results/${name}.log"
}
once() { "$PY" -u "$@" || { echo "FAILED: $*"; exit 1; }; }
F="257.8125 492.1875 750 1007.8125 1242.1875"
run analysis/prepare_mesh.py
run analysis/verify_room_equations.py
run analysis/verify_table1.py table _table; run analysis/verify_table1.py lam _lam; run analysis/verify_table1.py N _N
run analysis/verify_room_stage2_fill.py
run analysis/verify_room_gp_kernel.py
: > results/room_pipeline_allf.log
for f in $F; do once analysis/run_room_pipeline.py $f >> results/room_pipeline_allf.log; done
cat results/room_pipeline_allf.log
run analysis/verify_room_noise_model.py
run analysis/verify_room_dwell.py
run analysis/verify_room_refpair.py 750 _750
: > results/verify_room_refpair_allf.log
for f in 257.8125 492.1875 1007.8125 1242.1875; do once analysis/verify_room_refpair.py $f > results/_tmp.log; sed -n 1p results/_tmp.log >> results/verify_room_refpair_allf.log; done
for s in verify_room_recon verify_room_recon_final verify_room_recon_physics verify_room_recon_bias_r verify_room_rawsplit verify_room_flagweight verify_room_flag_threshold verify_room_euler_control; do run analysis/$s.py; done
run analysis/verify_room_gp_synthetic.py
for f in $F; do run analysis/run_room_gp.py $f _$("$PY" -c "print(int(round($f)))"); done
run analysis/verify_room_gp_noise_model.py
run analysis/run_gp_stack.py
run analysis/verify_room_localise_null.py
: > results/verify_room_polar.log
for f in 492.1875 750 1007.8125 1242.1875; do once analysis/verify_room_polar.py $f >> results/verify_room_polar.log; done
run analysis/verify_room_balloon.py 30
run analysis/verify_room_mesh_alignment.py
run analysis/verify_room_mesh_frame.py
for s in make_fig_validation make_fig_room make_fig_room2 make_fig_gp make_fig_stack3d make_wave_animation make_fig_flux3d make_fig_directivity make_fig_setup3d make_viewer3d; do run figures/$s.py; done
echo "done: numbers in results/*.log, figures in paperfigs/"
