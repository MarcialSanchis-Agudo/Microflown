#!/bin/sh
# Reproduce every number and figure of the paper from the raw Scan&Paint 3D export.
# Run from this folder:  PYTHON=$(which python) sh run_all.sh   (about 1.5-2 h on a laptop)
# Each script prints the numbers it produces; the output is kept in results/<script>.log.
# Errors appear on the terminal and the run stops at the first failing step.
PY=${PYTHON:-python3}
fail() { echo "FAILED: $1 (output so far in $2; error shown above)" >&2; exit 1; }
"$PY" -c "import numpy, scipy, matplotlib, PIL; from scipy.special import sph_harm" \
  || { echo "Missing packages for $PY: run  $PY -m pip install -r requirements.txt  (SciPy must be < 1.17)" >&2; exit 1; }
mkdir -p results paperfigs
run() { name=$(basename "$1" .py)$3; log="results/${name}.log"; echo "== $1 $2"
        "$PY" -u "$1" $2 > "$log" || fail "$1 $2" "$log"; cat "$log"; }
F="257.8125 492.1875 750 1007.8125 1242.1875"
run analysis/prepare_mesh.py
# 1. validation on exact fields (Sec. 4)
run analysis/verify_room_equations.py
run analysis/verify_table1.py table _table; run analysis/verify_table1.py lam _lam; run analysis/verify_table1.py N _N
run analysis/verify_room_stage2_fill.py
run analysis/verify_room_gp_kernel.py
run analysis/verify_room_gp_closedform.py
# 2. processing chain at five frequencies (Secs. 3, 5)
L=results/room_pipeline_allf.log; : > $L
for f in $F; do echo "== run_room_pipeline $f"; "$PY" -u analysis/run_room_pipeline.py $f >> $L || fail "run_room_pipeline $f" $L; done; cat $L
# 3. noise diagnosis and reconstruction checks (Secs. 5.2-5.4)
run analysis/verify_room_noise_model.py
run analysis/verify_room_dwell.py
run analysis/verify_room_refpair.py 750 _750
L=results/verify_room_refpair_allf.log; : > $L
for f in 257.8125 492.1875 1007.8125 1242.1875; do
  "$PY" -u analysis/verify_room_refpair.py $f > results/_tmp.log || fail "verify_room_refpair $f" results/_tmp.log
  sed -n 1p results/_tmp.log >> $L; done; rm -f results/_tmp.log; cat $L
for s in verify_room_recon verify_room_recon_final verify_room_recon_physics verify_room_recon_bias_r verify_room_rawsplit verify_room_flagweight verify_room_flag_threshold verify_room_euler_control; do run analysis/$s.py; done
# 4. joint pressure-velocity Gaussian process (Secs. 2.4, 4.3, 5.5)
run analysis/verify_room_gp_synthetic.py
for f in $F; do run analysis/run_room_gp.py $f _$("$PY" -c "print(int(round($f)))"); done
run analysis/verify_room_gp_noise_model.py
run analysis/run_gp_stack.py
# 5. energy flux, directivity, registration (Secs. 5.6-5.8)
run analysis/verify_room_localise_null.py
L=results/verify_room_polar.log; : > $L
for f in 492.1875 750 1007.8125 1242.1875; do "$PY" -u analysis/verify_room_polar.py $f >> $L || fail "verify_room_polar $f" $L; done; cat $L
run analysis/verify_room_balloon.py 30
run analysis/verify_room_mesh_alignment.py
run analysis/verify_room_mesh_frame.py
# 6. figures, animation, interactive viewer
for s in make_fig_validation make_fig_room make_fig_room2 make_fig_gp make_fig_stack3d make_wave_animation make_fig_flux3d make_fig_directivity make_fig_setup3d make_viewer3d; do run figures/$s.py; done
echo "done: numbers in results/*.log, figures in paperfigs/"
