#!/usr/bin/env python3
"""Why flag, and does the 5 % budget matter?  For flag fractions 2 / 5 / 10 / 20 % of the pressure channel
(750 Hz): median coherence and reference-pair scatter of flagged vs kept points (independent checks), and
the split-half random error of the reconstruction with the flagged points weighted zero (same halves and
targets as run_room_pipeline.py)."""
import sys, numpy as np; sys.path.insert(0, 'lib')
from room_stage2 import local_recon
Z = np.load('results/room_pipeline_750Hz.npz'); pos = Z['pos']; P = Z['p_raw']; k = 2 * np.pi * 750 / 343
sc = Z['score_p']; coh = Z['coh']; q = Z['ref_ratio']; qm = np.median(q.real) + 1j * np.median(q.imag); rp = np.abs(q / qm - 1)
rng = np.random.default_rng(4); qq = rng.choice(len(P), 800, replace=False); half = rng.random(len(P)) < 0.5
A_, B_ = np.flatnonzero(half), np.flatnonzero(~half)
for frac in (0.0, 0.02, 0.05, 0.10, 0.20):
    fl = sc >= np.quantile(sc, 1 - frac) if frac > 0 else np.zeros(len(sc), bool)
    w = Z['dwell'] * (~fl)
    pa = local_recon(pos, P, w, k, qq, n_nb=96, src=A_); pb = local_recon(pos, P, w, k, qq, n_nb=96, src=B_)
    e = np.median(np.abs(pa - pb) / np.sqrt(2) / np.abs(0.5 * (pa + pb)))
    if frac > 0:
        print(f"flag {frac*100:4.0f}%: coherence flagged {np.median(coh[fl]):.2f} vs kept {np.median(coh[~fl]):.2f}; ref-pair scatter {np.median(rp[fl]):.2f} vs {np.median(rp[~fl]):.2f}; split-half error {e*100:.1f}%")
    else:
        print(f"no flags : split-half error {e*100:.1f}%")
