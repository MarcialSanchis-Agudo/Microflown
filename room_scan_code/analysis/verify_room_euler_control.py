#!/usr/bin/env python3
"""Control for the Euler-consistency check: is the rise 0.51 -> 0.84 specific to the Helmholtz
reconstruction, or does any smoothing give it?  Non-physical smoother: dwell-weighted MEAN of the same 96
nearest points (flags weighted 0), applied to p and each u component independently; also the local
Helmholtz fit truncated at l = 0 (a weighted mean with j0 weights) and at l <= 1.  Euler consistency as in
the paper (1 500-point draw, seed 0, >= 8 neighbours within 8 cm, linear gradient fit)."""
import sys, numpy as np; sys.path.insert(0, 'lib')
from numpy.linalg import lstsq
from room_stage2 import local_recon
from scipy_free_interp import knn
Z = np.load('results/room_pipeline_750Hz.npz'); pos = Z['pos']; k = 2 * np.pi * 750 / 343; om = 2 * np.pi * 750
fl = {'p': Z['flag_p'], 'ux': Z['flag_ux'], 'uy': Z['flag_uy'], 'uz': Z['flag_uz']}; wd = Z['dwell']
idx, _ = knn(pos, pos, 96)
def mean_smooth(x, f):
    w = wd * (~f); W_ = w[idx]; return (W_ * x[idx]).sum(1) / np.maximum(W_.sum(1), 1e-12)
def euler(p, u):
    sel = np.random.default_rng(0).choice(len(pos), 1500, replace=False); cs = []
    for i in sel:
        d = np.linalg.norm(pos - pos[i], axis=1); nb = np.where((d > 0) & (d < 0.08))[0]
        if len(nb) < 8: continue
        X = np.column_stack([np.ones(len(nb)), pos[nb] - pos[i]]); g = lstsq(X, p[nb], rcond=None)[0][1:]
        ue = g / (1j * om * 1.204); cs.append(np.real(np.vdot(ue, u[i])) / (np.linalg.norm(ue) * np.linalg.norm(u[i]) + 1e-30))
    return np.median(cs)
P, U = Z['p_raw'], Z['u_raw']
print(f"raw                                   : {euler(P, U):.2f}")
Pm = mean_smooth(P, fl['p']); Um = np.column_stack([mean_smooth(U[:, j], fl[c]) for j, c in enumerate(('ux', 'uy', 'uz'))])
print(f"weighted mean of 96 points (no physics): {euler(Pm, Um):.2f}")
q = np.arange(len(pos))
for L in (0, 1):
    Pl = local_recon(pos, P, wd * (~fl['p']), k, q, n_nb=96, L=L)
    Ul = np.column_stack([local_recon(pos, U[:, j], wd * (~fl[c]), k, q, n_nb=96, L=L) for j, c in enumerate(('ux', 'uy', 'uz'))])
    print(f"local Helmholtz fit, l <= {L}             : {euler(Pl, Ul):.2f}")
print(f"local Helmholtz fit, l <= 2 (paper)     : {euler(Z['p'], Z['u']):.2f}")
