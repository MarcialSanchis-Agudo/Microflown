#!/usr/bin/env python3
"""Marginal-likelihood checks of the GP noise model on the room data (750 Hz, same quarter of the blocks):
(1) extended grid for beta_p (the first search hit its upper bound);
(2) dwell-dependent noise (var ~ 1/n_blocks) against uniform noise (n set to its median for every point),
    each with its own best beta: does dwell explain the data better?  Reported: mean log-likelihood per
    observation and its difference."""
import sys, numpy as np; sys.path.insert(0, 'lib')
from room_gp import gp_local
Z = np.load('results/room_pipeline_750Hz.npz'); pos = Z['pos']; nb = Z['dwell']; P = Z['p_raw']; U = Z['u_raw']
k = 2 * np.pi * 750 / 343; om = 2 * np.pi * 750
lvp = np.abs(Z['p']); lvu = np.linalg.norm(Z['u'], axis=1) / np.sqrt(3)
cell = np.floor((pos - pos.min(0)) / 0.25).astype(int); _, inv = np.unique(cell, axis=0, return_inverse=True)
sub = np.isin(inv, np.random.default_rng(0).choice(inv.max() + 1, (inv.max() + 1) // 4, replace=False))
def ll(bp, bu, nn):
    return gp_local(pos, P, U, nb if nn is None else nn, lvp, lvu, k, om, beta_p=bp, beta_u=bu, alpha=1.0, query_mask=sub, return_loglik=True)[-1]
res = {}
for lab, nn in (('dwell-dependent', None), ('uniform', np.full(len(pos), np.median(nb)))):
    best = max((ll(bp, bu, nn), bp, bu) for bp in (2.5, 3.5, 5.0, 7.0, 10.0) for bu in (1.8, 2.5, 3.5, 5.0))
    res[lab] = best; print(f"{lab:16s}: best beta_p {best[1]}, beta_u {best[2]}, mean log-likelihood per observation {best[0]:.4f}")
d = res['dwell-dependent'][0] - res['uniform'][0]
print(f"difference (dwell - uniform): {d:+.4f} per observation")
