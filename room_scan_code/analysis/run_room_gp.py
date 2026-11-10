#!/usr/bin/env python3
"""Joint p-u GP on the room data (750 Hz unless argv[1]).  Hyperparameters beta_p, beta_u by maximum marginal
likelihood on a quarter of the blocks (two-stage grid); the velocity calibration alpha is fixed at 1 (Euler's
equation with the sensors as calibrated) unless argv[2] == 'alpha-free'.  Then:
 (1) split-half random error (same halves and targets as run_room_pipeline.py), p only and p+u, vs local recon;
 (2) calibration on held-out data: 10 % of points withheld from training, predicted; the observed value should
     lie in the 90 % predictive interval (posterior + noise variance) 90 % of the time;
 (3) full reconstruction saved: mean and sd of p and u at every point."""
import sys, json, time, numpy as np; sys.path.insert(0, 'lib')
from room_gp import gp_local
from room_stage2 import local_recon
F0 = float(sys.argv[1]) if len(sys.argv) > 1 else 750.0
Z = np.load(f'results/room_pipeline_{int(round(F0))}Hz.npz'); pos = Z['pos']; nb = Z['dwell']; P = Z['p_raw']; U = Z['u_raw']
k = 2 * np.pi * F0 / 343; om = 2 * np.pi * F0
lvp = np.abs(Z['p']); lvu = np.linalg.norm(Z['u'], axis=1) / np.sqrt(3)
cell = np.floor((pos - pos.min(0)) / 0.25).astype(int); _, inv = np.unique(cell, axis=0, return_inverse=True)
sub = np.isin(inv, np.random.default_rng(0).choice(inv.max() + 1, (inv.max() + 1) // 4, replace=False))
def ll(bp, bu, al, use_u=True):
    return gp_local(pos, P, U, nb, lvp, lvu, k, om, beta_p=bp, beta_u=bu, alpha=al, use_u=use_u, query_mask=sub, return_loglik=True)[-1]
t0 = time.time()
best_p = max((ll(b, 1.0, 1.0, use_u=False), b) for b in (1.3, 1.8, 2.5, 3.5, 5.0, 7.0, 10.0))[1]
ALPHA_FREE = len(sys.argv) > 2 and sys.argv[2] == 'alpha-free'
BG = (1.3, 1.8, 2.5, 3.5, 5.0, 7.0, 10.0)
grid = [(ll(best_p, bu, al), bu, al) for bu in BG for al in ((0.6, 0.7, 0.8, 1.0, 1.25) if ALPHA_FREE else (1.0,))]
_, best_u, best_a = max(grid)
best_p2 = max((ll(b, best_u, best_a), b) for b in (1.3, 1.8, 2.5, 3.5, 5.0, 7.0, 10.0))[1]
print(f"[{F0:.0f} Hz] hyperparameters by max. marginal likelihood: beta_p {best_p2} (p-only fit: {best_p}), beta_u {best_u}, alpha {best_a}   [{time.time()-t0:.0f} s]")
res = dict(f=F0, beta_p=best_p2, beta_p_ponly=best_p, beta_u=best_u, alpha=best_a, grid_edge=bool(best_p2 in (1.3, 10.0) or best_u in (1.3, 10.0)), alpha_free=ALPHA_FREE)
# (1) split-half
rng = np.random.default_rng(4); qq = rng.choice(len(P), 800, replace=False); half = rng.random(len(P)) < 0.5
qm = np.zeros(len(P), bool); qm[qq] = True
for use_u, lab, bp in ((False, 'p only', best_p), (True, 'p+u', best_p2)):
    ma = gp_local(pos, P, U, nb, lvp, lvu, k, om, beta_p=bp, beta_u=best_u, alpha=best_a, use_u=use_u, query_mask=qm, train_mask=half)[0]
    mb = gp_local(pos, P, U, nb, lvp, lvu, k, om, beta_p=bp, beta_u=best_u, alpha=best_a, use_u=use_u, query_mask=qm, train_mask=~half)[0]
    e = np.median(np.abs(ma[qq] - mb[qq]) / np.sqrt(2) / np.abs(0.5 * (ma[qq] + mb[qq])))
    res['split_' + lab] = float(e); print(f"(1) split-half random error, GP {lab}: {e*100:.1f}%")
# (2) calibration on held-out points
ho = np.zeros(len(P), bool); ho[np.random.default_rng(7).choice(len(P), len(P) // 10, replace=False)] = True
for use_u, lab, bp in ((False, 'p only', best_p), (True, 'p+u', best_p2)):
    mp, sp, mu, su = gp_local(pos, P, U, nb, lvp, lvu, k, om, beta_p=bp, beta_u=best_u, alpha=best_a, use_u=use_u, query_mask=ho, train_mask=~ho)
    var = sp[ho] ** 2 + bp ** 2 * lvp[ho] ** 2 / np.maximum(nb[ho], 0.5)
    cov = np.mean(np.abs(P[ho] - mp[ho]) <= 1.517 * np.sqrt(var))
    res['cov90_' + lab] = float(cov); print(f"(2) held-out 90% predictive interval coverage, GP {lab}: {cov*100:.0f}%")
# (3) full reconstruction
mp, sp, mu, su = gp_local(pos, P, U, nb, lvp, lvu, k, om, beta_p=best_p2, beta_u=best_u, alpha=best_a, use_u=True)
res['rel_sd_median'] = float(np.median(sp / np.abs(mp)))
r = np.linalg.norm(pos, axis=1); q4 = np.percentile(r, [0, 25, 50, 75, 100])
res['rel_sd_by_distance'] = [float(np.median((sp / np.abs(mp))[(r >= a) & (r <= b)])) for a, b in zip(q4[:-1], q4[1:])]
print(f"(3) posterior sd / |p|: median {res['rel_sd_median']*100:.1f}%; by distance quartile {np.round(np.array(res['rel_sd_by_distance'])*100,1)}%")
np.savez(f'results/room_gp_{int(round(F0))}Hz.npz', mp=mp, sp=sp, mu=mu, su=su, pos=pos)
json.dump(res, open(f'results/room_gp_{int(round(F0))}Hz.json', 'w'))
