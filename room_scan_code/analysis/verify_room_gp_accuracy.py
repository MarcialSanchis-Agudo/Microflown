#!/usr/bin/env python3
"""What the GP error bars and the GP split-half error do and do not show (paper Sec. 5.5).
Uses the hyperparameters chosen by run_room_gp.py (results/room_gp_<f>Hz.json) and the same splits.
 A. Held-out points (same 10 % as run_room_gp.py): share of the posterior variance in the predictive
    variance; 90 % coverage with the full predictive variance and with the noise variance alone.
 B. Posterior sd tested directly: GP estimates from two disjoint halves at the same 800 targets; if the
    posterior sd is right, |m_a - m_b| / sqrt(sd_a^2 + sd_b^2) lies within sqrt(ln 10) 90 % of the time.
 C. Accuracy on held-out points: median |y - yhat| / level for the local fit and the GP
    (y is a single noisy reading, so this is dominated by the point noise).
 D. Amplitude: median |GP| / |local fit| and median |GP - local| / |local| over all points."""
import sys, json, numpy as np; sys.path.insert(0, 'lib')
from room_gp import gp_local
from room_stage2 import local_recon
F0 = float(sys.argv[1]) if len(sys.argv) > 1 else 750.0; F = int(round(F0))
Z = np.load(f'results/room_pipeline_{F}Hz.npz'); pos = Z['pos']; nb = Z['dwell']; P = Z['p_raw']; U = Z['u_raw']
J = json.load(open(f'results/room_gp_{F}Hz.json')); bp, bu, al = J['beta_p'], J['beta_u'], J['alpha']
k = 2 * np.pi * F0 / 343; om = 2 * np.pi * F0
lvp = np.abs(Z['p']); lvu = np.linalg.norm(Z['u'], axis=1) / np.sqrt(3)
R90 = np.sqrt(np.log(10))
print(f"[{F} Hz] beta_p {bp}, beta_u {bu}, alpha {al}")
# A
ho = np.zeros(len(P), bool); ho[np.random.default_rng(7).choice(len(P), len(P) // 10, replace=False)] = True
mp, sp, _, _ = gp_local(pos, P, U, nb, lvp, lvu, k, om, beta_p=bp, beta_u=bu, alpha=al, use_u=True, query_mask=ho, train_mask=~ho)
nv = bp ** 2 * lvp[ho] ** 2 / np.maximum(nb[ho], 0.5); pv = sp[ho] ** 2; r = np.abs(P[ho] - mp[ho])
print(f"A. posterior share of predictive variance: median {np.median(pv/(pv+nv)):.3f}, 90th pct {np.percentile(pv/(pv+nv),90):.3f}")
print(f"   90 % coverage, full predictive variance {100*np.mean(r <= R90*np.sqrt(pv+nv)):.0f} %; noise variance only {100*np.mean(r <= R90*np.sqrt(nv)):.0f} %")
# B
rng = np.random.default_rng(4); qq = rng.choice(len(P), 800, replace=False); half = rng.random(len(P)) < 0.5
qm = np.zeros(len(P), bool); qm[qq] = True
ma, sa, _, _ = gp_local(pos, P, U, nb, lvp, lvu, k, om, beta_p=bp, beta_u=bu, alpha=al, use_u=True, query_mask=qm, train_mask=half)
mb, sb, _, _ = gp_local(pos, P, U, nb, lvp, lvu, k, om, beta_p=bp, beta_u=bu, alpha=al, use_u=True, query_mask=qm, train_mask=~half)
z = np.abs(ma[qq] - mb[qq]) / np.sqrt(sa[qq] ** 2 + sb[qq] ** 2)
print(f"B. split-half difference within the 90 % radius of the posterior sd: {100*np.mean(z <= R90):.0f} % (calibrated: 90 %); median z {np.median(z):.2f} (calibrated: {np.sqrt(np.log(2)):.2f})")
# C
w = nb * (~Z['flag_p']); q = np.flatnonzero(ho); tr = np.flatnonzero(~ho)
loc = local_recon(pos, P, w, k, q, n_nb=96, src=tr)
mpo = gp_local(pos, P, U, nb, lvp, lvu, k, om, beta_p=J['beta_p_ponly'], beta_u=bu, alpha=al, use_u=False, query_mask=ho, train_mask=~ho)[0]
for lab, e in (('local fit', loc), ('GP p only', mpo[q]), ('GP p+u', mp[q])):
    d = np.abs(P[q] - e) / lvp[q]
    print(f"C. held-out |y - yhat| / level, {lab:9s}: median {np.median(d):.3f}, rms {np.sqrt(np.mean(d**2)):.2f}")
# D
G = np.load(f'results/room_gp_{F}Hz.npz'); g = G['mp']; l = Z['p']
print(f"D. all points: median |GP|/|local| {np.median(np.abs(g)/np.abs(l)):.2f}; median |GP - local|/|local| {np.median(np.abs(g-l)/np.abs(l)):.2f}")
