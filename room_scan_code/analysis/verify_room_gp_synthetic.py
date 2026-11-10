#!/usr/bin/env python3
"""Synthetic validation of the joint p-u GP at the real scan positions with the real dwell.
Truth: room-like field (monopole 10 cm below the origin + 3 image sources), e^{+iwt} convention, stored
velocity u_s = A grad p.  Noise: complex Gaussian, std = beta_true * |truth level| / sqrt(n_blocks), on p and on
each u component (u level = |u|/sqrt3).  Compared at all points: (A) local Helmholtz reconstruction of the paper
(96 neighbours, dwell weights); (B) GP, pressure only; (C) GP, pressure and velocity.  Hyperparameter beta is
estimated by maximum marginal likelihood on a subset of blocks (alpha fixed at 1 here).  Coverage: fraction of
points whose true value lies in the 90 % posterior interval (|error| <= 1.517 sd for a circular complex Gaussian)."""
import sys, time, numpy as np; sys.path.insert(0, 'lib')
from room_gp import gp_local
from room_stage2 import local_recon
Z = np.load('results/room_pipeline_750Hz.npz'); pos = Z['pos']; nb = Z['dwell']
f = 750.0; k = 2 * np.pi * f / 343; om = 2 * np.pi * f; A = 1 / (1j * om * 1.204)
srcs = [(np.array([0, 0, -0.1]), 1.0), (np.array([0, 0, -2.0]), 0.6), (np.array([0.9, 0, -0.1]), 0.5), (np.array([0, 2.6, -0.1]), 0.4)]
P0 = np.zeros(len(pos), complex); G0 = np.zeros((len(pos), 3), complex)
for s0, amp in srcs:
    d = pos - s0; r = np.linalg.norm(d, axis=1); e = np.exp(-1j * k * r) / r
    P0 += amp * e; G0 += amp * ((-1j * k - 1 / r) * e / r)[:, None] * d
U0 = A * G0
BT = 0.9; rng = np.random.default_rng(1)
lp, lu = np.abs(P0), np.linalg.norm(U0, axis=1) / np.sqrt(3)
cn = lambda n: (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
P = P0 + BT * lp / np.sqrt(nb) * cn(len(pos)); U = U0 + (BT * lu / np.sqrt(nb))[:, None] * np.column_stack([cn(len(pos)) for _ in range(3)])
print(f"noise: median relative error of the raw data {np.median(np.abs(P - P0) / np.abs(P0)) * 100:.0f}%")
# levels from a smooth estimate (as for the data): magnitude of the local Helmholtz reconstruction
q = np.arange(len(pos)); w = nb.copy()
PA = local_recon(pos, P, w, k, q, n_nb=96)
UA = np.column_stack([local_recon(pos, U[:, j], w, k, q, n_nb=96) for j in range(3)])
lvp, lvu = np.abs(PA), np.linalg.norm(UA, axis=1) / np.sqrt(3)
eA = np.abs(PA - P0) / np.abs(P0); print(f"(A) local Helmholtz (paper): median error {np.median(eA) * 100:.1f}%")
# hyperparameter by marginal likelihood on ~1/4 of the blocks
sub = np.zeros(len(pos), bool); cell = np.floor((pos - pos.min(0)) / 0.25).astype(int); _, inv = np.unique(cell, axis=0, return_inverse=True)
sub[np.isin(inv, np.random.default_rng(0).choice(inv.max() + 1, (inv.max() + 1) // 4, replace=False))] = True
for use_u, lab in ((False, '(B) GP, p only'), (True, '(C) GP, p and u')):
    best = None
    for bt in (0.5, 0.7, 0.9, 1.2, 1.6):
        out = gp_local(pos, P, U, nb, lvp, lvu, k, om, beta_p=bt, beta_u=bt, use_u=use_u, query_mask=sub, return_loglik=True)
        if best is None or out[-1] > best[1]: best = (bt, out[-1])
    t0 = time.time()
    mp, sp, mu, su = gp_local(pos, P, U, nb, lvp, lvu, k, om, beta_p=best[0], beta_u=best[0], use_u=use_u)
    e = np.abs(mp - P0); cov90 = np.mean(e <= 1.517 * sp)
    print(f"{lab}: beta by max. likelihood {best[0]} (true {BT}); median error {np.median(e / np.abs(P0)) * 100:.1f}%; "
          f"90% interval coverage {cov90 * 100:.0f}%; median predicted sd / |p| {np.median(sp / np.abs(P0)) * 100:.1f}%  [{time.time() - t0:.0f} s]")
    if use_u:
        eu = np.linalg.norm(mu - U0, axis=1) / np.linalg.norm(U0, axis=1); print(f"    velocity: median error {np.median(eu) * 100:.1f}% (local Helmholtz {np.median(np.linalg.norm(UA - U0, axis=1) / np.linalg.norm(U0, axis=1)) * 100:.1f}%)")
        np.savez('results/room_gp_synthetic.npz', P0=P0, mp=mp, sp=sp, PA=PA)
