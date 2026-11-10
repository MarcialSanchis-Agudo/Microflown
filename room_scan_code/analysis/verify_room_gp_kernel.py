#!/usr/bin/env python3
"""Checks of the joint p-u covariance in room_gp.py: (1) cross-covariances equal finite differences of the
j0 kernel; (2) the full covariance is Hermitian positive semi-definite; (3) a GP fitted to exact noise-free
p and u of a plane wave and of a monopole field reproduces them at held-out points."""
import sys, numpy as np; sys.path.insert(0, 'lib')
from room_gp import _cov_blocks, sph_j0
k = 2 * np.pi * 750 / 343; om = 2 * np.pi * 750; a = 1 / (1j * om * 1.204); s2 = 1.3
rng = np.random.default_rng(0); X1 = rng.normal(size=(6, 3)) * 0.2; X2 = rng.normal(size=(5, 3)) * 0.2
C = _cov_blocks(X1, X2, k, s2, a); n1, n2 = 6, 5
kf = lambda x1, x2: s2 * sph_j0(k * np.linalg.norm(x1 - x2))
h = 1e-5; err = 0
for i1 in range(n1):
    for i2 in range(n2):
        for j in range(3):
            e = np.zeros(3); e[j] = h
            dk2 = (kf(X1[i1], X2[i2] + e) - kf(X1[i1], X2[i2] - e)) / (2 * h)       # d/dx2_j
            dk1 = (kf(X1[i1] + e, X2[i2]) - kf(X1[i1] - e, X2[i2])) / (2 * h)       # d/dx1_j
            err = max(err, abs(C[i1, (j + 1) * n2 + i2] - np.conj(a) * dk2) / abs(a * k), abs(C[(j + 1) * n1 + i1, i2] - a * dk1) / abs(a * k))
            for i in range(3):
                f = np.zeros(3); f[i] = h
                d2 = (kf(X1[i1] + f, X2[i2] + e) - kf(X1[i1] + f, X2[i2] - e) - kf(X1[i1] - f, X2[i2] + e) + kf(X1[i1] - f, X2[i2] - e)) / (4 * h * h)
                err = max(err, abs(C[(i + 1) * n1 + i1, (j + 1) * n2 + i2] - abs(a) ** 2 * d2) / abs(a * k) ** 2)
print(f"(1) max relative deviation of cross-covariances from finite differences: {err:.2e}")
X = rng.normal(size=(40, 3)) * 0.15; Cs = _cov_blocks(X, X, k, s2, a)
herm = np.max(np.abs(Cs - Cs.conj().T)); ev = np.linalg.eigvalsh((Cs + Cs.conj().T) / 2)
print(f"(2) Hermitian deviation {herm:.1e}; smallest eigenvalue / largest {ev.min() / ev.max():.1e}")
# (3) exact fields: plane wave and monopole at 2 m, stored convention e^{+iwt}: p = exp(-i k.x), u_s = a grad p
Xt = rng.uniform(-0.15, 0.15, size=(150, 3)); Xq = rng.uniform(-0.08, 0.08, size=(30, 3))
for name, pf, gf in (('plane wave', lambda X: np.exp(-1j * k * X @ np.array([0.6, 0.0, 0.8])), lambda X: (-1j * k * np.array([0.6, 0, 0.8]))[None, :] * np.exp(-1j * k * X @ np.array([0.6, 0, 0.8]))[:, None]),
                     ('monopole 2 m away', None, None)):
    if pf is None:
        s0 = np.array([2.0, 0.5, -0.3])
        def pf(X): r = np.linalg.norm(X - s0, axis=1); return np.exp(-1j * k * r) / r
        def gf(X):
            d = X - s0; r = np.linalg.norm(d, axis=1); return ((-1j * k - 1 / r) * np.exp(-1j * k * r) / r ** 2)[:, None] * d
    pt, ut = pf(Xt), a * gf(Xt); y = np.concatenate([pt, ut[:, 0], ut[:, 1], ut[:, 2]]); s2_ = np.mean(np.abs(pt) ** 2)
    for use_u in (False, True):
        K = _cov_blocks(Xt, Xt, k, s2_, a); Kq = _cov_blocks(Xq, Xt, k, s2_, a)
        if not use_u: K = K[:150, :150]; Kq = Kq[:30, :150]; yy = y[:150]
        else: Kq = Kq[:30]; yy = y
        m = Kq @ np.linalg.solve(K + 1e-8 * s2_ * np.eye(len(K)), yy)
        print(f"(3) {name:18s} {'p and u' if use_u else 'p only '}: held-out relative error {np.median(np.abs(m - pf(Xq)) / np.abs(pf(Xq))):.1e}")
