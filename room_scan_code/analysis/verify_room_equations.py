#!/usr/bin/env python3
"""Numerical checks of the equations introduced in the room paper (Secs. 2-3)."""
import sys, numpy as np; sys.path.insert(0, 'lib')
from room_stage2 import _regular_basis, robust_norm
ok = []
k = 2 * np.pi * 750 / 343; r = np.linspace(0.01, 3, 400)
# (1) Im G = k j0(kr)/(4 pi)
G = np.exp(1j * k * r) / (4 * np.pi * r); j0 = np.sin(k * r) / (k * r)
ok.append(('Im G = k j0/4pi', np.max(np.abs(G.imag - k * j0 / (4 * np.pi)))))
# (2) regular basis solves Helmholtz: finite-difference Laplacian of each column + k^2 * column ~ 0
rng = np.random.default_rng(0); x0 = rng.normal(size=(50, 3)) * 0.1; h = 1e-3; worst = 0
for d in x0:
    f0 = _regular_basis(d[None], k)[0]; lap = -6 * f0
    for ax in range(3):
        for s in (1, -1):
            e = np.zeros(3); e[ax] = s * h; lap = lap + _regular_basis((d + e)[None], k)[0]
    lap /= h ** 2; worst = max(worst, np.max(np.abs(lap + k ** 2 * f0)) / k ** 2)
ok.append(('j_l Y_lm basis satisfies Helmholtz (rel. residual)', worst))
# (3) j_l(0) = delta_l0 -> basis at rho = 0 is (1, 0, ..., 0)
b0 = _regular_basis(np.zeros((1, 3)), k)[0]; ok.append(('basis at the node = e_0', np.max(np.abs(b0 - np.eye(9)[0]))))
# (4) median |complex Gaussian error| = 1.177 sigma (sigma per component)
z = rng.normal(size=10 ** 6) + 1j * rng.normal(size=10 ** 6); ok.append(('median |e| / sigma - 1.1774', abs(np.median(np.abs(z)) - 1.1774)))
# (5) Bendat-Piersol by simulation: H = Gxr/Grr from n blocks, gamma^2 = 0.7, n = 3 and 8
for g2, n in ((0.7, 3), (0.9, 8)):
    Ntr = 20000; R_ = (rng.normal(size=(Ntr, n)) + 1j * rng.normal(size=(Ntr, n))) / np.sqrt(2)
    Nn = (rng.normal(size=(Ntr, n)) + 1j * rng.normal(size=(Ntr, n))) / np.sqrt(2) * np.sqrt((1 - g2) / g2)
    X = R_ + Nn; H = (X * R_.conj()).sum(1) / (np.abs(R_) ** 2).sum(1)
    sim = np.median(np.abs(H - 1)); pred = 1.177 * np.sqrt((1 - g2) / (2 * n * g2))
    ok.append((f'BP median error gamma2={g2}, n={n}: sim {sim:.3f} vs Eq.6 {pred:.3f} (rel. diff)', abs(sim - pred) / pred))
# (6) flux-line intersection recovers a known point from exact radial lines
P = rng.normal(size=(500, 3)); s0 = np.array([0.3, -0.2, 0.1]); D = (P - s0) / np.linalg.norm(P - s0, axis=1)[:, None]
M = np.eye(3)[None] - D[:, :, None] * D[:, None, :]; xs = np.linalg.solve(M.sum(0), (M @ P[:, :, None]).sum(0)[:, 0])
ok.append(('x* recovers source of exact radial flux', np.linalg.norm(xs - s0)))
# (7) robust scale fallback: Gaussian -> ~1 either way
g = rng.normal(size=10 ** 5); ok.append(('robust_norm std on Gaussian - 1', abs(np.std(robust_norm(g)) - 1)))
h_ = np.maximum(g, 0); ok.append(('fallback used when MAD=0 (zeros > 50%): finite', 0.0 if np.all(np.isfinite(robust_norm(np.where(g > 0.5, g, 0)))) else 1.0))
for name, v in ok: print(f"{'PASS' if v < 0.06 else 'FAIL'}  {name}: {v:.2e}")
