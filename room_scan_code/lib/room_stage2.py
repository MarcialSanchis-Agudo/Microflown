#!/usr/bin/env python3
"""
room_stage2.py — Stage 2 (noise detection and filling) applied to the room scan.

Score: the four-feature anomaly score of microflown_speaker.ipynb (features_noise +
score_and_flags), re-implemented with NumPy: per channel (p, ux, uy, uz)
  magnitude deficit vs the 11-neighbour complex median, phase deviation (capped 120 deg),
  complex-correlation penalty; plus 0.5 x an impedance-consistency term shared by the node.
  Each term MAD-normalised, weights 1 / 0.5 / 1 and 0.6 / 0.4; top 5 % per channel flagged.
Fill: NOT the notebook's equivalent sources on the sensor nodes (singular self-term,
  see F13); instead a local regular Helmholtz expansion sum_{l<=2} c_lm j_l(kr) Y_lm about
  each flagged node, fitted per channel to its 24 nearest unflagged neighbours (ridge 1e-5).
  j_0(kr) = 4 pi Im G(r)/k is the regular part of the monopole kernel.
"""
import sys, numpy as np; sys.path.insert(0, 'lib')
from scipy_free_interp import knn

RHO, C = 1.204, 343.0
Z0 = RHO * C


def robust_norm(x):
    """(x - median) / robust scale.  The scale is 1.4826 MAD; when the MAD vanishes (a feature that is
    zero for more than half the nodes, e.g. the one-sided magnitude deficit) it falls back to
    1.2533 x the mean absolute deviation from the median (both equal sigma for a Gaussian)."""
    med = np.median(x); dev = np.abs(x - med)
    sc = 1.4826 * np.median(dev)
    if sc < 1e-12 * (np.mean(dev) + 1e-300): sc = 1.2533 * np.mean(dev)
    return (x - med) / (sc + 1e-300)


def cmedian(z, nb):
    return np.median(z[nb].real, 1) + 1j * np.median(z[nb].imag, 1)


def score(pos, P, U, k_nn=12, top=0.05):
    nb = knn(pos, pos, k_nn)[0][:, 1:]
    ch = {'p': P, 'ux': U[:, 0], 'uy': U[:, 1], 'uz': U[:, 2]}
    s = {}
    for n, x in ch.items():
        m = cmedian(x, nb)
        mag = np.maximum(0, 1 - (np.abs(x) + 1e-18) / (np.abs(m) + 1e-18))
        dph = np.abs(np.angle(x * np.conj(m)))
        corr = 1 - np.cos(dph) * np.minimum(1, (np.abs(x) + 1e-12) / (np.abs(m) + 1e-12))
        s[n] = np.maximum(0, robust_norm(mag) + robust_norm(corr) + 0.5 * robust_norm(np.minimum(dph, np.radians(120))))
    umag = np.linalg.norm(U, axis=1)
    imp_mag = np.abs(np.log10((np.abs(P) + 1e-18) / (Z0 * umag + 1e-18)))
    ub = np.take_along_axis(U, np.argmax(np.abs(U), 1)[:, None], 1)[:, 0]
    imp_phi = np.abs(np.angle(P * np.conj(ub)))
    # the sign of a Cartesian velocity component depends on the propagation direction, so the phase
    # difference is folded to [0, pi/2]: 0 for an active (in- or anti-phase) field, pi/2 for a reactive one
    imp_phi = np.minimum(imp_phi, np.pi - imp_phi)
    s_imp = 0.6 * robust_norm(imp_mag) + 0.4 * robust_norm(imp_phi)
    flags = {}
    for n in ch:
        s[n] = s[n] + 0.5 * s_imp
        flags[n] = s[n] >= np.quantile(s[n], 1 - top)
    return s, flags, s_imp


def _sph_jn(l, x):
    x = np.maximum(x, 1e-9)
    j0 = np.sin(x) / x
    if l == 0: return j0
    j1 = np.sin(x) / x**2 - np.cos(x) / x
    if l == 1: return j1
    return (3 / x**2 - 1) * np.sin(x) / x - 3 * np.cos(x) / x**2


def _regular_basis(d, k, L=2):
    """Real-valued regular solutions j_l(kr) * (solid-harmonic direction factors), l<=L, (n,9)."""
    r = np.linalg.norm(d, axis=1); n = d / np.maximum(r, 1e-12)[:, None]; x, y, z = n.T
    cols = [_sph_jn(0, k * r)]
    if L >= 1: cols += [_sph_jn(1, k * r) * c for c in (x, y, z)]
    if L >= 2: cols += [_sph_jn(2, k * r) * c for c in (x * y, y * z, x * z, x * x - y * y, 3 * z * z - 1)]
    return np.column_stack(cols)


def local_fill(pos, x, bad, k, n_nb=24, lam=1e-5, L=2, query=None):
    """Predict x at nodes `query` (default: bad) from the n_nb nearest nodes with ~bad."""
    q = np.flatnonzero(bad) if query is None else np.asarray(query)
    good = np.flatnonzero(~bad)
    idx, _ = knn(pos[q], pos[good], n_nb)
    out = np.empty(len(q), complex)
    for i, (qi, ii) in enumerate(zip(q, idx)):
        g = good[ii]; g = g[g != qi]
        B = _regular_basis(pos[g] - pos[qi], k, L)
        c = np.linalg.solve(B.T @ B + lam * np.trace(B.T @ B) / B.shape[1] * np.eye(B.shape[1]), B.T @ x[g])
        out[i] = c[0]                 # basis at the node itself: j0(0)=1, all others 0
    return out


def dwell_blocks(pos):
    """FFT blocks (2048 @ 48 kHz) of signal per node, from SampleCellMatrix (cells matched 100 %)."""
    from room_raw import h5, G
    S = h5().dataset(G + 'SampleCellMatrix').astype(np.int64)
    g = np.round(pos / 0.02).astype(int); key = lambda a: a[:, 0] * 10**8 + a[:, 1] * 10**4 + a[:, 2]
    kn = key(g - g.min(0)); ijk = S[:3].T; ks = key(ijk - ijk.min(0))
    dw = {}
    for kk, d in zip(ks, S[4] - S[3]): dw[kk] = dw.get(kk, 0) + d
    return np.array([dw[x] for x in kn]) / 2048.0


def local_recon(pos, x, w, k, query, n_nb=48, lam=1e-5, L=2, src=None, exclude_self=False):
    """Weighted local regular-Helmholtz fit at each query node from its n_nb nearest nodes in `src`."""
    src = np.arange(len(pos)) if src is None else np.asarray(src)
    idx, _ = knn(pos[query], pos[src], n_nb + (1 if exclude_self else 0))
    out = np.empty(len(query), complex)
    for i, (qi, ii) in enumerate(zip(query, idx)):
        g = src[ii]
        if exclude_self: g = g[g != qi][:n_nb]
        B = _regular_basis(pos[g] - pos[qi], k, L); W = w[g]
        BtW = B.T * W; A = BtW @ B
        c = np.linalg.solve(A + lam * np.trace(A) / A.shape[0] * np.eye(A.shape[0]), BtW @ x[g])
        out[i] = c[0]
    return out
