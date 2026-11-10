#!/usr/bin/env python3
"""
room_gp.py — joint pressure-velocity Gaussian-process reconstruction with dwell-dependent noise.

Prior: complex circular Gaussian process for the pressure with the Helmholtz (diffuse-field) covariance
    E[p(x) p(x')*] = s2 j0(k|x - x'|)                                   (Ueno et al. 2018)
Every sample of this process satisfies the Helmholtz equation.  The stored velocity follows Euler's equation
in the stored convention, u_s = alpha * A * grad p, A = 1/(i w rho0)  (alpha: calibration factor, ~1),
so velocity observations are linear functionals of the same process and enter with the analytic cross-
covariances below (derivatives of j0).
Noise: independent, heteroscedastic; for observation channel c at point n,
    var = beta_c^2 * level_c(n)^2 / n_blocks(n)
i.e. inversely proportional to the number of spectral averages (Eq. 6), scaled by the local signal level.
Estimation is local: the scan is divided into cubes of side B; each cube is predicted from the points within
B/2 + margin of its centre (at most NMAX, the nearest), with s2 = mean |p|^2 of the local level.
Returns posterior mean and standard deviation of p (and of u) at every scan point.
"""
import numpy as np


def sph_j0(x):
    x = np.asarray(x, float); out = np.ones_like(x); m = x > 1e-8
    out[m] = np.sin(x[m]) / x[m]; return out


def sph_j1(x):
    x = np.asarray(x, float); out = x / 3.0; m = x > 1e-4
    out[m] = np.sin(x[m]) / x[m] ** 2 - np.cos(x[m]) / x[m]; return out


def _cov_blocks(X1, X2, k, s2, a):
    """Covariance between [p, u_x, u_y, u_z](X1) and [p, u_x, u_y, u_z](X2), u = a * grad p (a complex).
    Returns a (4 n1, 4 n2) complex matrix ordered [p-block; ux-block; uy-block; uz-block]."""
    d = X1[:, None, :] - X2[None, :, :]; r = np.linalg.norm(d, axis=-1); kr = k * r
    j0 = sph_j0(kr); j1 = sph_j1(kr)
    with np.errstate(invalid='ignore', divide='ignore'):
        e = np.where(r[..., None] > 1e-9, d / r[..., None], 0.0)
    g = -k * j1                                   # f'(r) for f = j0(kr)
    # f''(r) = -k^2 (j0 - 2 j1/(kr)); limit r->0: -k^2/3
    with np.errstate(invalid='ignore', divide='ignore'):
        fpp = np.where(kr > 1e-4, -k ** 2 * (j0 - 2 * j1 / np.maximum(kr, 1e-12)), -k ** 2 / 3)
        fp_over_r = np.where(kr > 1e-4, g / np.maximum(r, 1e-12), -k ** 2 / 3)
    n1, n2 = len(X1), len(X2)
    C = np.empty((4 * n1, 4 * n2), complex)
    C[:n1, :n2] = s2 * j0
    for j in range(3):
        # Cov(p(x1), u_j(x2)) = conj(a) d/dx2_j k = -conj(a) f'(r) e_j
        C[:n1, (j + 1) * n2:(j + 2) * n2] = -np.conj(a) * s2 * g * e[..., j]
        # Cov(u_i(x1), p(x2)) = a d/dx1_i k = a f'(r) e_i
        C[(j + 1) * n1:(j + 2) * n1, :n2] = a * s2 * g * e[..., j]
    for i in range(3):
        for j in range(3):
            # Cov(u_i(x1), u_j(x2)) = |a|^2 d/dx1_i d/dx2_j k = -|a|^2 H_ij(d)
            H = fpp * e[..., i] * e[..., j] + fp_over_r * ((i == j) - e[..., i] * e[..., j])
            C[(i + 1) * n1:(i + 2) * n1, (j + 1) * n2:(j + 2) * n2] = -abs(a) ** 2 * s2 * H
    return C


def gp_local(pos, P, U, nblk, lvl_p, lvl_u, k, omega, rho=1.204, beta_p=0.6, beta_u=0.6, alpha=1.0,
             use_u=True, B=0.25, margin=0.12, NMAX=220, query_mask=None, return_loglik=False, train_mask=None):
    """Local joint GP.  P (N,), U (N,3) stored data; nblk: averages per point; lvl_p, lvl_u: local signal
    levels (|p| and |u|/sqrt3 of a smooth estimate).  Returns mean_p, sd_p, mean_u, sd_u (NaN where not
    queried) and, optionally, the summed log marginal likelihood of the blocks."""
    a = alpha / (1j * omega * rho)
    N = len(pos); mp = np.full(N, np.nan + 0j); sp = np.full(N, np.nan); mu = np.full((N, 3), np.nan + 0j); su = np.full((N, 3), np.nan)
    q_all = np.ones(N, bool) if query_mask is None else query_mask
    cell = np.floor((pos - pos.min(0)) / B).astype(int); keys, inv = np.unique(cell, axis=0, return_inverse=True)
    ll = 0.0; nobs = 0
    for b in range(len(keys)):
        qi = np.flatnonzero((inv == b) & q_all)
        if len(qi) == 0: continue
        c0 = pos[inv == b].mean(0)
        dist = np.max(np.abs(pos - c0), axis=1); okt = (dist < B / 2 + margin) if train_mask is None else ((dist < B / 2 + margin) & train_mask); tr = np.flatnonzero(okt)
        if len(tr) > NMAX: tr = tr[np.argsort(np.linalg.norm(pos[tr] - c0, axis=1))[:NMAX]]
        Xt = pos[tr]; nt = len(tr)
        s2 = np.mean(lvl_p[tr] ** 2)
        y = np.concatenate([P[tr]] + ([U[tr, j] for j in range(3)] if use_u else []))
        nv_p = beta_p ** 2 * lvl_p[tr] ** 2 / np.maximum(nblk[tr], 0.5)
        nv_u = beta_u ** 2 * lvl_u[tr] ** 2 / np.maximum(nblk[tr], 0.5)
        Cf = _cov_blocks(Xt, Xt, k, s2, a)
        if use_u:
            K = Cf; nv = np.concatenate([nv_p, nv_u, nv_u, nv_u])
        else:
            K = Cf[:nt, :nt]; nv = nv_p
        Ky = K + np.diag(nv) + 1e-9 * s2 * np.eye(len(nv))
        L = np.linalg.cholesky(Ky)
        alph = np.linalg.solve(L.conj().T, np.linalg.solve(L, y))
        if return_loglik:
            z = np.linalg.solve(L, y)
            ll += -np.real(np.vdot(z, z)) - 2 * np.sum(np.log(np.real(np.diag(L)))) - len(y) * np.log(np.pi); nobs += len(y)
        Xq = pos[qi]; Cq = _cov_blocks(Xq, Xt, k, s2, a)      # (4 nq, 4 nt)
        nq = len(qi)
        Kq = Cq if use_u else Cq[:, :nt]
        mean = Kq @ alph
        V = np.linalg.solve(L, Kq.conj().T)                    # (nobs, 4 nq)
        prior = np.concatenate([np.full(nq, s2), np.full(3 * nq, abs(a) ** 2 * s2 * k ** 2 / 3)])
        var = np.maximum(prior - np.sum(np.abs(V) ** 2, axis=0), 0)
        mp[qi] = mean[:nq]; sp[qi] = np.sqrt(var[:nq])
        for j in range(3):
            mu[qi, j] = mean[(j + 1) * nq:(j + 2) * nq]; su[qi, j] = np.sqrt(var[(j + 1) * nq:(j + 2) * nq])
    if return_loglik:
        return mp, sp, mu, su, ll / max(nobs, 1)
    return mp, sp, mu, su


def gp_predict(pos, P, U, nblk, lvp, lvu, k, omega, Xq, rho=1.204, beta_p=3.5, beta_u=2.5, alpha=1.0,
               B=0.25, margin=0.12, NMAX=220, with_u=False):
    """Joint p-u GP prediction of the pressure at arbitrary points Xq (same local scheme as gp_local: cubes of
    side B on the same lattice, training points within B/2 + margin of the cube centre).  Returns mean, sd and
    the prior sd sigma of the cube (NaN where a cube has fewer than 20 training points)."""
    a = alpha / (1j * omega * rho); o = pos.min(0)
    cq = np.floor((Xq - o) / B).astype(int); keys, inv = np.unique(cq, axis=0, return_inverse=True)
    m = np.full(len(Xq), np.nan + 0j); sd = np.full(len(Xq), np.nan); sig = np.full(len(Xq), np.nan); mu_ = np.full((len(Xq), 3), np.nan + 0j)
    for b in range(len(keys)):
        qi = np.flatnonzero(inv == b); c0 = o + (keys[b] + 0.5) * B
        tr = np.flatnonzero(np.max(np.abs(pos - c0), axis=1) < B / 2 + margin)
        if len(tr) < 20: continue
        if len(tr) > NMAX: tr = tr[np.argsort(np.linalg.norm(pos[tr] - c0, axis=1))[:NMAX]]
        Xt = pos[tr]; s2 = np.mean(lvp[tr] ** 2)
        y = np.concatenate([P[tr]] + [U[tr, j] for j in range(3)])
        nv = np.concatenate([beta_p ** 2 * lvp[tr] ** 2 / np.maximum(nblk[tr], 0.5)] + [beta_u ** 2 * lvu[tr] ** 2 / np.maximum(nblk[tr], 0.5)] * 3)
        L = np.linalg.cholesky(_cov_blocks(Xt, Xt, k, s2, a) + np.diag(nv) + 1e-9 * s2 * np.eye(len(nv)))
        nq = len(qi); Kf = _cov_blocks(Xq[qi], Xt, k, s2, a); Kq = Kf[:nq]
        al_ = np.linalg.solve(L.conj().T, np.linalg.solve(L, y)); m[qi] = Kq @ al_
        if with_u:
            mf = Kf @ al_
            for j in range(3): mu_[qi, j] = mf[(j + 1) * nq:(j + 2) * nq]
        V = np.linalg.solve(L, Kq.conj().T); sd[qi] = np.sqrt(np.maximum(s2 - np.sum(np.abs(V) ** 2, axis=0), 0)); sig[qi] = np.sqrt(s2)
    return (m, sd, sig, mu_) if with_u else (m, sd, sig)
