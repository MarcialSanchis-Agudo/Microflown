"""NumPy-only helpers: k-nearest-neighbour IDW onto a grid, and plain k-means."""
import numpy as np


def knn(xq, x, k, chunk=1500):
    """Indices and distances of the k nearest points of x (N,d) for each row of xq (M,d)."""
    idx = np.empty((len(xq), k), int); dst = np.empty((len(xq), k))
    for s in range(0, len(xq), chunk):
        d = np.linalg.norm(xq[s:s + chunk, None, :] - x[None, :, :], axis=-1)
        # deterministic: distances rounded to 1 nm, ties broken by point index (stable sort), so that
        # rounding-level changes of the coordinates cannot change which neighbours are chosen
        j = np.argsort(np.round(d, 9), axis=1, kind='stable')[:, :k]
        idx[s:s + chunk] = j; dst[s:s + chunk] = np.take_along_axis(d, j, 1)
    return idx, dst


def idw_grid(xy, val, gx, gy, k=8, power=2, maxdist=None):
    """IDW of scattered values onto the grid (gy, gx); NaN beyond maxdist of the nearest node."""
    GX, GY = np.meshgrid(gx, gy)
    q = np.column_stack([GX.ravel(), GY.ravel()])
    idx, d = knn(q, xy, k)
    w = 1.0 / np.maximum(d, 1e-9) ** power
    out = (w * val[idx]).sum(1) / w.sum(1)
    if maxdist is not None:
        out = np.where(d[:, 0] > maxdist, np.nan, out)
    return out.reshape(GY.shape)


def kmeans(x, k, seed=0, iters=200):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(np.linalg.norm(x[:, None, :] - c[None, :, :], axis=-1), 1)
        cn = np.array([x[lab == j].mean(0) for j in range(k)])
        if np.allclose(cn, c):
            break
        c = cn
    return lab, c


def gauss_smooth(F, sig_px):
    """Separable Gaussian smoothing of a 2-D array (NaN-aware, normalised)."""
    if sig_px <= 0:
        return F
    r = int(np.ceil(3 * sig_px)); x = np.arange(-r, r + 1); g = np.exp(-0.5 * (x / sig_px) ** 2); g /= g.sum()
    M = np.isfinite(F).astype(float); F0 = np.where(np.isfinite(F), F, 0.0)
    def conv(A):
        A = np.apply_along_axis(lambda v: np.convolve(v, g, mode='same'), 0, A)
        return np.apply_along_axis(lambda v: np.convolve(v, g, mode='same'), 1, A)
    num, den = conv(F0), conv(M)
    out = num / np.maximum(den, 1e-12)
    return np.where(den > 0.2, out, np.nan)
