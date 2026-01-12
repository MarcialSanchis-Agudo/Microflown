# mf_pipeline/mf_directivity.py

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.special import sph_harm, spherical_jn, spherical_yn

# ---------------------------
# Simple loaders/helpers used by other run_* scripts
# ---------------------------

def load_filled_npz(npz_path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    Z = np.load(npz_path, allow_pickle=True)
    pos = Z["pos"]
    p   = Z["p"]
    u   = Z["u"]
    meta = Z["meta"].item() if "meta" in Z.files else {}
    return pos, p, u, meta

def choose_metric(P: np.ndarray, U: np.ndarray, which: str = "u_mag") -> np.ndarray:
    which = which.lower()
    if which == "u_mag":
        return np.sqrt(np.abs(U[:,0])**2 + np.abs(U[:,1])**2 + np.abs(U[:,2])**2)
    if which == "ux": return np.abs(U[:,0])
    if which == "uy": return np.abs(U[:,1])
    if which == "uz": return np.abs(U[:,2])
    if which == "p":  return np.abs(P)
    raise ValueError(f"Unknown metric '{which}'")

# ---------------------------
# Angles, centers, utilities
# ---------------------------

def angles_from_pos(pos: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = pos - center[None, :]
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(np.clip(z / np.maximum(r, 1e-12), -1.0, 1.0))  # colatitude 0..pi
    phi = np.arctan2(y, x)                                           # azimuth -pi..pi
    return r, theta, phi

def real_sh_design(theta: np.ndarray, phi: np.ndarray, lmax: int) -> np.ndarray:
    N = theta.size
    L2 = (lmax + 1) ** 2
    A = np.empty((N, L2), dtype=float)
    idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y = sph_harm(m, l, phi, theta)  # complex (N,)
            if m < 0:
                A[:, idx] = np.sqrt(2.0) * (-1)**m * Y.imag
            elif m == 0:
                A[:, idx] = Y.real
            else:
                A[:, idx] = np.sqrt(2.0) * (-1)**m * Y.real
            idx += 1
    return A

def hankel1(l: int, z: np.ndarray) -> np.ndarray:
    return spherical_jn(l, z) + 1j * spherical_yn(l, z)

def radial_block(k: float, r: np.ndarray, lmax: int) -> np.ndarray:
    N = r.size
    L2 = (lmax + 1) ** 2
    H = np.empty((N, L2), dtype=complex)
    idx = 0
    for l in range(lmax + 1):
        hl = hankel1(l, k * r)  # (N,)
        for _m in range(-l, l + 1):
            H[:, idx] = hl
            idx += 1
    return H

def l_band_indices(lmax: int, l: int) -> np.ndarray:
    start = sum((2*lp + 1) for lp in range(l))
    end = start + (2*l + 1)
    return np.arange(start, end)

# ---------------------------
# Core complex SH near→far
# ---------------------------

@dataclass
class SHFitOptions:
    lmax: int = 8
    lam: float = 1e-3
    weights: Optional[np.ndarray] = None  # row weights (N,)

def fit_complex_SH_nearfield(
    p_meas: np.ndarray,
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    k: float,
    opt: SHFitOptions,
) -> np.ndarray:
    A = real_sh_design(theta, phi, opt.lmax).astype(complex)  # (N, L2)
    H = radial_block(k, r, opt.lmax)                          # (N, L2)
    M = H * A                                                 # Hadamard per column

    y = p_meas.astype(complex)
    if opt.weights is not None:
        w = opt.weights.reshape(-1, 1)
        M = w * M
        y = w.flatten() * y

    L2 = M.shape[1]
    reg = opt.lam * np.eye(L2, dtype=complex)
    a = np.linalg.solve(M.conj().T @ M + reg, M.conj().T @ y)
    return a  # complex, shape (L2,)

def eval_complex_SH_at_radius(
    a: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    k: float,
    r_eval: float,
    lmax: int,
) -> np.ndarray:
    A = real_sh_design(theta, phi, lmax).astype(complex)          # (N, L2)
    H = radial_block(k, np.array([r_eval]), lmax)                  # (1, L2)
    H = np.repeat(H, theta.size, axis=0)                           # (N, L2)
    return (H * A) @ a                                             # complex (N,)


def eval_complex_SH_points(
    a: np.ndarray,
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    k: float,
    lmax: int,
) -> np.ndarray:
    """
    Evaluate SH coefficients at arbitrary points (r, theta, phi) with per-point radius.
    """
    A = real_sh_design(theta, phi, lmax).astype(complex)  # (N, L2)
    H = radial_block(k, r, lmax)                          # (N, L2)
    return (H * A) @ a                                   # complex (N,)

# ---------------------------
# Acoustic-center (optional)
# ---------------------------

def dipole_energy(a: np.ndarray, lmax: int) -> float:
    idx = l_band_indices(lmax, 1)
    return float(np.vdot(a[idx], a[idx]).real)

def auto_center_gridsearch(
    pos: np.ndarray,
    p_meas: np.ndarray,
    f0: float,
    c: float,
    lmax: int,
    lam: float,
    span_mm: float = 6.0,
    step_mm: float = 2.0,
) -> np.ndarray:
    k = 2.0 * np.pi * f0 / c
    mm = 1e-3
    grid = np.arange(-span_mm, span_mm + 1e-9, step_mm) * mm
    best_c = np.zeros(3)
    best_e = np.inf
    for dx in grid:
        for dy in grid:
            for dz in grid:
                center = np.array([dx, dy, dz])
                r, th, ph = angles_from_pos(pos, center)
                a = fit_complex_SH_nearfield(p_meas, r, th, ph, k, SHFitOptions(lmax=lmax, lam=lam))
                e = dipole_energy(a, lmax)
                if e < best_e:
                    best_e = e
                    best_c = center.copy()
    return best_c

# ---------------------------
# Map + polar helpers
# ---------------------------

def grid_lonlat(res_lon: int = 361, res_lat: int = 181) -> Tuple[np.ndarray, np.ndarray]:
    lon_deg = np.linspace(-180.0, 180.0, res_lon)
    lat_deg = np.linspace(-90.0, 90.0, res_lat)
    return lon_deg, lat_deg

def eval_on_lonlat_map(a: np.ndarray, k: float, r_eval: float, lmax: int,
                       res_lon: int = 361, res_lat: int = 181) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_deg, lat_deg = grid_lonlat(res_lon, res_lat)
    lon, lat = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    Lon, Lat = np.meshgrid(lon, lat)
    theta = np.pi/2 - Lat  # colat
    phi = Lon
    V = eval_complex_SH_at_radius(a, theta.ravel(), phi.ravel(), k, r_eval, lmax)
    return lon_deg, lat_deg, V.reshape(Lat.shape)

def horizontal_polar_from_map(lon_deg: np.ndarray, lat_deg: np.ndarray, V: np.ndarray, elev_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    j = int(np.argmin(np.abs(lat_deg - elev_deg)))
    return lon_deg, V[j, :].ravel()

def vertical_polar_from_map(lon_deg: np.ndarray, lat_deg: np.ndarray, V: np.ndarray, az_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    i = int(np.argmin(np.abs(lon_deg - az_deg)))
    return lat_deg, V[:, i].ravel()
