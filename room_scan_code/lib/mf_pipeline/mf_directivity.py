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
# Joint pressure-velocity SH fitting
# ---------------------------
#
# Time convention: e^{-i omega t}.  With h_l^(1)(kr) ~ e^{+ikr} this makes the
# radial basis outgoing, and the linearised Euler equation
#     rho * du/dt = -grad(p)      ->      u = grad(p) / (i*omega*rho)
# which yields the correct positive specific impedance p/u_r = +rho*c for an
# outgoing plane/spherical wave in the far field.

def sh_design_at_pos(pos: np.ndarray, center: np.ndarray, k: float, lmax: int) -> np.ndarray:
    """SH pressure design matrix M (N, L2) at arbitrary Cartesian positions."""
    r, theta, phi = angles_from_pos(pos, center)
    A = real_sh_design(theta, phi, lmax).astype(complex)
    H = radial_block(k, r, lmax)
    return H * A


def velocity_design_cartesian(
    pos: np.ndarray,
    center: np.ndarray,
    k: float,
    lmax: int,
    omega: float,
    rho: float = 1.2,
    h: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Design matrices (M_ux, M_uy, M_uz), each (N, L2) complex, mapping SH
    coefficients to the three Cartesian particle-velocity components.

    The gradient is taken by central finite differences of the pressure design
    matrix in Cartesian coordinates.  This is deliberate: the analytic
    spherical-gradient form carries a 1/(r sin(theta)) factor that is singular
    on the polar axis, whereas the Cartesian difference is regular everywhere
    the field itself is regular.  With h = 1e-5 m the truncation and roundoff
    errors are both ~1e-7 relative or below for k r of order unity.
    """
    out = []
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = h
        Mp = sh_design_at_pos(pos + step[None, :], center, k, lmax)
        Mm = sh_design_at_pos(pos - step[None, :], center, k, lmax)
        out.append((Mp - Mm) / (2.0 * h))
    fac = 1.0 / (1j * omega * rho)
    return tuple(fac * g for g in out)


def fit_complex_SH_nearfield_pu(
    p_meas: np.ndarray,
    u_meas: np.ndarray,
    pos: np.ndarray,
    center: np.ndarray,
    k: float,
    omega: float,
    opt: SHFitOptions,
    rho: float = 1.2,
    c: float = 343.0,
    w_u: float = 1.0,
) -> np.ndarray:
    """
    Joint pressure + particle-velocity SH fit.

    Stacks the pressure equation and the three Cartesian velocity equations
    into a single system of 4N equations for L2 unknowns:

        [   M_p  ]         [   p    ]
        [ s M_ux ]  a  =    [ s u_x ]
        [ s M_uy ]         [ s u_y ]
        [ s M_uz ]         [ s u_z ]

    with s = w_u * Z0 and Z0 = rho*c the characteristic impedance, so that the
    pressure [Pa] and velocity [m/s] blocks contribute in comparable units.

    Velocity is not redundant with pressure here: u constrains grad(p)
    independently, which is what disambiguates angular structure that a
    pressure-only fit leaves poorly conditioned.

    Parameters
    ----------
    p_meas : (N,) complex pressure
    u_meas : (N,3) complex Cartesian particle velocity
    pos    : (N,3) measurement positions
    center : (3,) expansion centre
    w_u    : relative weight of the velocity block; w_u=0 reproduces the
             pressure-only fit of fit_complex_SH_nearfield.

    Returns
    -------
    a : (L2,) complex SH coefficients
    """
    Mp = sh_design_at_pos(pos, center, k, opt.lmax)
    Mux, Muy, Muz = velocity_design_cartesian(pos, center, k, opt.lmax, omega, rho)

    s = w_u * rho * c

    M = np.vstack([Mp, s * Mux, s * Muy, s * Muz])
    y = np.concatenate([
        p_meas.astype(complex),
        s * u_meas[:, 0].astype(complex),
        s * u_meas[:, 1].astype(complex),
        s * u_meas[:, 2].astype(complex),
    ])

    if opt.weights is not None:
        w = np.tile(opt.weights, 4).reshape(-1, 1)
        M = w * M
        y = w.flatten() * y

    reg = opt.lam * np.eye(M.shape[1], dtype=complex)
    return np.linalg.solve(M.conj().T @ M + reg, M.conj().T @ y)


def eval_velocity_at_pos(
    a: np.ndarray,
    pos: np.ndarray,
    center: np.ndarray,
    k: float,
    lmax: int,
    omega: float,
    rho: float = 1.2,
) -> np.ndarray:
    """Predict Cartesian particle velocity (N,3) complex from SH coefficients."""
    Mux, Muy, Muz = velocity_design_cartesian(pos, center, k, lmax, omega, rho)
    return np.column_stack([Mux @ a, Muy @ a, Muz @ a])


def angular_coverage(pos: np.ndarray, center: np.ndarray, nside_lon: int = 72,
                     nside_lat: int = 36) -> float:
    """
    Fraction of the full sphere spanned by the measurement directions,
    estimated as the solid-angle-weighted fraction of occupied bins on a
    lon/lat grid.  Returns a value in [0, 1].
    """
    _, theta, phi = angles_from_pos(pos, center)
    lon_edges = np.linspace(-np.pi, np.pi, nside_lon + 1)
    lat_edges = np.linspace(0.0, np.pi, nside_lat + 1)
    ci = np.clip(np.digitize(phi, lon_edges) - 1, 0, nside_lon - 1)
    ri = np.clip(np.digitize(theta, lat_edges) - 1, 0, nside_lat - 1)
    occupied = np.zeros((nside_lat, nside_lon), dtype=bool)
    occupied[ri, ci] = True
    # solid angle of each latitude band row
    d_omega = (np.cos(lat_edges[:-1]) - np.cos(lat_edges[1:])) * (2 * np.pi / nside_lon)
    return float((occupied * d_omega[:, None]).sum() / (4 * np.pi))


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
