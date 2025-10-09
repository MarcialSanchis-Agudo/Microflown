# mf_pipeline/mf_directivity.py
import numpy as np
from pathlib import Path
from scipy.special import sph_harm

def to_spherical(P: np.ndarray, center=None):
    if center is None: center = P.mean(axis=0)
    C = P - center
    x,y,z = C[:,0],C[:,1],C[:,2]
    r = np.sqrt(x*x+y*y+z*z)+1e-12
    theta = np.arccos(np.clip(z/r,-1,1))
    phi   = np.arctan2(y,x)
    return r, theta, phi, center

def design_SH(theta, phi, lmax):
    N = theta.size; cols = (lmax+1)**2
    A = np.zeros((N, cols), float); c = 0
    for l in range(lmax+1):
        A[:, c] = sph_harm(0, l, phi, theta).real; c += 1
        for m in range(1, l+1):
            Y = sph_harm(m, l, phi, theta)
            A[:, c] = np.sqrt(2)*Y.real; c += 1
            A[:, c] = np.sqrt(2)*Y.imag; c += 1
    return A

def fit_SH(P: np.ndarray, v: np.ndarray, lmax=8, lam=1e-3, center=None):
    _, th, ph, C = to_spherical(P, center)
    A = design_SH(th, ph, lmax)
    ATA, ATv = A.T@A, A.T@v
    coeffs = np.linalg.solve(ATA + lam*np.eye(ATA.shape[0]), ATv)
    return coeffs, dict(lmax=lmax, center=C)

def eval_SH_map(coeffs, lmax=8, res=(361,181)):
    lon = np.linspace(-np.pi, np.pi, res[0])
    lat = np.linspace(-np.pi/2, np.pi/2, res[1])
    Lon, Lat = np.meshgrid(lon, lat)
    th = np.pi/2 - Lat; ph = Lon
    A = design_SH(th.ravel(), ph.ravel(), lmax)
    V = (A @ coeffs).reshape(Lat.shape)
    return np.rad2deg(lon), np.rad2deg(lat), V

def load_filled_npz(npz_path: Path):
    Z = np.load(npz_path, allow_pickle=True)
    return Z["pos"], Z["p"], Z["u"], Z["meta"].item()

def choose_metric(P_fill: np.ndarray, U_fill: np.ndarray, which: str="u_mag"):
    which = which.lower()
    if which == "u_mag":
        return np.linalg.norm(U_fill, axis=1)
    if which == "ux":
        return np.abs(U_fill[:,0])
    if which == "uy":
        return np.abs(U_fill[:,1])
    if which == "uz":
        return np.abs(U_fill[:,2])
    if which == "p":
        return np.abs(P_fill)
    raise ValueError("which must be one of: u_mag, ux, uy, uz, p")

# --- spherical sampling & polar cuts ---
def unit_sphere_grid(n_theta=181, n_phi=361):
    th = np.linspace(0.0, np.pi, n_theta)
    ph = np.linspace(-np.pi, np.pi, n_phi)
    Th, Ph = np.meshgrid(th, ph, indexing="ij")
    return Th, Ph

def eval_on_sphere(coeffs, lmax=8, n_theta=181, n_phi=361):
    Th, Ph = unit_sphere_grid(n_theta=n_theta, n_phi=n_phi)
    A = design_SH(Th.ravel(), Ph.ravel(), lmax)
    V = (A @ coeffs).reshape(Th.shape)
    return Th, Ph, V

def polar_cut_azimuth(coeffs, lmax=8, elevation_deg=0.0, n=361):
    phi = np.linspace(-np.pi, np.pi, n)
    theta = np.deg2rad(90.0 - elevation_deg) * np.ones_like(phi)
    A = design_SH(theta, phi, lmax)
    v = A @ coeffs
    return np.rad2deg(phi), v

def polar_cut_elevation(coeffs, lmax=8, azimuth_deg=0.0, n=181):
    theta = np.linspace(0.0, np.pi, n)
    phi = np.deg2rad(azimuth_deg) * np.ones_like(theta)
    A = design_SH(theta, phi, lmax)
    v = A @ coeffs
    elev_deg = 90.0 - np.rad2deg(theta)
    return elev_deg, v
