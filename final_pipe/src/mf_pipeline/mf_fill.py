# mf_pipeline/mf_fill.py
import numpy as np, h5py
from pathlib import Path
from dataclasses import dataclass
from scipy.spatial import cKDTree
from scipy.interpolate import Rbf
from mf_pipeline.mf_config import H5_PATH, H5_GROUP, EXPORTS

@dataclass
class SliceData:
    f0: float
    Pk: np.ndarray   # (N,) complex
    Uk: np.ndarray   # (N,3) complex
    pos: np.ndarray  # (N,3) float

def load_positions() -> np.ndarray:
    with h5py.File(H5_PATH, "r") as f:
        pos = np.array(f[f"{H5_GROUP}/POSITION"])
    return pos

def load_slice_complex(f_target: float) -> SliceData:
    with h5py.File(H5_PATH, "r") as f:
        freq = np.array(f[f"{H5_GROUP}/FREQUENCY_VECTOR"]).ravel()
        k = int(np.argmin(np.abs(freq - f_target)))
        Rp = np.array(f[f"{H5_GROUP}/REAL_TFpref1"])[k]     # (N,)
        Ip = np.array(f[f"{H5_GROUP}/IMAG_TFpref1"])[k]
        Ru = np.array(f[f"{H5_GROUP}/REAL_TFxyzref1"])[k]   # (N,3)
        Iu = np.array(f[f"{H5_GROUP}/IMAG_TFxyzref1"])[k]
        pos = np.array(f[f"{H5_GROUP}/POSITION"])
    Pk = Rp + 1j*Ip
    Uk = Ru + 1j*Iu
    return SliceData(float(freq[k]), Pk, Uk, pos)

def robust_flags(values: np.ndarray, nbr_idx: np.ndarray, tau: float=3.5) -> np.ndarray:
    med = np.array([np.median(values[nbr_idx[i]]) for i in range(len(values))])
    mad = np.array([np.median(np.abs(values[nbr_idx[i]] - med[i])) for i in range(len(values))])
    sig = 1.4826*mad + 1e-12
    z = np.abs(values - med) / sig
    return z > tau

def build_knn(pos: np.ndarray, k: int=12) -> np.ndarray:
    tree = cKDTree(pos)
    _, idx = tree.query(pos, k=min(k, len(pos)))
    return idx

def fill_channel(pos: np.ndarray, x: np.ndarray, noisy_mask: np.ndarray, smooth: float=0.2) -> np.ndarray:
    x2 = x.copy()
    tree = cKDTree(pos)
    for i in np.flatnonzero(noisy_mask):
        _, ii = tree.query(pos[i], k=min(24, len(pos)))
        ii = np.atleast_1d(ii)
        ii = ii[~noisy_mask[ii]] if np.any(~noisy_mask[ii]) else ii
        rbf_r = Rbf(pos[ii,0], pos[ii,1], pos[ii,2], np.real(x[ii]), function="multiquadric", smooth=smooth)
        rbf_i = Rbf(pos[ii,0], pos[ii,1], pos[ii,2], np.imag(x[ii]), function="multiquadric", smooth=smooth)
        x2[i] = rbf_r(pos[i,0], pos[i,1], pos[i,2]) + 1j*rbf_i(pos[i,0], pos[i,1], pos[i,2])
    return x2

def smooth_velocity_field(pos: np.ndarray, U: np.ndarray, smooth: float=0.1) -> np.ndarray:
    """RBF-smooth each velocity component over the full point cloud."""
    pos = np.asarray(pos, float)
    U = np.asarray(U)
    if U.shape[1] != 3:
        raise ValueError(f"U must have shape (N,3), got {U.shape}")
    U_sm = np.zeros_like(U, dtype=complex)
    for k in range(3):
        rbf_r = Rbf(pos[:,0], pos[:,1], pos[:,2], np.real(U[:,k]), function="multiquadric", smooth=smooth)
        rbf_i = Rbf(pos[:,0], pos[:,1], pos[:,2], np.imag(U[:,k]), function="multiquadric", smooth=smooth)
        U_sm[:,k] = rbf_r(pos[:,0], pos[:,1], pos[:,2]) + 1j*rbf_i(pos[:,0], pos[:,1], pos[:,2])
    return U_sm

def noiseaware_fill_arrays(
    pos: np.ndarray,
    Pk: np.ndarray,
    Uk: np.ndarray,
    tau: float = 3.5,
    k_nn: int = 12,
    smooth: float = 0.2,
):
    """
    Denoise when you already have arrays (pos, Pk, Uk) from an NPZ.
    Returns a dict with p_filled, u_filled, flags.
    """
    # ensure shapes/dtypes
    pos = np.asarray(pos, float)
    Pk  = np.asarray(Pk)
    Uk  = np.asarray(Uk)
    if Pk.ndim != 1 or Uk.ndim != 2 or Uk.shape[1] != 3 or Pk.shape[0] != pos.shape[0] or Uk.shape[0] != pos.shape[0]:
        raise ValueError(f"Incompatible shapes: pos={pos.shape}, Pk={Pk.shape}, Uk={Uk.shape}")

    nbr_idx = build_knn(pos, k=k_nn)

    f_p  = robust_flags(np.abs(Pk),      nbr_idx, tau=tau)
    f_ux = robust_flags(np.abs(Uk[:,0]), nbr_idx, tau=tau)
    f_uy = robust_flags(np.abs(Uk[:,1]), nbr_idx, tau=tau)
    f_uz = robust_flags(np.abs(Uk[:,2]), nbr_idx, tau=tau)

    P_fill  = fill_channel(pos, Pk,       noisy_mask=f_p,  smooth=smooth)
    Ux_fill = fill_channel(pos, Uk[:,0],  noisy_mask=f_ux, smooth=smooth)
    Uy_fill = fill_channel(pos, Uk[:,1],  noisy_mask=f_uy, smooth=smooth)
    Uz_fill = fill_channel(pos, Uk[:,2],  noisy_mask=f_uz, smooth=smooth)
    U_fill  = np.vstack([Ux_fill, Uy_fill, Uz_fill]).T

    flags = dict(p=f_p, ux=f_ux, uy=f_uy, uz=f_uz)
    return dict(p_filled=P_fill, u_filled=U_fill, flags=flags)


def noiseaware_fill(f_target: float, tau: float=3.5, k_nn: int=12, smooth: float=0.2, tag: str="noiseaware") -> Path:
    S = load_slice_complex(f_target)
    pos, Pk, Uk = S.pos, S.Pk, S.Uk

    nbr_idx = build_knn(pos, k=k_nn)

    f_p  = robust_flags(np.abs(Pk),            nbr_idx, tau=tau)
    f_ux = robust_flags(np.abs(Uk[:,0]),       nbr_idx, tau=tau)
    f_uy = robust_flags(np.abs(Uk[:,1]),       nbr_idx, tau=tau)
    f_uz = robust_flags(np.abs(Uk[:,2]),       nbr_idx, tau=tau)

    P_fill  = fill_channel(pos, Pk, noisy_mask=f_p,  smooth=smooth)
    Ux_fill = fill_channel(pos, Uk[:,0], noisy_mask=f_ux, smooth=smooth)
    Uy_fill = fill_channel(pos, Uk[:,1], noisy_mask=f_uy, smooth=smooth)
    Uz_fill = fill_channel(pos, Uk[:,2], noisy_mask=f_uz, smooth=smooth)
    U_fill  = np.vstack([Ux_fill, Uy_fill, Uz_fill]).T

    out = EXPORTS / f"filled_arrays_{tag}_{int(round(S.f0))}Hz.npz"
    flags = dict(p=f_p, ux=f_ux, uy=f_uy, uz=f_uz)
    np.savez(
        out,
        p=P_fill, u=U_fill,
        p_raw=Pk, u_raw=Uk,
        flags=flags,
        pos=pos,
        meta=dict(f0=S.f0, tau=tau, k_nn=k_nn, smooth=smooth, tag=tag)
    )
    return out
