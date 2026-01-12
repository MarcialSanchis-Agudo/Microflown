# mf_pipeline/mf_csm.py
import numpy as np
from typing import Tuple, Optional


def propagate_csm(G: np.ndarray, Sqq: np.ndarray) -> np.ndarray:
    """
    Far-field CSM from source CSM via transfer matrix G.
      G   : (M receivers, N sources) complex
      Sqq : (N, N) source CSM
    Returns:
      Spp : (M, M) far-field CSM
    """
    return G @ Sqq @ G.conj().T


def spl_from_csm(Spp: np.ndarray, pref: float = 20e-6) -> np.ndarray:
    """
    SPL (dB) from CSM diagonal.
      Spp : (M,M) complex (power spectral density)
    """
    diag = np.real(np.diag(Spp))
    return 10.0 * np.log10(np.maximum(diag, 1e-30) / (pref ** 2))


def coherence_matrix(Spp: np.ndarray) -> np.ndarray:
    """
    Magnitude-squared coherence matrix from CSM.
    """
    diag = np.real(np.diag(Spp))
    denom = np.sqrt(np.maximum(diag, 1e-30))
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.abs(Spp) / (denom[:, None] * denom[None, :])
    coh[np.isnan(coh)] = 0.0
    coh[np.isinf(coh)] = 0.0
    return coh


def intensity_from_gradients(
    G: np.ndarray,
    dG: Tuple[np.ndarray, np.ndarray, np.ndarray],
    Sqq: np.ndarray,
    omega: float,
    rho0: float = 1.2041,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute active intensity components from pressure/velocity cross-CSMs using Green's gradients.
      G   : (M,N) pressure transfer matrix
      dG  : tuple of (dGdx, dGdy, dGdz), each (M,N)
      Sqq : (N,N) source CSM
      omega : angular frequency (rad/s)
      rho0  : air density (kg/m^3)
    Returns:
      Ix, Iy, Iz arrays of length M (W/m^2), real parts of 0.5 * E[p v*].
    """
    factor = -1j / (omega * rho0)
    dGdx, dGdy, dGdz = dG
    Spvx = factor * (G @ Sqq @ dGdx.conj().T)
    Spvy = factor * (G @ Sqq @ dGdy.conj().T)
    Spvz = factor * (G @ Sqq @ dGdz.conj().T)
    Ix = 0.5 * np.real(np.diag(Spvx))
    Iy = 0.5 * np.real(np.diag(Spvy))
    Iz = 0.5 * np.real(np.diag(Spvz))
    return Ix, Iy, Iz
