# mf_pipeline/mf_vector.py
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class VectorPropagator:
    p: np.ndarray          # complex pressure (N,)
    u: np.ndarray          # complex particle velocity (N,3)
    pos: np.ndarray        # positions (N,3)
    f0: float
    rho0: float = 1.2041

    def complex_intensity(self) -> np.ndarray:
        """
        Complex acoustic intensity vector (W/m^2), per sample.
        J = 0.5 * p * conj(u) (component-wise).
        Returns shape (N,3).
        """
        return 0.5 * self.p[:, None] * np.conj(self.u)

    def active_intensity(self) -> np.ndarray:
        return np.real(self.complex_intensity())

    def reactive_intensity(self) -> np.ndarray:
        return np.imag(self.complex_intensity())

    def radial_normals(self) -> np.ndarray:
        """Unit vectors from origin to each sample (fallback when no surface normal is provided)."""
        r = np.linalg.norm(self.pos, axis=1, keepdims=True) + 1e-12
        return self.pos / r

    def impedance(self, normals: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Specific acoustic impedance Z = p / (u · n).
        normals: (N,3) unit normals. If None, uses radial normals.
        Returns complex array (N,).
        """
        n = normals if normals is not None else self.radial_normals()
        un = np.sum(self.u * np.conj(n), axis=1)  # project onto normal
        return self.p / (un + 1e-12)

    def q_local(self, normals: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Local radiation efficiency metric Q_local = |I| / (|Q| + eps).
        Returns shape (N,).
        """
        I = np.linalg.norm(self.active_intensity(), axis=1)
        Q = np.linalg.norm(self.reactive_intensity(), axis=1)
        return I / (Q + 1e-12)


def build_csm(X: np.ndarray) -> np.ndarray:
    """
    Build cross-spectral matrix from multichannel complex spectra.
      X: (M channels, nfft bins) or (M, T) single-frequency snapshots.
    Returns Sxx: (M,M) complex
    """
    return X @ np.conj(X.T) / X.shape[1] if X.ndim == 2 else np.outer(X, np.conj(X))
