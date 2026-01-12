import numpy as np, h5py
from pathlib import Path
from typing import Optional, Tuple, List

# Name hints
COORD_KEYS = ("points","point","coords","coordinates","xyz","geometry","nodes","position")
PRESS_KEYS = ("pressure","PRESSURE","p","P","Pressure")
VELO_KEYS  = ("velocity","VELOCITY","U","u","Velocity")
FREQ_KEYS  = ("freq","frequency","FREQ","FREQUENCY","omega","Omega","frequency_vector")

def _score(nm: str, keys) -> int:
    n = nm.lower()
    return sum(k in n for k in keys)

def _list_datasets(f: h5py.File) -> List[Tuple[str, h5py.Dataset]]:
    out = []
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out.append((name, obj))
    f.visititems(lambda n,o: visit(n,o))
    return out

def _coord_candidates(f: h5py.File, N_hint: Optional[int]=None):
    cands = []
    for name, ds in _list_datasets(f):
        if ds.ndim == 2 and np.issubdtype(ds.dtype, np.floating) and (ds.shape[1]==3 or ds.shape[0]==3):
            if (N_hint is None) or (N_hint in ds.shape):
                cands.append((name, ds))
    # prefer coordinate-like names
    cands.sort(key=lambda x: (_score(x[0], COORD_KEYS), -len(x[0])), reverse=True)
    return cands

def _find_coords_h5(f: h5py.File, N_hint: Optional[int]=None) -> np.ndarray:
    cands = _coord_candidates(f, N_hint)
    if not cands:
        raise RuntimeError("No coordinate-like (N,3) or (3,N) dataset found.")
    name, ds = cands[0]
    arr = np.array(ds)
    return arr if arr.shape[1]==3 else arr.T

def _find_freq_vec(f: h5py.File) -> Optional[np.ndarray]:
    best = None; best_sc = -1
    for name, ds in _list_datasets(f):
        if ds.ndim in (1,2):
            sc = _score(name, FREQ_KEYS)
            if sc > best_sc:
                best, best_sc = ds, sc
    if best is None:
        return None
    fv = np.array(best).astype(float)
    return fv.ravel()

def _closest_idx(fvec: np.ndarray, f0: float) -> int:
    return int(np.argmin(np.abs(fvec - float(f0))))

def _find_pressure(f: h5py.File):
    best = None; best_sc = -1
    for name, ds in _list_datasets(f):
        if ds.ndim in (1,2):
            sc = _score(name, PRESS_KEYS)
            if sc > best_sc:
                best, best_sc = (name, ds), sc
    if best is None:
        raise RuntimeError("PRESSURE dataset not found.")
    return best

def _find_velocity(f: h5py.File):
    best = None; best_sc = -1
    for name, ds in _list_datasets(f):
        if ds.ndim in (1,2,3):
            sc = _score(name, VELO_KEYS)
            if sc > best_sc:
                best, best_sc = (name, ds), sc
    return best  # may be None

def load_paraview_h5(h5_path: Path, f0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust loader for ParaView H5 where (per your probe):
      POSITION:           (Npts,3)          -> coords
      PRESSURE:           (Nfreq, Npts)     -> slice nearest f0
      FREQUENCY_VECTOR:   (Nfreq,1) / (Nfreq,) / (1,Nfreq)
      VELOCITY (optional):(Nfreq, Npts, 3) or (Npts, 3, Nfreq) or (3, Npts, Nfreq)
    Returns:
      pos (Npts,3), P (Npts,), U (Npts,3)  [complex128]
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as f:
        # --- 1) Coordinates: prefer dataset whose name contains 'POSITION'
        pos = None
        for name, ds in _list_datasets(f):
            if "position" in name.lower() and ds.ndim==2 and 3 in ds.shape:
                arr = np.array(ds)
                pos = arr if arr.shape[1]==3 else arr.T
                break
        if pos is None:
            # fallback to any (N,3)/(3,N) coord-like dataset
            pos = _find_coords_h5(f)
        pos = pos.astype(float)
        Npts = pos.shape[0]

        # --- 2) Frequency vector (optional but present in your files)
        fvec = _find_freq_vec(f)

        # --- 3) Pressure
        p_name, p_ds = _find_pressure(f)
        Praw = np.array(p_ds)

        if Praw.ndim == 1:
            if Praw.shape[0] != Npts:
                raise RuntimeError(f"Pressure length {Praw.shape[0]} != Npts {Npts}")
            P = Praw.astype(float)
        elif Praw.ndim == 2:
            r0, r1 = Praw.shape
            # two orientations are common: (Nfreq, Npts) or (Npts, Nfreq)
            if r1 == Npts and (fvec is None or r0 != fvec.size):
                # treat as (Nfreq, Npts) but without freq vec -> take first row
                P = Praw[0, :].astype(float)
            elif r0 == Npts and (fvec is None or r1 != fvec.size):
                # treat as (Npts, Nfreq) but without freq vec -> take first column
                P = Praw[:, 0].astype(float)
            elif fvec is not None and r0 == fvec.size and r1 == Npts:
                # (Nfreq, Npts) with freq vec → slice
                idx = _closest_idx(fvec, f0)
                P = Praw[idx, :].astype(float)
            elif fvec is not None and r1 == fvec.size and r0 == Npts:
                # (Npts, Nfreq) with freq vec → slice
                idx = _closest_idx(fvec, f0)
                P = Praw[:, idx].astype(float)
            else:
                raise RuntimeError(f"2D pressure shape {Praw.shape} not aligned with Npts={Npts} (freq len={None if fvec is None else fvec.size})")
        else:
            raise RuntimeError(f"Unsupported pressure rank: {Praw.ndim}")

        # --- 4) Velocity (optional)
        U = np.zeros((Npts, 3), float)
        u_found = _find_velocity(f)
        if u_found is not None:
            u_name, u_ds = u_found
            Uraw = np.array(u_ds)
            if Uraw.ndim == 1:
                if Uraw.shape[0] == Npts:
                    U[:,0] = Uraw.astype(float)
            elif Uraw.ndim == 2:
                if Uraw.shape == (Npts, 3):
                    U = Uraw.astype(float)
                elif Uraw.shape == (3, Npts):
                    U = Uraw.T.astype(float)
                elif fvec is not None:
                    if Uraw.shape[0] == Npts:          # (Npts, Nfreq)
                        idx = _closest_idx(fvec, f0) if Uraw.shape[1]==fvec.size else 0
                        U[:,0] = Uraw[:, idx].astype(float)
                    elif Uraw.shape[1] == Npts:        # (Nfreq, Npts)
                        idx = _closest_idx(fvec, f0) if Uraw.shape[0]==fvec.size else 0
                        U[:,0] = Uraw[idx, :].astype(float)
            elif Uraw.ndim == 3:
                if fvec is not None:
                    idx = _closest_idx(fvec, f0)
                    if Uraw.shape[0] == fvec.size and Uraw.shape[1:] == (Npts,3):      # (Nfreq,Npts,3)
                        U = Uraw[idx, :, :].astype(float)
                    elif Uraw.shape[2] == fvec.size and Uraw.shape[:2] == (Npts,3):     # (Npts,3,Nfreq)
                        U = Uraw[:, :, idx].astype(float)
                    elif Uraw.shape[2] == fvec.size and Uraw.shape[:2] == (3,Npts):     # (3,Npts,Nfreq)
                        U = np.moveaxis(Uraw[:, :, idx], 0, 1).astype(float)

        return pos.astype(float), P.astype(np.complex128), U.astype(np.complex128)



"""
# mf_pipeline/mf_loader_paraview.py
import numpy as np
import meshio
from pathlib import Path

# Try common field names
P_NAMES  = ["p", "pressure", "Pressure", "PRESSURE", "P"]
U_NAMES  = ["U", "u", "velocity", "Velocity", "VELOCITY", "v", "V"]

def _pick(data_dict, names):
    for n in names:
        if n in data_dict:
            return data_dict[n]
    return None

def load_paraview_xdmf(xdmf_path: Path):
#    Returns:
##      pos  : (N,3) float array
#      P    : (N,) complex or float (converted to complex)
#      U    : (N,3) complex or float (optional; None if not found)
#    Notes:
#      - If fields are real, cast to complex (imag=0).
#      - If only magnitude exists, it is used as real part.
    mesh = meshio.read(str(xdmf_path))
    # Geometry: points (N,3)
    pos = np.asarray(mesh.points, dtype=float)

    # Combine all point_data dicts (some XDMFs carry time-sets)
    # mesh.point_data is Dict[str, np.ndarray]
    p_arr = _pick(mesh.point_data, P_NAMES)
    u_arr = _pick(mesh.point_data, U_NAMES)

    if p_arr is None and u_arr is None:
        raise ValueError(f"No pressure/velocity fields found in {xdmf_path.name}. "
                         f"Available: {list(mesh.point_data.keys())}")

    # Normalize shapes
    if p_arr is not None:
        p_arr = np.asarray(p_arr)
        if p_arr.ndim == 2 and p_arr.shape[1] == 1:
            p_arr = p_arr[:, 0]
        P = p_arr.astype(float)
        P = P.astype(np.complex128)  # imag=0
    else:
        P = np.zeros(pos.shape[0], dtype=np.complex128)

    U = None
    if u_arr is not None:
        U = np.asarray(u_arr)
        # If vector stored as (N,1,3) or (N,3,1) -> squeeze to (N,3)
        U = np.squeeze(U)
        if U.ndim == 1:  # bad writer -> scalar; upcast to (N,3) zeros
            U3 = np.zeros((pos.shape[0], 3), dtype=float)
            U3[:, 0] = U
            U = U3
        if U.shape[1] != 3:
            raise ValueError(f"Velocity field must have 3 components, got shape {U.shape}")
        U = U.astype(float).astype(np.complex128)  # imag=0
    return pos, P, U
"""