#!/usr/bin/env python3
# Apply RBF smoothing to velocity components in a filled_arrays_* NPZ.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np
import os
from mf_pipeline.mf_config import EXPORTS

try:
    from scipy.interpolate import Rbf
    _HAVE_RBF = True
except Exception:
    _HAVE_RBF = False
    Rbf = None

OUTDIR = Path(os.environ.get("MF_OUTDIR", EXPORTS)).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Smooth velocity field in a filled_arrays_* NPZ using RBF.")
    ap.add_argument("--npz", type=Path, required=True, help="Input filled_arrays_* NPZ.")
    ap.add_argument("--smooth", type=float, default=0.1, help="RBF smooth parameter.")
    ap.add_argument("--out", type=Path, default=None, help="Output NPZ path.")
    args = ap.parse_args()

    if not _HAVE_RBF:
        raise SystemExit("SciPy Rbf not available; install SciPy to run velocity smoothing.")

    Z = np.load(args.npz, allow_pickle=True)
    pos = Z["pos"]
    P = Z["p"]
    U = Z["u"]
    meta = Z["meta"].item() if "meta" in Z else {}

    # Smooth each velocity component
    U_sm = np.zeros_like(U, dtype=complex)
    for k in range(3):
        rbf_r = Rbf(pos[:,0], pos[:,1], pos[:,2], np.real(U[:,k]), function="multiquadric", smooth=args.smooth)
        rbf_i = Rbf(pos[:,0], pos[:,1], pos[:,2], np.imag(U[:,k]), function="multiquadric", smooth=args.smooth)
        U_sm[:,k] = rbf_r(pos[:,0], pos[:,1], pos[:,2]) + 1j*rbf_i(pos[:,0], pos[:,1], pos[:,2])

    out = args.out or OUTDIR / f"{args.npz.stem}_velsmooth.npz"
    meta2 = dict(meta)
    meta2["vel_smooth"] = args.smooth

    np.savez(out, p=P, u=U_sm, p_raw=Z["p_raw"] if "p_raw" in Z else P,
             u_raw=U, flags=Z["flags"] if "flags" in Z else {},
             pos=pos, meta=meta2)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
