#!/usr/bin/env python3
"""
Convert Microflown *.psd (HDF5) to filled_arrays_* NPZ with complex P/U.
Uses Process1 TFpref1_i/j and TFxyzref1_i/j plus datPostition and FreqVar.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np
import h5py
import os
from mf_pipeline.mf_config import EXPORTS

OUTDIR = Path(os.environ.get("MF_OUTDIR", EXPORTS)).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Convert Microflown PSD H5 -> NPZ at f0.")
    ap.add_argument("--psd", type=Path, required=True)
    ap.add_argument("--f0", type=float, required=True)
    ap.add_argument("--group", type=str, default="Process1")
    ap.add_argument("--tag", type=str, default="psd")
    args = ap.parse_args()

    with h5py.File(args.psd, "r") as f:
        g = f[args.group]
        pos = np.array(g["datPostition"]).T  # (3,N) -> (N,3)
        freq = np.array(g["FreqVar"]).ravel()
        k = int(np.argmin(np.abs(freq - args.f0)))
        f0 = float(freq[k])

        # Transfer functions (complex)
        P_re = np.array(g["TFpref1_i"])[:, 0, k]
        P_im = np.array(g["TFpref1_j"])[:, 0, k]
        U_re = np.array(g["TFxyzref1_i"])[:, :, k]
        U_im = np.array(g["TFxyzref1_j"])[:, :, k]
        P = P_re + 1j * P_im
        U = U_re + 1j * U_im

    out = OUTDIR / f"filled_arrays_{args.tag}_{int(round(f0))}Hz.npz"
    np.savez(out, p=P, u=U, p_raw=P.copy(), u_raw=U.copy(),
             pos=pos, meta=dict(f0=f0, tag=args.tag, source="microflown_psd"))
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
