# mf_pipeline/run_paraview_h5_to_npz.py
#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np
import os
from mf_pipeline.mf_config import EXPORTS
from mf_pipeline.mf_loader_paraview import load_paraview_h5

OUTDIR = Path(os.environ.get("MF_OUTDIR", EXPORTS)).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Convert ParaView HDF5 -> pipeline NPZ (pos,P,U at f0).")
    ap.add_argument("--h5", type=Path, required=True, help="ParaView .h5 file (low/high).")
    ap.add_argument("--f0", type=float, required=True, help="Frequency in Hz.")
    ap.add_argument("--tag", type=str, required=True, help="Short tag for outputs (e.g. low1024_50).")
    args = ap.parse_args()

    pos, P, U = load_paraview_h5(args.h5, args.f0)
    N = pos.shape[0]
    flags = dict(p=np.zeros(N,bool), ux=np.zeros(N,bool), uy=np.zeros(N,bool), uz=np.zeros(N,bool))
    out = OUTDIR / f"filled_arrays_{args.tag}_{int(round(args.f0))}Hz.npz"
    np.savez(out, p=P, u=U, p_raw=P.copy(), u_raw=U.copy(),
             flags=flags, pos=pos, meta=dict(f0=float(args.f0), tag=args.tag, source="paraview_h5"))
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
