#!/usr/bin/env python3
# mf_pipeline/run_fill.py

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np
import os
from mf_pipeline.mf_fill import noiseaware_fill, noiseaware_fill_arrays
from mf_pipeline.mf_config import EXPORTS

OUTDIR = Path(os.environ.get("MF_OUTDIR", EXPORTS)).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)

def _load_from_npz(npz_path: Path):
    Z = np.load(npz_path, allow_pickle=True)
    pos  = Z["pos"]
    P    = Z["p_raw"] if "p_raw" in Z.files else Z["p"]
    U    = Z["u_raw"] if "u_raw" in Z.files else Z["u"]
    meta = Z["meta"].item() if "meta" in Z.files else {}
    return pos, P, U, meta

def main():
    ap = argparse.ArgumentParser(description="Noise-aware fill for Microflown-like point clouds.")
    ap.add_argument("--f0", type=float, help="Frequency (Hz). Required if not using --npz-in.")
    ap.add_argument("--npz-in", type=Path, default=None, help="Load pos/p/u from this NPZ instead of H5.")
    ap.add_argument("--tau", type=float, default=3.5)
    ap.add_argument("--k-nn", type=int, default=12)
    ap.add_argument("--smooth", type=float, default=0.2)
    ap.add_argument("--tag", type=str, default="noiseaware")
    args = ap.parse_args()

    if args.npz_in is not None:
        pos, P, U, meta = _load_from_npz(args.npz_in)
        if args.f0 is None and "f0" not in meta:
            raise SystemExit("--f0 is required if NPZ lacks meta['f0']")
        f0 = float(meta.get("f0", args.f0))
        outd = noiseaware_fill_arrays(pos, P, U, tau=args.tau, k_nn=args.k_nn, smooth=args.smooth)
        out_path = OUTDIR / f"filled_arrays_{args.tag}_{int(round(f0))}Hz.npz"
        np.savez(out_path,
                 p=outd["p_filled"], u=outd["u_filled"],
                 p_raw=P, u_raw=U,
                 flags=outd["flags"], pos=pos,
                 meta=dict(f0=f0, tau=args.tau, k_nn=args.k_nn, smooth=args.smooth, tag=args.tag, source="npz"))
        print(f"Saved: {out_path}")
        return

    if args.f0 is None:
        raise SystemExit("--f0 required when not using --npz-in")
    out_path = noiseaware_fill(args.f0, tau=args.tau, k_nn=args.k_nn, smooth=args.smooth, tag=args.tag)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
