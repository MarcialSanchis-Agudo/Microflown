#!/usr/bin/env python3
"""
Simple Tikhonov inverse holography:
  q = (G^H G + lam I)^{-1} G^H p
Inputs: G_npz (with G), p vector (from NPZ)
Saves q and a back-projected p_hat for comparison.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np
import os
from mf_pipeline.mf_config import EXPORTS

OUTDIR = Path(os.environ.get("MF_OUTDIR", EXPORTS)).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Tikhonov inverse holography: solve for q from p and G.")
    ap.add_argument("--G-npz", type=Path, required=True, help="NPZ containing G (shape MxN). Key: G")
    ap.add_argument("--p-npz", type=Path, required=True, help="NPZ containing p vector. Key: p")
    ap.add_argument("--lam", type=float, default=1e-3, help="Tikhonov regularization lambda.")
    ap.add_argument("--tag", type=str, default="inv")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    G = np.load(args.G_npz, allow_pickle=True)["G"]
    p = np.load(args.p_npz, allow_pickle=True)["p"]
    M, N = G.shape
    if p.shape[0] != M:
        raise SystemExit(f"p length {p.shape[0]} != G rows {M}")

    A = G.conj().T @ G
    reg = args.lam * np.eye(N, dtype=complex)
    rhs = G.conj().T @ p
    q = np.linalg.solve(A + reg, rhs)
    p_hat = G @ q

    out_npz = args.out or OUTDIR / f"tikhonov_q_{args.tag}.npz"
    np.savez(out_npz, q=q, p_hat=p_hat, meta=dict(lam=args.lam, source_p=str(args.p_npz), source_G=str(args.G_npz)))
    print(f"[save] {out_npz}")


if __name__ == "__main__":
    main()
