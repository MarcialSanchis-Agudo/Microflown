#!/usr/bin/env python3
"""
Vector-acoustic diagnostics from a filled_arrays_* NPZ:
  - Complex/active/reactive intensity
  - Impedance using radial normals
  - Q_local = |I| / |Q|

Saves a summary NPZ and simple histograms for I/Q/Z/Q_local.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from mf_pipeline.mf_config import EXPORTS
from mf_pipeline.mf_vector import VectorPropagator

OUTDIR = Path(os.environ.get("MF_OUTDIR", EXPORTS)).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)


def save_hist(data, title, xlabel, out_png):
    fig, ax = plt.subplots(figsize=(5,3), dpi=150)
    ax.hist(data, bins=80, log=True, color="C0")
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Vector-acoustic metrics from filled_arrays_* NPZ.")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--rho0", type=float, default=1.2041)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    Z = np.load(args.npz, allow_pickle=True)
    p = Z["p"]
    u = Z["u"]
    pos = Z["pos"]
    meta = Z["meta"].item() if "meta" in Z else {}
    f0 = float(meta.get("f0", 0.0))

    vp = VectorPropagator(p=p, u=u, pos=pos, f0=f0, rho0=args.rho0)
    J = vp.complex_intensity()
    I = np.real(J); Q = np.imag(J)
    Zspec = vp.impedance()
    q_local = vp.q_local()

    out_npz = args.out or OUTDIR / f"vector_metrics_{args.npz.stem}.npz"
    np.savez(out_npz,
             I=I, Q=Q, J=J, Z=Zspec, q_local=q_local,
             meta=dict(f0=f0, rho0=args.rho0, source=str(args.npz)))
    print(f"[save] {out_npz}")

    # Histograms
    save_hist(np.linalg.norm(I, axis=1), f"|I| @ {f0:.0f} Hz", "|I| [W/m^2]", OUTDIR / f"{args.npz.stem}_I_hist.png")
    save_hist(np.linalg.norm(Q, axis=1), f"|Q| @ {f0:.0f} Hz", "|Q| [W/m^2]", OUTDIR / f"{args.npz.stem}_Q_hist.png")
    save_hist(np.abs(Zspec), f"|Z| @ {f0:.0f} Hz", "|Z| [Pa·s/m]", OUTDIR / f"{args.npz.stem}_Z_hist.png")
    save_hist(q_local, f"Q_local @ {f0:.0f} Hz", "Q_local", OUTDIR / f"{args.npz.stem}_Qlocal_hist.png")


if __name__ == "__main__":
    main()
