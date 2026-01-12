#!/usr/bin/env python3
"""
Propagate a source CSM through a transfer matrix G to get far-field CSM, SPL,
and optional intensity if Green's gradients are provided.
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
from mf_pipeline.mf_csm import propagate_csm, spl_from_csm, coherence_matrix, intensity_from_gradients

OUTDIR = Path(os.environ.get("MF_OUTDIR", EXPORTS)).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="CSM propagation: Spp = G Sqq G^H (and optional intensity).")
    ap.add_argument("--G-npz", type=Path, required=True,
                    help="NPZ containing G (and optionally dGdx,dGdy,dGdz). Keys: G[, dGdx, dGdy, dGdz].")
    ap.add_argument("--csm", type=Path, required=True, help="NPZ containing Sqq (source CSM). Key: Sqq.")
    ap.add_argument("--f0", type=float, required=True, help="Frequency (Hz) for omega/rho calculations.")
    ap.add_argument("--rho0", type=float, default=1.2041, help="Air density (kg/m^3).")
    ap.add_argument("--pref", type=float, default=20e-6, help="Reference pressure for SPL.")
    ap.add_argument("--tag", type=str, default="csm")
    ap.add_argument("--out-npz", type=Path, default=None, help="Output NPZ path for Spp/SPL.")
    args = ap.parse_args()

    Gd = np.load(args.G_npz, allow_pickle=True)
    if "G" not in Gd:
        raise SystemExit("G_npz must contain key 'G'.")
    G = Gd["G"]
    have_grad = all(k in Gd for k in ("dGdx", "dGdy", "dGdz"))

    Sqq = np.load(args.csm, allow_pickle=True)["Sqq"]
    Spp = propagate_csm(G, Sqq)
    spl_db = spl_from_csm(Spp, pref=args.pref)
    coh = coherence_matrix(Spp)

    base = OUTDIR / f"Spp_{args.tag}_{int(round(args.f0))}Hz"
    out_npz = args.out_npz or (base.with_suffix(".npz"))
    np.savez(out_npz, Spp=Spp, SPL_dB=spl_db, coh=coh,
             meta=dict(f0=args.f0, rho0=args.rho0, pref=args.pref, tag=args.tag))
    print(f"[save] {out_npz}")

    # SPL plot (receiver index)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(np.arange(len(spl_db)), spl_db, marker="o", linewidth=1)
    ax.set_xlabel("Receiver index")
    ax.set_ylabel("SPL [dB]")
    ax.set_title(f"SPL @ {args.f0:.0f} Hz")
    fig.tight_layout()
    spl_png = base.with_name(base.name + "_spl.png")
    fig.savefig(spl_png, bbox_inches="tight", dpi=160)
    print(f"[save] {spl_png}")

    # Optional intensity if gradients provided
    if have_grad:
        omega = 2.0 * np.pi * args.f0
        dG = (Gd["dGdx"], Gd["dGdy"], Gd["dGdz"])
        Ix, Iy, Iz = intensity_from_gradients(G, dG, Sqq, omega=omega, rho0=args.rho0)
        intens_csv = base.with_name(base.name + "_intensity.csv")
        np.savetxt(intens_csv, np.column_stack([Ix, Iy, Iz]), delimiter=",", header="Ix,Iy,Iz", comments="")
        print(f"[save] {intens_csv}")


if __name__ == "__main__":
    main()
