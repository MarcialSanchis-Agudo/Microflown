#!/usr/bin/env python3
# mf_pipeline/run_plot_source.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mf_pipeline.mf_config import EXPORTS

OUTDIR = Path(os.environ.get("MF_OUTDIR", EXPORTS)).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)

def _metric(P, U, which):
    which = which.lower()
    if which == "p":
        return np.abs(P).ravel(), "|P|", "viridis", None
    if which == "p_phase":
        return np.angle(P).ravel(), "phase(P) [rad]", "twilight", (-np.pi, np.pi)
    if which == "u_mag":
        return np.linalg.norm(U, axis=1).ravel(), "|U|", "viridis", None
    if which == "ux":
        return np.abs(U[:,0]).ravel(), "|Ux|", "viridis", None
    if which == "uy":
        return np.abs(U[:,1]).ravel(), "|Uy|", "viridis", None
    if which == "uz":
        return np.abs(U[:,2]).ravel(), "|Uz|", "viridis", None
    raise ValueError("metric not recognized")

def main():
    ap = argparse.ArgumentParser(description="3D view of near-field samples (color by |P| or |U|).")
    ap.add_argument("--npz", type=Path, required=True, help="filled_arrays_<tag>_<Hz>Hz.npz")
    ap.add_argument("--metric", type=str, default="u_mag",
                    choices=["u_mag","p","p_phase","ux","uy","uz"])
    ap.add_argument("--subsample", type=int, default=1, help="plot every k-th sample (vectors)")
    ap.add_argument("--scale", type=float, default=0.02, help="quiver scale for velocity arrows")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    Z = np.load(args.npz, allow_pickle=True)
    pos  = Z["pos"]                          # (N,3)
    P    = Z["p"]                            # complex (N,)
    U    = Z["u"]                            # may be complex (N,3)
    meta = Z["meta"].item() if "meta" in Z else {}
    f0   = float(meta.get("f0", 0.0))

    # if U is complex, use real part for direction (phase at sample)
    Uplot = np.real(U) if np.iscomplexobj(U) else U

    val, label, cmap, fixed_lim = _metric(P, Uplot, args.metric)
    if fixed_lim is None:
        vmin, vmax = np.percentile(val, [5, 99.5])
    else:
        vmin, vmax = fixed_lim

    # 3D scatter
    fig = plt.figure(figsize=(8,7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=val, s=10, cmap=cmap,
                   vmin=vmin, vmax=vmax)
    cb = fig.colorbar(p, ax=ax, shrink=0.65, label=label)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(f"Near-field samples @ {f0:.0f} Hz (color = |{args.metric}|)")

    # velocity quivers (subsampled)
    step = max(1, args.subsample)
    ax.quiver(pos[::step,0], pos[::step,1], pos[::step,2],
              Uplot[::step,0], Uplot[::step,1], Uplot[::step,2],
              length=args.scale, normalize=True, color="white", linewidths=0.5)

    ax.view_init(elev=22, azim=45)
    fig.tight_layout()
    out = args.out or args.npz.with_name(args.npz.stem + f"_{args.metric}_3d.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"[save] {out}")

    # 2D projections (XY, XZ, YZ) with a single right-side colorbar
    fig2, axs = plt.subplots(1,3, figsize=(12,4), dpi=160)
    for ax2, xdat, ydat, xlabel, ylabel, title in [
        (axs[0], pos[:,0], pos[:,1], "x [m]", "y [m]", "XY projection"),
        (axs[1], pos[:,0], pos[:,2], "x [m]", "z [m]", "XZ projection"),
        (axs[2], pos[:,1], pos[:,2], "y [m]", "z [m]", "YZ projection"),
    ]:
        ax2.scatter(xdat, ydat, c=val, s=5, cmap=cmap, vmin=vmin, vmax=vmax)
        ax2.set_xlabel(xlabel); ax2.set_ylabel(ylabel); ax2.set_title(title)

    # reserve right margin for colorbar
    fig2.tight_layout(rect=[0,0,0.94,1])
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    cax = fig2.add_axes([0.945, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    fig2.colorbar(sm, cax=cax, label=label)

    out2 = args.npz.with_name(args.npz.stem + f"_{args.metric}_projections.png")
    fig2.savefig(out2, bbox_inches="tight")
    print(f"[save] {out2}")

if __name__ == "__main__":
    main()
