#!/usr/bin/env python3
# mf_pipeline/run_nearfield_slices.py
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

OUTDIR = Path(os.environ.get("MF_OUTDIR", EXPORTS)).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)

try:
    from scipy.interpolate import Rbf
    _HAVE_RBF = True
except Exception:
    _HAVE_RBF = False
    Rbf = None

def main():
    ap = argparse.ArgumentParser(description="RBF grid and orthogonal slices of |P| or |U| (or phase).")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--metric", type=str, default="u_mag", choices=["u_mag","p","p_phase"])
    ap.add_argument("--n", type=int, default=120, help="grid points per axis")
    ap.add_argument("--smooth", type=float, default=0.15)
    ap.add_argument("--clip", type=float, nargs=2, default=[5.0, 99.5],
                    help="percentiles for color scaling (low high)")
    ap.add_argument("--db", action="store_true", help="show color in dB (relative to max)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    Z = np.load(args.npz, allow_pickle=True)
    pos = Z["pos"]; P = Z["p"]; U = Z["u"]
    meta = Z["meta"].item() if "meta" in Z else {}
    f0 = float(meta.get("f0",0))

    # --- Metric selection ---
    if args.metric == "u_mag":
        y = np.sqrt((np.abs(U)**2).sum(axis=1))
        label = r"$|\mathbf{U}|$"
        is_phase = False
    elif args.metric == "p":
        y = np.abs(P).ravel()
        label = r"$|P|$"
        is_phase = False
    else:  # p_phase
        y = np.angle(P).ravel()
        label = r"$\angle P$ [rad]"
        is_phase = True

    # quick sanity print
    print(f"[metric] {args.metric}: min={y.min():.3e}, median={np.median(y):.3e}, max={y.max():.3e}")

    x, yv, z = pos[:,0], pos[:,1], pos[:,2]

    # NumPy 2.0-safe ptp
    pad = 0.05 * np.array([np.ptp(x), np.ptp(yv), np.ptp(z)])
    xmin,xmax = x.min()-pad[0], x.max()+pad[0]
    ymin,ymax = yv.min()-pad[1], yv.max()+pad[1]
    zmin,zmax = z.min()-pad[2], z.max()+pad[2]

    xi = np.linspace(xmin, xmax, args.n)
    yi = np.linspace(ymin, ymax, args.n)
    zi = np.linspace(zmin, zmax, args.n)

    if not _HAVE_RBF:
        raise RuntimeError("SciPy Rbf not available; install SciPy or ask me for an IDW fallback.")

    # Interpolate magnitude/phase (diagnostics only)
    rbf = Rbf(x, yv, z, y, function="multiquadric", smooth=args.smooth)

    # planes nearest to 0
    z0 = zi[np.argmin(np.abs(zi))]
    y0 = yi[np.argmin(np.abs(yi))]
    x0 = xi[np.argmin(np.abs(xi))]

    X, Y = np.meshgrid(xi, yi, indexing="xy")
    ZX = rbf(X.ravel(), Y.ravel(), np.full_like(X, z0).ravel()).reshape(X.shape)

    X2, Z2 = np.meshgrid(xi, zi, indexing="xy")
    YX = rbf(X2.ravel(), np.full_like(X2, y0).ravel(), Z2.ravel()).reshape(X2.shape)

    Y3, Z3 = np.meshgrid(yi, zi, indexing="xy")
    XX = rbf(np.full_like(Y3, x0).ravel(), Y3.ravel(), Z3.ravel()).reshape(Y3.shape)

    # scale
    if is_phase:
        ZXp, YXp, XXp = ZX, YX, XX
        vmin, vmax = -np.pi, np.pi
        cmap = "twilight"
        cbar_label = label
    elif args.db:
        def to_db(A):
            m = 20*np.log10(np.maximum(A, 1e-12))
            return m - np.nanmax(m)  # relative to local max
        ZXp, YXp, XXp = to_db(ZX), to_db(YX), to_db(XX)
        vmin, vmax = -40, 0  # show 40 dB dynamic range
        cmap = "viridis"
        cbar_label = f"{label} [dB rel.]"
    else:
        vlow, vhigh = np.percentile(y, args.clip)
        ZXp, YXp, XXp = ZX, YX, XX
        vmin, vmax = vlow, vhigh
        cmap = "viridis"
        cbar_label = label

    fig, axs = plt.subplots(1,3, figsize=(13,4), dpi=160)

    im0 = axs[0].imshow(ZXp.T, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="equal",
                        vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0].set_title(f"Slice z≈{z0:.3f} m (XY)")
    axs[0].set_xlabel("x [m]"); axs[0].set_ylabel("y [m]")

    im1 = axs[1].imshow(YXp.T, origin="lower", extent=[xmin,xmax,zmin,zmax], aspect="equal",
                        vmin=vmin, vmax=vmax, cmap=cmap)
    axs[1].set_title(f"Slice y≈{y0:.3f} m (XZ)")
    axs[1].set_xlabel("x [m]"); axs[1].set_ylabel("z [m]")

    im2 = axs[2].imshow(XXp.T, origin="lower", extent=[ymin,ymax,zmin,zmax], aspect="equal",
                        vmin=vmin, vmax=vmax, cmap=cmap)
    axs[2].set_title(f"Slice x≈{x0:.3f} m (YZ)")
    axs[2].set_xlabel("y [m]"); axs[2].set_ylabel("z [m]")

    # right-side colorbar
    fig.tight_layout(rect=[0,0,0.94,1])
    cax = fig.add_axes([0.945, 0.15, 0.015, 0.7])
    fig.colorbar(im2, cax=cax, label=cbar_label)

    out = args.out or args.npz.with_name(args.npz.stem + f"_{args.metric}_slices.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"[save] {out}")

if __name__ == "__main__":
    main()
