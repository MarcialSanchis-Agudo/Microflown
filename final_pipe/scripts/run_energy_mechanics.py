#!/usr/bin/env python3
"""
Energy-mechanics plots from a filled_arrays_* NPZ:
  1) Reactivity scatter: kr vs |Q|/|I|
  2) Q_local histogram (|I|/|Q|)
  3) Active/Reactive streamlines (XY plane)
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

try:
    from scipy.interpolate import Rbf
    _HAVE_RBF = True
except Exception:
    _HAVE_RBF = False
    Rbf = None

OUTDIR = Path(os.environ.get("MF_OUTDIR", EXPORTS)).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Energy mechanics plots (reactivity scatter + streamlines).")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--c", type=float, default=343.0)
    ap.add_argument("--center", type=float, nargs=3, default=None)
    ap.add_argument("--grid", type=int, default=60, help="Grid resolution for streamlines.")
    ap.add_argument("--max-points", type=int, default=4000, help="Downsample points for scatter.")
    args = ap.parse_args()

    Z = np.load(args.npz, allow_pickle=True)
    p = Z["p"]; u = Z["u"]; pos = Z["pos"]
    meta = Z["meta"].item() if "meta" in Z else {}
    f0 = float(meta.get("f0", 0.0))

    center = np.array(args.center, float) if args.center is not None else np.zeros(3)
    r = np.linalg.norm(pos - center[None, :], axis=1)
    k = 2.0 * np.pi * f0 / args.c
    kr = k * r

    J = 0.5 * p[:, None] * np.conj(u)
    I = np.linalg.norm(np.real(J), axis=1)
    Q = np.linalg.norm(np.imag(J), axis=1)
    ratio = Q / (I + 1e-12)
    q_local = I / (Q + 1e-12)

    # 1) Reactivity scatter
    idx = np.arange(len(kr))
    if len(idx) > args.max_points:
        idx = np.random.default_rng(0).choice(idx, size=args.max_points, replace=False)
    fig, ax = plt.subplots(figsize=(5.5,4), dpi=160)
    ax.scatter(kr[idx], ratio[idx], s=6, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("kr")
    ax.set_ylabel("|Q|/|I|")
    ax.set_title(f"Reactivity scatter @ {f0:.0f} Hz")
    fig.tight_layout()
    out_scatter = OUTDIR / f"{args.npz.stem}_reactivity_scatter.png"
    fig.savefig(out_scatter, bbox_inches="tight")
    print(f"[save] {out_scatter}")

    # 2) Q_local histogram
    fig2, ax2 = plt.subplots(figsize=(5,3.2), dpi=160)
    ax2.hist(q_local, bins=80, log=True)
    ax2.set_xlabel("Q_local = |I|/|Q|")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Q_local histogram @ {f0:.0f} Hz")
    fig2.tight_layout()
    out_hist = OUTDIR / f"{args.npz.stem}_Qlocal_hist.png"
    fig2.savefig(out_hist, bbox_inches="tight")
    print(f"[save] {out_hist}")

    # 3) Streamlines (XY plane near z=0)
    if not _HAVE_RBF:
        print("[skip] SciPy Rbf not available; streamlines skipped.")
        return

    z0 = np.median(pos[:, 2])
    x = pos[:, 0]; y = pos[:, 1]; z = pos[:, 2]
    grid_n = args.grid
    xi = np.linspace(x.min(), x.max(), grid_n)
    yi = np.linspace(y.min(), y.max(), grid_n)
    X, Y = np.meshgrid(xi, yi, indexing="xy")

    # Active
    Ix = np.real(J[:, 0]); Iy = np.real(J[:, 1])
    rbf_Ix = Rbf(x, y, z, Ix, function="multiquadric", smooth=0.2)
    rbf_Iy = Rbf(x, y, z, Iy, function="multiquadric", smooth=0.2)
    Ipx = rbf_Ix(X.ravel(), Y.ravel(), np.full_like(X, z0).ravel()).reshape(X.shape)
    Ipy = rbf_Iy(X.ravel(), Y.ravel(), np.full_like(X, z0).ravel()).reshape(X.shape)

    # Reactive
    Qx = np.imag(J[:, 0]); Qy = np.imag(J[:, 1])
    rbf_Qx = Rbf(x, y, z, Qx, function="multiquadric", smooth=0.2)
    rbf_Qy = Rbf(x, y, z, Qy, function="multiquadric", smooth=0.2)
    Qpx = rbf_Qx(X.ravel(), Y.ravel(), np.full_like(X, z0).ravel()).reshape(X.shape)
    Qpy = rbf_Qy(X.ravel(), Y.ravel(), np.full_like(X, z0).ravel()).reshape(X.shape)

    fig3, axs = plt.subplots(1, 2, figsize=(10,4), dpi=160)
    axs[0].streamplot(xi, yi, Ipx, Ipy, density=1.2, color="C0")
    axs[0].set_title("Active intensity (Re)")
    axs[0].set_xlabel("x [m]"); axs[0].set_ylabel("y [m]")
    axs[0].set_aspect("equal", "box")

    axs[1].streamplot(xi, yi, Qpx, Qpy, density=1.2, color="C1")
    axs[1].set_title("Reactive intensity (Im)")
    axs[1].set_xlabel("x [m]"); axs[1].set_ylabel("y [m]")
    axs[1].set_aspect("equal", "box")

    fig3.tight_layout()
    out_stream = OUTDIR / f"{args.npz.stem}_intensity_streamlines_xy.png"
    fig3.savefig(out_stream, bbox_inches="tight")
    print(f"[save] {out_stream}")


if __name__ == "__main__":
    main()
