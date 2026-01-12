#!/usr/bin/env python3
"""
Clean 3D isosurface plot from scattered near-field data.
Uses RBF interpolation onto a grid, then marching cubes.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from scipy.interpolate import Rbf
    _HAVE_RBF = True
except Exception:
    _HAVE_RBF = False
    Rbf = None

try:
    from skimage.measure import marching_cubes
    _HAVE_MC = True
except Exception:
    _HAVE_MC = False
    marching_cubes = None


def main():
    ap = argparse.ArgumentParser(description="3D isosurface plot from NPZ (cleaner view).")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--metric", type=str, default="u_mag", choices=["u_mag", "p"])
    ap.add_argument("--grid", type=int, default=40)
    ap.add_argument("--smooth", type=float, default=0.2)
    ap.add_argument("--levels", type=float, nargs="+", default=[85, 95],
                    help="Isosurface percentiles (e.g., 85 95).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not _HAVE_RBF or not _HAVE_MC:
        raise SystemExit("Requires SciPy Rbf and scikit-image (marching_cubes).")

    Z = np.load(args.npz, allow_pickle=True)
    pos = Z["pos"]; P = Z["p"]; U = Z["u"]

    if args.metric == "u_mag":
        val = np.sqrt((np.abs(U) ** 2).sum(axis=1))
        label = "|U|"
    else:
        val = np.abs(P).ravel()
        label = "|P|"

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    grid_n = args.grid
    xi = np.linspace(x.min(), x.max(), grid_n)
    yi = np.linspace(y.min(), y.max(), grid_n)
    zi = np.linspace(z.min(), z.max(), grid_n)
    X, Y, Zg = np.meshgrid(xi, yi, zi, indexing="xy")

    rbf = Rbf(x, y, z, val, function="multiquadric", smooth=args.smooth)
    V = rbf(X.ravel(), Y.ravel(), Zg.ravel()).reshape(X.shape)

    fig = plt.figure(figsize=(7, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, pctl in enumerate(args.levels):
        level = np.percentile(val, pctl)
        verts, faces, _, _ = marching_cubes(V, level=level, spacing=(xi[1]-xi[0], yi[1]-yi[0], zi[1]-zi[0]))
        color = to_rgba(colors[i % len(colors)], alpha=0.35)
        ax.plot_trisurf(verts[:, 0] + xi.min(),
                        verts[:, 1] + yi.min(),
                        faces,
                        verts[:, 2] + zi.min(),
                        color=color, linewidth=0.2, edgecolor="k")

    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(f"Isosurfaces of {label}")
    ax.view_init(elev=20, azim=45)
    fig.tight_layout()
    out = args.out or args.npz.with_name(args.npz.stem + f"_{args.metric}_isosurface.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
