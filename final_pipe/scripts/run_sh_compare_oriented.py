#!/usr/bin/env python3
"""
Compare SH extrapolation vs measured for an oriented NPZ.
Outputs:
  - error vs distance plot
  - measured vs predicted |P| slices (XY + XZ) and error maps
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mf_pipeline.mf_directivity import angles_from_pos, fit_complex_SH_nearfield, SHFitOptions, eval_complex_SH_points


def kmeans3(x, seed=0, iters=60):
    rng = np.random.default_rng(seed)
    cent = x[rng.choice(len(x), 3, replace=False)]
    for _ in range(iters):
        dist = ((x[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        lab = dist.argmin(axis=1)
        new_cent = np.array([x[lab == i].mean(axis=0) for i in range(3)])
        if np.allclose(new_cent, cent, atol=1e-5):
            break
        cent = new_cent
    return lab, cent


def voxel_average_scalar(pos, val, grid):
    gx, gy, gz = grid
    ix = np.clip(np.digitize(pos[:, 0], gx) - 1, 0, len(gx) - 2)
    iy = np.clip(np.digitize(pos[:, 1], gy) - 1, 0, len(gy) - 2)
    iz = np.clip(np.digitize(pos[:, 2], gz) - 1, 0, len(gz) - 2)
    out = np.full((len(gx)-1, len(gy)-1, len(gz)-1), np.nan)
    count = np.zeros(out.shape, dtype=int)
    for a, b, c, v in zip(ix, iy, iz, val):
        if np.isnan(out[a, b, c]):
            out[a, b, c] = v
        else:
            out[a, b, c] += v
        count[a, b, c] += 1
    out = out / np.maximum(count, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description="SH compare on oriented NPZ.")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--f0", type=float, required=True)
    ap.add_argument("--c", type=float, default=343.0)
    ap.add_argument("--lmax", type=int, default=8)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--grid", type=int, default=26)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="sh_compare")
    args = ap.parse_args()

    Z = np.load(args.npz, allow_pickle=True)
    pos = Z["pos"]
    P = Z["P"]
    src = np.array([0.0, 0.0, 0.0])

    labels, cent = kmeans3(pos)
    train_id = int(np.argmin(np.linalg.norm(cent - src[None, :], axis=1)))
    train_mask = labels == train_id
    test_mask = ~train_mask

    k = 2.0 * np.pi * args.f0 / args.c
    r, th, ph = angles_from_pos(pos[train_mask], src)
    a = fit_complex_SH_nearfield(P[train_mask], r, th, ph, k, SHFitOptions(lmax=args.lmax, lam=args.lam))

    r_all, th_all, ph_all = angles_from_pos(pos, src)
    P_pred = eval_complex_SH_points(a, r_all, th_all, ph_all, k, args.lmax)

    # error vs distance
    train_pts = pos[train_mask]
    test_pts = pos[test_mask]
    d2 = ((test_pts[:, None, :] - train_pts[None, :, :])**2).sum(axis=2)
    dmin = np.sqrt(d2.min(axis=1))
    rel_err = np.abs(P_pred[test_mask] - P[test_mask]) / np.maximum(np.abs(P[test_mask]), 1e-12)

    # error vs distance with bin medians
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.2), dpi=140)
    ax.scatter(dmin, rel_err, s=10, alpha=0.5, color="#444444")
    bins = np.linspace(dmin.min(), dmin.max(), 8)
    bin_cent = 0.5 * (bins[:-1] + bins[1:])
    med = []
    for i in range(len(bins)-1):
        m = (dmin >= bins[i]) & (dmin < bins[i+1])
        med.append(np.median(rel_err[m]) if np.any(m) else np.nan)
    ax.plot(bin_cent, med, "-o", color="#d62728", lw=2, ms=4, label="median")
    ax.set_xlabel("distance to training [m]")
    ax.set_ylabel("rel |ΔP| / |P|")
    ax.set_title(f"SH error vs distance ({args.f0:.0f} Hz)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.outdir / f"{args.tag}_err_vs_dist.png", bbox_inches="tight")
    plt.close(fig)

    # slice comparisons
    xmin, ymin, zmin = pos.min(0)
    xmax, ymax, zmax = pos.max(0)
    gx = np.linspace(xmin, xmax, args.grid)
    gy = np.linspace(ymin, ymax, args.grid)
    gz = np.linspace(zmin, zmax, args.grid)
    xc = 0.5 * (gx[:-1] + gx[1:])
    yc = 0.5 * (gy[:-1] + gy[1:])
    zc = 0.5 * (gz[:-1] + gz[1:])
    iz = len(zc)//2
    iy = len(yc)//2
    meas = voxel_average_scalar(pos, np.abs(P), (gx, gy, gz))
    pred = voxel_average_scalar(pos, np.abs(P_pred), (gx, gy, gz))
    err = np.abs(pred - meas) / np.maximum(meas, 1e-12)
    vmax = np.nanpercentile(meas, 95)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=140)
    X, Y = np.meshgrid(xc, yc, indexing="xy")
    Zm, Xz = np.meshgrid(zc, xc, indexing="xy")
    axes[0,0].contourf(X, Y, meas[:, :, iz].T, levels=20, cmap="viridis", vmin=0, vmax=vmax)
    axes[0,0].set_title("Measured |P| (XY)")
    axes[0,1].contourf(X, Y, pred[:, :, iz].T, levels=20, cmap="viridis", vmin=0, vmax=vmax)
    axes[0,1].set_title("Predicted |P| (XY)")
    axes[0,2].contourf(X, Y, err[:, :, iz].T, levels=20, cmap="magma")
    axes[0,2].set_title("Rel error (XY)")

    axes[1,0].contourf(Xz, Zm, meas[:, iy, :].T, levels=20, cmap="viridis", vmin=0, vmax=vmax)
    axes[1,0].set_title("Measured |P| (XZ)")
    axes[1,1].contourf(Xz, Zm, pred[:, iy, :].T, levels=20, cmap="viridis", vmin=0, vmax=vmax)
    axes[1,1].set_title("Predicted |P| (XZ)")
    axes[1,2].contourf(Xz, Zm, err[:, iy, :].T, levels=20, cmap="magma")
    axes[1,2].set_title("Rel error (XZ)")
    for ax in axes.ravel():
        ax.set_xlabel("x [m]"); ax.set_ylabel("y/z [m]")
    fig.tight_layout()
    fig.savefig(args.outdir / f"{args.tag}_slices.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
