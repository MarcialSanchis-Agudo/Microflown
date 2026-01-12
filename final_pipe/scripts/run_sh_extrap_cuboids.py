#!/usr/bin/env python3
"""
Physics-based extrapolation across cuboids using spherical harmonics (pressure only).
- Cluster points into 3 cuboids; pick train cuboid closest to source (default 0,0,0).
- Fit SH coefficients on training cuboid at f0.
- Predict pressure at held-out cuboids; report magnitude/phase errors vs distance.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np
import h5py
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


def main():
    ap = argparse.ArgumentParser(description="SH near->other cuboid extrapolation (pressure only).")
    ap.add_argument("--psd", type=Path, required=True)
    ap.add_argument("--group", type=str, default="Process1")
    ap.add_argument("--f0", type=float, default=1008.0)
    ap.add_argument("--c", type=float, default=343.0)
    ap.add_argument("--lmax", type=int, default=8)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--source", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Source center for SH expansion.")
    ap.add_argument("--train-cluster", type=int, default=None, help="Force train cluster (0/1/2). Default: nearest to source.")
    ap.add_argument("--outdir", type=Path, default=Path("figs/examples/martial_measurements"))
    ap.add_argument("--tag", type=str, default="sh_extrap")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.psd, "r") as f:
        g = f[args.group]
        freq = np.array(g["FreqVar"]).ravel()
        pos = np.array(g["datPostition"]).T
        Pk = np.array(g["TFpref1_i"])[:, 0, :] + 1j * np.array(g["TFpref1_j"])[:, 0, :]

    fi = np.argmin(np.abs(freq - args.f0))
    fsel = freq[fi]
    p = Pk[:, fi]

    labels, cent = kmeans3(pos)
    src = np.asarray(args.source, dtype=float)
    if args.train_cluster is None:
        train_id = int(np.argmin(np.linalg.norm(cent - src[None, :], axis=1)))
    else:
        train_id = int(args.train_cluster)

    train_mask = labels == train_id
    test_mask = ~train_mask

    k = 2.0 * np.pi * fsel / args.c
    r_train, th_train, ph_train = angles_from_pos(pos[train_mask], src)
    a = fit_complex_SH_nearfield(p[train_mask], r_train, th_train, ph_train, k,
                                 SHFitOptions(lmax=args.lmax, lam=args.lam))

    # predict on all points
    r_all, th_all, ph_all = angles_from_pos(pos, src)
    p_pred = eval_complex_SH_points(a, r_all, th_all, ph_all, k, args.lmax)

    # errors on held-out
    p_true = p[test_mask]
    p_hat = p_pred[test_mask]
    rel_err = np.abs(p_hat - p_true) / np.maximum(np.abs(p_true), 1e-12)
    phase_err = np.angle(p_hat) - np.angle(p_true)
    phase_err = (phase_err + np.pi) % (2 * np.pi) - np.pi

    # distance to training set
    train_pts = pos[train_mask]
    test_pts = pos[test_mask]
    d2 = ((test_pts[:, None, :] - train_pts[None, :, :]) ** 2).sum(axis=2)
    dmin = np.sqrt(d2.min(axis=1))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=140)
    axes[0].scatter(dmin, rel_err, s=10, alpha=0.7, c=np.angle(p_hat), cmap="twilight")
    axes[0].set_xlabel("distance to training [m]"); axes[0].set_ylabel("rel |Δp|/|p|")
    axes[0].set_title(f"SH extrap error @ {fsel:.1f} Hz")
    axes[1].hist(rel_err, bins=80, color="#444", alpha=0.85)
    axes[1].set_xlabel("rel |Δp|/|p|"); axes[1].set_ylabel("count")
    axes[1].set_title("Held-out magnitude error")
    axes[2].hist(phase_err, bins=80, color="#777", alpha=0.85)
    axes[2].set_xlabel("phase error [rad]"); axes[2].set_ylabel("count")
    axes[2].set_title("Held-out phase error")
    fig.tight_layout()
    out_png = args.outdir / f"{args.tag}_train{train_id}_{int(fsel)}Hz.png"
    fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)

    np.savez(args.outdir / f"{args.tag}_train{train_id}_{int(fsel)}Hz.npz",
             freq=fsel, pos=pos, labels=labels, train_id=train_id,
             p_true=p_true, p_pred=p_hat, rel_err=rel_err, phase_err=phase_err, dmin=dmin,
             source=src, lmax=args.lmax, lam=args.lam)
    print(f"[save] {out_png}")
    print(f"[info] mean rel err={rel_err.mean():.3f}, median={np.median(rel_err):.3f}")


if __name__ == "__main__":
    main()
