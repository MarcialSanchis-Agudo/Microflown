#!/usr/bin/env python3
"""
Extrapolate acoustic field from one cuboid (training cluster) to the others.
- Clusters the measured points into three cuboids.
- Fits an RBF interpolant on the training cuboid (complex pressure and velocity).
- Predicts on the held-out cuboids, reports errors vs distance to training.
- Optional room mesh overlay for context.
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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


def load_obj(mesh_path, face_stride=1, max_faces=None):
    verts, faces = [], []
    with open(mesh_path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                _, xs, ys, zs = line.strip().split()[:4]
                verts.append([float(xs), float(ys), float(zs)])
            elif line.startswith("f "):
                toks = line.strip().split()[1:]
                idx = []
                for t in toks:
                    t = t.split("/")[0]
                    if t:
                        idx.append(int(t) - 1)
                if len(idx) >= 3:
                    for k in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[k], idx[k + 1]])
    if face_stride > 1:
        faces = faces[::face_stride]
    if max_faces is not None and len(faces) > max_faces:
        step = max(1, len(faces) // max_faces)
        faces = faces[::step][:max_faces]
    return np.asarray(verts), np.asarray(faces)


def fit_rbf(x, y, z, val, smooth=0.1, function="multiquadric"):
    from scipy.interpolate import Rbf
    rbf_re = Rbf(x, y, z, val.real, function=function, smooth=smooth)
    rbf_im = Rbf(x, y, z, val.imag, function=function, smooth=smooth)
    return lambda xx, yy, zz: rbf_re(xx, yy, zz) + 1j * rbf_im(xx, yy, zz)


def main():
    ap = argparse.ArgumentParser(description="Extrapolate from one cuboid to the others.")
    ap.add_argument("--psd", type=Path, required=True, help="Microflown PSD file")
    ap.add_argument("--group", type=str, default="Process1")
    ap.add_argument("--f0", type=float, default=1008.0, help="Target frequency (Hz)")
    ap.add_argument("--train-cluster", type=int, default=None,
                    help="Cluster index to train on (0/1/2). If omitted, use nearest-to-source cluster.")
    ap.add_argument("--source", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="Source position (x y z) to choose nearest cluster if train-cluster not set.")
    ap.add_argument("--smooth", type=float, default=0.2, help="RBF smoothness")
    ap.add_argument("--mesh-obj", type=Path, default=None, help="Optional OBJ mesh to overlay.")
    ap.add_argument("--mesh-stride", type=int, default=40)
    ap.add_argument("--mesh-maxfaces", type=int, default=60000)
    ap.add_argument("--outdir", type=Path, default=Path("figs/examples/martial_measurements"))
    ap.add_argument("--tag", type=str, default="cuboid_extrap")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.psd, "r") as f:
        g = f[args.group]
        freq = np.array(g["FreqVar"]).ravel()
        pos = np.array(g["datPostition"]).T  # (N,3)
        Pk = np.array(g["TFpref1_i"])[:, 0, :] + 1j * np.array(g["TFpref1_j"])[:, 0, :]
        Uk = (np.array(g["TFxyzref1_i"]) + 1j * np.array(g["TFxyzref1_j"]))  # (N,3,F)

    fi = np.argmin(np.abs(freq - args.f0))
    fsel = freq[fi]
    p = Pk[:, fi]
    u = Uk[:, :, fi]

    labels, cent = kmeans3(pos)
    # If train-cluster not provided, pick cluster whose centroid is closest to the given source
    if args.train_cluster is None:
        src = np.asarray(args.source, dtype=float)
        dcent = np.linalg.norm(cent - src[None, :], axis=1)
        train_id = int(np.argmin(dcent))
    else:
        train_id = int(args.train_cluster)

    train_mask = labels == train_id
    test_mask = ~train_mask

    # Fit RBF on training cuboid
    rbf_p = fit_rbf(pos[train_mask, 0], pos[train_mask, 1], pos[train_mask, 2],
                    p[train_mask], smooth=args.smooth)
    pred_all = rbf_p(pos[:, 0], pos[:, 1], pos[:, 2])

    # Errors on held-out points
    p_true = p[test_mask]
    p_pred = pred_all[test_mask]
    mag_err = np.abs(p_pred - p_true)
    rel_err = mag_err / np.maximum(np.abs(p_true), 1e-12)
    phase_err = np.angle(p_pred) - np.angle(p_true)
    phase_err = (phase_err + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

    # distance from each test point to nearest training point
    train_pts = pos[train_mask]
    test_pts = pos[test_mask]
    d2 = ((test_pts[:, None, :] - train_pts[None, :, :]) ** 2).sum(axis=2)
    dmin = np.sqrt(d2.min(axis=1))

    # plots
    mesh = None
    if args.mesh_obj and args.mesh_obj.exists():
        try:
            mesh = load_obj(args.mesh_obj, face_stride=args.mesh_stride,
                            max_faces=args.mesh_maxfaces)
        except Exception:
            mesh = None

    fig = plt.figure(figsize=(12, 4.5), dpi=140)
    ax = fig.add_subplot(131, projection="3d")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for k in range(3):
        m = labels == k
        ax.scatter(pos[m, 0], pos[m, 1], pos[m, 2],
                   s=6, alpha=0.6, color=colors[k], label=f"cuboid {k}")
    if mesh is not None:
        verts, faces = mesh
        coll = Poly3DCollection(verts[faces], facecolor="#cccccc", alpha=0.05, edgecolor="none")
        ax.add_collection3d(coll)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(f"Room + cuboids (train={train_id})")
    ax.legend(loc="best")

    ax2 = fig.add_subplot(132)
    sc = ax2.scatter(dmin, rel_err, c=np.angle(p_pred), s=10, cmap="twilight", alpha=0.8)
    ax2.set_xlabel("distance to training set [m]")
    ax2.set_ylabel("relative |Δp| / |p_true|")
    ax2.set_title(f"Extrapolation error @ {fsel:.1f} Hz")
    fig.colorbar(sc, ax=ax2, label="phase pred [rad]")

    ax3 = fig.add_subplot(133)
    ax3.hist(phase_err, bins=60, color="#444444", alpha=0.85)
    ax3.set_xlabel("phase error [rad]")
    ax3.set_ylabel("count")
    ax3.set_title("Held-out phase error")

    fig.tight_layout()
    out_png = outdir / f"{args.tag}_cluster_train{train_id}_{int(fsel)}Hz.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    np.savez(outdir / f"{args.tag}_train{train_id}_{int(fsel)}Hz.npz",
             freq=fsel, pos=pos, labels=labels, train_id=train_id,
             p_true=p_true, p_pred=p_pred, rel_err=rel_err, phase_err=phase_err, dmin=dmin)
    print(f"[save] {out_png}")
    print(f"[info] mean rel err={rel_err.mean():.3f}, median={np.median(rel_err):.3f}")


if __name__ == "__main__":
    main()
