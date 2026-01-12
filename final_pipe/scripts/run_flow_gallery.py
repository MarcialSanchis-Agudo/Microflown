#!/usr/bin/env python3
"""
Flow / vortex gallery from Microflown PSD:
- cluster points into 3 cuboids; train interpolant on nearest-to-source cluster
- interpolate P and U to a grid
- compute complex intensity J = 0.5 * P * conj(U)
- plot active/reactive quivers on slices and a 3D room view (optional mesh overlay)
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
    ap = argparse.ArgumentParser(description="Flow gallery from PSD (P, U -> intensity).")
    ap.add_argument("--psd", type=Path, required=True)
    ap.add_argument("--group", type=str, default="Process1")
    ap.add_argument("--f0", type=float, default=1008.0)
    ap.add_argument("--grid", type=int, default=26)
    ap.add_argument("--smooth", type=float, default=0.15)
    ap.add_argument("--source", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="Source position (x y z) to pick nearest cuboid for training.")
    ap.add_argument("--mesh-obj", type=Path, default=None)
    ap.add_argument("--mesh-stride", type=int, default=60)
    ap.add_argument("--mesh-maxfaces", type=int, default=40000)
    ap.add_argument("--outdir", type=Path, default=Path("figs/examples/martial_measurements"))
    ap.add_argument("--tag", type=str, default="flow_gallery")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

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
    src = np.asarray(args.source, dtype=float)
    train_id = int(np.argmin(np.linalg.norm(cent - src[None, :], axis=1)))
    train_mask = labels == train_id

    # Fit RBF interpolants on training cuboid
    rbf_p = fit_rbf(pos[train_mask, 0], pos[train_mask, 1], pos[train_mask, 2],
                    p[train_mask], smooth=args.smooth)
    rbf_ux = fit_rbf(pos[train_mask, 0], pos[train_mask, 1], pos[train_mask, 2],
                     u[train_mask, 0], smooth=args.smooth)
    rbf_uy = fit_rbf(pos[train_mask, 0], pos[train_mask, 1], pos[train_mask, 2],
                     u[train_mask, 1], smooth=args.smooth)
    rbf_uz = fit_rbf(pos[train_mask, 0], pos[train_mask, 1], pos[train_mask, 2],
                     u[train_mask, 2], smooth=args.smooth)

    # grid within measurement extents
    xmin, ymin, zmin = pos.min(axis=0)
    xmax, ymax, zmax = pos.max(axis=0)
    gx = np.linspace(xmin, xmax, args.grid)
    gy = np.linspace(ymin, ymax, args.grid)
    gz = np.linspace(zmin, zmax, args.grid)
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="xy")

    Pgrid = rbf_p(X, Y, Z)
    Ux = rbf_ux(X, Y, Z)
    Uy = rbf_uy(X, Y, Z)
    Uz = rbf_uz(X, Y, Z)
    Ugrid = np.stack([Ux, Uy, Uz], axis=-1)

    J = 0.5 * Pgrid[..., None] * np.conj(Ugrid)
    I = np.real(J)  # active intensity
    Q = np.imag(J)  # reactive

    # slice indices
    iz = args.grid // 2
    iy = args.grid // 2

    def quiver_slice(ax, plane="xy"):
        if plane == "xy":
            ax.quiver(X[:, :, iz], Y[:, :, iz], I[:, :, iz, 0], I[:, :, iz, 1],
                      color="#ff7f0e", alpha=0.8, angles="xy", scale_units="xy", scale=1.0)
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title(f"Active intensity XY z~{gz[iz]:.2f} m")
        elif plane == "xz":
            ax.quiver(X[:, iy, :], Z[:, iy, :], I[:, iy, :, 0], I[:, iy, :, 2],
                      color="#1f77b4", alpha=0.8, angles="xy", scale_units="xy", scale=1.0)
            ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]"); ax.set_title(f"Active intensity XZ y~{gy[iy]:.2f} m")

    # histograms of |Q|/|I|
    qratio = np.linalg.norm(Q.reshape(-1, 3), axis=1) / np.maximum(np.linalg.norm(I.reshape(-1, 3), axis=1), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=140)
    quiver_slice(axes[0], "xy")
    quiver_slice(axes[1], "xz")
    axes[2].hist(qratio, bins=80, color="#444444", alpha=0.85)
    axes[2].set_xlabel("|Q| / |I|"); axes[2].set_ylabel("count")
    axes[2].set_title("Reactive vs active ratio")
    fig.tight_layout()
    out_png = args.outdir / f"{args.tag}_{int(fsel)}Hz_slices.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # 3D overview with mesh
    mesh = None
    if args.mesh_obj and args.mesh_obj.exists():
        try:
            mesh = load_obj(args.mesh_obj, face_stride=args.mesh_stride, max_faces=args.mesh_maxfaces)
        except Exception:
            mesh = None

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax3d = fig.add_subplot(111, projection="3d")
    if mesh is not None:
        verts, faces = mesh
        coll = Poly3DCollection(verts[faces], facecolor="#cccccc", alpha=0.07, edgecolor="none")
        ax3d.add_collection3d(coll)
    # plot a subset of grid vectors for legibility
    step = max(1, args.grid // 10)
    ax3d.quiver(X[::step, ::step, ::step], Y[::step, ::step, ::step], Z[::step, ::step, ::step],
                I[::step, ::step, ::step, 0], I[::step, ::step, ::step, 1], I[::step, ::step, ::step, 2],
                length=0.15, color="#ff7f0e", alpha=0.7, linewidth=0.6, normalize=False)
    ax3d.set_xlim(xmin, xmax); ax3d.set_ylim(ymin, ymax); ax3d.set_zlim(zmin, zmax)
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.set_title(f"Active intensity vectors @ {fsel:.1f} Hz (train cuboid {train_id})")
    fig.tight_layout()
    out_vec = args.outdir / f"{args.tag}_{int(fsel)}Hz_vectors.png"
    fig.savefig(out_vec, bbox_inches="tight")
    plt.close(fig)

    print(f"[save] {out_png}")
    print(f"[save] {out_vec}")


if __name__ == "__main__":
    main()
