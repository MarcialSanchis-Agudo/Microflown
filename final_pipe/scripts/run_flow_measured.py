#!/usr/bin/env python3
"""
Flow/vortex visualization using measured P,U only (no free-form RBF extrapolation).
- Load Microflown PSD (Process1) P and U
- Compute intensity J = 0.5 * P * conj(U) on measured points
- Voxel-average onto a coarse grid for streamplots
- Render:
  * XY and XZ active-intensity streamplots
  * 3D quiver of active intensity on measured points (subsampled), optional room mesh overlay
  * |Q|/|I| histogram
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


def rotation_from_vectors(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2 + 1e-12))
    return R


def fit_floor_normal(points, frac=0.2):
    zmin = np.quantile(points[:, 2], frac)
    subset = points[points[:, 2] <= zmin + 1e-6]
    if subset.shape[0] < 10:
        subset = points
    c = subset.mean(axis=0)
    X = subset - c
    cov = X.T @ X / max(len(subset) - 1, 1)
    w, v = np.linalg.eigh(cov)
    normal = v[:, 0]
    return normal


def dominant_direction(vecs):
    """Principal direction of a set of vectors (rows)."""
    cov = vecs.T @ vecs / max(len(vecs) - 1, 1)
    w, v = np.linalg.eigh(cov)
    return v[:, np.argmax(w)]


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


def voxel_average(pos, vec, grid):
    """Average vectors into voxels defined by grid (edges per axis)."""
    gx, gy, gz = grid
    idx_x = np.clip(np.digitize(pos[:, 0], gx) - 1, 0, len(gx) - 2)
    idx_y = np.clip(np.digitize(pos[:, 1], gy) - 1, 0, len(gy) - 2)
    idx_z = np.clip(np.digitize(pos[:, 2], gz) - 1, 0, len(gz) - 2)
    acc = {}
    for i, j, k, v in zip(idx_x, idx_y, idx_z, vec):
        acc.setdefault((i, j, k), []).append(v)
    vx = np.zeros((len(gx)-1, len(gy)-1, len(gz)-1, 3), dtype=float)
    mask = np.zeros(vx.shape[:3], dtype=bool)
    for (i, j, k), vals in acc.items():
        vals = np.stack(vals, axis=0)
        vx[i, j, k] = vals.mean(axis=0)
        mask[i, j, k] = True
    return vx, mask


def compute_alignment_basis(points):
    """Return centroid and orthonormal basis (columns) that aligns smallest-variance axis to z."""
    c = points.mean(axis=0)
    X = points - c
    cov = X.T @ X / max(len(points) - 1, 1)
    w, v = np.linalg.eigh(cov)
    # eigenvalues ascending; columns of v are eigenvectors
    order = np.argsort(w)
    basis = v[:, order]  # smallest variance -> first column
    return c, basis


def apply_alignment(x, c, basis):
    """Center and rotate points: x_new = basis.T @ (x - c)."""
    return (x - c) @ basis


def main():
    ap = argparse.ArgumentParser(description="Measured-flow visualization (no extrapolation).")
    ap.add_argument("--psd", type=Path, required=True)
    ap.add_argument("--group", type=str, default="Process1")
    ap.add_argument("--f0", type=float, default=1008.0)
    ap.add_argument("--grid", type=int, default=20, help="Coarse grid per axis for voxel averaging.")
    ap.add_argument("--mesh-obj", type=Path, default=None, help="Optional room mesh to overlay.")
    ap.add_argument("--mesh-stride", type=int, default=80)
    ap.add_argument("--mesh-maxfaces", type=int, default=30000)
    ap.add_argument("--quiver-step", type=int, default=6, help="Subsample measured points for 3D quiver (higher = fewer).")
    ap.add_argument("--quiver-scale", type=float, default=0.12, help="Arrow length scaling for 3D quiver.")
    ap.add_argument("--stream-density", type=float, default=1.2, help="Matplotlib streamplot density.")
    ap.add_argument("--z-slices", type=float, nargs="+", default=None, help="Explicit z slices (meters) for XY streamplots. Defaults to mid-plane.")
    ap.add_argument("--y-slices", type=float, nargs="+", default=None, help="Explicit y slices (meters) for XZ streamplots. Defaults to mid-plane.")
    ap.add_argument("--align-mesh", action="store_true", help="Level mesh/points using mesh PCA (smallest variance axis -> z).")
    ap.add_argument("--center-to", type=float, nargs=3, default=None,
                    help="Translate points/vectors so this coordinate becomes the origin (no rotation).")
    ap.add_argument("--level-floor", action="store_true", help="Rotate mesh/points so floor normal aligns to +z (fit on mesh low verts).")
    ap.add_argument("--vertical-axis", choices=["auto", "x", "y", "z"], default="auto",
                    help="Treat this axis as vertical before leveling (permute so chosen axis -> z).")
    ap.add_argument("--align-long-axis", action="store_true",
                    help="After floor leveling, rotate longest horizontal axis to +x.")
    ap.add_argument("--align-intensity-axis", action="store_true",
                    help="After floor leveling, rotate dominant active intensity (f0) to +x (keeps z vertical).")
    ap.add_argument("--align-intensity-axis", action="store_true",
                    help="After floor leveling, rotate dominant active intensity to +x (keeps z vertical).")
    ap.add_argument("--outdir", type=Path, default=Path("figs/examples/martial_measurements"))
    ap.add_argument("--tag", type=str, default="flow_measured")
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
    P = Pk[:, fi]
    U = Uk[:, :, fi]

    mesh = None
    mesh_c = None
    mesh_basis = None
    R_level = None
    perm = np.arange(3)
    if args.mesh_obj and args.mesh_obj.exists():
        try:
            mesh = load_obj(args.mesh_obj, face_stride=args.mesh_stride, max_faces=args.mesh_maxfaces)
            # vertical axis permutation
            if args.vertical_axis != "auto":
                mapping = {"x": 0, "y": 1, "z": 2}
                vert = mapping[args.vertical_axis]
            else:
                rng = mesh[0].max(axis=0) - mesh[0].min(axis=0)
                vert = int(np.argmin(rng))
            if vert == 0:
                perm = np.array([1, 2, 0])
            elif vert == 1:
                perm = np.array([0, 2, 1])
            else:
                perm = np.array([0, 1, 2])
            mesh = (mesh[0][:, perm], mesh[1])
            pos = pos[:, perm]
            U = U[:, perm, :]
            if args.level_floor:
                n = fit_floor_normal(mesh[0])
                R_level = rotation_from_vectors(n, np.array([0.0, 0.0, 1.0]))
                mesh = (mesh[0] @ R_level.T, mesh[1])
            if args.align_long_axis:
                cov = (mesh[0] - mesh[0].mean(axis=0)).T @ (mesh[0] - mesh[0].mean(axis=0)) / max(len(mesh[0]) - 1, 1)
                w_long, v_long = np.linalg.eigh(cov)
                v_long = v_long[:, np.argmax(w_long)]
                zhat = np.array([0.0, 0.0, 1.0])
                v_proj = v_long - np.dot(v_long, zhat) * zhat
                if np.linalg.norm(v_proj) > 1e-9:
                    R_long = rotation_from_vectors(v_proj, np.array([1.0, 0.0, 0.0]))
                    mesh = (mesh[0] @ R_long.T, mesh[1])
                    if R_level is not None:
                        R_level = R_long @ R_level
            if args.align_mesh:
                mesh_c, mesh_basis = compute_alignment_basis(mesh[0])
                verts_aligned = apply_alignment(mesh[0], mesh_c, mesh_basis)
                mesh = (verts_aligned, mesh[1])
        except Exception:
            mesh = None

    # alignment on points if mesh-based align requested
    if args.level_floor and R_level is not None:
        pos = pos @ R_level.T
        U = np.einsum("ij,njf->nif", R_level, U)
    if args.align_long_axis and R_level is not None:
        pos = pos @ R_level.T
        U = np.einsum("ij,njf->nif", R_level, U)
    if args.align_intensity_axis:
        # compute active intensity direction at f0 and align its dominant horizontal component to +x
        J_tmp = 0.5 * (Pk[:, fi])[:, None] * np.conj(Uk[:, :, fi])
        I_tmp = np.real(J_tmp)
        dir_dom = dominant_direction(I_tmp)
        zhat = np.array([0.0, 0.0, 1.0])
        dir_proj = dir_dom - np.dot(dir_dom, zhat) * zhat
        if np.linalg.norm(dir_proj) > 1e-9:
            R_int = rotation_from_vectors(dir_proj, np.array([1.0, 0.0, 0.0]))
            pos = pos @ R_int.T
            U = np.einsum("ij,njf->nif", R_int, U)
    if args.align_mesh and mesh_basis is not None:
        pos = apply_alignment(pos, mesh_c, mesh_basis)
        U = np.einsum("ij,njf->nif", mesh_basis, U)
    # center shift if requested
    if args.center_to is not None:
        shift = np.asarray(args.center_to, dtype=float)
        pos = pos - shift[None, :]
        # vectors unaffected by translation
        if mesh is not None and mesh_basis is None:  # only translate when not already aligned
            mesh = (mesh[0] - shift[None, :], mesh[1])

    J = 0.5 * P[:, None] * np.conj(U)
    I = np.real(J)
    Q = np.imag(J)

    # voxel average for streamplots
    xmin, ymin, zmin = pos.min(axis=0)
    xmax, ymax, zmax = pos.max(axis=0)
    gx = np.linspace(xmin, xmax, args.grid)
    gy = np.linspace(ymin, ymax, args.grid)
    gz = np.linspace(zmin, zmax, args.grid)
    vox, mask = voxel_average(pos, I, (gx, gy, gz))
    # center coords of voxels
    xc = 0.5 * (gx[:-1] + gx[1:])
    yc = 0.5 * (gy[:-1] + gy[1:])
    zc = 0.5 * (gz[:-1] + gz[1:])

    # map desired slices to nearest indices
    def nearest_idx(arr, val):
        return int(np.argmin(np.abs(arr - val)))
    if args.z_slices:
        iz_list = [nearest_idx(zc, z) for z in args.z_slices]
    else:
        iz_list = [len(zc) // 2]
    if args.y_slices:
        iy_list = [nearest_idx(yc, y) for y in args.y_slices]
    else:
        iy_list = [len(yc) // 2]

    def stream(ax, plane="xy", idx=0):
        if plane == "xy":
            iz = iz_list[idx]
            X, Y = np.meshgrid(xc, yc, indexing="xy")
            Uxy = vox[:, :, iz, 0].T  # transpose to match meshgrid
            Vxy = vox[:, :, iz, 1].T
            ax.streamplot(X, Y, Uxy, Vxy, density=args.stream_density, color="#ff7f0e", linewidth=1.0, arrowsize=1.1)
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title(f"Active intensity XY (z~{zc[iz]:.2f} m)")
        elif plane == "xz":
            iy = iy_list[idx]
            X, Z = np.meshgrid(xc, zc, indexing="xy")
            Uxz = vox[:, iy, :, 0].T
            Vxz = vox[:, iy, :, 2].T
            ax.streamplot(X, Z, Uxz, Vxz, density=args.stream_density, color="#1f77b4", linewidth=1.0, arrowsize=1.1)
            ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]"); ax.set_title(f"Active intensity XZ (y~{yc[iy]:.2f} m)")

    qratio = np.linalg.norm(Q, axis=1) / np.maximum(np.linalg.norm(I, axis=1), 1e-12)

    n_xy = len(iz_list)
    n_xz = len(iy_list)
    cols = n_xy + n_xz + 1
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4.2), dpi=140)
    axes = np.atleast_1d(axes)
    idx_ax = 0
    for i in range(n_xy):
        stream(axes[idx_ax], "xy", idx=i)
        idx_ax += 1
    for i in range(n_xz):
        stream(axes[idx_ax], "xz", idx=i)
        idx_ax += 1
    axes[idx_ax].hist(qratio, bins=80, color="#444444", alpha=0.85)
    axes[idx_ax].set_xlabel("|Q| / |I|"); axes[idx_ax].set_ylabel("count"); axes[idx_ax].set_title("Reactive vs active ratio (measured)")
    fig.tight_layout()
    out_slice = args.outdir / f"{args.tag}_{int(fsel)}Hz_slices.png"
    fig.savefig(out_slice, bbox_inches="tight"); plt.close(fig)

    mesh = None
    if args.mesh_obj and args.mesh_obj.exists():
        try:
            mesh = load_obj(args.mesh_obj, face_stride=args.mesh_stride, max_faces=args.mesh_maxfaces)
            if args.align_mesh and mesh_basis is not None:
                verts_aligned = apply_alignment(mesh[0], mesh_c, mesh_basis)
                mesh = (verts_aligned, mesh[1])
        except Exception:
            mesh = None

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax3d = fig.add_subplot(111, projection="3d")
    if mesh is not None:
        verts, faces = mesh
        coll = Poly3DCollection(verts[faces], facecolor="#cccccc", alpha=0.07, edgecolor="none")
        ax3d.add_collection3d(coll)
    # subsample points for readability
    step = max(1, args.quiver_step)
    ax3d.quiver(pos[::step, 0], pos[::step, 1], pos[::step, 2],
                I[::step, 0], I[::step, 1], I[::step, 2],
                length=args.quiver_scale, color="#ff7f0e", alpha=0.6, linewidth=0.5, normalize=False)
    ax3d.set_xlim(xmin, xmax); ax3d.set_ylim(ymin, ymax); ax3d.set_zlim(zmin, zmax)
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.set_title(f"Active intensity quiver @ {fsel:.1f} Hz (measured points)")
    fig.tight_layout()
    out_vec = args.outdir / f"{args.tag}_{int(fsel)}Hz_vectors.png"
    fig.savefig(out_vec, bbox_inches="tight"); plt.close(fig)

    print(f"[save] {out_slice}")
    print(f"[save] {out_vec}")


if __name__ == "__main__":
    main()
