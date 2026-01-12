#!/usr/bin/env python3
"""
Flow/phase plots from an oriented NPZ (pos, P, U).
Generates:
  - flow slices with |P| contours + streamlines (XY + XZ)
  - phase slices (XY + XZ)
  - 3D quiver with room mesh backdrop (optional)
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import distance_transform_edt


def load_obj(mesh_path):
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
    return np.asarray(verts), np.asarray(faces)


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
    with np.errstate(invalid="ignore"):
        out = out / np.maximum(count, 1)
    return out


def voxel_average_vec(pos, vec, grid):
    gx, gy, gz = grid
    ix = np.clip(np.digitize(pos[:, 0], gx) - 1, 0, len(gx) - 2)
    iy = np.clip(np.digitize(pos[:, 1], gy) - 1, 0, len(gy) - 2)
    iz = np.clip(np.digitize(pos[:, 2], gz) - 1, 0, len(gz) - 2)
    out = np.full((len(gx)-1, len(gy)-1, len(gz)-1, 3), np.nan)
    count = np.zeros(out.shape[:3], dtype=int)
    for a, b, c, v in zip(ix, iy, iz, vec):
        if np.isnan(out[a, b, c, 0]):
            out[a, b, c] = v
        else:
            out[a, b, c] += v
        count[a, b, c] += 1
    with np.errstate(invalid="ignore"):
        out = out / np.maximum(count[..., None], 1)
    return out


def fill_nan_nearest(arr2d):
    mask = np.isnan(arr2d)
    if not np.any(mask):
        return arr2d
    idx = np.array(np.nonzero(~mask)).T
    if idx.size == 0:
        return np.nan_to_num(arr2d)
    dist, inds = distance_transform_edt(mask, return_indices=True)
    filled = arr2d[tuple(inds)]
    return filled


def main():
    ap = argparse.ArgumentParser(description="Flow/phase plots from oriented NPZ.")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--mesh-obj", type=Path, default=None)
    ap.add_argument("--mesh-stride", type=int, default=5)
    ap.add_argument("--grid", type=int, default=28)
    ap.add_argument("--n-slices", type=int, default=4)
    ap.add_argument("--stream-density", type=float, default=2.0)
    ap.add_argument("--contour-levels", type=int, default=24)
    ap.add_argument("--quiver-step", type=int, default=3, help="Overlay quiver every N grid cells.")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="flow")
    args = ap.parse_args()

    Z = np.load(args.npz, allow_pickle=True)
    pos = Z["pos"]
    P = Z["P"]
    U = Z["U"]

    f0 = float(Z["f0"]) if "f0" in Z else None
    args.outdir.mkdir(parents=True, exist_ok=True)

    J = 0.5 * P[:, None] * np.conj(U)
    I = np.real(J)
    Pmag = np.abs(P)
    Pph = np.angle(P)

    xmin, ymin, zmin = pos.min(0)
    xmax, ymax, zmax = pos.max(0)
    gx = np.linspace(xmin, xmax, args.grid)
    gy = np.linspace(ymin, ymax, args.grid)
    gz = np.linspace(zmin, zmax, args.grid)
    xc = 0.5 * (gx[:-1] + gx[1:])
    yc = 0.5 * (gy[:-1] + gy[1:])
    zc = 0.5 * (gz[:-1] + gz[1:])

    Igrid = voxel_average_vec(pos, I, (gx, gy, gz))
    Pmag_grid = voxel_average_scalar(pos, Pmag, (gx, gy, gz))
    Pph_grid = voxel_average_scalar(pos, Pph, (gx, gy, gz))

    z_slices = np.linspace(zc[int(0.15*len(zc))], zc[int(0.85*len(zc))], args.n_slices)
    y_slices = np.linspace(yc[int(0.15*len(yc))], yc[int(0.85*len(yc))], args.n_slices)
    iz_list = [int(np.argmin(np.abs(zc - z))) for z in z_slices]
    iy_list = [int(np.argmin(np.abs(yc - y))) for y in y_slices]

    # Flow slices
    cols = args.n_slices * 2
    fig, axes = plt.subplots(2, args.n_slices, figsize=(4*args.n_slices, 7), dpi=140)
    for i, iz in enumerate(iz_list):
        X, Y = np.meshgrid(xc, yc, indexing="xy")
        Uxy = Igrid[:, :, iz, 0].T
        Vxy = Igrid[:, :, iz, 1].T
        mag = Pmag_grid[:, :, iz].T
        ax = axes[0, i]
        mag_f = fill_nan_nearest(mag)
        ax.contourf(X, Y, mag_f, levels=args.contour_levels, cmap="viridis")
        ax.streamplot(X, Y, Uxy, Vxy, density=args.stream_density, color="w", linewidth=0.8)
        if args.quiver_step > 0:
            qs = args.quiver_step
            ax.quiver(X[::qs, ::qs], Y[::qs, ::qs],
                      Uxy[::qs, ::qs], Vxy[::qs, ::qs],
                      color="k", alpha=0.6, scale=40)
        ax.set_title(f"XY z~{zc[iz]:.2f} m")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    for i, iy in enumerate(iy_list):
        X, Zm = np.meshgrid(xc, zc, indexing="xy")
        Uxz = Igrid[:, iy, :, 0].T
        Vxz = Igrid[:, iy, :, 2].T
        mag = Pmag_grid[:, iy, :].T
        ax = axes[1, i]
        mag_f = fill_nan_nearest(mag)
        ax.contourf(X, Zm, mag_f, levels=args.contour_levels, cmap="viridis")
        ax.streamplot(X, Zm, Uxz, Vxz, density=args.stream_density, color="w", linewidth=0.8)
        if args.quiver_step > 0:
            qs = args.quiver_step
            ax.quiver(X[::qs, ::qs], Zm[::qs, ::qs],
                      Uxz[::qs, ::qs], Vxz[::qs, ::qs],
                      color="k", alpha=0.6, scale=40)
        ax.set_title(f"XZ y~{yc[iy]:.2f} m")
        ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    fig.suptitle(f"Flow slices with |P| contours ({f0:.0f} Hz)" if f0 else "Flow slices with |P| contours")
    fig.tight_layout()
    fig.savefig(args.outdir / f"{args.tag}_flow_slices.png", bbox_inches="tight")
    plt.close(fig)

    # Phase slices
    fig, axes = plt.subplots(2, args.n_slices, figsize=(4*args.n_slices, 7), dpi=140)
    for i, iz in enumerate(iz_list):
        X, Y = np.meshgrid(xc, yc, indexing="xy")
        ph = Pph_grid[:, :, iz].T
        ax = axes[0, i]
        ph_f = fill_nan_nearest(ph)
        ax.contourf(X, Y, ph_f, levels=args.contour_levels, cmap="twilight")
        ax.set_title(f"Phase XY z~{zc[iz]:.2f} m")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    for i, iy in enumerate(iy_list):
        X, Zm = np.meshgrid(xc, zc, indexing="xy")
        ph = Pph_grid[:, iy, :].T
        ax = axes[1, i]
        ph_f = fill_nan_nearest(ph)
        ax.contourf(X, Zm, ph_f, levels=args.contour_levels, cmap="twilight")
        ax.set_title(f"Phase XZ y~{yc[iy]:.2f} m")
        ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    fig.suptitle(f"Phase slices ({f0:.0f} Hz)" if f0 else "Phase slices")
    fig.tight_layout()
    fig.savefig(args.outdir / f"{args.tag}_phase_slices.png", bbox_inches="tight")
    plt.close(fig)

    # 3D colored points by |I| magnitude (cleaner than dense quiver)
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    if args.mesh_obj:
        verts, faces = load_obj(args.mesh_obj)
        if args.mesh_stride > 1 and faces.size > 0:
            faces = faces[::args.mesh_stride]
        if "center" in Z and "R_floor" in Z and "R_int" in Z:
            center = np.asarray(Z["center"], dtype=float)
            Rf = np.asarray(Z["R_floor"], dtype=float)
            Ri = np.asarray(Z["R_int"], dtype=float)
            verts = (verts - center[None, :]) @ Rf.T
            verts = verts @ Ri.T
        coll = Poly3DCollection(verts[faces], facecolor="#bbbbbb", alpha=0.08, edgecolor="none")
        ax.add_collection3d(coll)
    I_mag = np.linalg.norm(I, axis=1)
    step = max(1, len(pos) // 3000)
    sc = ax.scatter(pos[::step,0], pos[::step,1], pos[::step,2],
                    c=I_mag[::step], cmap="viridis", s=6, alpha=0.9)
    ax.scatter([0],[0],[0], color="magenta", s=60, depthshade=False)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(f"|I| colored points ({f0:.0f} Hz)" if f0 else "|I| colored points")
    fig.colorbar(sc, ax=ax, shrink=0.65, label="|I|")
    fig.tight_layout()
    fig.savefig(args.outdir / f"{args.tag}_intensity_3d.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
