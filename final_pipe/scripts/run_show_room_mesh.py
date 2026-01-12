#!/usr/bin/env python3
"""
Render the scanned room OBJ with its texture to a PNG (no interactive viewer).
Uses a light-weight OBJ/UV parser + matplotlib Poly3DCollection.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_textured_mesh(obj_path):
    obj_path = Path(obj_path)
    verts = []
    uvs = []
    faces_v = []
    faces_uv = []
    with open(obj_path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()[:4]
                verts.append([float(x), float(y), float(z)])
            elif line.startswith("vt "):
                _, u, v = line.strip().split()[:3]
                uvs.append([float(u), float(v)])
            elif line.startswith("f "):
                toks = line.strip().split()[1:]
                v_idx = []
                uv_idx = []
                for t in toks:
                    parts = t.split("/")
                    if len(parts) >= 2:
                        v_idx.append(int(parts[0]) - 1)
                        uv_idx.append(int(parts[1]) - 1)
                if len(v_idx) >= 3:
                    for k in range(1, len(v_idx) - 1):
                        faces_v.append([v_idx[0], v_idx[k], v_idx[k + 1]])
                        faces_uv.append([uv_idx[0], uv_idx[k], uv_idx[k + 1]])
    return np.asarray(verts), np.asarray(uvs), np.asarray(faces_v), np.asarray(faces_uv)


def sample_face_colors(uvs, faces_uv, image):
    w, h = image.size
    arr = np.asarray(image.convert("RGB"))
    cols = []
    for uv_idx in faces_uv:
        uv = uvs[uv_idx]
        u_mean = np.clip(uv[:, 0].mean(), 0, 1)
        v_mean = np.clip(uv[:, 1].mean(), 0, 1)
        px = int(u_mean * (w - 1))
        py = int((1 - v_mean) * (h - 1))  # OBJ v runs up
        cols.append(arr[py, px] / 255.0)
    return np.asarray(cols)


def compute_alignment_basis(points):
    c = points.mean(axis=0)
    X = points - c
    cov = X.T @ X / max(len(points) - 1, 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)
    basis = v[:, order]
    return c, basis


def apply_alignment(x, c, basis):
    return (x - c) @ basis


def rotation_from_vectors(a, b):
    """Return rotation matrix R such that R @ a = b (a,b normalized)."""
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
    """Estimate floor normal by PCA of lowest fraction of points (z by heuristic)."""
    zmin = np.quantile(points[:, 2], frac)
    subset = points[points[:, 2] <= zmin + 1e-6]
    if subset.shape[0] < 10:
        subset = points
    c = subset.mean(axis=0)
    X = subset - c
    cov = X.T @ X / max(len(subset) - 1, 1)
    w, v = np.linalg.eigh(cov)
    normal = v[:, 0]  # smallest variance
    return normal


def main():
    ap = argparse.ArgumentParser(description="Render textured room OBJ to PNG.")
    ap.add_argument("--obj", type=Path, required=True)
    ap.add_argument("--texture", type=Path, default=None,
                    help="Optional override texture (otherwise use .mtl referenced image if found).")
    ap.add_argument("--stride", type=int, default=8, help="Take every Nth face to lighten the mesh.")
    ap.add_argument("--max-faces", type=int, default=80000, help="Upper bound on faces drawn.")
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=35.0)
    ap.add_argument("--alpha", type=float, default=1.0, help="Mesh opacity.")
    ap.add_argument("--align", action="store_true", help="Level mesh by PCA (smallest variance axis -> z).")
    ap.add_argument("--marker", type=float, nargs=3, default=None, help="Optional point to highlight (e.g., source).")
    ap.add_argument("--center-to-marker", action="store_true", help="If marker given, translate mesh so marker -> origin.")
    ap.add_argument("--level-floor", action="store_true", help="Rotate mesh so floor normal aligns to +z (fit on low vertices).")
    ap.add_argument("--vertical-axis", choices=["auto", "x", "y", "z"], default="auto",
                    help="Treat this axis as vertical before any leveling (permute so chosen axis -> z).")
    ap.add_argument("--align-long-axis", action="store_true",
                    help="After floor leveling, rotate so longest variance axis aligns with +x (keeps z vertical).")
    ap.add_argument("--out", type=Path, default=Path("figs/examples/martial_measurements/room_mesh.png"))
    args = ap.parse_args()

    verts, uvs, f_v, f_uv = load_textured_mesh(args.obj)
    if args.stride > 1:
        f_v = f_v[::args.stride]
        f_uv = f_uv[::args.stride]
    if f_v.shape[0] > args.max_faces:
        step = max(1, f_v.shape[0] // args.max_faces)
        f_v = f_v[::step][:args.max_faces]
        f_uv = f_uv[::step][:args.max_faces]

    tex = args.texture
    if tex is None:
        # attempt to find texture referenced next to OBJ
        candidate = list(args.obj.parent.glob("*.jpg")) + list(args.obj.parent.glob("*.png"))
        tex = candidate[0] if candidate else None
    if tex is None or not Path(tex).exists():
        raise SystemExit("Texture image not found; pass --texture.")

    colors = sample_face_colors(uvs, f_uv, Image.open(tex))

    R_level = None
    # axis permutation
    perm = np.arange(3)
    if args.vertical_axis != "auto":
        mapping = {"x": 0, "y": 1, "z": 2}
        vert = mapping[args.vertical_axis]
    else:
        rng = verts.max(axis=0) - verts.min(axis=0)
        vert = int(np.argmin(rng))  # smallest span -> vertical
    # build permutation that moves chosen axis to index 2 (z)
    if vert == 0:
        perm = np.array([1, 2, 0])
    elif vert == 1:
        perm = np.array([0, 2, 1])
    elif vert == 2:
        perm = np.array([0, 1, 2])
    verts = verts[:, perm]
    if args.marker is not None:
        args.marker = np.asarray(args.marker)[perm]

    if args.level_floor:
        n = fit_floor_normal(np.asarray(verts))
        R_level = rotation_from_vectors(n, np.array([0.0, 0.0, 1.0]))
        verts = verts @ R_level.T
        if args.marker is not None:
            args.marker = np.asarray(args.marker) @ R_level.T
    # align longest horizontal axis to +x (keep z vertical)
    if args.align_long_axis:
        cov = (verts - verts.mean(axis=0)).T @ (verts - verts.mean(axis=0)) / max(len(verts) - 1, 1)
        w_long, v_long = np.linalg.eigh(cov)
        v_long = v_long[:, np.argmax(w_long)]
        zhat = np.array([0.0, 0.0, 1.0])
        v_proj = v_long - np.dot(v_long, zhat) * zhat
        if np.linalg.norm(v_proj) > 1e-9:
            R_long = rotation_from_vectors(v_proj, np.array([1.0, 0.0, 0.0]))
            verts = verts @ R_long.T
            if args.marker is not None:
                args.marker = np.asarray(args.marker) @ R_long.T
            if R_level is not None:
                R_level = R_long @ R_level

    if args.align:
        c, basis = compute_alignment_basis(verts)
        verts = apply_alignment(verts, c, basis)
        tris = verts[f_v]
        if args.marker is not None:
            args.marker = (np.asarray(args.marker) - c) @ basis
    else:
        tris = verts[f_v]
        if args.center_to_marker and args.marker is not None:
            shift = np.asarray(args.marker, dtype=float)
            verts = verts - shift[None, :]
            tris = verts[f_v]
            args.marker = np.zeros(3, dtype=float)
    fig = plt.figure(figsize=(8, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(tris, facecolors=colors, edgecolor="none", linewidth=0, alpha=args.alpha)
    ax.add_collection3d(coll)
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.view_init(elev=args.elev, azim=args.azim)
    if args.marker is not None:
        m = np.asarray(args.marker, dtype=float).ravel()
        ax.scatter([m[0]], [m[1]], [m[2]], color="magenta", s=60, depthshade=False, label="marker")
    ax.set_title("Scanned room (textured OBJ)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
