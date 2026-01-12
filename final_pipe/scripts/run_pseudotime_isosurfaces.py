#!/usr/bin/env python3
"""
Pseudo-time 3D isosurface sequence from Microflown PSD (complex TFs over a scanned grid).
Generates frame images and optionally an MP4 if ffmpeg is available.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np
import h5py
import os
import subprocess
from mf_pipeline.mf_config import EXPORTS

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

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUTDIR = Path(os.environ.get("MF_OUTDIR", EXPORTS)).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_obj(mesh_path, face_stride=1, max_faces=None):
    """Minimal OBJ loader (vertices + faces only)."""
    verts = []
    faces = []
    with open(mesh_path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = []
                for tok in line.strip().split()[1:]:
                    tok = tok.split("/")[0]
                    if tok:
                        idx.append(int(tok) - 1)  # OBJ is 1-indexed
                if len(idx) >= 3:
                    # triangulate simple polygons fan-style
                    for k in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[k], idx[k + 1]])
    if face_stride > 1:
        faces = faces[::face_stride]
    if max_faces is not None and len(faces) > max_faces:
        step = max(1, len(faces) // max_faces)
        faces = faces[::step][:max_faces]
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=int)
    return verts, faces


def main():
    ap = argparse.ArgumentParser(description="Pseudo-time isosurface sequence from PSD.")
    ap.add_argument("--psd", type=Path, required=True, help="Microflown PSD H5 (Process1).")
    ap.add_argument("--group", type=str, default="Process1")
    ap.add_argument("--metric", choices=["p", "u_mag"], default="p")
    ap.add_argument("--fmin", type=float, default=None)
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--max-freqs", type=int, default=200, help="Max number of frequency bins to use.")
    ap.add_argument("--n-time", type=int, default=32, help="Number of pseudo-time frames.")
    ap.add_argument("--grid", type=int, default=30, help="Grid resolution per axis.")
    ap.add_argument("--smooth", type=float, default=0.3)
    ap.add_argument("--levels", type=float, nargs="+", default=[85, 95], help="Isosurface percentiles.")
    ap.add_argument("--abs", dest="use_abs", action="store_true", help="Use absolute value for isosurfaces.")
    ap.add_argument("--mesh-obj", type=Path, default=None, help="Optional OBJ mesh to overlay (room scan).")
    ap.add_argument("--mesh-stride", type=int, default=8, help="Take every Nth face to lighten mesh.")
    ap.add_argument("--mesh-maxfaces", type=int, default=100000, help="Upper bound on faces drawn.")
    ap.add_argument("--mesh-alpha", type=float, default=0.1, help="Mesh transparency.")
    ap.add_argument("--mesh-color", type=str, default="#666666", help="Mesh color.")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--tag", type=str, default="pseudotime")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    if not _HAVE_RBF or not _HAVE_MC:
        raise SystemExit("Requires SciPy Rbf and scikit-image (marching_cubes).")

    outdir = args.outdir or OUTDIR / f"{args.tag}_frames"
    outdir.mkdir(parents=True, exist_ok=True)
    mesh = None
    if args.mesh_obj:
        try:
            mesh = load_obj(args.mesh_obj, face_stride=args.mesh_stride, max_faces=args.mesh_maxfaces)
            print(f"[mesh] loaded {args.mesh_obj} verts={mesh[0].shape[0]} faces={mesh[1].shape[0]}")
        except Exception as exc:
            print(f"[mesh] failed to load {args.mesh_obj}: {exc}")
            mesh = None

    with h5py.File(args.psd, "r") as f:
        g = f[args.group]
        pos = np.array(g["datPostition"]).T  # (N,3)
        freq = np.array(g["FreqVar"]).ravel()
        # complex TFs
        P_re = np.array(g["TFpref1_i"])[:, 0, :]
        P_im = np.array(g["TFpref1_j"])[:, 0, :]
        Pk = P_re + 1j * P_im  # (N, F)
        if args.metric == "u_mag":
            U_re = np.array(g["TFxyzref1_i"])  # (N,3,F)
            U_im = np.array(g["TFxyzref1_j"])
            Uk = U_re + 1j * U_im
        else:
            Uk = None

    # frequency selection
    mask = np.ones(freq.shape, dtype=bool)
    if args.fmin is not None:
        mask &= freq >= args.fmin
    if args.fmax is not None:
        mask &= freq <= args.fmax
    freq_sel = freq[mask]
    if freq_sel.size == 0:
        raise SystemExit("No frequencies in the requested range.")

    idx = np.where(mask)[0]
    if idx.size > args.max_freqs:
        # select highest-energy bins by mean magnitude
        energy = np.mean(np.abs(Pk[:, idx]), axis=0)
        top = np.argsort(energy)[-args.max_freqs:]
        idx = idx[top]
        freq_sel = freq[idx]

    # build pseudo-time
    fmin = float(freq_sel.min())
    T = 1.0 / fmin
    t = np.linspace(0.0, T, args.n_time, endpoint=False)
    exp_mat = np.exp(1j * 2.0 * np.pi * t[:, None] * freq_sel[None, :])  # (T,F)

    Pk_sel = Pk[:, idx].T  # (F,N)
    if args.metric == "u_mag":
        Uk_sel = Uk[:, :, idx].transpose(2, 0, 1)  # (F,N,3)

    # precompute grid
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    xi = np.linspace(x.min(), x.max(), args.grid)
    yi = np.linspace(y.min(), y.max(), args.grid)
    zi = np.linspace(z.min(), z.max(), args.grid)
    X, Y, Zg = np.meshgrid(xi, yi, zi, indexing="xy")
    xlims = [x.min(), x.max()]
    ylims = [y.min(), y.max()]
    zlims = [z.min(), z.max()]
    if mesh is not None:
        xlims = [min(xlims[0], mesh[0][:, 0].min()), max(xlims[1], mesh[0][:, 0].max())]
        ylims = [min(ylims[0], mesh[0][:, 1].min()), max(ylims[1], mesh[0][:, 1].max())]
        zlims = [min(zlims[0], mesh[0][:, 2].min()), max(zlims[1], mesh[0][:, 2].max())]

    # determine global level reference
    # use magnitude envelope
    env = np.mean(np.abs(Pk_sel), axis=0)
    if args.metric == "u_mag":
        env = np.mean(np.linalg.norm(Uk_sel, axis=2), axis=0)
    levels = [np.percentile(env, p) for p in args.levels]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ti, tt in enumerate(t):
        # instantaneous field
        p_t = 2.0 * np.real(exp_mat[ti:ti+1, :] @ Pk_sel)[0]  # (N,)
        if args.metric == "u_mag":
            u_t = 2.0 * np.real(exp_mat[ti:ti+1, :, None] * Uk_sel).sum(axis=0)
            val = np.linalg.norm(u_t, axis=1)
        else:
            val = p_t
        if args.use_abs:
            val = np.abs(val)

        rbf = Rbf(x, y, z, val, function="multiquadric", smooth=args.smooth)
        V = rbf(X.ravel(), Y.ravel(), Zg.ravel()).reshape(X.shape)

        fig = plt.figure(figsize=(6, 5), dpi=160)
        ax = fig.add_subplot(111, projection="3d")

        for i, level in enumerate(levels):
            try:
                verts, faces, _, _ = marching_cubes(V, level=level,
                                                    spacing=(xi[1]-xi[0], yi[1]-yi[0], zi[1]-zi[0]))
            except Exception:
                continue
            color = to_rgba(colors[i % len(colors)], alpha=0.35)
            ax.plot_trisurf(verts[:, 0] + xi.min(),
                            verts[:, 1] + yi.min(),
                            faces,
                            verts[:, 2] + zi.min(),
                            color=color, linewidth=0.2, edgecolor="k")

        if mesh is not None and mesh[1].size > 0:
            m_verts, m_faces = mesh
            coll = Poly3DCollection(m_verts[m_faces], facecolor=args.mesh_color,
                                    edgecolor="none", alpha=args.mesh_alpha)
            ax.add_collection3d(coll)

        ax.set_xlim(xlims); ax.set_ylim(ylims); ax.set_zlim(zlims)

        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.set_title(f"{args.metric} pseudo-time t={tt*1000:.2f} ms")
        ax.view_init(elev=20, azim=45)
        fig.tight_layout()

        out = outdir / f"frame_{ti:03d}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

    # optional mp4
    mp4 = outdir.parent / f"{args.tag}.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(outdir / "frame_%03d.png"),
            "-pix_fmt", "yuv420p", str(mp4)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[save] {mp4}")
    except Exception:
        print("[info] ffmpeg not available; frames saved in", outdir)


if __name__ == "__main__":
    main()
