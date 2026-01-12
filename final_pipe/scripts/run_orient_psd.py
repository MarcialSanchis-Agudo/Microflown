#!/usr/bin/env python3
"""
Build oriented NPZs from Microflown PSD.
Orientation:
  - translate so source proxy is at origin
  - rotate floor normal -> +z (fit from lowest 10% z)
  - rotate dominant active-intensity direction (projected to floor) -> +x
"""
import argparse
from pathlib import Path
import numpy as np
import h5py


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
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2 + 1e-12))


def main():
    ap = argparse.ArgumentParser(description="Orient PSD data and save per-frequency NPZs.")
    ap.add_argument("--psd", type=Path, required=True)
    ap.add_argument("--freqs", type=float, nargs="+", required=True)
    ap.add_argument("--center", type=float, nargs=3, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    center = np.asarray(args.center, dtype=float)
    args.outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.psd, "r") as f:
        g = f["Process1"]
        freq_all = np.array(g["FreqVar"]).ravel()
        pos = np.array(g["datPostition"]).T
        for f0 in args.freqs:
            fi = int(np.argmin(np.abs(freq_all - f0)))
            P = g["TFpref1_i"][:, 0, fi] + 1j * g["TFpref1_j"][:, 0, fi]
            Ux = g["TFxyzref1_i"][:, 0, fi] + 1j * g["TFxyzref1_j"][:, 0, fi]
            Uy = g["TFxyzref1_i"][:, 1, fi] + 1j * g["TFxyzref1_j"][:, 1, fi]
            Uz = g["TFxyzref1_i"][:, 2, fi] + 1j * g["TFxyzref1_j"][:, 2, fi]
            U = np.stack([Ux, Uy, Uz], axis=1)

            pos_c = pos - center
            # floor normal from lowest 10% z
            zq = np.quantile(pos_c[:, 2], 0.1)
            sub = pos_c[pos_c[:, 2] <= zq]
            _, _, Vt = np.linalg.svd(sub - sub.mean(0), full_matrices=False)
            n_floor = Vt[-1] / np.linalg.norm(Vt[-1])
            Rz = rotation_from_vectors(n_floor, np.array([0.0, 0.0, 1.0]))
            pos1 = pos_c @ Rz.T
            U1 = U @ Rz.T

            # dominant active intensity direction
            J = 0.5 * P[:, None] * np.conj(U1)
            I = np.real(J)
            cov = I.T @ I / (len(I) - 1)
            w, v = np.linalg.eigh(cov)
            vdir = v[:, np.argmax(w)]
            proj = vdir - np.dot(vdir, np.array([0.0, 0.0, 1.0])) * np.array([0.0, 0.0, 1.0])
            if np.linalg.norm(proj) > 1e-9:
                Rint = rotation_from_vectors(proj, np.array([1.0, 0.0, 0.0]))
            else:
                Rint = np.eye(3)
            pos2 = pos1 @ Rint.T
            U2 = U1 @ Rint.T

            folder = args.outdir / f"oriented_{int(f0)}Hz"
            folder.mkdir(parents=True, exist_ok=True)
            np.savez(folder / f"martial_oriented_{int(f0)}Hz.npz",
                     pos=pos2, P=P, U=U2, R_floor=Rz, R_int=Rint, center=center, f0=f0)
            print("saved", folder)


if __name__ == "__main__":
    main()
