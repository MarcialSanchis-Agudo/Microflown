#!/usr/bin/env python3
"""
Inspect an H5 (ParaView export) for dataset names/shapes and report whether
pressure/velocity have separate real/imag datasets or only real-valued fields.
"""
import sys
from pathlib import Path
import argparse
import numpy as np
import h5py


def main():
    ap = argparse.ArgumentParser(description="Inspect H5 contents and basic ranges for pressure/velocity.")
    ap.add_argument("--h5", type=Path, required=True)
    ap.add_argument("--group", type=str, default=None,
                    help="Group to inspect (e.g., Proc-3_3D_Cuboid,25_mm_resolution). If omitted, list top-level.")
    args = ap.parse_args()

    h5_path = Path(args.h5)
    if not h5_path.exists():
        raise SystemExit(f"H5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if args.group is None:
            print("Top-level groups/datasets:")
            for k in f.keys():
                print(" ", k)
            return

        if args.group not in f:
            raise SystemExit(f"Group '{args.group}' not found. Available: {list(f.keys())}")

        g = f[args.group]
        print(f"Contents of group '{args.group}':")
        for k in g.keys():
            print(" ", k, g[k].shape, g[k].dtype)

        # Try to find pressure/velocity datasets
        if "PRESSURE" in g:
            P = np.array(g["PRESSURE"])
            print(f"\nPRESSURE: shape {P.shape}, dtype {P.dtype}, min/max {P.min():.3e}/{P.max():.3e}")
            print("  imag max", np.abs(np.imag(P)).max())
        if "VELOCITY" in g:
            U = np.array(g["VELOCITY"])
            print(f"VELOCITY: shape {U.shape}, dtype {U.dtype}, min/max {U.min():.3e}/{U.max():.3e}")
            print("  imag max", np.abs(np.imag(U)).max())
        for name in ("REAL_TFpref1","IMAG_TFpref1","REAL_TFxyzref1","IMAG_TFxyzref1"):
            if name in g:
                arr = np.array(g[name])
                print(f"{name}: shape {arr.shape}, dtype {arr.dtype}, min/max {arr.min():.3e}/{arr.max():.3e}")


if __name__ == "__main__":
    main()
