#!/usr/bin/env python3
"""
Quick inspection of a filled_arrays_* NPZ: real/imag ranges, min/median/max, and nonzero imaginary fraction.
"""
import argparse
import numpy as np
from pathlib import Path


def summarize_complex(arr, name: str):
    real = arr.real
    imag = arr.imag
    mag = np.abs(arr)
    lines = []
    lines.append(f"{name}: shape={arr.shape}")
    lines.append(f"  real min/median/max: {real.min():.3e}, {np.median(real):.3e}, {real.max():.3e}")
    lines.append(f"  imag min/median/max: {imag.min():.3e}, {np.median(imag):.3e}, {imag.max():.3e}")
    lines.append(f"  |.|   min/median/max: {mag.min():.3e}, {np.median(mag):.3e}, {mag.max():.3e}")
    nonzero = np.mean(np.abs(imag) > 1e-6)
    lines.append(f"  fraction |imag|>1e-6: {nonzero:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser(description="Inspect NPZ (complex stats for p/u).")
    ap.add_argument("--npz", type=Path, required=True)
    args = ap.parse_args()

    Z = np.load(args.npz, allow_pickle=True)
    lines = []
    lines += summarize_complex(Z["p"], "p")
    lines += summarize_complex(Z["u"], "u (vector flattened)")
    if "p_raw" in Z:
        lines += summarize_complex(Z["p_raw"], "p_raw")
    if "u_raw" in Z:
        lines += summarize_complex(Z["u_raw"], "u_raw (vector flattened)")
    if "meta" in Z:
        lines.append(f"meta: {Z['meta'].item()}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
