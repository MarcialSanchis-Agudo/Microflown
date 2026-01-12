#!/usr/bin/env python3
"""Plot histograms of imaginary magnitudes for p and u components in a filled_arrays_* NPZ."""
import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="Imaginary-part histograms for p and u in NPZ.")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    ap.add_argument("--bins", type=int, default=50)
    args = ap.parse_args()

    Z = np.load(args.npz, allow_pickle=True)
    p = Z["p"]
    u = Z["u"]

    p_im = np.abs(np.imag(p)).ravel()
    u_im = np.abs(np.imag(u)).ravel()

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    axs[0].hist(p_im, bins=args.bins, log=True, color="C0")
    axs[0].set_title("|Im{P}|")
    axs[0].set_xlabel("Magnitude"); axs[0].set_ylabel("Count")

    axs[1].hist(u_im, bins=args.bins, log=True, color="C1")
    axs[1].set_title("|Im{U}| (all components)")
    axs[1].set_xlabel("Magnitude"); axs[1].set_ylabel("Count")

    fig.suptitle(f"Imaginary parts: {args.npz.name}", fontsize=12)
    fig.tight_layout()
    out = args.out or args.npz.with_name(args.npz.stem + "_imag_hist.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
