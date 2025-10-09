#!/usr/bin/env python3
import argparse, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Batch: fill + directivity for many frequencies")
    ap.add_argument("--freqs", type=float, nargs="+", required=True, help="List of frequencies in Hz")
    ap.add_argument("--metric", type=str, default="u_mag", help="u_mag | ux | uy | uz | p")
    args = ap.parse_args()

    for f0 in args.freqs:
        # 1) fill
        code = subprocess.call([sys.executable, "run_fill.py", "--f0", str(f0)])
        if code != 0:
            print(f"[WARN] fill failed for f0={f0}")
            continue
        npz = Path("exports") / f"filled_arrays_noiseaware_{int(round(f0))}Hz.npz"
        if not npz.exists():
            print(f"[WARN] NPZ not found: {npz}")
            continue
        # 2) directivity
        code = subprocess.call([sys.executable, "run_directivity.py", "--npz", str(npz), "--metric", args.metric])
        if code != 0:
            print(f"[WARN] directivity failed for {npz}")

if __name__ == "__main__":
    main()
