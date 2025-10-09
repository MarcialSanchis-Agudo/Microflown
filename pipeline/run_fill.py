#!/usr/bin/env python3
# mf_pipeline/run_fill.py
import argparse, sys, pathlib
# Script-mode shim: allow "python mf_pipeline/run_fill.py"
if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from mf_pipeline.mf_fill import noiseaware_fill

def main():
    ap = argparse.ArgumentParser(description="Noise-aware fill for Microflown speaker scans")
    ap.add_argument("--f0", type=float, required=True, help="Target frequency in Hz (e.g., 1007.8)")
    ap.add_argument("--tau", type=float, default=3.5, help="Robust z-score threshold")
    ap.add_argument("--k-nn", dest="k_nn", type=int, default=12, help="Neighbors for local stats")
    ap.add_argument("--smooth", type=float, default=0.2, help="RBF smoothing")
    ap.add_argument("--tag", type=str, default="noiseaware", help="Output tag (default: noiseaware)")
    args = ap.parse_args()

    out = noiseaware_fill(args.f0, tau=args.tau, k_nn=args.k_nn, smooth=args.smooth, tag=args.tag)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
