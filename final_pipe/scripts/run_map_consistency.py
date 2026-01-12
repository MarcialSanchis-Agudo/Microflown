#!/usr/bin/env python3
"""
Compute simple consistency stats across far-field maps:
  - Peak direction (az/elev) per map
  - Normalized correlation of magnitudes vs a reference map
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np


def load_map(path: Path):
    d = np.load(path, allow_pickle=True)
    return d["lon_deg"], d["lat_deg"], d["V"]


def peak_direction(lon_deg, lat_deg, V):
    mag = np.abs(V)
    idx = np.argmax(mag)
    i, j = np.unravel_index(idx, mag.shape)  # lat x lon
    return float(lon_deg[j]), float(lat_deg[i]), float(mag[i, j])


def mag_correlation(Vref, Vother):
    a = np.abs(Vref).ravel()
    b = np.abs(Vother).ravel()
    # normalize to unit norm to get cosine similarity
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def main():
    ap = argparse.ArgumentParser(description="Check peak direction and correlation across lon/lat maps.")
    ap.add_argument("--maps", nargs="+", required=True, help="Paths to *_map_lonlat.npz.")
    ap.add_argument("--labels", nargs="+", required=True, help="Labels same length as maps.")
    ap.add_argument("--out", type=Path, default=None, help="Optional text file to save the summary.")
    args = ap.parse_args()
    if len(args.maps) != len(args.labels):
        raise SystemExit("Number of --maps must equal number of --labels")

    lon_ref, lat_ref, Vref = load_map(Path(args.maps[0]))
    peak_ref = peak_direction(lon_ref, lat_ref, Vref)

    lines = []
    lines.append("# Map consistency summary")
    lines.append(f"Reference: {args.labels[0]}  peak az={peak_ref[0]:.2f}°, elev={peak_ref[1]:.2f}°")
    lines.append("label, peak_az_deg, peak_el_deg, corr_vs_ref")

    for path, lab in zip(args.maps, args.labels):
        lon, lat, V = load_map(Path(path))
        pk = peak_direction(lon, lat, V)
        corr = mag_correlation(Vref, V)
        lines.append(f"{lab}, {pk[0]:.3f}, {pk[1]:.3f}, {corr:.4f}")

    txt = "\n".join(lines)
    print(txt)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(txt)
        print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
