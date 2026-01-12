#!/usr/bin/env python3
"""
Plot phase map with magnitude mask for readability.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="Phase map with magnitude mask.")
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mag-cut", type=float, default=5.0, help="Percentile for magnitude mask.")
    args = ap.parse_args()

    d = np.load(args.map, allow_pickle=True)
    lon = d["lon_deg"]
    lat = d["lat_deg"]
    V = d["V"]
    mag = np.abs(V)
    ph = np.angle(V)
    thr = np.percentile(mag, args.mag_cut)
    ph_masked = np.where(mag >= thr, ph, np.nan)

    Lon, Lat = np.meshgrid(lon, lat)
    plt.figure(figsize=(8, 4), dpi=160)
    plt.contourf(Lon, Lat, ph_masked, levels=24, cmap="twilight")
    plt.xlabel("azimuth [deg]")
    plt.ylabel("elevation [deg]")
    plt.title("Phase map (masked)")
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")


if __name__ == "__main__":
    main()
