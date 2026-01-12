#!/usr/bin/env python3
"""
Plot magnitude and wrapped phase of a far-field lon/lat map.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mf_pipeline.mf_directivity import grid_lonlat


def main():
    ap = argparse.ArgumentParser(description="Plot magnitude and phase of *_map_lonlat.npz.")
    ap.add_argument("--map", type=Path, required=True, help="*_map_lonlat.npz from run_directivity.")
    ap.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    args = ap.parse_args()

    d = np.load(args.map, allow_pickle=True)
    lon = d["lon_deg"]; lat = d["lat_deg"]; V = d["V"]

    mag_db = 20*np.log10(np.maximum(np.abs(V), 1e-12))
    mag_db -= np.nanmax(mag_db)  # normalize to 0 dB
    phase = np.angle(V)

    fig, axs = plt.subplots(1, 2, figsize=(10,4), dpi=160)
    im0 = axs[0].imshow(mag_db, origin="lower",
                        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                        aspect="auto", cmap="plasma", vmin=-40, vmax=0)
    axs[0].set_title("Magnitude (dB, norm)"); axs[0].set_xlabel("Azimuth [deg]"); axs[0].set_ylabel("Elevation [deg]")
    fig.colorbar(im0, ax=axs[0], shrink=0.8, label="dB")

    im1 = axs[1].imshow(phase, origin="lower",
                        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                        aspect="auto", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axs[1].set_title("Wrapped phase"); axs[1].set_xlabel("Azimuth [deg]"); axs[1].set_ylabel("Elevation [deg]")
    fig.colorbar(im1, ax=axs[1], shrink=0.8, label="rad")

    fig.tight_layout()
    out = args.out or args.map.with_name(args.map.stem + "_phase.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
