#!/usr/bin/env python3
"""
Report the direction (az/elev) of a user-specified "on-axis" near-field point
and compare to the far-field peak direction. Can optionally auto-select the
near-field point whose direction is closest to the far-field peak.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np
from mf_pipeline.mf_directivity import angles_from_pos


def peak_direction_from_map(path: Path):
    d = np.load(path, allow_pickle=True)
    lon = d["lon_deg"]; lat = d["lat_deg"]; V = d["V"]
    mag = np.abs(V)
    idx = np.argmax(mag)
    i, j = np.unravel_index(idx, mag.shape)
    return float(lon[j]), float(lat[i])


def main():
    ap = argparse.ArgumentParser(description="Check alignment of an on-axis near-field point vs far-field peak.")
    ap.add_argument("--npz", type=Path, required=True, help="filled_arrays_* NPZ (near-field).")
    ap.add_argument("--map", type=Path, required=True, help="far-field *_map_lonlat.npz.")
    ap.add_argument("--index", type=int, default=0, help="Index of near-field sample expected to be on-axis.")
    ap.add_argument("--find-nearest", action="store_true",
                    help="Ignore --index and instead pick the near-field point whose direction is closest to the far-field peak.")
    ap.add_argument("--center", type=float, nargs=3, default=None, help="Manual center dx dy dz [m].")
    ap.add_argument("--auto-center", action="store_true", help="Use center stored in map meta if present.")
    args = ap.parse_args()

    Z = np.load(args.npz, allow_pickle=True)
    pos = Z["pos"]
    meta = Z["meta"].item() if "meta" in Z else {}

    center = np.zeros(3)
    if args.center is not None:
        center = np.array(args.center, float)
    elif args.auto_center:
        c_meta = meta.get("center", None)
        if c_meta is not None:
            center = np.array(c_meta, float)

    r, th, ph = angles_from_pos(pos, center)

    az_peak, el_peak = peak_direction_from_map(args.map)

    if args.find_nearest:
        # Convert peak direction to unit vector
        azp = np.deg2rad(az_peak)
        elp = np.deg2rad(el_peak)
        vp = np.array([np.cos(elp)*np.cos(azp), np.cos(elp)*np.sin(azp), np.sin(elp)])
        # Convert all near-field points to unit vectors
        az_all = ph
        el_all = np.pi/2 - th
        vn = np.vstack([np.cos(el_all)*np.cos(az_all),
                        np.cos(el_all)*np.sin(az_all),
                        np.sin(el_all)]).T  # (N,3)
        dots = vn @ vp
        idx = int(np.argmax(dots))
    else:
        if args.index < 0 or args.index >= len(pos):
            raise SystemExit(f"--index must be in [0, {len(pos)-1}]")
        idx = int(args.index)

    th_i, ph_i = th[idx], ph[idx]
    az_onaxis = np.rad2deg(ph_i)
    el_onaxis = 90.0 - np.rad2deg(th_i)

    print(f"Center used: {center}")
    print(f"Chosen near-field index {idx}: az={az_onaxis:.2f}°, el={el_onaxis:.2f}°")
    print(f"Far-field peak: az={az_peak:.2f}°, el={el_peak:.2f}°")
    print(f"Delta: d_az={az_peak-az_onaxis:.2f}°, d_el={el_peak-el_onaxis:.2f}°")


if __name__ == "__main__":
    main()
