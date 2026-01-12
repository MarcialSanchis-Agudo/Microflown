#!/usr/bin/env python3
"""
Estimate forward directions from both near-field intensity and far-field peak,
and report the nearest near-field sample to each.
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


def unit_from_az_el(az_deg, el_deg):
    az = np.deg2rad(az_deg); el = np.deg2rad(el_deg)
    return np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])


def az_el_from_unit(v):
    v = v / (np.linalg.norm(v) + 1e-12)
    az = np.rad2deg(np.arctan2(v[1], v[0]))
    el = np.rad2deg(np.arcsin(v[2]))
    return az, el


def nearest_idx_to_dir(az_target, el_target, th, ph):
    az_all = ph
    el_all = np.pi/2 - th
    vt = unit_from_az_el(az_target, el_target)
    vn = np.vstack([np.cos(el_all)*np.cos(az_all),
                    np.cos(el_all)*np.sin(az_all),
                    np.sin(el_all)]).T  # (N,3)
    dots = vn @ vt
    idx = int(np.argmax(dots))
    best_az = np.rad2deg(ph[idx]); best_el = 90.0 - np.rad2deg(th[idx])
    return idx, best_az, best_el


def main():
    ap = argparse.ArgumentParser(description="Compare forward directions: near-field intensity vs far-field peak.")
    ap.add_argument("--npz", type=Path, required=True, help="Near-field filled_arrays_* NPZ.")
    ap.add_argument("--map", type=Path, required=True, help="Far-field *_map_lonlat.npz.")
    ap.add_argument("--center", type=float, nargs=3, default=None, help="Manual center dx dy dz [m].")
    ap.add_argument("--auto-center", action="store_true", help="Use center stored in near-field meta if present.")
    ap.add_argument("--p-cut", type=float, default=0.0,
                    help="Discard points with |p| below this percentile (0-100).")
    ap.add_argument("--u-cut", type=float, default=0.0,
                    help="Discard points with |U| below this percentile (0-100).")
    ap.add_argument("--weight-by-p", action="store_true",
                    help="Weight intensity average by |p| to emphasize stronger points.")
    args = ap.parse_args()

    Z = np.load(args.npz, allow_pickle=True)
    pos = Z["pos"]
    P = Z["p"]
    U = Z["u"]
    meta = Z["meta"].item() if "meta" in Z else {}

    center = np.zeros(3)
    if args.center is not None:
        center = np.array(args.center, float)
    elif args.auto_center:
        c_meta = meta.get("center", None)
        if c_meta is not None:
            center = np.array(c_meta, float)

    r, th, ph = angles_from_pos(pos, center)

    # Apply magnitude-based masking to reduce low-SNR/reactive influence
    mask = np.ones(len(P), dtype=bool)
    if args.p_cut > 0:
        thr = np.percentile(np.abs(P), args.p_cut)
        mask &= (np.abs(P) >= thr)
    if args.u_cut > 0:
        thr_u = np.percentile(np.linalg.norm(U, axis=1), args.u_cut)
        mask &= (np.linalg.norm(U, axis=1) >= thr_u)

    if not np.any(mask):
        raise SystemExit("All points masked; lower --p-cut/--u-cut.")

    P_use = P[mask]; U_use = U[mask]

    # Near-field active intensity direction (average Re{p u*}), optionally weighted by |p|
    weights = np.abs(P_use) if args.weight_by_p else 1.0
    i_vec = np.real(P_use[:, None] * np.conj(U_use)) * weights[:, None]
    i_avg = np.sum(i_vec, axis=0)
    az_i, el_i = az_el_from_unit(i_avg)

    # Far-field peak direction
    az_peak, el_peak = peak_direction_from_map(args.map)

    idx_i, az_i_nf, el_i_nf = nearest_idx_to_dir(az_i, el_i, th, ph)
    idx_pk, az_pk_nf, el_pk_nf = nearest_idx_to_dir(az_peak, el_peak, th, ph)

    lines = []
    lines.append(f"Center used: {center}")
    lines.append(f"Mask kept {mask.sum()}/{len(mask)} points (p_cut={args.p_cut}%, u_cut={args.u_cut}%).")
    lines.append(f"Near-field active-intensity direction: az={az_i:.2f}°, el={el_i:.2f}°")
    lines.append(f"  Nearest sample idx={idx_i}: az={az_i_nf:.2f}°, el={el_i_nf:.2f}°")
    lines.append(f"Far-field peak direction:           az={az_peak:.2f}°, el={el_peak:.2f}°")
    lines.append(f"  Nearest sample idx={idx_pk}: az={az_pk_nf:.2f}°, el={el_pk_nf:.2f}°")
    lines.append(f"Deltas (peak - intensity): d_az={az_peak-az_i:.2f}°, d_el={el_peak-el_i:.2f}°")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
