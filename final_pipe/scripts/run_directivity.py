#!/usr/bin/env python3
# mf_pipeline/run_directivity.py

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from mf_pipeline.mf_config import EXPORTS

from mf_pipeline.mf_directivity import (
    SHFitOptions,
    angles_from_pos,
    fit_complex_SH_nearfield,
    eval_on_lonlat_map,
    horizontal_polar_from_map,
    vertical_polar_from_map,
    auto_center_gridsearch,
)

OUTDIR = Path(os.environ.get("MF_OUTDIR", EXPORTS)).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)

def save_map_npz(out_path: Path, lon_deg, lat_deg, V):
    np.savez(out_path, lon_deg=lon_deg, lat_deg=lat_deg, V=V)

def plot_polar(angle_deg, values, title, out_png, db=False, normalize=True):
    if db:
        mag = 20*np.log10(np.maximum(np.abs(values), 1e-12))
        if normalize:
            mag -= mag.max()
        vals = mag
    else:
        vals = np.abs(values) / (np.max(np.abs(values)) + 1e-12) if normalize else np.abs(values)
    theta = np.deg2rad(angle_deg)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='polar')
    ax.plot(theta, vals)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def _angular_bin_median_complex(r, th, ph, p, bin_deg: float):
    """Median-bin complex p over (theta,phi) bins of size bin_deg (degrees)."""
    if bin_deg is None or bin_deg <= 0:
        return r, th, ph, p

    th_deg = np.rad2deg(th)                         # [0..180]
    ph_deg = (np.rad2deg(ph) + 360.0) % 360.0       # [0..360)
    b = float(bin_deg)
    th_bins = np.arange(0.0, 180.0 + b, b)
    ph_bins = np.arange(0.0, 360.0 + b, b)

    ith = np.clip(np.digitize(th_deg, th_bins) - 1, 0, len(th_bins) - 2)
    iph = np.clip(np.digitize(ph_deg, ph_bins) - 1, 0, len(ph_bins) - 2)
    key = ith * (len(ph_bins) - 1) + iph
    uniq = np.unique(key)

    rb, thb, phb, pb = [], [], [], []
    for k in uniq:
        m = (key == k)
        rb.append(np.median(r[m]))
        thb.append(np.deg2rad(np.median(th_deg[m])))
        phb.append(np.deg2rad(np.median(ph_deg[m])))
        pb.append(np.median(np.real(p[m])) + 1j*np.median(np.imag(p[m])))
    return np.array(rb), np.array(thb), np.array(phb), np.array(pb)

def main():
    ap = argparse.ArgumentParser(description="Complex SH near→far directivity")
    ap.add_argument("--npz", type=Path, required=True, help="filled_arrays_*.npz from run_fill.py")
    ap.add_argument("--field", type=str, default="complex_p", choices=["complex_p"])
    ap.add_argument("--lmax", type=int, default=10)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--c", type=float, default=343.0)
    ap.add_argument("--rfar", type=float, default=1.0, help="Radius to evaluate [m]")
    ap.add_argument("--db", action="store_true", help="Plot in dB (normalized unless --no-norm)")
    ap.add_argument("--res-lon", type=int, default=361)
    ap.add_argument("--res-lat", type=int, default=181)
    ap.add_argument("--center", type=float, nargs=3, default=None, help="Manual center shift [m]: dx dy dz")
    ap.add_argument("--auto-center", action="store_true", help="Search a few mm around (0,0,0)")
    ap.add_argument("--bin-deg", type=float, default=5.0, help="Angular median-binning size (deg). 0 disables.")
    ap.add_argument("--no-norm", action="store_true", help="Plot absolute dB (no per-curve normalization).")
    ap.add_argument("--tag", type=str, default="complex")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    pos  = data["pos"]
    meta = data["meta"].item()
    f0   = float(meta["f0"])
    k    = 2.0 * np.pi * f0 / args.c
    p    = data["p"]

    if args.center is not None:
        center = np.array(args.center, dtype=float)
    elif args.auto_center:
        center = auto_center_gridsearch(pos, p, f0, args.c, args.lmax, args.lam)
        print(f"[auto-center] chosen center (m): {center}")
    else:
        center = np.zeros(3)

    r, th, ph = angles_from_pos(pos, center)

    # median-binning before SH fit
    r_b, th_b, ph_b, p_b = _angular_bin_median_complex(r, th, ph, p, args.bin_deg)

    opt = SHFitOptions(lmax=args.lmax, lam=args.lam)
    a = fit_complex_SH_nearfield(p_b, r_b, th_b, ph_b, k, opt)

    lon_deg, lat_deg, V = eval_on_lonlat_map(a, k, args.rfar, args.lmax, args.res_lon, args.res_lat)

    base = f"filled_arrays_{args.tag}_{int(round(f0))}Hz_p"
    out_npz = OUTDIR / f"{base}_map_lonlat.npz"
    save_map_npz(out_npz, lon_deg, lat_deg, V)
    print(f"[save] {out_npz}")

    normalize = not args.no_norm
    lon_h, Vh = horizontal_polar_from_map(lon_deg, lat_deg, V, elev_deg=0.0)
    plot_polar(lon_h, Vh, f"Horizontal @ {f0:.0f} Hz (r={args.rfar} m)",
               OUTDIR / f"{base}_polar_horizontal_0deg.png", db=True, normalize=normalize)

    lat_v, Vv = vertical_polar_from_map(lon_deg, lat_deg, V, az_deg=0.0)
    plot_polar(lat_v, Vv, f"Vertical @ {f0:.0f} Hz (az=0°, r={args.rfar} m)",
               OUTDIR / f"{base}_polar_vertical_az0deg.png", db=True, normalize=normalize)
    print("[done]")

if __name__ == "__main__":
    main()
