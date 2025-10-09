#!/usr/bin/env python3
# mf_pipeline/run_directivity.py
import argparse, sys, pathlib
# Script-mode shim
if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from mf_pipeline.mf_config import EXPORTS
from mf_pipeline.mf_directivity import (
    load_filled_npz, choose_metric, fit_SH, eval_SH_map,
    eval_on_sphere, polar_cut_azimuth, polar_cut_elevation
)

def _to_db(x, floor_db=-60.0):
    x = np.abs(x)
    x = x / (x.max() + 1e-18)
    return 20.0 * np.log10(np.maximum(x, 10.0**(floor_db/20.0)))

def main():
    ap = argparse.ArgumentParser(description="Build directivity from filled NPZ")
    ap.add_argument("--npz", type=pathlib.Path, required=True, help="Path to .../exports/filled_arrays_noiseaware_{Hz}.npz")
    ap.add_argument("--metric", type=str, default="u_mag", help="u_mag | ux | uy | uz | p")
    ap.add_argument("--lmax", type=int, default=8, help="Spherical-harmonics degree")
    ap.add_argument("--lam", type=float, default=1e-3, help="Tikhonov regularization")
    ap.add_argument("--res_lon", type=int, default=361, help="Longitude samples")
    ap.add_argument("--res_lat", type=int, default=181, help="Latitude samples")
    ap.add_argument("--db", action="store_true", help="Plot/save in dB (normalized)")
    args = ap.parse_args()

    # Always save into absolute EXPORTS
    stem = args.npz.stem
    base = (EXPORTS / f"{stem}_{args.metric}").resolve()

    pos, P_fill, U_fill, meta = load_filled_npz(args.npz)
    v = choose_metric(P_fill, U_fill, which=args.metric)

    coeffs, info = fit_SH(pos, v, lmax=args.lmax, lam=args.lam, center=None)

    # Equirectangular map (lon/lat)
    lon, lat, Vmap = eval_SH_map(coeffs, lmax=info["lmax"], res=(args.res_lon, args.res_lat))
    Vplot = _to_db(Vmap) if args.db else Vmap
    vlabel = "dB (norm)" if args.db else "value"

    # Heatmap PNG
    plt.figure(figsize=(7.0, 3.2))
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    plt.imshow(Vplot, origin="lower", extent=extent, aspect="auto")
    plt.xlabel("Azimuth [deg]"); plt.ylabel("Elevation [deg]")
    ttl = f"Directivity @ {meta.get('f0','?'):.1f} Hz ({args.metric})"
    if args.db: ttl += " [dB]"
    plt.title(ttl)
    cb = plt.colorbar(); cb.set_label(vlabel)
    out_png_map = base.with_name(base.name + "_directivity.png")
    plt.tight_layout(); plt.savefig(out_png_map, dpi=180)
    print("Saved:", out_png_map)

    # Save equirect arrays
    out_npz_map = base.with_name(base.name + "_map_lonlat.npz")
    np.savez(out_npz_map, lon_deg=lon, lat_deg=lat, V=Vmap)
    print("Saved:", out_npz_map)

    # True (theta, phi) sphere grid
    Th, Ph, Vsph = eval_on_sphere(coeffs, lmax=info["lmax"], n_theta=args.res_lat, n_phi=args.res_lon)
    out_npz_sph = base.with_name(base.name + "_sphere_theta_phi.npz")
    np.savez(out_npz_sph, theta=Th, phi=Ph, V=Vsph)  # radians
    print("Saved:", out_npz_sph)

    # Horizontal polar (elev=0°)
    phi_deg, v_h = polar_cut_azimuth(coeffs, lmax=info["lmax"], elevation_deg=0.0, n=361)
    y_h = _to_db(v_h) if args.db else np.abs(v_h)
    out_png_h = base.with_name(base.name + "_polar_horizontal_0deg.png")
    plt.figure(figsize=(4.8, 4.8)); plt.polar(np.deg2rad(phi_deg), y_h)
    plt.title(f"Horizontal polar @ {meta.get('f0','?'):.1f} Hz" + (" [dB]" if args.db else ""))
    plt.tight_layout(); plt.savefig(out_png_h, dpi=180)
    print("Saved:", out_png_h)

    # Vertical polar (az=0°)
    elev_deg, v_v = polar_cut_elevation(coeffs, lmax=info["lmax"], azimuth_deg=0.0, n=181)
    y_v = _to_db(v_v) if args.db else np.abs(v_v)
    out_png_v = base.with_name(base.name + "_polar_vertical_az0deg.png")
    plt.figure(figsize=(5.8, 4.4)); plt.plot(elev_deg, y_v)
    plt.xlabel("Elevation [deg]"); plt.ylabel("dB (norm)" if args.db else "|value|")
    plt.title(f"Vertical cut (az=0°) @ {meta.get('f0','?'):.1f} Hz")
    plt.tight_layout(); plt.savefig(out_png_v, dpi=180)
    print("Saved:", out_png_v)

if __name__ == "__main__":
    main()
