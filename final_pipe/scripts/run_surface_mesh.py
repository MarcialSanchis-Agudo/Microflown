#!/usr/bin/env python3
# Render far-field maps as 3D spherical surfaces.
# Inputs are ..._p_map_lonlat.npz that contain lon_deg, lat_deg, V (complex values on sphere).
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def load_map(path):
    d = np.load(path, allow_pickle=True)
    lon = d["lon_deg"]
    lat = d["lat_deg"]
    V   = d["V"]
    # Ensure 2D arrays (lat x lon)
    if V.ndim == 1:
        # fallback (shouldn't happen with our pipeline)
        V = V.reshape(lat.size, lon.size)
    return lon, lat, V

def complex_to_db(V, mode="norm", vmin=None, vmax=None):
    """Return dB field and (vmin, vmax) for color scaling."""
    mag = np.abs(V) + 1e-12
    if mode == "norm":
        ref = np.max(mag)
        db = 20.0 * np.log10(mag / ref)
        if vmin is None: vmin = -30.0
        if vmax is None: vmax = 0.0
    elif mode == "abs":
        db = 20.0 * np.log10(mag)
        if vmin is None: vmin = float(np.nanpercentile(db, 1.0))
        if vmax is None: vmax = float(np.nanpercentile(db, 99.0))
    else:
        raise ValueError("mode must be 'norm' or 'abs'")
    return db, (vmin, vmax)

def lonlat_to_xyz(lon_deg, lat_deg, r=1.0):
    """Create unit-sphere XYZ grids from lon/lat degrees (meshgrid)."""
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    Lon, Lat = np.meshgrid(lon, lat)  # (lat x lon)
    X = r * np.cos(Lat) * np.cos(Lon)
    Y = r * np.cos(Lat) * np.sin(Lon)
    Z = r * np.sin(Lat)
    return X, Y, Z

def main():
    ap = argparse.ArgumentParser(description="Plot spherical surface maps from *_map_lonlat.npz.")
    ap.add_argument("--maps", type=Path, nargs="+", required=True,
                    help="Paths to *_map_lonlat.npz files (lon_deg, lat_deg, V).")
    ap.add_argument("--titles", type=str, nargs="*", default=None,
                    help="Titles per panel (same length as --maps).")
    ap.add_argument("--mode", choices=["norm","abs"], default="norm",
                    help="'norm' = normalize each map to 0 dB; 'abs' = absolute dB.")
    ap.add_argument("--vmin", type=float, default=None, help="Color scale min (dB).")
    ap.add_argument("--vmax", type=float, default=None, help="Color scale max (dB).")
    ap.add_argument("--cols", type=int, default=3, help="Number of columns.")
    ap.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    ap.add_argument("--outdir", type=Path, default=Path("figs") / "paper_figs",
                    help="Directory for the figure if --out is not given.")
    # Colorbar placement (figure coords)
    ap.add_argument("--cb-right", type=float, default=0.90, help="Colorbar left position")
    ap.add_argument("--cb-width", type=float, default=0.02, help="Colorbar width")
    ap.add_argument("--cb-bottom", type=float, default=0.18, help="Colorbar bottom")
    ap.add_argument("--cb-height", type=float, default=0.64, help="Colorbar height")
    args = ap.parse_args()

    n = len(args.maps)
    cols = max(1, args.cols)
    rows = int(np.ceil(n / cols))

    # Default titles = stems
    if args.titles is None or len(args.titles) != n:
        titles = [p.stem.replace("_", " ") for p in args.maps]
    else:
        titles = args.titles

    # Prepare figure
    fig = plt.figure(figsize=(4.0*cols, 4.0*rows))
    # leave room on the right for colorbar
    fig.subplots_adjust(right=min(0.88, args.cb_right - 0.02), wspace=0.05, hspace=0.1)

    mappables = []
    for i, (p, ttl) in enumerate(zip(args.maps, titles), start=1):
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        lon, lat, V = load_map(p)
        db, (vmin, vmax) = complex_to_db(V, mode=args.mode, vmin=args.vmin, vmax=args.vmax)
        X, Y, Z = lonlat_to_xyz(lon, lat, r=1.0)

        # Color mapping
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        facecolors = cm.plasma(norm(db))

        # Reduce wire clutter
        ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
                        facecolors=facecolors, linewidth=0, antialiased=False, shade=False)

        # minimal look
        ax.set_axis_off()
        ax.set_title(ttl, pad=8)

        # match a sensible view
        ax.view_init(elev=20, azim=-60)

        # store a mappable for the colorbar
        # create a ScalarMappable with the same norm/cmap
        m = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
        m.set_array([])
        mappables.append(m)

    # Colorbar at far right
    if mappables:
        cax = fig.add_axes([args.cb_right, args.cb_bottom, args.cb_width, args.cb_height])
        cb = fig.colorbar(mappables[-1], cax=cax)
        cb.set_label("dB" if args.mode == "abs" else "dB (normalized)", rotation=90)

    # Output
    if args.out is None:
        args.outdir.mkdir(parents=True, exist_ok=True)
        out = args.outdir / ("surfaces_panel_abs.png" if args.mode=="abs" else "surfaces_panel_norm.png")
    else:
        out = args.out
        out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[save] {out}")

if __name__ == "__main__":
    main()
