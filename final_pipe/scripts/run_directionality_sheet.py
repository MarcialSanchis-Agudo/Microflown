#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mf_pipeline.mf_config import EXPORTS

OUTDIR = Path(os.environ.get("MF_OUTDIR", EXPORTS)).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)

def load_map(path: Path):
    d = np.load(path, allow_pickle=True)
    return d["lon_deg"], d["lat_deg"], d["V"]

def nearest_idx(arr, val): return int(np.argmin(np.abs(arr - val)))

def extract_horizontal(lon_deg, lat_deg, V, elev_deg=0.0):
    j = nearest_idx(lat_deg, elev_deg); return lon_deg, V[j, :].ravel()

def extract_vertical(lon_deg, lat_deg, V, az_deg=0.0):
    i = nearest_idx(lon_deg, az_deg);   return lat_deg, V[:, i].ravel()

def to_db(y): return 20*np.log10(np.maximum(np.abs(y), 1e-12))

def pretty_polar(ax, title):
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(range(0, 360, 30))
    ax.set_rlabel_position(90)
    ax.set_rticks([0, -10, -20, -30, -40, -50])
    ax.set_rmin(-50); ax.set_rmax(0)
    ax.grid(True, alpha=0.25)
    ax.set_title(title, pad=14, fontsize=16, fontweight="bold")

def main():
    ap = argparse.ArgumentParser(description="Directionality Response sheet (multi-frequency or multi-radius).")
    ap.add_argument("--maps", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--elev", type=float, default=0.0)
    ap.add_argument("--az", type=float, default=0.0)
    ap.add_argument("--normalize", action="store_true", default=True,
                    help="Normalize each curve to its own 0 dB (default ON).")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.add_argument("--ref-first", action="store_true",
                    help="NEW: Normalize all curves to the FIRST curve's max (nice for distance sweeps).")
    ap.add_argument("--out", type=Path, default=Path(f"{OUTDIR}/directionality_response.png"))
    args = ap.parse_args()
    if len(args.maps) != len(args.labels):
        raise SystemExit("Number of --maps must equal number of --labels")

    fig = plt.figure(figsize=(12, 5), dpi=200)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.3], wspace=0.25)
    axL = fig.add_subplot(gs[0,0], projection="polar")
    axR = fig.add_subplot(gs[0,1], projection="polar")
    axLegend = fig.add_subplot(gs[0,2]); axLegend.axis("off")
    pretty_polar(axL, f"Horizontal (elev = {args.elev:.0f}°)")
    pretty_polar(axR, f"Vertical (az = {args.az:.0f}°)")

    handles = []
    # NEW: store ref max from first curve (horizontal) if requested
    ref_max = None

    for idx, (path, lab) in enumerate(zip(args.maps, args.labels)):
        lon_deg, lat_deg, V = load_map(Path(path))
        ang_h, y_h = extract_horizontal(lon_deg, lat_deg, V, elev_deg=args.elev)
        ang_v, y_v = extract_vertical (lon_deg, lat_deg, V, az_deg=args.az)

        r_h = to_db(y_h)
        r_v = to_db(y_v)

        if args.normalize:
            if args.ref_first:
                if ref_max is None:
                    ref_max = np.max(r_h)  # first curve horizontal max
                r_h = r_h - ref_max
                r_v = r_v - ref_max
            else:
                r_h = r_h - np.max(r_h)
                r_v = r_v - np.max(r_v)

        # full-circle vertical by mirroring
        ang_v_full = np.concatenate([ang_v, 180 - ang_v[-2:0:-1]])
        r_v_full   = np.concatenate([r_v,   r_v[-2:0:-1]])

        th_h = np.deg2rad(ang_h)
        th_v = np.deg2rad(ang_v_full)

        (hline,) = axL.plot(th_h, r_h, linewidth=2, label=f"{lab}")
        axR.plot(th_v, r_v_full, linewidth=2, color=hline.get_color(), label=f"{lab}")
        handles.append(hline)

    axLegend.legend(handles=handles, labels=list(args.labels),
                    loc="center left", frameon=False, fontsize=12)
    fig.suptitle("DIRECTIONALITY RESPONSE", x=0.52, y=1.02, fontsize=22, fontweight="bold")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[save] {args.out}")

if __name__ == "__main__":
    main()
