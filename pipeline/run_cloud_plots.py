#!/usr/bin/env python3
# mf_pipeline/run_cloud_plots.py
import argparse, sys, pathlib, csv
# Script-mode shim
if __package__ is None or __package__ == '':
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from mf_pipeline.mf_config import EXPORTS
from mf_pipeline.mf_directivity import load_filled_npz, choose_metric
from mf_pipeline.mf_plots import (
    plot_cloud_before_after, plot_hist_residuals,
    plot_delta_series, plot_raw_vs_filled
)

def main():
    ap = argparse.ArgumentParser(description="Point-cloud plots with noisy highlighting and per-point deltas")
    ap.add_argument("--npz", type=pathlib.Path, required=True, help="Path to .../exports/filled_arrays_noiseaware_{Hz}.npz")
    ap.add_argument("--metric", type=str, default="u_mag", help="u_mag | ux | uy | uz | p")
    args = ap.parse_args()

    # Output base in absolute EXPORTS
    stem = args.npz.stem
    base = (EXPORTS / f"{stem}_{args.metric}").resolve()

    # Load filled arrays
    pos, P_fill, U_fill, meta = load_filled_npz(args.npz)

    # Load optional raw + flags (for Δ plots and highlighting)
    Z = np.load(args.npz, allow_pickle=True)
    P_raw = Z["p_raw"] if "p_raw" in Z.files else None
    U_raw = Z["u_raw"] if "u_raw" in Z.files else None
    flags = Z["flags"].item() if "flags" in Z.files else {}
    f_p  = flags.get("p", None); f_ux = flags.get("ux", None)
    f_uy = flags.get("uy", None); f_uz = flags.get("uz", None)

    # Metric for filled/raw
    v_fill = choose_metric(P_fill, U_fill, which=args.metric)
    if P_raw is not None and U_raw is not None:
        v_raw = choose_metric(P_raw, U_raw, which=args.metric)
    else:
        print("[WARN] raw arrays not found in NPZ; using filled for both (Δ=0).")
        v_raw = v_fill.copy()

    # Mask for highlighting
    mask = None
    if args.metric == "p" and f_p is not None: mask = f_p
    if args.metric == "ux" and f_ux is not None: mask = f_ux
    if args.metric == "uy" and f_uy is not None: mask = f_uy
    if args.metric == "uz" and f_uz is not None: mask = f_uz
    if args.metric == "u_mag" and (f_ux is not None) and (f_uy is not None) and (f_uz is not None):
        mask = (f_ux | f_uy | f_uz)

    qty = {"u_mag":"|U|","ux":"|Ux|","uy":"|Uy|","uz":"|Uz|","p":"|P|"}[args.metric]
    dv = v_fill - v_raw

    # Save figures
    out_cloud = base.with_name(base.name + "_cloud_before_after.png")
    fig = plot_cloud_before_after(pos, v_raw, v_fill, mask_noisy=mask, quantity_label=qty)
    fig.savefig(out_cloud, dpi=200, bbox_inches="tight")

    out_hist = base.with_name(base.name + "_residual_hist.png")
    fig2 = plot_hist_residuals(v_raw, v_fill, mask_noisy=mask, quantity_label=qty)
    fig2.savefig(out_hist, dpi=200, bbox_inches="tight")

    out_delta = base.with_name(base.name + "_delta_series.png")
    fig3 = plot_delta_series(dv, mask_noisy=mask, quantity_label=qty)
    fig3.savefig(out_delta, dpi=200, bbox_inches="tight")

    out_scatt = base.with_name(base.name + "_raw_vs_filled.png")
    fig4 = plot_raw_vs_filled(v_raw, v_fill, quantity_label=qty)
    fig4.savefig(out_scatt, dpi=200, bbox_inches="tight")

    # CSV with per-point Δ
    out_csv = base.with_name(base.name + "_delta.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "raw", "filled", "delta", "abs_delta",
                    "is_noisy" if mask is not None else "is_noisy(n/a)"])
        for i in range(dv.size):
            is_noisy = bool(mask[i]) if mask is not None else ""
            w.writerow([i, float(v_raw[i]), float(v_fill[i]), float(dv[i]), float(abs(dv[i])), is_noisy])

    print("Saved:")
    print(" ", out_cloud)
    print(" ", out_hist)
    print(" ", out_delta)
    print(" ", out_scatt)
    print(" ", out_csv)

if __name__ == "__main__":
    main()
