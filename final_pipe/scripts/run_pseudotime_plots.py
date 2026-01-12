#!/usr/bin/env python3
"""
Pseudo-time Hovmöller + backflow plots using multiple frequency-domain NPZs.
This synthesizes time series by summing discrete complex tones.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from mf_pipeline.mf_config import EXPORTS

OUTDIR = Path(os.environ.get("MF_OUTDIR", EXPORTS)).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)


def parse_freq(path: Path, meta):
    if "f0" in meta:
        return float(meta["f0"])
    stem = path.stem
    for token in stem.split("_"):
        if token.endswith("Hz"):
            try:
                return float(token.replace("Hz", ""))
            except Exception:
                pass
    raise ValueError(f"Cannot infer frequency from {path}")


def main():
    ap = argparse.ArgumentParser(description="Pseudo-time Hovmöller + backflow from multi-frequency NPZs.")
    ap.add_argument("--npzs", nargs="+", required=True, help="List of NPZs at different frequencies.")
    ap.add_argument("--n-time", type=int, default=256)
    ap.add_argument("--n-r", type=int, default=40)
    ap.add_argument("--center", type=float, nargs=3, default=None)
    ap.add_argument("--tag", type=str, default="band")
    args = ap.parse_args()

    # Load all frequencies
    Ps = []
    Us = []
    freqs = []
    pos = None
    for p in args.npzs:
        Z = np.load(p, allow_pickle=True)
        meta = Z["meta"].item() if "meta" in Z else {}
        f0 = parse_freq(Path(p), meta)
        freqs.append(f0)
        Ps.append(Z["p"])
        Us.append(Z["u"])
        if pos is None:
            pos = Z["pos"]
        else:
            if Z["pos"].shape != pos.shape or np.max(np.abs(Z["pos"] - pos)) > 1e-6:
                raise SystemExit("Positions differ across NPZs; cannot synthesize pseudo-time.")

    freqs = np.array(freqs, float)
    order = np.argsort(freqs)
    freqs = freqs[order]
    Pk = np.stack(Ps, axis=0)[order]  # (F,N)
    Uk = np.stack(Us, axis=0)[order]  # (F,N,3)

    fmin = freqs.min()
    T = 1.0 / fmin
    t = np.linspace(0.0, T, args.n_time, endpoint=False)
    exp_mat = np.exp(1j * 2.0 * np.pi * t[:, None] * freqs[None, :])  # (T,F)

    # Time series from discrete tones (real signal)
    p_t = 2.0 * np.real(exp_mat @ Pk)            # (T,N)
    u_t = 2.0 * np.real(exp_mat @ Uk.reshape(len(freqs), -1)).reshape(args.n_time, -1, 3)

    center = np.array(args.center, float) if args.center is not None else np.zeros(3)
    r = np.linalg.norm(pos - center[None, :], axis=1)
    n = (pos - center[None, :]) / (r[:, None] + 1e-12)

    # Instantaneous intensity along radial normal
    un_t = np.sum(u_t * n[None, :, :], axis=2)   # (T,N)
    i_t = p_t * un_t                              # (T,N)

    # Hovmöller: bin by radius
    r_bins = np.linspace(r.min(), r.max(), args.n_r + 1)
    r_cent = 0.5 * (r_bins[:-1] + r_bins[1:])
    H = np.zeros((args.n_r, args.n_time))
    for i in range(args.n_r):
        m = (r >= r_bins[i]) & (r < r_bins[i+1])
        if np.any(m):
            H[i, :] = np.mean(i_t[:, m], axis=1)

    fig, ax = plt.subplots(figsize=(6,4), dpi=160)
    im = ax.imshow(H, origin="lower", aspect="auto",
                   extent=[t.min(), t.max(), r_cent.min(), r_cent.max()],
                   cmap="RdBu_r")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Radius r [m]")
    ax.set_title("Hovmöller: I(r,t) (pseudo-time)")
    fig.colorbar(im, ax=ax, label="I (arb.)")
    fig.tight_layout()
    out_hov = OUTDIR / f"hovmoller_{args.tag}.png"
    fig.savefig(out_hov, bbox_inches="tight")
    print(f"[save] {out_hov}")

    # Backflow time series at max |p|
    idx = int(np.argmax(np.mean(np.abs(Pk), axis=0)))
    series = i_t[:, idx]
    fig2, ax2 = plt.subplots(figsize=(6,3), dpi=160)
    ax2.plot(t, series, color="k", linewidth=1)
    ax2.fill_between(t, 0, series, where=series >= 0, color="green", alpha=0.3, label="outflow")
    ax2.fill_between(t, 0, series, where=series < 0, color="red", alpha=0.3, label="backflow")
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("P(t)·u(t)")
    ax2.set_title("Backflow time series (pseudo-time)")
    ax2.legend(frameon=False)
    fig2.tight_layout()
    out_back = OUTDIR / f"backflow_{args.tag}.png"
    fig2.savefig(out_back, bbox_inches="tight")
    print(f"[save] {out_back}")


if __name__ == "__main__":
    main()
