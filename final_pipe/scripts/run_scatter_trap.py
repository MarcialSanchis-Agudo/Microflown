#!/usr/bin/env python3
"""
Scattering vs Trapping analysis from time-domain recordings (PRJ/H5).
Requires a config JSON that maps measurements to REF/CVX/CNV and distances.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import json
import numpy as np
import h5py
from scipy.signal import welch, resample_poly
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from mf_pipeline.mf_config import EXPORTS

OUTDIR = Path(os.environ.get("MF_OUTDIR", EXPORTS)).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_prj_measurement(prj_path: Path, meas_name: str):
    with h5py.File(prj_path, "r") as f:
        g = f[meas_name]
        x = np.array(g["TimeData"])  # (channels, samples)
        fs = float(g["daqinfo/ObjInfo"][0]["SampleRate"])
    return x, fs


def frac_delay_fft(x, fs, delay_s):
    # frequency-domain fractional delay
    n = x.size
    freqs = np.fft.rfftfreq(n, d=1/fs)
    X = np.fft.rfft(x)
    X *= np.exp(-1j * 2*np.pi * freqs * delay_s)
    return np.fft.irfft(X, n=n)


def align_to_ref(sig, ref, fs):
    # cross-correlation to estimate delay
    corr = np.correlate(sig, ref, mode="full")
    lag = np.argmax(corr) - (len(ref) - 1)
    delay = -lag / fs
    sig_aligned = frac_delay_fft(sig, fs, delay)
    return sig_aligned, delay


def window_energy(x, fs, t0, t1):
    i0 = int(max(0, t0 * fs))
    i1 = int(min(len(x), t1 * fs))
    if i1 <= i0:
        return 0.0
    return np.sum(x[i0:i1] ** 2) / fs


def exp_decay(t, a, tau):
    return a * np.exp(-t / tau)


def run_one_setup(data, fs, cfg, label):
    ref_ch = cfg["channels"].get("ref", None)
    p_ch = cfg["channels"]["p"]
    u_chs = cfg["channels"].get("u", [])

    sig = data[p_ch]
    ref = data[ref_ch] if ref_ch is not None else None

    if ref is not None:
        sig, delay = align_to_ref(sig, ref, fs)
    else:
        delay = 0.0

    return sig, ref, delay


def main():
    ap = argparse.ArgumentParser(description="Scattering vs trapping pipeline (time domain).")
    ap.add_argument("--config", type=Path, required=True, help="JSON config with measurement mapping.")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    outdir = args.outdir or OUTDIR / cfg.get("tag", "scatter_trap")
    outdir.mkdir(parents=True, exist_ok=True)

    prj = Path(cfg["prj"])
    c = cfg.get("c", 343.0)
    r1 = cfg["distances"]["r1"]
    r2 = cfg["distances"]["r2"]

    windows = cfg["windows"]
    t_pad = windows["t_pad"]
    t_direct = windows["t_direct"]
    t_late_start = windows["t_late_start"]
    t_late_end = windows["t_late_end"]

    results = {}

    for setup in ["REF", "CVX", "CNV"]:
        results[setup] = {}
        for rlab in ["r1", "r2"]:
            meas = cfg["setups"][setup][rlab]
            data, fs = load_prj_measurement(prj, meas)
            sig, ref, delay = run_one_setup(data, fs, cfg, setup)
            results[setup][rlab] = dict(sig=sig, ref=ref, fs=fs, delay=delay)

    # Distance law check (direct RMS)
    def direct_rms(x, fs, r):
        t0 = r / c
        t1 = t0 + t_pad
        t2 = t0 + t_direct
        seg = x[int(t1*fs):int(t2*fs)]
        return np.sqrt(np.mean(seg**2) + 1e-12)

    dl = {}
    for setup in ["REF", "CVX", "CNV"]:
        x1 = results[setup]["r1"]["sig"]; fs1 = results[setup]["r1"]["fs"]
        x2 = results[setup]["r2"]["sig"]; fs2 = results[setup]["r2"]["fs"]
        if fs1 != fs2:
            x2 = resample_poly(x2, int(fs1), int(fs2)); fs2 = fs1
        rms1 = direct_rms(x1, fs1, r1)
        rms2 = direct_rms(x2, fs2, r2)
        dl[setup] = 20*np.log10(rms1 / rms2)

    # Scattered field PSD
    def scattered_psd(obj, ref, fs):
        if ref is None:
            return None
        p_scat = obj - ref
        f, Pxx = welch(p_scat, fs=fs, nperseg=4096)
        return f, Pxx

    psd = {}
    for setup in ["CVX", "CNV"]:
        psd[setup] = {}
        for rlab in ["r1", "r2"]:
            obj = results[setup][rlab]["sig"]
            ref = results["REF"][rlab]["sig"]
            fs = results[setup][rlab]["fs"]
            f, Pxx = scattered_psd(obj, ref, fs)
            psd[setup][rlab] = (f, Pxx)

    # Gain map G(f) = 10log10(Scat_CNV / Scat_CVX)
    gain = {}
    for rlab in ["r1", "r2"]:
        f = psd["CNV"][rlab][0]
        g = 10*np.log10(np.maximum(psd["CNV"][rlab][1], 1e-30) /
                        np.maximum(psd["CVX"][rlab][1], 1e-30))
        gain[rlab] = (f, g)

    # Late energy + decay
    late = {}
    for setup in ["CVX", "CNV"]:
        late[setup] = {}
        for rlab in ["r1", "r2"]:
            x = results[setup][rlab]["sig"]
            fs = results[setup][rlab]["fs"]
            t0 = (r1 if rlab == "r1" else r2) / c
            e = window_energy(x, fs, t0 + t_late_start, t0 + t_late_end)
            late[setup][rlab] = e

    # Save plots
    # Distance law
    fig, ax = plt.subplots(figsize=(4,3), dpi=160)
    ax.bar(["REF","CVX","CNV"], [dl["REF"], dl["CVX"], dl["CNV"]])
    ax.axhline(20*np.log10(r2/r1), color="k", linestyle="--", label="expected")
    ax.set_ylabel("ΔL (dB)")
    ax.set_title("Distance-law (direct window)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "distance_law.png", bbox_inches="tight")

    # Gain map
    for rlab in ["r1", "r2"]:
        f, g = gain[rlab]
        fig, ax = plt.subplots(figsize=(5,3), dpi=160)
        ax.plot(f, g)
        ax.set_xlabel("Hz"); ax.set_ylabel("G(f) [dB]")
        ax.set_title(f"Gain map CNV/CVX @ {rlab}")
        fig.tight_layout()
        fig.savefig(outdir / f"gain_map_{rlab}.png", bbox_inches="tight")

    # Late energy ratio
    fig, ax = plt.subplots(figsize=(4,3), dpi=160)
    ratio = late["CNV"]["r1"] / (late["CVX"]["r1"] + 1e-12)
    ax.bar(["r1"], [ratio])
    ax.set_ylabel("E_late CNV/CVX")
    ax.set_title("Late energy ratio")
    fig.tight_layout()
    fig.savefig(outdir / "late_energy_ratio.png", bbox_inches="tight")

    # Save numeric summary
    summary = dict(distance_law=dl, late_energy=late)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] {outdir}")


if __name__ == "__main__":
    main()
