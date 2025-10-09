# mf_pipeline/mf_plots.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def _scatter3(ax, P, c, title, highlight=None, s=18, cb_label="value"):
    sc = ax.scatter(P[:,0], P[:,1], P[:,2], c=c, s=s)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(title)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(cb_label)
    if highlight is not None and np.any(highlight):
        sel = np.flatnonzero(highlight)
        ax.scatter(P[sel,0], P[sel,1], P[sel,2], facecolors="none", edgecolors="k", s=80, linewidths=1.4)

def plot_cloud_before_after(P, v_raw, v_fill, mask_noisy=None, quantity_label="|U|"):
    fig = plt.figure(figsize=(14,4.5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    _scatter3(ax1, P, v_raw, f"Raw ({quantity_label})", highlight=mask_noisy, cb_label=quantity_label)
    _scatter3(ax2, P, v_fill, f"Filled ({quantity_label})", highlight=mask_noisy, cb_label=quantity_label)

    dv = v_fill - v_raw
    norm = TwoSlopeNorm(vcenter=0.0)
    sc = ax3.scatter(P[:,0], P[:,1], P[:,2], c=dv, s=18, norm=norm)
    ax3.set_title("Difference (filled - raw)")
    ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]"); ax3.set_zlabel("z [m]")
    cb = plt.colorbar(sc, ax=ax3, shrink=0.8, pad=0.02)
    cb.set_label(f"Delta {quantity_label}")
    fig.tight_layout()
    return fig

def plot_hist_residuals(v_raw, v_fill, mask_noisy=None, quantity_label="|U|"):
    dv = v_fill - v_raw
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.hist(dv, bins=40, alpha=0.7, label="All")
    if mask_noisy is not None and np.any(mask_noisy):
        ax.hist(dv[mask_noisy], bins=40, alpha=0.7, label="Noisy subset")
    ax.set_xlabel(f"Delta {quantity_label} (filled - raw)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_delta_series(dv, mask_noisy=None, quantity_label="|U|"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,3.6))
    n = dv.size
    idx = np.arange(n)
    ax1.plot(idx, dv, linewidth=1.0)
    ax1.set_title("Per-point Delta")
    ax1.set_xlabel("Sensor index")
    ax1.set_ylabel(f"Delta {quantity_label}")
    if mask_noisy is not None and np.any(mask_noisy):
        ax1.scatter(idx[mask_noisy], dv[mask_noisy], s=12, color="C3", label="Noisy")
        ax1.legend()

    mag = np.abs(dv)
    order = np.argsort(mag)
    ax2.plot(np.arange(n), mag[order], linewidth=1.0)
    ax2.set_title("Sorted |Delta|")
    ax2.set_xlabel("Rank")
    ax2.set_ylabel(f"|Delta {quantity_label}|")
    fig.tight_layout()
    return fig

def plot_raw_vs_filled(v_raw, v_fill, quantity_label="|U|"):
    fig, ax = plt.subplots(figsize=(4.8,4.6))
    ax.scatter(v_raw, v_fill, s=8, alpha=0.8)
    lims = [min(v_raw.min(), v_fill.min()), max(v_raw.max(), v_fill.max())]
    ax.plot(lims, lims, linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"Raw {quantity_label}")
    ax.set_ylabel(f"Filled {quantity_label}")
    ax.set_title("Raw vs Filled")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig
