#!/usr/bin/env python3
"""make_fig_directivity.py — Fig. 11: directivity of the loudspeaker in the room.
Row 1: directivity balloons (ICSV 2012 spherical projection, |p| r, shell 0.2-0.4 m, 30-deg bins,
       radius = linear pressure re maximum, colour dB 0 to -10; grey = fewer than 3 points) at 4 frequencies.
Row 2: polar cuts on the same shell (|elevation| < 20 deg / vertical x-z plane, 15-deg sectors,
       >= 5 points) at the 4 frequencies, and at 750 Hz the 0.8-1.2 m shell with the exterior
       SH prediction from 0.15-0.5 m.  Inputs: room_balloon.json, room_polar_<f>Hz.json."""
import sys, json, numpy as np
sys.path.insert(0, 'lib')
import paperstyle as S
import matplotlib, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
B = json.load(open('results/room_balloon.json')); ae = np.radians(B['az_edges']); ee = np.radians(B['el_edges'])   # 30-deg bins (verify_room_balloon.py 30)
FR = [492, 750, 1008, 1242]; CM = matplotlib.colormaps["turbo"]; FLR = -9.0
W = S.TEXTW; H = 5.4; fig = plt.figure(figsize=(W, H))
for k, f in enumerate(FR):
    dB = np.array([[np.nan if v is None else v for v in row] for row in B[str(f)]['dB']], float)
    rad = 10 ** (dB / 20)                                              # radius: linear pressure re maximum
    nl, na = dB.shape; RV = np.zeros((nl + 1, na + 1))
    for i in range(nl + 1):
        for j in range(na + 1):
            nb = [rad[ii, jj % na] for ii in (i - 1, i) for jj in (j - 1, j) if 0 <= ii < nl and np.isfinite(rad[ii, jj % na])]
            RV[i, j] = np.mean(nb) if nb else np.nanmean(rad)
    RV[0, :] = RV[0, :].mean(); RV[-1, :] = RV[-1, :].mean()          # single radius at each pole (no wedges)
    AZ, EL = np.meshgrid(ae, ee)
    X, Y, Zs = RV * np.cos(EL) * np.cos(AZ), RV * np.cos(EL) * np.sin(AZ), RV * np.sin(EL)
    fc = CM((np.clip(dB, FLR, 0) - FLR) / (-FLR)); fc[~np.isfinite(dB)] = (0.86, 0.86, 0.86, 1.0)
    a = fig.add_axes([0.01 + k * 0.245, 0.58, 0.24, 0.37], projection='3d')
    a.plot_surface(X, Y, Zs, facecolors=fc, rstride=1, cstride=1, linewidth=0.3, edgecolor=(0.25, 0.25, 0.25, 0.35), shade=False, antialiased=True)
    for v_, lab in (((1.15, 0, 0), '$x$'), ((0, 1.15, 0), '$y$'), ((0, 0, 1.08), '$z$')):
        a.plot([v_[0] * 0.87, v_[0]], [v_[1] * 0.87, v_[1]], [v_[2] * 0.87, v_[2]], color='k', lw=0.8); a.text(v_[0] * 1.07, v_[1] * 1.07, v_[2] * 1.07, lab, fontsize=7)
    a.set_xlim(-1, 1); a.set_ylim(-1, 1); a.set_zlim(-1, 1); a.set_box_aspect((1, 1, 1), zoom=1.5); a.view_init(elev=20, azim=125); a.set_axis_off()
    a.text2D(0.02, 1.06, f'({"abcd"[k]})', transform=a.transAxes, ha='left', va='top', fontweight='bold', fontsize=9)
    a.text2D(0.17, 1.055, f'{f} Hz', transform=a.transAxes, ha='left', va='top', fontsize=8.5)
cax = fig.add_axes([0.30, 0.575, 0.40, 0.016])
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(FLR, 0), cmap=CM), cax=cax, orientation='horizontal')
cb.set_ticks([-8, -6, -4, -2, 0]); cb.ax.tick_params(labelsize=7, length=2); cb.outline.set_linewidth(0.6)
cax.set_title('level re maximum [dB];  radius: linear pressure;  grey: fewer than 3 points', fontsize=7.5, pad=3)
# row 2: polars
J = {f: json.load(open(f'results/room_polar_{f}Hz.json')) for f in FR}; cen = np.radians(J[750]['bins_deg'])
def arrp(x): return np.array([np.nan if v is None else v for v in x], float)
def pax(xl, t, title):
    a = fig.add_axes([xl, 0.10, 0.25, 0.31], projection='polar'); a.set_ylim(-20, 2); a.set_yticks([-20, -15, -10, -5, 0])
    a.set_yticklabels(['', '$-15$', '$-10$', '$-5$', '0 dB'], fontsize=6); a.set_rlabel_position(112)
    a.set_xticks(np.radians(np.arange(0, 360, 45))); a.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'], fontsize=6.5)
    a.grid(color='0.82', lw=0.5); a.spines['polar'].set_linewidth(0.6)
    a.text(-0.20, 1.21, f'({t})', transform=a.transAxes, fontweight='bold', fontsize=9, va='top'); a.text(0.5, 1.21, title, transform=a.transAxes, ha='center', fontsize=7.5); return a
def closed(v):
    vv = np.append(v, v[0]); cc = np.append(cen, cen[0] + 2 * np.pi); return cc, np.maximum(vv, -20)
cols = {492: S.GREEN, 750: S.BLUE, 1008: S.ORANGE, 1242: S.RED}
for t, plane, title in (('e', 'horizontal', 'horizontal plane, 0.2–0.4 m'), ('f', 'vertical', 'vertical $x$–$z$ plane, 0.2–0.4 m')):
    a = pax(0.04 if t == 'e' else 0.37, t, title)
    for f in FR:
        cc, vv = closed(arrp(J[f][f'{plane}_inner'][0])); a.plot(cc, vv, '-', color=cols[f], lw=1.2, label=f'{f} Hz')
    if t == 'e': a.legend(frameon=False, fontsize=6.5, loc='upper left', bbox_to_anchor=(0.0, -0.13), ncol=4, columnspacing=0.8, handlelength=1.2)
a = pax(0.70, 'g', 'horizontal, 750 Hz: 0.2–0.4 m vs 0.8–1.2 m')
j = J[750]
v, lo, hi = (arrp(x) for x in j['horizontal_inner']); cc, vv = closed(v); a.plot(cc, vv, '-', color=S.BLUE, lw=1.2, label='0.2–0.4 m')
cc, l1 = closed(lo); cc, h1 = closed(hi); a.fill_between(cc, l1, np.minimum(h1, 2), color=S.BLUE, alpha=0.18, lw=0, label='90 % interval')
v, lo, hi = (arrp(x) for x in j['horizontal_outer']); ok = np.isfinite(v)
a.plot(cen[ok], np.maximum(v[ok], -20), 's', color=S.ORANGE, ms=3.5, label='0.8–1.2 m')
for th_, l_, h_ in zip(cen[ok], np.maximum(lo[ok], -20), np.minimum(hi[ok], 2)): a.plot([th_, th_], [l_, h_], color=S.ORANGE, lw=1.0)
pr_ = arrp(j['horizontal_pred_L2']); okp = np.isfinite(pr_); a.plot(cen[okp], np.maximum(pr_[okp], -20), 'x', color='k', ms=4.5, mew=1.0, label='exterior expansion')
a.legend(frameon=False, fontsize=6.3, loc='upper left', bbox_to_anchor=(-0.25, -0.12), ncol=2, columnspacing=0.6, handlelength=1.0)
fig.savefig('paperfigs/fig_polar.pdf', dpi=300); fig.savefig('results/fig_polar_preview.png', dpi=160); plt.close(fig)
print('wrote fig_polar (directivity)')
