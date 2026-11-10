#!/usr/bin/env python3
"""
make_fig_room.py — the room-scan figures (750 Hz unless stated), house style.

  paperfigs/fig_room_noise.pdf   noise diagnosis: dwell map, noise vs dwell, flags vs coherence
  paperfigs/fig_room_recon.pdf   raw vs reconstructed pressure, Euler consistency
  paperfigs/fig_room_flux.pdf    energy flux and source localisation
  paperfigs/fig_room_shadow.pdf  line-of-sight geometry and the shadow test

Inputs are produced and checked by the scripts in results/ (VERIFICATION.md, section H):
room_pipeline_750Hz.npz (results/run_room_pipeline.py).
Every number printed here is one the text quotes.  Run from the repo root.
"""
import sys
import numpy as np
sys.path.insert(0, 'lib')
import paperstyle as S
import matplotlib.pyplot as plt
from room_raw import raw
from room_stage2 import dwell_blocks, local_fill
from scipy_free_interp import idw_grid, gauss_smooth, knn

K = 2 * np.pi * 750.0 / 343.0
FLOOR = -1.369                                   # floor height in the source frame (mesh)
Z = np.load('results/room_pipeline_750Hz.npz')
S2 = np.load('results/room_pipeline_750Hz.npz')
pos = Z['pos']; r = np.linalg.norm(pos, axis=1)
Rw = raw(750.0); nb = dwell_blocks(Rw['pos'])     # raw frame, same node order
zc = pos[:, 2]; hsl = np.abs(zc - np.median(zc)) < 0.18 * np.ptp(zc)
XL, YL = (-2.80, 0.55), (-0.62, 1.32)


def flux(p, u):
    return -0.5 * np.real(p[:, None] * np.conj(u))


# ════════════════════════════════════════════════════════════ Fig: noise diagnosis
q = Rw['p_ref1'] / Rw['p_ref2']; qm = np.median(q.real) + 1j * np.median(q.imag); scat = np.abs(q / qm - 1)
g2r = Rw['coh_r1r2']
edges = np.percentile(nb, [0, 20, 40, 60, 80, 100])
eH = Z['node_err']; H = np.arange(len(pos))          # leave-one-out node error, all nodes (run_room_pipeline.py)
mid, s_obs, e_obs, g_med = [], [], [], []
for a, b in zip(edges[:-1], edges[1:]):
    m = (nb >= a) & (nb <= b); mh = (nb[H] >= a) & (nb[H] <= b)
    mid.append(np.median(nb[m])); s_obs.append(np.median(scat[m])); e_obs.append(np.median(eH[mh])); g_med.append(np.median(g2r[m]))
mid, s_obs, e_obs = map(np.array, (mid, s_obs, e_obs))
print("[noise] dwell quintile medians (blocks):", np.round(mid, 1))
print("[noise] ref-pair scatter:", np.round(s_obs, 3), " leave-one-out node error:", np.round(e_obs, 2), " coh(r1,r2):", np.round(g_med, 3))
cp = S2['coh']; fl = S2['flag_p']
print(f"[noise] coh(p,ref1) median flagged {np.median(cp[fl]):.3f} vs kept {np.median(cp[~fl]):.3f}; "
      f"single-block cells with coh == 1: {np.mean(np.isclose(Rw['coh_r1r2'][nb < 1.5], 1.0)) * 100:.0f}%")

R = S.Row(W=S.TEXTW, H=2.36, left=0.58, bottom=0.48)
a = R.add_map(XL, YL, 2.0)
o = np.argsort(nb)
sc = a.scatter(pos[o, 0], pos[o, 1], c=np.log10(nb[o] * 2048 / 48000 * 1000), cmap=S.SEQ, s=1.2, lw=0, rasterized=True)
S.source(a); a.set_xlabel('$x$ [m]'); a.set_ylabel('$y$ [m]'); S.tag(a, 'a')
S.cbar(R.fig, sc, R.add_cbar(), r'dwell [ms]', ticks=[np.log10(v) for v in (50, 100, 200, 500)]).set_ticklabels(['50', '100', '200', '500'])
R.gap(0.95)
b = R.add_plot(1.85)
nn = np.logspace(0, np.log10(40), 60); gg = np.median(g2r[nb >= 1.5])
nn = np.logspace(0, np.log10(12), 60)
b.loglog(nn, 100 * 1.177 * np.sqrt((1 - gg) / (nn * gg)), color='0.35', lw=1.0, ls='--', label='Eq. (6), ratio')
b.loglog(mid[1:], 100 * s_obs[1:], 'o', color=S.BLUE, ms=4.5, label='reference pair')
b.loglog(mid[:1], 100 * s_obs[:1], 'o', mfc='white', mec=S.BLUE, ms=4.5, label='ref. pair, ~1 block')
b.loglog(mid, 100 * e_obs, 's', color=S.ORANGE, ms=4.5, label='probe pressure')
b.set_xlim(0.9, 12); b.set_ylim(1.5, 200); b.set_xlabel('FFT blocks per cell'); b.set_ylabel('median relative error [%]')
b.grid(color='0.9', lw=0.5, which='major'); b.legend(frameon=False, loc='lower center', fontsize=6.0, ncol=2, columnspacing=0.5, handlelength=1.0, handletextpad=0.4); S.tag(b, 'b', x=0.84)
R.gap(0.66)
c = R.add_plot(R.W - R.x - 0.08)
c.hist(nb, bins=np.logspace(0, np.log10(60), 30), color=S.GREY)
c.set_xscale('log'); c.axvline(np.median(nb), color='k', lw=1.0)
for nreq, lab_ in ((7.7, r'$\gamma^2{=}0.9$'), (29.7, r'$\gamma^2{=}0.7$')):
    c.axvline(nreq, color=S.RED, lw=0.9, ls='--'); c.text(nreq * 0.92, c.get_ylim()[1] * 0.97, lab_, fontsize=7.5, color=S.RED, va='top', ha='right', rotation=90)
c.set_xlabel('FFT blocks per cell'); c.set_ylabel('number of cells'); S.tag(c, 'c', x=0.80)
print(f"[noise] cells with < 1.5 blocks: {np.mean(nb < 1.5) * 100:.0f}%; >= 7.7 blocks: {np.mean(nb >= 7.7) * 100:.1f}%; >= 29.7: {np.mean(nb >= 29.7) * 100:.1f}%")
R.save('paperfigs/fig_room_noise.pdf', 'results/fig_room_noise_preview.png')

# ════════════════════════════════════════════════════════════ Fig: reconstruction
lr, lc = np.log10(np.abs(Z['p_raw'])), np.log10(np.abs(Z['p']))
vmin, vmax = np.percentile(lc[hsl], [2, 99.5])
from numpy.linalg import lstsq
om = 2 * np.pi * 750; sel = np.random.default_rng(0).choice(len(pos), 1500, replace=False)
EU = {}
for lab, p, u in (('raw', Z['p_raw'], Z['u_raw']), ('rec', Z['p'], Z['u'])):
    cs = []
    for i in sel:
        d = np.linalg.norm(pos - pos[i], axis=1); nbh = np.where((d > 0) & (d < 0.08))[0]
        if len(nbh) < 8: continue
        X = np.column_stack([np.ones(len(nbh)), pos[nbh] - pos[i]]); g = lstsq(X, p[nbh], rcond=None)[0][1:]
        ue = g / (1j * om * 1.204); cs.append(np.real(np.vdot(ue, u[i])) / (np.linalg.norm(ue) * np.linalg.norm(u[i]) + 1e-30))
    EU[lab] = np.array(cs)
print(f"[recon] Euler corr median raw {np.median(EU['raw']):+.2f}, reconstructed {np.median(EU['rec']):+.2f}")
R = S.Row(W=S.TEXTW, H=2.30, left=0.58, bottom=0.42)
for j, (L, t) in enumerate(((lr, 'a'), (lc, 'b'))):
    ax = R.add_map(XL, YL, 2.0); o = np.argsort(L[hsl]); P_ = pos[hsl][o]
    m_ = ax.scatter(P_[:, 0], P_[:, 1], c=L[hsl][o], cmap=S.SEQ, vmin=vmin, vmax=vmax, s=1.3, lw=0, rasterized=True)
    S.source(ax); ax.set_xlabel('$x$ [m]'); S.tag(ax, t)
    if j == 0: ax.set_ylabel('$y$ [m]'); R.gap(0.18)
    else: ax.set_yticklabels([])
S.cbar(R.fig, m_, R.add_cbar(), r'$\log_{10}|p|$')
R.gap(0.62)
c = R.add_plot(R.W - R.x - 0.08)
bins = np.linspace(-1, 1, 41)
c.hist(EU['raw'], bins=bins, density=True, color=S.GREY, label='raw')
c.hist(EU['rec'], bins=bins, density=True, histtype='step', color=S.ORANGE, lw=1.4, label='reconstructed')
c.set_xlabel(r'corr$(\mathbf{u},\,-\nabla p/\mathrm{i}\omega\rho_0)$'); c.set_ylabel('density'); c.set_xlim(-1, 1)
c.legend(frameon=False, loc='upper left', fontsize=7, bbox_to_anchor=(0.0, 0.88)); S.tag(c, 'c')
R.save('paperfigs/fig_room_recon.pdf', 'results/fig_room_recon_preview.png')

# ════════════════════════════════════════════════════════════ Fig: flux and localisation
def xstar(P, D, w):
    A = np.zeros((3, 3)); bb = np.zeros(3)
    for x, d, wi in zip(P, D, w):
        M = np.eye(3) - np.outer(d, d); A += wi * M; bb += wi * M @ x
    return np.linalg.solve(A, bb)
I = flux(Z['p'], Z['u']); Im = np.linalg.norm(I, axis=1); Dn = I / np.maximum(Im, 1e-30)[:, None]
far = r > 0.2
xs = xstar(pos[far], Dn[far], Im[far]); xu = xstar(pos, Dn, np.ones(len(pos)))
out = np.sum(Dn * pos / np.maximum(r, 1e-9)[:, None], 1) > 0
print(f"[flux] outward {out.mean() * 100:.0f}% of nodes; x* (|I|-weighted, r > 0.2 m) = {np.round(xs, 3)} ({np.linalg.norm(xs) * 100:.0f} cm); "
      f"unweighted {np.round(xu, 2)} ({np.linalg.norm(xu) * 100:.0f} cm)")
gx = np.linspace(pos[:, 0].min(), pos[:, 0].max(), 220); gy = np.linspace(pos[:, 1].min(), pos[:, 1].max(), 160)
Ix = idw_grid(pos[hsl, :2], I[hsl, 0], gx, gy, maxdist=0.10); Iy = idw_grid(pos[hsl, :2], I[hsl, 1], gx, gy, maxdist=0.10)
LI = 10 * np.log10(np.hypot(Ix, Iy))
vs = np.abs(pos[:, 1] - 0.1) < 0.15; gz = np.linspace(-0.92, 0.72, 120)
Jx = idw_grid(pos[vs][:, [0, 2]], I[vs, 0], gx, gz, maxdist=0.10); Jz = idw_grid(pos[vs][:, [0, 2]], I[vs, 2], gx, gz, maxdist=0.10)
LJ = 10 * np.log10(np.hypot(Jx, Jz)); Lmax = np.nanmax(np.concatenate([LI[np.isfinite(LI)], LJ[np.isfinite(LJ)]])); LI -= Lmax; LJ -= Lmax
lv = (-35, 0)
m = (r > 0.15) & (r < 2.6); sr = np.polyfit(np.log(r[m]), np.log(np.abs(Z['p_raw'][m])), 1); sc_ = np.polyfit(np.log(r[m]), np.log(np.abs(Z['p'][m])), 1)
print(f"[flux] |p| ~ r^{sr[0]:.2f} (raw), r^{sc_[0]:.2f} (reconstructed)")
R = S.Row(W=S.TEXTW, H=2.0, left=0.58, bottom=0.42)
a = R.add_map(XL, YL, 1.9)
pm = a.pcolormesh(gx, gy, np.ma.masked_invalid(LI), cmap=S.SEQ, vmin=lv[0], vmax=lv[1], shading='auto', rasterized=True)
a.streamplot(gx, gy, np.ma.masked_invalid(Ix), np.ma.masked_invalid(Iy), color='white', density=1.2, linewidth=0.5, arrowsize=0.55)
S.source(a)

a.set_xlabel('$x$ [m]'); a.set_ylabel('$y$ [m]'); S.tag(a, 'a')
R.gap(0.66)
bz = R.add_map(XL, (-0.92, 0.72), 1.9)
bz.pcolormesh(gx, gz, np.ma.masked_invalid(LJ), cmap=S.SEQ, vmin=lv[0], vmax=lv[1], shading='auto', rasterized=True)
bz.streamplot(gx, gz, np.ma.masked_invalid(Jx), np.ma.masked_invalid(Jz), color='white', density=1.0, linewidth=0.5, arrowsize=0.55)
S.source(bz)
bz.set_xlabel('$x$ [m]'); bz.set_ylabel('$z$ [m]'); S.tag(bz, 'b')
cb = S.cbar(R.fig, pm, R.add_cbar(), r'$|\mathbf{I}|$ [dB re max]', ticks=[-30, -20, -10, 0])
R.gap(0.80)
c = R.add_plot(R.W - R.x - 0.08)
ss = np.random.default_rng(1).choice(len(r), 2500, replace=False)
c.loglog(r[ss], np.abs(Z['p_raw'][ss]), '.', ms=1.4, color=S.GREY, rasterized=True, label='raw')
c.loglog(r[ss], np.abs(Z['p'][ss]), '.', ms=1.4, color=S.ORANGE, rasterized=True, label='reconstructed')
rr = np.logspace(np.log10(0.15), np.log10(2.6), 20); c.loglog(rr, np.exp(np.polyval(sc_, np.log(rr))), color='k', lw=1.0, label=f'$r^{{{sc_[0]:.2f}}}$')
c.set_xlabel('$r$ [m]'); c.set_ylabel('$|p|$ [relative]'); c.grid(color='0.9', lw=0.5, which='major')
c.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.9, loc='lower left', fontsize=6.5, markerscale=5); S.tag(c, 'c', x=0.72)
R.save('paperfigs/fig_room_flux.pdf', 'results/fig_room_flux_preview.png')

print("wrote fig_room_noise/recon/flux/shadow")
