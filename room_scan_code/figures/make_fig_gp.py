#!/usr/bin/env python3
"""make_fig_gp.py — headline figure: the joint pressure-velocity Gaussian process at 750 Hz predicted on
continuous slices through the scanned volume (not only at the scan points).
(a) mean level, dB re max, shown where the posterior sd is below 60 % of the prior sd (the data constrain
    the field); (b) posterior sd relative to the prior sd (1 = no information); (c) exact room-like field with
    dwell-dependent noise at the real positions: error vs truth; (d) split-half error on the data, 5 frequencies.
Slices: horizontal plane at the median scan height, vertical x-z plane at y = 0.1 m, 2.5 cm mesh; prediction in
10 cm sub-cubes, each from the scan points within 24.5 cm (max-norm) of its centre (same window as the data fits).
Inputs: room_pipeline_<f>Hz.npz, room_gp_<f>Hz.json, room_gp_synthetic.log, room_pipeline_allf.log."""
import sys, json, re, numpy as np
sys.path.insert(0, 'lib')
import paperstyle as S
import matplotlib, matplotlib.pyplot as plt
from room_gp import gp_predict
FR = [258, 492, 750, 1008, 1242]
Z = np.load('results/room_pipeline_750Hz.npz'); pos = Z['pos']; k = 2 * np.pi * 750 / 343; om = 2 * np.pi * 750
lvp = np.abs(Z['p']); lvu = np.linalg.norm(Z['u'], axis=1) / np.sqrt(3)
J = {f: json.load(open(f'results/room_gp_{f}Hz.json')) for f in FR}
hp = J[750]
lo = pos.min(0) - 0.05; hi = pos.max(0) + 0.05; h = 0.025
xs, ys, zs = (np.arange(lo[d], hi[d] + h, h) for d in range(3))
zc, yc = np.median(pos[:, 2]), 0.1
Xh, Yh = np.meshgrid(xs, ys, indexing='ij'); Xv, Zv = np.meshgrid(xs, zs, indexing='ij')
Qh = np.column_stack([Xh.ravel(), Yh.ravel(), np.full(Xh.size, zc)]); Qv = np.column_stack([Xv.ravel(), np.full(Xv.size, yc), Zv.ravel()])
Xh, Yh = np.meshgrid(xs, ys, indexing='ij')
Qh = np.column_stack([Xh.ravel(), Yh.ravel(), np.full(Xh.size, zc)])
m, sd, sig, mu = gp_predict(pos, Z['p_raw'], Z['u_raw'], Z['dwell'], lvp, lvu, k, om, Qh, beta_p=hp['beta_p'], beta_u=hp['beta_u'], alpha=hp['alpha'], B=0.10, margin=0.195, with_u=True)
I = -0.5 * np.real(m[:, None] * np.conj(mu))                       # physical flux from the stored-convention velocity
shp = Xh.shape; m, sd, sig = m.reshape(shp), sd.reshape(shp), sig.reshape(shp); Ix, Iy = I[:, 0].reshape(shp), I[:, 1].reshape(shp)
rr = sd / sig; ok = np.isfinite(rr); good = ok & (rr < 0.6)
Ldb = 20 * np.log10(np.abs(m) / np.nanmax(np.abs(m)[good]))
print(f"[gp headline] horizontal slice: {good.sum() / ok.sum() * 100:.0f}% of predicted cells with sd < 0.6 prior sd")
Rr = np.hypot(Xh, Yh); outward = (Ix * Xh + Iy * Yh) > 0
print(f"[gp headline] flux from the GP points away from the origin in {np.mean(outward[good]) * 100:.0f}% of well-constrained cells")
np.savez('results/gp_slice_h_750Hz.npz', xs=xs, ys=ys, zc=zc, m=m, sd=sd, sig=sig, Ix=Ix, Iy=Iy)
syn = {}
for line in open('results/verify_room_gp_synthetic.log'):
    mm = re.match(r'\((A|B|C)\).*?median error ([0-9.]+)%(?:; 90% interval coverage ([0-9]+)%)?', line)
    if mm: syn[mm.group(1)] = (float(mm.group(2)), None if mm.group(3) is None else float(mm.group(3)))
M = {int(round(json.loads(l)['f'])): json.loads(l) for l in open('results/room_pipeline_allf.log')}
W = S.TEXTW; H = 4.45; fig = plt.figure(figsize=(W, H))
XL_, YL_ = (xs[0], xs[-1]), (ys[0], ys[-1]); mw = 2.6; mh = mw * (YL_[1] - YL_[0]) / (XL_[1] - XL_[0]); y1 = 2.55
a = fig.add_axes([0.6 / W, y1 / H, mw / W, mh / H])
L1 = np.where(ok, Ldb, np.nan)
pa = a.pcolormesh(xs, ys, L1.T, cmap=S.SEQ, vmin=-35, vmax=0, shading='auto', rasterized=True)
a.pcolormesh(xs, ys, np.where(ok & ~good, 1.0, np.nan).T, cmap=matplotlib.colors.ListedColormap([(1, 1, 1, 0.65)]), shading='auto', rasterized=True)
Ixm = np.where(good, Ix, np.nan); Iym = np.where(good, Iy, np.nan)
a.streamplot(xs, ys, Ixm.T, Iym.T, color='white', density=1.4, linewidth=0.6, arrowsize=0.6)
S.source(a); a.set_xlim(*XL_); a.set_ylim(*YL_); a.set_aspect('equal'); a.set_xlabel('$x$ [m]'); a.set_ylabel('$y$ [m]'); S.tag(a, 'a')
S.cbar(fig, pa, fig.add_axes([(0.6 + mw + 0.06) / W, y1 / H, 0.09 / W, mh / H]), r'$|p|$ [dB re max]', ticks=[-30, -15, 0])
xb = 0.6 + mw + 0.06 + 0.09 + 0.6
b = fig.add_axes([xb / W, y1 / H, mw / W, mh / H])
pb = b.pcolormesh(xs, ys, np.where(ok, rr, np.nan).T, cmap='viridis', vmin=0, vmax=1, shading='auto', rasterized=True)
sl = np.abs(pos[:, 2] - zc) < 0.1
b.scatter(pos[sl, 0], pos[sl, 1], s=0.15, c='white', alpha=0.5, lw=0, rasterized=True)
S.source(b); b.set_xlim(*XL_); b.set_ylim(*YL_); b.set_aspect('equal'); b.set_xlabel('$x$ [m]'); b.set_yticklabels([]); S.tag(b, 'b')
S.cbar(fig, pb, fig.add_axes([(xb + mw + 0.06) / W, y1 / H, 0.09 / W, mh / H]), 'sd / prior sd', ticks=[0, 0.5, 1])
c = fig.add_axes([0.1, 0.09, 0.3, 0.33])
vals = [syn['A'][0], syn['B'][0], syn['C'][0]]
c.bar(range(3), vals, color=['0.6', S.BLUE, S.ORANGE], width=0.6)
for i_, key in enumerate('BC'):
    c.text(i_ + 1, vals[i_ + 1] + 0.3, f"90 % interval\nholds {syn[key][1]:.0f} %", ha='center', fontsize=6.3)
c.set_xticks(range(3)); c.set_xticklabels(['local Helmholtz\n(Eq. 4)', 'GP, $p$ only', 'GP, $p$ and $\\mathbf{u}$'], fontsize=7)
c.set_ylabel('median error vs truth [%]'); c.set_ylim(0, max(vals) * 1.45); c.grid(color='0.9', lw=0.5, axis='y'); S.tag(c, 'c', x=0.88)
d = fig.add_axes([0.53, 0.09, 0.42, 0.33]); xf = np.arange(5)
d.plot(xf, [M[f]['split_rec'] * 100 for f in FR], 's-', color='0.5', ms=4, label='local Helmholtz')
d.plot(xf, [J[f]['split_p only'] * 100 for f in FR], 'o--', color=S.BLUE, ms=3.5, lw=0.9, label='GP, $p$ only')
d.plot(xf, [J[f]['split_p+u'] * 100 for f in FR], 'o-', color=S.ORANGE, ms=4, label='GP, $p$ and $\\mathbf{u}$')
for i_, f in enumerate(FR):
    d.text(i_ + 0.2, J[f]['split_p+u'] * 100 - 8.5, f"{J[f]['cov90_p+u']*100:.0f}%", ha='center', fontsize=6, color=S.ORANGE)
d.set_xticks(xf); d.set_xticklabels(FR, fontsize=7); d.set_xlabel('$f$ [Hz]'); d.set_ylabel('split-half error [%]')
d.set_ylim(0, 95); d.legend(frameon=False, fontsize=6.5, loc='upper left', ncol=3, columnspacing=0.8, handlelength=1.5); d.grid(color='0.9', lw=0.5); S.tag(d, 'd', x=0.93)
fig.savefig('paperfigs/fig_gp.pdf', dpi=300); fig.savefig('results/fig_gp_preview.png', dpi=140); plt.close(fig)
print('wrote fig_gp (headline)')
