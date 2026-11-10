#!/usr/bin/env python3
"""make_wave_animation.py — the reconstructed sound wave at 750 Hz propagating through the room.
Physical pressure p(t) = Re(p e^{+i w t}) (stored e^{+iwt} convention) on the horizontal plane at the median scan
height, from the joint p-u Gaussian process (gp_slice_h_750Hz.npz).  Spreading is compensated (p r) so that the
wavefronts stay visible across the room; cells where the posterior sd exceeds 60 % of the prior are left blank.
One period in 30 frames, looped.  Writes room_wave_750Hz.gif and a 4-frame strip for the paper."""
import sys, numpy as np
sys.path.insert(0, 'lib')
import paperstyle as S
import matplotlib, matplotlib.pyplot as plt
from PIL import Image
D = np.load('results/gp_slice_h_750Hz.npz'); xs, ys = D['xs'], D['ys']; m = D['m']; sd = D['sd']; sig = D['sig']
X, Y = np.meshgrid(xs, ys, indexing='ij'); r = np.maximum(np.hypot(X, Y), 0.15)
good = np.isfinite(sd) & (sd / sig < 0.6)
q = np.where(good, m * r, np.nan); A = np.nanpercentile(np.abs(q), 95)
def frame(phi, ax):
    v = np.real(q * np.exp(1j * phi)) / A
    im = ax.pcolormesh(xs, ys, v.T, cmap='RdBu_r', vmin=-1, vmax=1, shading='gouraud', rasterized=True)
    S.source(ax); ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1]); ax.set_aspect('equal')
    return im
frames = []
for i in range(30):
    fig = plt.figure(figsize=(7.2, 3.0)); ax = fig.add_axes([0.08, 0.15, 0.8, 0.78])
    im = frame(2 * np.pi * i / 30, ax); ax.set_xlabel('$x$ [m]'); ax.set_ylabel('$y$ [m]')
    ax.set_title('Reconstructed sound wave, 750 Hz (pressure $\\times$ distance)', fontsize=9)
    cax = fig.add_axes([0.9, 0.15, 0.02, 0.78]); cb = fig.colorbar(im, cax=cax); cb.set_ticks([-1, 0, 1]); cb.ax.tick_params(labelsize=7)
    fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])); plt.close(fig)
frames[0].save('room_wave_750Hz.gif', save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=True)
# strip for the paper: four phases
fig = plt.figure(figsize=(S.TEXTW, 1.55))
for j, ph in enumerate((0, 90, 180, 270)):
    ax = fig.add_axes([0.06 + j * 0.225, 0.2, 0.215, 0.72]); im = frame(np.radians(ph), ax)
    ax.set_title(f'$\\omega t = {ph}^\\circ$', fontsize=8); ax.set_xlabel('$x$ [m]', fontsize=7); ax.tick_params(labelsize=6)
    if j: ax.set_yticklabels([])
    else: ax.set_ylabel('$y$ [m]', fontsize=7)
cax = fig.add_axes([0.965, 0.2, 0.012, 0.72]); cb = fig.colorbar(im, cax=cax); cb.set_ticks([-1, 0, 1]); cb.ax.tick_params(labelsize=6)
fig.savefig('paperfigs/fig_wave_strip.pdf', dpi=300); fig.savefig('results/fig_wave_strip_preview.png', dpi=140); plt.close(fig)
# check (Sec. 5.5): does the reconstructed phase advance outward, at about the acoustic wavenumber?
h = xs[1] - xs[0]
gx = np.angle(m[2:, 1:-1] * np.conj(m[:-2, 1:-1])) / (2 * h); gy = np.angle(m[1:-1, 2:] * np.conj(m[1:-1, :-2])) / (2 * h)
Xc, Yc = X[1:-1, 1:-1], Y[1:-1, 1:-1]; R = np.hypot(Xc, Yc)
g = good[1:-1, 1:-1] & good[2:, 1:-1] & good[:-2, 1:-1] & good[1:-1, 2:] & good[1:-1, :-2] & (R > 0.3)
kr = -(gx * Xc + gy * Yc) / R; k = 2 * np.pi * 750 / 343
print(f"[wave] frames {len(frames)}; phase advances outward in {np.mean(kr[g] > 0) * 100:.0f}% of well-constrained cells beyond 0.3 m")
for a_, b_ in ((0.3, 0.8), (0.8, 1.5), (1.5, 2.8)):
    s_ = g & (R >= a_) & (R < b_); print(f"[wave]   {a_}-{b_} m: local radial wavenumber median {np.median(kr[s_]):.1f} rad/m (k = {k:.1f})")
