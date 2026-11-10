#!/usr/bin/env python3
"""
make_fig_flux3d.py — 3-D view of the reconstructed field at 750 Hz (ParaView-style, Matplotlib only).

  (a) orthogonal slices of the pressure level (dB re max) through the source: horizontal plane at the
      median scan height and vertical x-z plane at y = 0.1 m, with the scan point cloud;
  (b) 3-D energy-flux streamlines, one seed per occupied 20 cm cell of the scanned volume, integrated forwards
      and backwards along the reconstructed intensity (RK2, 1 cm steps), coloured by |I| in dB re max.
The reconstructed p and I (run_room_pipeline.py) are evaluated by inverse-distance weighting of the nearest
scan points (8 for the slices, drawn on a 2.5 cm mesh and left empty farther than 10 cm from any point; 16 for
the streamlines, which stop where the nearest point is farther than 12 cm).
Prints the streamline statistics quoted in the caption.
"""
import sys, numpy as np
sys.path.insert(0, 'lib')
import paperstyle as S
import matplotlib, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy_free_interp import knn

Z = np.load('results/room_pipeline_750Hz.npz'); pos = Z['pos']; p = Z['p']; u = Z['u']
I = -0.5 * np.real(p[:, None] * np.conj(u)); r = np.linalg.norm(pos, axis=1)
lo = pos.min(0) - 0.02; hi = pos.max(0) + 0.02
Lp = 20 * np.log10(np.abs(p) / np.abs(p).max()); Imax = np.linalg.norm(I, axis=1).max()
MAXD = 0.12

def idw(q, vals, k=16):
    idx, dd = knn(q, pos, k, chunk=200); w = 1 / np.maximum(dd, 1e-6) ** 2
    out = np.einsum('nk,nk...->n...', w, vals[idx]) / w.sum(1).reshape((-1,) + (1,) * (vals.ndim - 1))
    return out, dd[:, 0]

def trace_all(seeds, step=0.01, nmax=400, sgn=1.0):
    x = seeds.copy(); alive = np.ones(len(x), bool); paths = [[s_.copy()] for s_ in x]; mags = [[] for _ in x]
    for _ in range(nmax):
        if not alive.any(): break
        ia = np.flatnonzero(alive)
        v1, d1 = idw(x[ia], I); v1 = sgn * v1; n1 = np.linalg.norm(v1, axis=1)
        ok = (d1 < MAXD) & (n1 > 0)
        xm = x[ia] + 0.5 * step * v1 / np.maximum(n1, 1e-30)[:, None]
        v2, d2 = idw(xm, I); v2 = sgn * v2; n2 = np.linalg.norm(v2, axis=1); ok &= (d2 < MAXD) & (n2 > 0)
        xn = x[ia] + step * v2 / np.maximum(n2, 1e-30)[:, None]
        for jj, i_ in enumerate(ia):
            if ok[jj]: x[i_] = xn[jj]; paths[i_].append(xn[jj].copy()); mags[i_].append(n2[jj])
            else: alive[i_] = False
    return [(np.array(P_), np.array(M_)) for P_, M_ in zip(paths, mags)]

rng = np.random.default_rng(0)
# one seed per occupied 20 cm cell of the scanned volume, traced forwards and backwards along the flux
vox = np.floor((pos - lo) / 0.20).astype(int); _, first = np.unique(vox[rng.permutation(len(pos))], axis=0, return_index=True)
perm = rng.permutation(len(pos)); _, fi = np.unique(vox[perm], axis=0, return_index=True); seeds = pos[perm[fi]]
fw = trace_all(seeds, nmax=250, sgn=1.0); bw = trace_all(seeds, nmax=250, sgn=-1.0)
lines = []
for (Pf, Mf), (Pb, Mb) in zip(fw, bw):
    P_ = np.concatenate([Pb[::-1], Pf[1:]]); M_ = np.concatenate([Mb[::-1], Mf])
    if len(P_) > 10: lines.append((P_, M_))
cellsL = set(); outw = []
for P_, _ in lines:
    for c in np.unique(np.floor((P_ - lo) / 0.20).astype(int), axis=0): cellsL.add(tuple(c))
    d_ = np.diff(P_, axis=0); mid = 0.5 * (P_[1:] + P_[:-1]); outw.append(np.sum(np.einsum('ij,ij->i', d_, mid) > 0) / len(d_))
cov = len(cellsL & set(map(tuple, np.unique(vox, axis=0)))) / len(np.unique(vox, axis=0))
print(f"[flux3d] seeds {len(seeds)} (one per occupied 20 cm cell); streamlines kept {len(lines)}; they cross {cov*100:.0f}% of the occupied cells; "
      f"fraction of streamline length pointing away from the origin: median {np.median(outw)*100:.0f}%")
lens = np.array([np.sum(np.linalg.norm(np.diff(P_, axis=0), axis=1)) for P_, _ in lines])
ends = np.array([np.linalg.norm(P_[-1]) for P_, _ in lines])
np.savez('results/flux3d_lines.npz', *[np.column_stack([P_[1:], 10 * np.log10(np.maximum(M_, 1e-30) / Imax)]) for P_, M_ in lines])
print(f"[flux3d] {len(lines)} streamlines from {len(seeds)} seeds (gaps up to {MAXD*100:.0f} cm bridged); "
      f"length median {np.median(lens):.2f} m (max {lens.max():.2f}); end distance from origin median {np.median(ends):.2f} m (max {ends.max():.2f})")
hs = 0.025; ax_ = [np.arange(lo[d], hi[d] + hs, hs) for d in range(3)]
def slice_vals(Q):
    v, d0 = idw(Q, Lp, k=8); v[d0 > 0.10] = np.nan; return v
W = S.TEXTW * 0.8; H = 3.4; fig = plt.figure(figsize=(W, H))
cmL = matplotlib.colormaps[S.SEQ]; normL = matplotlib.colors.Normalize(-40, 0)
cmI = matplotlib.colormaps['viridis']; normI = matplotlib.colors.Normalize(-30, 0)
def frame(a):
    a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1]); a.set_zlim(lo[2], hi[2])
    a.set_box_aspect(tuple(hi - lo), zoom=1.12); a.view_init(elev=24, azim=-58)
    for x_ in (a.xaxis, a.yaxis, a.zaxis):
        x_.set_pane_color((1, 1, 1, 0)); x_._axinfo['grid']['color'] = (0.9, 0.9, 0.9, 1); x_._axinfo['grid']['linewidth'] = 0.4
    a.tick_params(labelsize=6, pad=-2); a.set_xlabel('$x$ [m]', labelpad=-4, fontsize=7); a.set_ylabel('$y$ [m]', labelpad=-4, fontsize=7); a.set_zlabel('$z$ [m]', labelpad=-6, fontsize=7)
    a.set_zticks([-0.5, 0, 0.5]); a.set_yticks([-0.5, 0, 0.5, 1.0])
# (b) streamlines
b = fig.add_axes([0.0, 0.0, 1.0, 0.92], projection='3d', computed_zorder=False)
segs, cols = [], []
for P_, M_ in lines:
    s_ = np.stack([P_[:-1], P_[1:]], 1); segs.append(s_); cols.append(10 * np.log10(np.maximum(M_, 1e-30) / Imax))
segs = np.concatenate(segs); cols = np.concatenate(cols)
lc = Line3DCollection(segs, colors=cmI(normI(cols)), linewidths=0.7, alpha=0.9, zorder=3); b.add_collection3d(lc)
b.scatter(pos[::4, 0], pos[::4, 1], pos[::4, 2], s=0.4, c='0.55', alpha=0.25, lw=0, depthshade=False, zorder=0)
b.scatter([0], [0], [0], marker='*', s=110, c='white', edgecolors='k', linewidths=0.8, depthshade=False, zorder=10)
frame(b)
cb2a = fig.add_axes([0.35, 0.93, 0.3, 0.022]); cb2 = fig.colorbar(matplotlib.cm.ScalarMappable(normI, cmI), cax=cb2a, orientation='horizontal')
cb2.set_ticks([-30, -15, 0]); cb2.ax.tick_params(labelsize=6, length=2); cb2.outline.set_linewidth(0.5); cb2a.set_title('$|\\mathbf{I}|$ [dB re max]', fontsize=7, pad=2)
fig.savefig('paperfigs/fig_flux3d.pdf', dpi=300); fig.savefig('results/fig_flux3d_preview.png', dpi=160); plt.close(fig)
print('wrote fig_flux3d')
