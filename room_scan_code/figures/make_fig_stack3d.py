#!/usr/bin/env python3
"""make_fig_stack3d.py — the reconstructed field in 3-D as a stack of horizontal planes (750 Hz, joint p-u GP).
Each plane: level in dB re the global maximum (blank where posterior sd > 0.6 prior sd) with streamlines of the
in-plane energy flux computed from the predicted pressure and velocity.  Vertical spacing exaggerated x2.5 for
legibility; tick labels give true heights.  Inputs: gp_plane_z*.npz (run_gp_stack.py)."""
import sys, glob, numpy as np
sys.path.insert(0, 'lib')
import paperstyle as S
import matplotlib, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
P = [np.load(f) for f in sorted(glob.glob('results/gp_plane_z*.npz'), key=lambda f: float(np.load(f)['z']))]
EX = 2.5
mref = max(np.nanmax(np.abs(p['m'])[np.isfinite(p['sd']) & (p['sd'] / p['sig'] < 0.6)]) for p in P)
cm = matplotlib.colormaps[S.SEQ]; nrm = matplotlib.colors.Normalize(-35, 0)
fig = plt.figure(figsize=(S.TEXTW, 4.6)); a = fig.add_axes([-0.06, -0.06, 1.05, 1.12], projection='3d', computed_zorder=False)
for p in P:
    xs, ys, z = p['xs'], p['ys'], float(p['z']); X, Y = np.meshgrid(xs, ys, indexing='ij')
    good = np.isfinite(p['sd']) & (p['sd'] / p['sig'] < 0.6)
    L = 20 * np.log10(np.abs(p['m']) / mref); fc = cm(nrm(np.nan_to_num(L, nan=-99))); fc[~good] = (1, 1, 1, 0); fc[good, 3] = 0.92
    a.plot_surface(X, Y, np.full_like(X, z * EX), facecolors=fc, rstride=1, cstride=1, linewidth=0, shade=False, antialiased=False, zorder=1)
    f2, a2 = plt.subplots()
    st = a2.streamplot(xs, ys, np.where(good, p['Ix'], np.nan).T, np.where(good, p['Iy'], np.nan).T, density=1.0, linewidth=0.5, arrowsize=0.01)
    segs = [np.column_stack([s_[:, 0], s_[:, 1], np.full(len(s_), z * EX + 0.004)]) for s_ in st.lines.get_segments()]
    plt.close(f2)
    a.add_collection3d(Line3DCollection(segs, colors='white', linewidths=0.45, alpha=0.9, zorder=3), autolim=False)
zs_ = [float(p['z']) for p in P]
a.plot([0, 0], [0, 0], [min(zs_) * EX, max(zs_) * EX], color='k', lw=0.8, ls=':', zorder=4)
a.scatter([0], [0], [0], marker='*', s=130, c='white', edgecolors='k', linewidths=0.9, depthshade=False, zorder=10)
xs, ys = P[0]['xs'], P[0]['ys']
a.set_xlim(xs[0], xs[-1]); a.set_ylim(ys[0], ys[-1]); a.set_zlim(min(zs_) * EX - 0.1, max(zs_) * EX + 0.1)
a.set_box_aspect((xs[-1] - xs[0], ys[-1] - ys[0], (max(zs_) - min(zs_)) * EX + 0.2), zoom=1.05); a.view_init(elev=24, azim=-62)
a.set_zticks([z * EX for z in zs_]); a.set_zticklabels([f'{z:+.2f}' for z in zs_], fontsize=6)
for x_ in (a.xaxis, a.yaxis, a.zaxis):
    x_.set_pane_color((1, 1, 1, 0)); x_._axinfo['grid']['color'] = (0.92, 0.92, 0.92, 1); x_._axinfo['grid']['linewidth'] = 0.4
a.tick_params(labelsize=6, pad=-2); a.set_xlabel('$x$ [m]', labelpad=-3, fontsize=7.5); a.set_ylabel('$y$ [m]', labelpad=-3, fontsize=7.5); a.set_zlabel('$z$ [m]', labelpad=-5, fontsize=7.5)
cax = fig.add_axes([0.06, 0.12, 0.015, 0.3]); cb = fig.colorbar(matplotlib.cm.ScalarMappable(nrm, cm), cax=cax, orientation='vertical')
cb.set_ticks([-30, -15, 0]); cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.5); cax.set_title('$|p|$\n[dB re max]', fontsize=7, pad=4)
fig.savefig('paperfigs/fig_stack3d.pdf', dpi=300); fig.savefig('results/fig_stack3d_preview.png', dpi=150); plt.close(fig)
print('wrote fig_stack3d; planes at', zs_)
