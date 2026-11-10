#!/usr/bin/env python3
"""make_fig_setup3d.py — Fig. 1: photographs, the photogrammetric room model recorded by the system
(vertices in their texture colour, in the model's own frame), and the scan in the acoustic frame
(origin at the 750 Hz pressure maximum next to the loudspeaker).  The two are shown side by side,
not overlaid, because their registration could not be verified (verify_room_mesh_frame.py)."""
import sys, numpy as np
sys.path.insert(0, 'lib')
import paperstyle as S
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
V = np.load('results/room_mesh_vertices.npy').astype(float); Cc = np.load('results/room_mesh_colors.npy') / 255.0
M = np.column_stack([V[:, 0], V[:, 2], V[:, 1]]); hm = M[:, 2] - np.percentile(M[:, 2], 1)
box = (hm < 2.3); rng = np.random.default_rng(0); idx = rng.choice(np.flatnonzero(box), 160000, replace=False)
Z = np.load('results/room_pipeline_750Hz.npz'); pos = Z['pos']
def photo(ax, path, aspect):
    im = plt.imread(path); h, w = im.shape[:2]; tw = int(h * aspect)
    if tw < w: x0 = (w - tw) // 2; im = im[:, x0:x0 + tw]
    ax.imshow(im); ax.set_axis_off()
W = S.TEXTW; H = 2.3; fig = plt.figure(figsize=(W, H)); ph = 2.0; pw = ph * 0.45; y0 = 0.15
pics = ['data/photo_room.jpeg', 'data/photo_probe.jpeg',
        'data/photo_furniture.jpeg']
x = 0.03
for t, pth in zip('abc', pics):
    a = fig.add_axes([x / W, y0 / H, pw / W, ph / H]); photo(a, pth, 0.45); S.tag(a, t); x += pw + 0.04
def style3d(a):
    for ax_ in (a.xaxis, a.yaxis, a.zaxis):
        ax_.set_pane_color((1, 1, 1, 0)); ax_._axinfo['grid']['color'] = (0.9, 0.9, 0.9, 1); ax_._axinfo['grid']['linewidth'] = 0.4
    a.tick_params(labelsize=5.5, pad=-3)
# (d) room model
a = fig.add_axes([(x - 0.1) / W, -0.05, 2.15 / W, 1.1], projection='3d', computed_zorder=False)
a.scatter(M[idx, 0], M[idx, 1], hm[idx], c=Cc[idx], s=0.15, lw=0, rasterized=True, depthshade=False)
a.set_box_aspect((np.ptp(M[idx, 0]), np.ptp(M[idx, 1]), 2.3 * 2.0), zoom=1.15); a.view_init(elev=35, azim=-60)
a.set_axis_off()
a.text2D(0.03, 0.93, '(d)', transform=a.transAxes, fontweight='bold', fontsize=9)
a.text2D(0.5, 0.93, 'room model', transform=a.transAxes, ha='center', fontsize=7)
# (e) scan in the acoustic frame
xe = x - 0.1 + 2.15 - 0.1; e = fig.add_axes([xe / W, 0.02, (W - xe - 0.4) / W, 0.93], projection='3d', computed_zorder=False)
L = 20 * np.log10(np.abs(Z['p']) / np.abs(Z['p']).max()); o = np.argsort(L)
sc = e.scatter(pos[o, 0], pos[o, 1], pos[o, 2], c=L[o], cmap=S.SEQ, vmin=-40, vmax=0, s=0.6, lw=0, rasterized=True, depthshade=False)
e.scatter([0], [0], [0], marker='*', s=90, c='white', edgecolors='k', linewidths=0.8, depthshade=False, zorder=20)
e.set_box_aspect((np.ptp(pos[:, 0]), np.ptp(pos[:, 1]), np.ptp(pos[:, 2])), zoom=1.05); e.view_init(elev=25, azim=-60)
e.set_xlabel('$x$ [m]', labelpad=-7, fontsize=6.5); e.set_ylabel('$y$', labelpad=-8, fontsize=6.5); e.set_zlabel('$z$', labelpad=-8, fontsize=6.5)
e.set_zticks([-0.5, 0, 0.5]); e.set_xticks([-2, -1, 0]); e.set_yticks([-0.5, 0.5]); style3d(e)
e.text2D(0.0, 0.97, '(e)', transform=e.transAxes, fontweight='bold', fontsize=9)
e.text2D(0.55, 0.99, 'scan, acoustic frame', transform=e.transAxes, ha='center', fontsize=7)
cax = fig.add_axes([(xe + 1.3) / W, 0.78, 0.75 / W, 0.035])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal'); cb.set_ticks([-40, -20, 0]); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.5)
cax.set_title('$|p|$ [dB re max]', fontsize=6.5, pad=2)
fig.savefig('paperfigs/fig_setup.pdf', dpi=300); fig.savefig('results/fig_setup_preview.png', dpi=160); plt.close(fig)
print('[setup3d] room model and scan side by side')
