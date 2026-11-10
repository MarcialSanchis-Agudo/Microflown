#!/usr/bin/env python3
"""Is the photogrammetric mesh in the same frame as the scan, and where are the armchairs?
Mapping used in the paper: acoustic (x, y, z) = mesh (x, z, y); floor at z = -1.369 m.
(1) floor: plane fit to the lowest vertices under the scan footprint (tilt, height);
(2) scan points vs mesh: fraction of points within 3 cm of a mesh vertex (a point inside a
    surface would mean misalignment), compared with the same scan shifted rigidly by 0.3 m;
(3) loudspeaker stand: mesh vertices within 0.25 m horizontally of the origin, by height;
(4) armchairs from their texture colour (mustard: hue 25-50 deg, saturation > 0.45, value > 0.35),
    0.1-1.3 m above the floor: two largest clusters, centroid, footprint, top height (99th pct),
    and whether the blocked scan points lie behind them as seen from the source."""
import sys, numpy as np; sys.path.insert(0, 'lib')
from scipy_free_interp import knn
V = np.load('results/room_mesh_vertices.npy').astype(float); C = np.load('results/room_mesh_colors.npy') / 255.0
P = np.column_stack([V[:, 0], V[:, 2], V[:, 1]]); FL = -1.369
Z = np.load('results/room_pipeline_750Hz.npz'); pos = Z['pos']
# (1) floor
fp = (P[:, 0] > pos[:, 0].min()) & (P[:, 0] < pos[:, 0].max()) & (P[:, 1] > pos[:, 1].min()) & (P[:, 1] < pos[:, 1].max())
low = fp & (P[:, 2] < np.percentile(P[fp, 2], 5))
A = np.column_stack([P[low, 0], P[low, 1], np.ones(low.sum())]); cf = np.linalg.lstsq(A, P[low, 2], rcond=None)[0]
tilt = np.degrees(np.arctan(np.hypot(cf[0], cf[1])))
print(f"(1) floor under the scan: z = {cf[2]:+.3f} m at the origin (paper {FL}), tilt {tilt:.1f} deg; lowest scan point {pos[:,2].min()-FL:.2f} m above it")
# (2) scan points inside surfaces?
sub = np.random.default_rng(0).choice(len(P), 200000, replace=False)
_, d = knn(pos, P[sub], 1, chunk=40); f_in = np.mean(d[:, 0] < 0.03)
shifts = [(0.3, 0, 0), (-0.3, 0, 0), (0, 0.3, 0), (0, -0.3, 0), (0, 0, 0.3), (0, 0, -0.3)]
fs = []
for s_ in shifts:
    _, ds = knn(pos + np.array(s_), P[sub], 1, chunk=40); fs.append(np.mean(ds[:, 0] < 0.03))
print(f"(2) scan points within 3 cm of a mesh vertex: {f_in*100:.1f}% (as aligned) vs {np.min(fs)*100:.1f}-{np.max(fs)*100:.1f}% for the scan shifted by 0.3 m")
# (3) loudspeaker stand
cyl = np.hypot(P[:, 0], P[:, 1]) < 0.25; h = P[cyl, 2] - FL
hist, e = np.histogram(h, bins=np.arange(0, 2.01, 0.2))
print("(3) mesh vertices within 0.25 m of the vertical through the origin, per 0.2 m of height:", dict(zip([f'{a:.1f}' for a in e[:-1]], hist.tolist())))
# (4) armchairs by colour
mx = C.max(1); mn = C.min(1); sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0)
hue = np.zeros(len(C)); r_, g_, b_ = C.T; dd = np.maximum(mx - mn, 1e-9)
m_r = mx == r_; hue[m_r] = (60 * ((g_ - b_) / dd) % 360)[m_r]
m_g = (mx == g_) & ~m_r; hue[m_g] = (60 * ((b_ - r_) / dd) + 120)[m_g]
hh = P[:, 2] - FL
chair = (hue > 25) & (hue < 50) & (sat > 0.45) & (mx > 0.35) & (hh > 0.1) & (hh < 1.3) & (P[:, 0] > -3.6) & (P[:, 0] < 1.2) & (P[:, 1] > -1.4) & (P[:, 1] < 2.0)
Q = P[chair]
b = 0.05; ij = np.floor(Q[:, :2] / b).astype(int); keys, inv, cnt = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
occ = {tuple(k): c for k, c in zip(keys, cnt) if c >= 5}
lab = {}; comps = []
for k in occ:
    if k in lab: continue
    st = [k]; lab[k] = len(comps); comp = [k]
    while st:
        a = st.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                n = (a[0] + dx, a[1] + dy)
                if n in occ and n not in lab: lab[n] = len(comps); st.append(n); comp.append(n)
    comps.append(comp)
comps.sort(key=lambda c: -sum(occ[k] for k in c))
print(f"(4) mustard-coloured vertices 0.1-1.3 m high: {chair.sum()}; clusters (>= 5 vertices per 5 cm cell): {len(comps)}")
bl = np.zeros((0, 3))   # line-of-sight analysis not part of the paper
res = []
for c in comps[:4]:
    cells = np.array(c); inside = np.isin(inv, [np.flatnonzero((keys == cc).all(1))[0] for cc in c])
    q = Q[inside]; cen = q[:, :2].mean(0); ext = np.ptp(q[:, :2], 0); top = np.percentile(q[:, 2] - FL, 99)
    ang_c = np.arctan2(cen[1], cen[0]); rc = np.hypot(*cen)
    ab = np.arctan2(bl[:, 1], bl[:, 0]); rb = np.hypot(bl[:, 0], bl[:, 1])
    behind = np.sum((np.abs(np.angle(np.exp(1j * (ab - ang_c)))) * rc < 0.5) & (rb > rc))
    print(f"    cluster: {len(q)} vertices, centre ({cen[0]:+.2f}, {cen[1]:+.2f}) m, footprint {ext[0]:.2f} x {ext[1]:.2f} m, top {top:.2f} m; "
          f"blocked points behind it (within 0.5 m of its bearing, farther from the source): {behind}")
    res.append((cen, top))
np.save('results/room_chairs_colour.npy', np.array([[c[0][0], c[0][1], c[1]] for c in res]))

# (5) under the inherited mapping, how many scan points lie within 5 cm (3-D) of model surfaces 1.0-1.8 m high?
band = (P[:, 2] - FL > 1.0) & (P[:, 2] - FL < 1.8) & (P[:, 0] > -3.6) & (P[:, 0] < 1.2) & (P[:, 1] > -1.4) & (P[:, 1] < 2.0)
_, d5 = knn(pos, P[band], 1, chunk=40)
print(f"(5) scan points within 5 cm of model surfaces 1.0-1.8 m high: {np.sum(d5[:,0] < 0.05)} ({np.mean(d5[:,0] < 0.05)*100:.1f}%); within 10 cm: {np.mean(d5[:,0] < 0.10)*100:.1f}%")
