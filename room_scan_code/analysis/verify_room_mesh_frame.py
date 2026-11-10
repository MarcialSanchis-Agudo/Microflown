#!/usr/bin/env python3
"""Which frame is the photogrammetric mesh in?  Test all 48 signed axis permutations T of the RAW probe
positions (tracking frame of the .psd) against the OBJ vertices, no translation: fraction of scan points
inside the mesh's 1-99 % bounding box, below its floor (1st pct of the vertical axis), and within 3 cm of a
mesh vertex.  The same scores for the ORIENTED positions with the mapping used so far (mesh x, z, y)."""
import sys, itertools, numpy as np; sys.path.insert(0, 'lib')
from scipy_free_interp import knn
from room_raw import raw
V = np.load('results/room_mesh_vertices.npy').astype(float)
X = raw(750.0)['pos']; Y = np.load('results/room_pipeline_750Hz.npz')['pos']
sub = np.random.default_rng(0).choice(len(V), 120000, replace=False); Vs = V[sub]
lo, hi = np.percentile(V, 1, 0), np.percentile(V, 99, 0)
pts = X[::6]
def score(Pt, up):
    inside = np.all((Pt > lo) & (Pt < hi), 1).mean()
    below = (Pt[:, up] < lo[up]).mean()
    _, d = knn(Pt, Vs, 1, chunk=40); near = (d[:, 0] < 0.03).mean(); med = np.median(d[:, 0])
    return inside, below, near, med
res = []
for perm in itertools.permutations(range(3)):
    for sg in itertools.product((1, -1), repeat=3):
        Pt = pts[:, perm] * np.array(sg)
        res.append((perm, sg) + score(Pt, 1))
res.sort(key=lambda t: (-t[2], t[3], t[4]))
print("raw probe positions, signed permutations (mesh vertical = y): best 6 by fraction inside the room")
for r in res[:6]: print(f"  perm {r[0]} sign {r[1]}: inside {r[2]*100:5.1f}%  below floor {r[3]*100:5.1f}%  within 3 cm of mesh {r[4]*100:4.1f}%  median distance {r[5]*100:5.1f} cm")
M = np.column_stack([Y[:, 0], Y[:, 2], Y[:, 1]])[::6]
s = score(M, 1)
print(f"oriented positions with the paper's mapping: inside {s[0]*100:.1f}%  below floor {s[1]*100:.1f}%  within 3 cm {s[2]*100:.1f}%  median distance {s[3]*100:.1f} cm")
