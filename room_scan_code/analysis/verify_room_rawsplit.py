#!/usr/bin/env python3
"""Like-for-like raw baseline for the split-half residual: same 800 query points and halves as
verify_room_flagweight.py; each half represented by its node nearest to the query point
(no averaging). |pA - pB| / sqrt2 / |mean|, median."""
import sys, numpy as np; sys.path.insert(0,'lib')
from room_raw import raw
from scipy_free_interp import knn
R=raw(750.0); pos=R['pos']; P=R['p_ref1']
rng=np.random.default_rng(4); q=rng.choice(len(P),800,replace=False); half=rng.random(len(P))<0.5
A=np.flatnonzero(half); B=np.flatnonzero(~half)
ia,da=knn(pos[q],pos[A],1); ib,db=knn(pos[q],pos[B],1); pa=P[A[ia[:,0]]]; pb=P[B[ib[:,0]]]
e=np.abs(pa-pb)/np.sqrt(2)/np.abs(0.5*(pa+pb))
print(f"raw split-half (nearest node per half, median separation {np.median(da[:,0]+db[:,0])*100:.1f} cm): median {np.median(e)*100:.0f}%")
