#!/usr/bin/env python3
"""Reconstruction bias versus distance from the source, on the known room-like field
(monopole at the origin + 3 images) at the real node positions (oriented frame), for n_nb = 24/48/96,
magnitude (dB) and complex relative error, in distance bands."""
import sys, numpy as np; sys.path.insert(0,'lib')
from room_stage2 import local_recon
Z=np.load('results/room_pipeline_750Hz.npz'); pos=Z['pos']; r=np.linalg.norm(pos,axis=1); k=2*np.pi*750/343
Gf=lambda s: (lambda d: np.exp(-1j*k*d)/d)(np.linalg.norm(pos-np.array(s),axis=1))
F=Gf([0,0,0])+0.6*Gf([0,0,-1.9])+0.5*Gf([0.9,0,0])+0.4*Gf([0,2.6,0])
w=Z['dwell']*(~Z['flag_p']); bands=((0,0.2),(0.2,0.4),(0.4,0.7),(0.7,1.2),(1.2,3))
rng=np.random.default_rng(0)
for n in (24,48,96):
    q=np.concatenate([rng.choice(np.flatnonzero((r>=a)&(r<b)),min(300,((r>=a)&(r<b)).sum()),replace=False) for a,b in bands])
    pr=local_recon(pos,F,w,k,q,n_nb=n); e=np.abs(pr-F[q])/np.abs(F[q]); db=20*np.log10(np.abs(pr)/np.abs(F[q]))
    row=[]
    for a,b in bands:
        m=(r[q]>=a)&(r[q]<b); row.append(f"{a:.1f}-{b:.1f} m: {np.median(e[m])*100:5.1f}% / {np.median(db[m]):+.2f} dB")
    print(f"n_nb={n:3d} | "+" | ".join(row))
