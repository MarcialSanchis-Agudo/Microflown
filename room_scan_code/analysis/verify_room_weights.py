#!/usr/bin/env python3
"""Split-half random error of the local Helmholtz fit (Eq. 4) at 750 Hz for three weightings:
recorded length (dwell, the default), the exact number of 50 %-overlap averages
K = floor((L-2048)/1024)+1, and uniform. Same split and query points for all three."""
import sys, numpy as np; sys.path.insert(0,'lib')
from room_stage2 import local_recon
Z=np.load('results/room_pipeline_750Hz.npz')
pos=Z['pos']; P=Z['p_raw']; nb=Z['dwell']; fl=Z['flag_p']; k=2*np.pi*750/343
rng=np.random.default_rng(4); qq=rng.choice(len(P),800,replace=False); half=rng.random(len(P))<0.5
A,B=np.flatnonzero(half),np.flatnonzero(~half)
K=np.floor((nb*2048-2048)/1024+1e-9)+1
for lab,w in (('recorded length (default)',nb*(~fl)),('exact count K, 50 % overlap',K*(~fl)),('uniform',1.0*(~fl))):
    pa=local_recon(pos,P,w,k,qq,n_nb=96,src=A); pb=local_recon(pos,P,w,k,qq,n_nb=96,src=B)
    print(f"{lab:28s} split-half random error {100*np.median(np.abs(pa-pb)/np.sqrt(2)/np.abs(0.5*(pa+pb))):.1f} %")
