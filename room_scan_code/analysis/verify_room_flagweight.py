#!/usr/bin/env python3
"""Does giving flagged nodes zero weight change the reconstruction's residual error?
Split-half (96 neighbours per half, dwell weights) with and without the Stage-2 flags; same halves,
same query points. Also the same split-half metric with 6 neighbours per half (minimal averaging)."""
import sys, numpy as np; sys.path.insert(0,'lib')
from room_stage2 import local_recon, dwell_blocks
from room_raw import raw
R=raw(750.0); pos=R['pos']; P=R['p_ref1']; k=2*np.pi*R['f']/343; nb=dwell_blocks(pos)
fl=np.load('results/room_pipeline_750Hz.npz')['flag_p']
rng=np.random.default_rng(4); q=rng.choice(len(P),800,replace=False); half=rng.random(len(P))<0.5
A=np.flatnonzero(half); B=np.flatnonzero(~half)
def sh(w,n):
    pa=local_recon(pos,P,w,k,q,n_nb=n,src=A); pb=local_recon(pos,P,w,k,q,n_nb=n,src=B)
    return np.median(np.abs(pa-pb)/np.sqrt(2)/np.abs(0.5*(pa+pb)))*100
print(f"split-half residual, 96 per half: flags weighted 0 -> {sh(nb*(~fl),96):.1f}%, flags kept -> {sh(nb,96):.1f}%")
print(f"split-half residual,  6 per half (minimal averaging), flags kept -> {sh(nb,6):.1f}%")
