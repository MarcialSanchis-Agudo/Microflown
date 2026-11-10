#!/usr/bin/env python3
"""Is flux-line localisation more than the |I| weighting pulling x* towards the loudest node?
For nodes beyond a distance cut r > r_c from the origin (node of max |p|), |I|-weighted and
unweighted intersection x* (Eq. 5) for the data vs a null with the same positions and weights
but uniformly random flux directions (50 draws)."""
import sys, numpy as np
Z=np.load('results/room_pipeline_750Hz.npz'); pos=Z['pos']; r=np.linalg.norm(pos,axis=1)
def xstar(P,D,w):
    M=np.eye(3)[None]-D[:,:,None]*D[:,None,:]; A=(w[:,None,None]*M).sum(0); b=(w[:,None,None]*M@P[:,:,None]).sum(0)[:,0]
    return np.linalg.solve(A,b)
rng=np.random.default_rng(0)
print(f"{'data':14s} {'r_c':>5} {'nodes':>6} | {'|I|-w x* [cm]':>13} {'null med (5-95%)':>18} | {'unweighted':>10} {'null med':>9}")
for lab,p,u in (('raw',Z['p_raw'],Z['u_raw']),('reconstructed',Z['p'],Z['u'])):
    I=-0.5*np.real(p[:,None]*np.conj(u)); m=np.linalg.norm(I,axis=1); D=I/np.maximum(m,1e-30)[:,None]
    for rc in (0.0,0.2,0.3,0.5,1.0):
        s=r>rc; P=pos[s]; w=m[s]
        d=np.linalg.norm(xstar(P,D[s],w))*100; du=np.linalg.norm(xstar(P,D[s],np.ones(s.sum())))*100
        nw=[];nu=[]
        for _ in range(50):
            R=rng.standard_normal((s.sum(),3)); R/=np.linalg.norm(R,axis=1)[:,None]
            nw.append(np.linalg.norm(xstar(P,R,w))*100); nu.append(np.linalg.norm(xstar(P,R,np.ones(s.sum())))*100)
        print(f"{lab:14s} {rc:5.1f} {s.sum():6d} | {d:13.1f} {np.median(nw):8.1f} ({np.percentile(nw,5):.0f}-{np.percentile(nw,95):.0f}) | {du:10.1f} {np.median(nu):9.1f}",flush=True)
