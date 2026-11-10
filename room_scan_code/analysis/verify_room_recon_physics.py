#!/usr/bin/env python3
"""Raw vs Stage-2 reconstruction (750 Hz, oriented frame, source at origin).
(a) Euler consistency |corr(u, grad p/(i w rho))| — p and u are reconstructed INDEPENDENTLY,
    so agreement is an independent physics check; (b) |p| ~ r^s; (c) flux outward;
(d) bearing of the intensity-weighted mean flux vs the true bearing from the cloud centroid."""
import sys, numpy as np; sys.path.insert(0,'lib')
from numpy.linalg import lstsq
N_NB=int(sys.argv[1]) if len(sys.argv)>1 else 96
Z=np.load(f'results/room_pipeline_750Hz.npz'); pos=Z['pos']; r=np.linalg.norm(pos,axis=1)
om=2*np.pi*750; rho=1.204; rng=np.random.default_rng(0); sel=rng.choice(len(pos),1500,replace=False)
def ang(v): return np.degrees(np.arctan2(v[1],v[0])), np.degrees(np.arcsin(np.clip(v[2]/np.linalg.norm(v),-1,1)))
c=pos.mean(0); tru=-c/np.linalg.norm(c)
print(f"n_nb = {N_NB}; true bearing from centroid az {ang(tru)[0]:.1f}, el {ang(tru)[1]:.1f}")
for lab,p,u in (('raw',Z['p_raw'],Z['u_raw']),('reconstructed',Z['p'],Z['u'])):
    cs=[]
    for i in sel:
        d=np.linalg.norm(pos-pos[i],axis=1); nb=np.where((d>0)&(d<0.08))[0]
        if len(nb)<8: continue
        X=np.column_stack([np.ones(len(nb)),pos[nb]-pos[i]]); g=lstsq(X,p[nb],rcond=None)[0][1:]
        ue=g/(1j*om*rho); cs.append(np.real(np.vdot(ue,u[i]))/(np.linalg.norm(ue)*np.linalg.norm(u[i])+1e-30))
    cs=np.array(cs)
    m=(r>0.15)&(r<2.6); s=np.polyfit(np.log(r[m]),np.log(np.abs(p[m])),1)[0]
    I=-0.5*np.real(p[:,None]*np.conj(u)); w=np.linalg.norm(I,axis=1); rh=pos/np.maximum(r,1e-9)[:,None]
    out=np.sum(I*rh,1)>0; dh=I.sum(0)/w.sum(); dh/=np.linalg.norm(dh)
    err=np.degrees(np.arccos(np.clip(np.dot(-dh,tru),-1,1)))
    print(f"{lab:14s}: Euler corr median {np.median(cs):+.2f} (sign-agree {np.mean(cs<0)*100:.0f}% as -1 convention) | |p|~r^{s:.2f} | "
          f"outward {out.mean()*100:.0f}% nodes, {np.sum(w*out)/w.sum()*100:.0f}% |I| | bearing az {ang(-dh)[0]:.1f} el {ang(-dh)[1]:.1f} err {err:.1f} deg")
