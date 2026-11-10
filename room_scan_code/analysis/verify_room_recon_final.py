#!/usr/bin/env python3
"""Numbers quoted for the chosen reconstruction (n_nb = 96, dwell-weighted, L = 2, flagged channels
weighted 0): (1) neighbourhood radius; (2) bias on a known room-like field at the real positions;
(3) residual random error by split-half (two random halves, 96 neighbours each, |pA-pB|/sqrt2);
(4) raw node error by the same split-half construction with 1 neighbour-free comparison
(held-out local prediction, from verify_room_noise_model.py); (5) averaging needed for a target error."""
import sys, numpy as np; sys.path.insert(0,'lib')
from room_stage2 import local_recon, dwell_blocks
from room_raw import raw, frame_transform
from scipy_free_interp import knn
R=raw(750.0); pos=R['pos']; P=R['p_ref1']; k=2*np.pi*R['f']/343; nb=dwell_blocks(pos)
S2=np.load('results/room_pipeline_750Hz.npz'); w=nb*(~S2['flag_p'])
_,d=knn(pos[::10],pos,96); print(f"(1) radius of the 96-node neighbourhood: median {np.median(d[:,-1])*100:.1f} cm (lambda/{2*np.pi/k/np.median(d[:,-1]):.1f})")
Rm, t_ = frame_transform(); src = -Rm.T @ t_                 # origin of the analysis frame, in raw coordinates
Gf=lambda s: (lambda r: np.exp(1j*k*r)/r)(np.linalg.norm(pos-s,axis=1))
F=Gf(src)+0.6*Gf(src+Rm.T@[0,0,-1.9])+0.5*Gf(src+Rm.T@[0.9,0,0])+0.4*Gf(src+Rm.T@[0,2.6,0])
rng=np.random.default_rng(4); q=rng.choice(len(P),800,replace=False)
b=np.abs(local_recon(pos,F,w,k,q,n_nb=96)-F[q])/np.abs(F[q]); print(f"(2) bias on the known room-like field: median {np.median(b)*100:.1f}%, p90 {np.percentile(b,90)*100:.1f}%")
half=rng.random(len(P))<0.5
pa=local_recon(pos,P,w,k,q,n_nb=96,src=np.flatnonzero(half)); pb=local_recon(pos,P,w,k,q,n_nb=96,src=np.flatnonzero(~half))
e=np.abs(pa-pb)/np.sqrt(2)/np.abs(0.5*(pa+pb)); print(f"(3) residual random error (split-half): median {np.median(e)*100:.0f}%")
# (5) Bendat-Piersol: median |e| of one TF estimate = 1.177*sqrt((1-g2)/(2 n g2)); solve for n
for g2 in (0.7,0.9):
    for et in (0.1,0.2):
        n=1.177**2*(1-g2)/(2*g2*et**2); T=n*2048/48000
        print(f"(5) coherence {g2}, target {et*100:.0f}%: n = {n:.1f} blocks = {T:.2f} s per cell -> sweep <= {0.02/T*100:.1f} cm/s for 2 cm cells")
print(f"    achieved: median dwell {np.median(nb)*2048/48000:.3f} s per cell = {0.02/(np.median(nb)*2048/48000)*100:.0f} cm/s")
