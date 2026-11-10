#!/usr/bin/env python3
"""Choose the neighbourhood of the dwell-weighted local Helmholtz reconstruction.
Bias: known room-like field at the real node positions (noise-free).
Noise: split-half on the real data — nodes split into two random halves A/B; reconstruct at
common query points from A only and from B only; independent noise => residual noise of one
reconstruction = |pA - pB| / sqrt(2)."""
import sys, numpy as np; sys.path.insert(0,'lib')
from room_stage2 import local_recon, dwell_blocks
from room_raw import raw, frame_transform
R=raw(750.0); pos=R['pos']; P=R['p_ref1']; k=2*np.pi*R['f']/343
nb=dwell_blocks(pos); w=nb.copy()
s0=np.array([np.median(pos[:,0]),np.max(pos[:,1]),np.median(pos[:,2])])  # arbitrary; field shape only
# synthetic room-like field: loudspeaker estimate = acoustic centre used for the oriented frame
Rm, t_ = frame_transform(); src = -Rm.T @ t_                 # origin of the analysis frame, in raw coordinates
Gf=lambda s: (lambda r: np.exp(1j*k*r)/r)(np.linalg.norm(pos-s,axis=1))
F=Gf(src)+0.6*Gf(src+[0,0,-1.9])+0.5*Gf(src+[0.9,0,0])+0.4*Gf(src+[0,2.6,0])
rng=np.random.default_rng(4); q=rng.choice(len(P),800,replace=False)
half=rng.random(len(P))<0.5; Aset=np.flatnonzero(half); Bset=np.flatnonzero(~half)
print(f"{'n_nb':>5} {'bias (synthetic)':>17} {'noise (split-half)':>19} {'raw node noise':>15}")
for n in (12,24,48,96,192):
    b=np.median(np.abs(local_recon(pos,F,w,k,q,n_nb=n)-F[q])/np.abs(F[q]))
    pa=local_recon(pos,P,w,k,q,n_nb=n//2,src=Aset); pb=local_recon(pos,P,w,k,q,n_nb=n//2,src=Bset)
    s=np.median(np.abs(pa-pb)/np.sqrt(2)/np.abs(0.5*(pa+pb)))
    print(f"{n:5d} {b*100:16.2f}% {s*100:18.1f}%")
