#!/usr/bin/env python3
"""Check local_fill on known fields at the real room node positions (750 Hz):
monopole at the loudspeaker, plus a monopole + 3 image-like sources (room-like field)."""
import sys, numpy as np; sys.path.insert(0,'lib')
from room_stage2 import local_fill
from scipy_free_interp import knn
from room_raw import raw, to_frame
pos=to_frame(raw(750.0)['pos'])[0]; k=2*np.pi*750/343
def G(src): r=np.linalg.norm(pos-src,axis=1); return np.exp(1j*k*r)/r
fields={'monopole':G(np.zeros(3)),
        'room-like':G(np.zeros(3))+0.6*G(np.array([0,0,-1.9]))+0.5*G(np.array([0.9,0,0]))+0.4*G(np.array([0,2.6,0]))}
rng=np.random.default_rng(0); q=rng.choice(len(pos),600,replace=False); bad=np.zeros(len(pos),bool); bad[q]=True
nn=knn(pos,pos,2)[1][:,1]; print(f"node spacing: median NN distance {np.median(nn)*100:.1f} cm, lambda={2*np.pi/k*100:.0f} cm")
for n,f in fields.items():
    pr=local_fill(pos,f,bad,k,query=q)
    e=np.abs(pr-f[q])/np.abs(f[q])
    # IDW baseline
    good=np.flatnonzero(~bad); idx,d=knn(pos[q],pos[good],8); w=1/np.maximum(d,1e-9)**2
    idw=(w*f[good][idx]).sum(1)/w.sum(1); ei=np.abs(idw-f[q])/np.abs(f[q])
    print(f"{n:10s}: local j_l fill median rel. error {np.median(e)*100:.2f}% (p90 {np.percentile(e,90)*100:.2f}%) | IDW {np.median(ei)*100:.2f}%")
