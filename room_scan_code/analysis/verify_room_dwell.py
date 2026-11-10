#!/usr/bin/env python3
"""Does the node-scale error scale with the dwell time (number of FFT averages) per cell?
Cells mapped to nodes via SampleCellMatrix grid indices (100 % match). Held-out local
Helmholtz prediction error vs total dwell samples, and the Bendat-Piersol TF estimate
variance (1-g2)/(2 n g2) with n = dwell/2048."""
import sys, numpy as np; sys.path.insert(0,'lib')
from room_raw import raw, h5, G
from room_stage2 import local_fill
S=h5().dataset(G+'SampleCellMatrix').astype(np.int64)
F0=float(sys.argv[1]) if len(sys.argv)>1 else 750.0
R=raw(F0); pos=R['pos']; P=R['p_ref1']; coh=R['coh_p_ref1']; k=2*np.pi*R['f']/343
g=np.round(pos/0.02).astype(int); key=lambda a: a[:,0]*10**8+a[:,1]*10**4+a[:,2]
kn=key(g-g.min(0)); ijk=S[:3].T; ks=key(ijk-ijk.min(0))
dw={}; 
for kk,d in zip(ks,S[4]-S[3]): dw[kk]=dw.get(kk,0)+d
dwell=np.array([dw[x] for x in kn]); nblk=dwell/2048
print(f"f={R['f']:.1f} Hz; dwell per node: median {np.median(dwell)/48000*1000:.0f} ms = {np.median(nblk):.1f} blocks")
rng=np.random.default_rng(3); H=rng.choice(len(P),2000,replace=False); bad=np.zeros(len(P),bool); bad[H]=True
pr=local_fill(pos,P,bad,k,query=H); e=np.abs(pr-P[H])/np.abs(pr)
edges=np.percentile(nblk[H],[0,20,40,60,80,100]); print(f"{'blocks':>14} {'n':>5} {'median err':>10} {'median coh':>10}")
mids=[];meds=[]
for a,b in zip(edges[:-1],edges[1:]):
    m=(nblk[H]>=a)&(nblk[H]<=b); mids.append(np.median(nblk[H][m])); meds.append(np.median(e[m]))
    print(f"{a:6.1f}-{b:6.1f} {m.sum():5d} {np.median(e[m]):10.3f} {np.median(coh[H][m]):10.3f}")
sl=np.polyfit(np.log(mids),np.log(meds),1)[0]; print(f"log-log slope of median error vs blocks: {sl:+.2f}  (pure averaging noise: -0.50)")
g2=np.clip(coh[H],1e-3,0.999); pred=np.sqrt(2*(1-g2)/(2*np.maximum(nblk[H],1)*g2))
print(f"corr(log err, log BP-predicted err) = {np.corrcoef(np.log(e),np.log(pred))[0,1]:+.2f}; median predicted {np.median(pred):.2f} vs observed {np.median(e):.2f}")
