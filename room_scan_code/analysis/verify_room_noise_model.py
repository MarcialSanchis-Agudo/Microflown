#!/usr/bin/env python3
"""What is the node-scale noise of the room scan? Held-out local Helmholtz prediction
(10 % of nodes, from 24 neighbours) at five frequencies, split into magnitude and phase,
and against coherence. Segment bookkeeping from SampleCellMatrix."""
import sys, numpy as np; sys.path.insert(0,'lib')
from room_raw import raw, h5, G
from room_stage2 import local_fill
S=h5().dataset(G+'SampleCellMatrix').astype(np.int64)
seg=S[4]-S[3]; cells=S[0]*10**8+S[1]*10**4+S[2]; _,cnt=np.unique(cells,return_counts=True)
print(f"segments {S.shape[1]}, cells {len(cnt)}; segments per cell: 1 -> {np.mean(cnt==1)*100:.0f}%, >=2 -> {np.mean(cnt>=2)*100:.0f}%")
print(f"segment length [samples @48 kHz]: median {np.median(seg):.0f} ({np.median(seg)/48000:.3f} s), p10 {np.percentile(seg,10):.0f}, p90 {np.percentile(seg,90):.0f}")
print(f"=> 2048-pt blocks per segment (no overlap): median {np.median(seg)/2048:.1f}")
rng=np.random.default_rng(2)
print(f"\n{'f':>7} {'kept med coh':>12} | {'complex err':>11} {'|p| err':>8} {'phase err':>9} | err by coh: <0.5 / 0.5-0.8 / >=0.8 | k*2cm")
for f0 in (257.8125,492.1875,750.0,1007.8125,1242.1875):
    R=raw(f0); k=2*np.pi*R['f']/343; pos=R['pos']; P=R['p_ref1']; coh=R['coh_p_ref1']
    H=rng.choice(len(P),len(P)//10,replace=False); bad=np.zeros(len(P),bool); bad[H]=True
    pr=local_fill(pos,P,bad,k,query=H)
    ec=np.abs(pr-P[H])/np.abs(pr); em=np.abs(np.log(np.abs(P[H])/np.abs(pr))); ep=np.abs(np.angle(P[H]*np.conj(pr)))
    c=coh[H]; b=[np.median(ec[(c>=lo)&(c<hi)]) for lo,hi in ((0,.5),(.5,.8),(.8,1.01))]
    print(f"{R['f']:7.1f} {np.median(coh):12.3f} | {np.median(ec):11.3f} {np.median(em):8.3f} {np.degrees(np.median(ep)):8.1f}d | {b[0]:.2f} / {b[1]:.2f} / {b[2]:.2f} | {k*0.02:.2f} rad")
