#!/usr/bin/env python3
"""Reference-pair test of the short-dwell noise model. H1/H2 = (P/R1)/(P/R2) = R2/R1 does not
depend on the probe; for two fixed references its true value is one constant. Its per-cell
scatter is therefore pure transfer-function estimation noise. Compare with the Bendat-Piersol
random error of a TF estimate, per real component s = sqrt((1-g2)/(2 n g2)), with n the number of
averages (dwell blocks, no overlap, or 2*blocks-1 for 50 % overlap), and |e| median = 1.177 s*sqrt(2)
for the ratio of two TF estimates (errors of H1 and H2 roughly independent)."""
import sys, numpy as np; sys.path.insert(0,'lib')
from room_raw import raw
from room_stage2 import dwell_blocks
F0=float(sys.argv[1]) if len(sys.argv)>1 else 750.0
R=raw(F0); nb=dwell_blocks(R['pos']); g2=np.clip(R['coh_r1r2'],1e-3,0.9999)
q=R['p_ref1']/R['p_ref2']; qm=np.median(q.real)+1j*np.median(q.imag); e=np.abs(q/qm-1)
print(f"f={R['f']:.1f} Hz: coh(ref1,ref2) median {np.median(g2):.3f}; ratio scatter median {np.median(e):.3f}")
edges=np.percentile(nb,[0,20,40,60,80,100])
print(f"{'blocks':>13} {'n':>5} {'obs':>6} {'BP n=blk':>9} {'BP n=2blk-1':>11} {'med coh':>8}")
for a,b in zip(edges[:-1],edges[1:]):
    m=(nb>=a)&(nb<=b); gg=np.median(g2[m]); n1=np.median(nb[m]); n2=2*n1-1
    bp=lambda n: 1.177*np.sqrt(2)*np.sqrt((1-gg)/(2*max(n,1)*gg))
    print(f"{a:5.1f}-{b:5.1f} {m.sum():5d} {np.median(e[m]):6.3f} {bp(n1):9.3f} {bp(n2):11.3f} {gg:8.3f}")
mids=[];obs=[]
for a,b in zip(edges[1:-2],edges[2:-1]):
    m=(nb>=a)&(nb<=b); mids.append(np.median(nb[m])); obs.append(np.median(e[m]))
print(f"log-log slope of scatter vs blocks (excluding the 1-block quintile, where coh is trivially 1): {np.polyfit(np.log(mids),np.log(obs),1)[0]:+.2f}")
