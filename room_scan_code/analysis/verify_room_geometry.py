#!/usr/bin/env python3
"""Numbers describing the scan and the raw field (paper Secs. 2.5, 3, 5.1, 5.2), 750 Hz, analysis frame.
Input: results/room_pipeline_750Hz.npz (pos, p_raw, node_err)."""
import numpy as np
Z=np.load('results/room_pipeline_750Hz.npz'); pos=Z['pos']; p=Z['p_raw']; e=Z['node_err']
k=2*np.pi*750/343; r=np.linalg.norm(pos,axis=1)
print("extent x, y, z [m]:", np.round(np.ptp(pos,axis=0),2), "; z from", round(pos[:,2].min(),2), "to", round(pos[:,2].max(),2))
print(f"distance from origin: {r.min():.2f}-{r.max():.2f} m, median {np.median(r):.2f} m")
zc=pos[:,2]; hsl=np.abs(zc-np.median(zc))<0.18*np.ptp(zc)
print(f"horizontal slice: median height {np.median(zc):+.3f} m, half-width {0.18*np.ptp(zc):.2f} m, {hsl.sum()} points")
m=(r>0.3)&(r<2.6)
for s,lab in ((+1,'p r e^{+ikr}'),(-1,'p r e^{-ikr}')):
    ph=np.angle(p[m]*r[m]*np.exp(s*1j*k*r[m])); print(f"phase coherence of {lab} over 0.3-2.6 m: {np.abs(np.mean(np.exp(1j*ph))):.3f}")
m2=(r>0.15)&(r<2.6); print(f"|p| ~ r^{np.polyfit(np.log(r[m2]),np.log(np.abs(p[m2])),1)[0]:.2f} over 0.15-2.6 m (raw)")
print(f"point error: median {100*np.median(e):.0f} %; fraction above 50 %: {100*np.mean(e>0.5):.0f} %")
q=np.percentile(r,[25,75]); print(f"point error by distance: below {q[0]:.2f} m {100*np.median(e[r<q[0]]):.0f} %, beyond {q[1]:.2f} m {100*np.median(e[r>q[1]]):.0f} %")
