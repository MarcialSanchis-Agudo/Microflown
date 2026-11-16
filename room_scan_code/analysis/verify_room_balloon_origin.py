#!/usr/bin/env python3
"""Sensitivity of the near-field balloons (verify_room_balloon.py, 30 deg bins, shell 0.2-0.4 m) to the
choice of origin.  (a) An ideal monopole placed delta away from the origin, sampled at the real scan
positions and processed exactly like the data (|p| r about the origin): the range it shows is spurious.
(b) The measured balloons recomputed with the origin moved by 5 cm along +-x, +-y, +-z."""
import numpy as np
BIN=30; ae=np.arange(-180,181,BIN); ee=np.arange(-90,91,BIN)
def balloon(pos,P,c):
    x=pos-c; r=np.linalg.norm(x,axis=1); m=(r>=0.2)&(r<0.4)
    az=np.degrees(np.arctan2(x[:,1],x[:,0])); el=np.degrees(np.arcsin(np.clip(x[:,2]/np.maximum(r,1e-9),-1,1)))
    V=[]
    for i in range(len(ee)-1):
        for j in range(len(ae)-1):
            s=m&(el>=ee[i])&(el<ee[i+1])&(az>=ae[j])&(az<ae[j+1])
            if s.sum()>=3: V.append(np.median(np.abs(P[s])*r[s]))
    V=np.array(V); return 20*np.log10(V.max()/V.min())
Z=np.load('results/room_pipeline_750Hz.npz'); pos=Z['pos']
dirs=np.vstack([np.eye(3),-np.eye(3)])
for d in (0.05,0.074):
    rg=[balloon(pos,1/np.linalg.norm(pos-d*u,axis=1),np.zeros(3)) for u in dirs]
    print(f"(a) ideal monopole {100*d:.1f} cm from the origin: spurious range {min(rg):.1f}-{max(rg):.1f} dB (analytic at r=0.3 m: {20*np.log10((0.3+d)/(0.3-d)):.1f} dB)")
for F in (492,750,1008,1242):
    Z=np.load(f'results/room_pipeline_{F}Hz.npz'); P=Z['p_raw']
    r0=balloon(pos,P,np.zeros(3)); rs=[balloon(pos,P,0.05*u) for u in dirs]
    print(f"(b) {F} Hz: range {r0:.1f} dB at the origin; {min(rs):.1f}-{max(rs):.1f} dB with the origin moved by 5 cm")
