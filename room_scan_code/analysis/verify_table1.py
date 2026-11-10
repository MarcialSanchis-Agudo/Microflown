#!/usr/bin/env python3
"""Table 1 and the Section-4 sweeps, reproduced. Canonical monopole / x-dipole, positions uniform
over the sphere with r ~ U[0.10, 0.88] m, circular complex Gaussian noise at 40 dB SNR (relative to
the mean |p|), exterior SH fit (L_max = 8, lambda = 1e-3), evaluated at r = 1 m on the 1 deg x 1 deg
lon-lat grid (361 x 181). Error = RMS over grid points of (|V|/max|V| - E/max E), in %.
Ten realisations (positions AND noise re-drawn) per frequency."""
import sys, numpy as np
sys.path.insert(0,'lib')
from mf_pipeline.mf_directivity import angles_from_pos, fit_complex_SH_nearfield, SHFitOptions, eval_on_lonlat_map
C=343.0
lon,lat,_=eval_on_lonlat_map(np.zeros(1),1.0,1.0,0)          # default 361 x 181
LO,LA=np.meshgrid(np.radians(lon),np.radians(lat)); EX={'mono':np.ones_like(LA),'dip':np.abs(np.cos(LA)*np.cos(LO))}
print(f"grid {len(lon)} x {len(lat)} (step {lon[1]-lon[0]:.1f} deg)")
def run(f,seed,N=3000,L=8,lam=1e-3):
    k=2*np.pi*f/C; rng=np.random.default_rng(seed)
    r=rng.uniform(.10,.88,N); th=np.arccos(rng.uniform(-1,1,N)); ph=rng.uniform(-np.pi,np.pi,N)
    pos=np.column_stack([r*np.sin(th)*np.cos(ph),r*np.sin(th)*np.sin(ph),r*np.cos(th)]); rr,tt,pp=angles_from_pos(pos,np.zeros(3))
    s={'mono':np.exp(1j*k*rr)/rr,'dip':pos[:,0]*np.exp(1j*k*rr)/rr**2*(1j*k-1/rr)}; out={}
    for n,x in s.items():
        sd=np.mean(np.abs(x))/10**(40/20); x=x+sd*(rng.standard_normal(N)+1j*rng.standard_normal(N))/np.sqrt(2)
        a=fit_complex_SH_nearfield(x,rr,tt,pp,k,SHFitOptions(lmax=L,lam=lam))
        V=np.abs(eval_on_lonlat_map(a,k,1.0,L)[2]); E=EX[n]
        out[n]=float(np.sqrt(np.mean((V/V.max()-E/E.max())**2)))*100
    return out
mode=sys.argv[1] if len(sys.argv)>1 else 'table'
if mode=='table':
    print(f"{'f':>7} | mono mean +/- sd | dip mean +/- sd")
    for f in (257.8125,492.1875,750.0,1007.8125,1242.1875,1500.0):
        R=[run(f,s) for s in range(10)]; m=np.array([x['mono'] for x in R]); d=np.array([x['dip'] for x in R])
        print(f"{f:7.1f} | {m.mean():.3f} +/- {m.std(ddof=1):.3f} | {d.mean():.3f} +/- {d.std(ddof=1):.3f}",flush=True)
elif mode=='lam':
    for lam in (1e-8,1e-6,1e-4,1e-3,1e-2,1e-1,1.0):
        R=[run(750.0,s,lam=lam) for s in range(5)]
        print(f"lambda {lam:7.0e}: mono {np.mean([x['mono'] for x in R]):.3f}%, dip {np.mean([x['dip'] for x in R]):.3f}%",flush=True)
elif mode=='N':
    for N in (200,500,1000,2000,3000,6000):
        R=[run(750.0,s,N=N) for s in range(5)]
        print(f"N {N:5d}: mono {np.mean([x['mono'] for x in R]):.3f}%, dip {np.mean([x['dip'] for x in R]):.3f}%",flush=True)
