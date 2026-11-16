#!/usr/bin/env python3
"""Averaging scheme of the export and its consequences (paper Secs. 2.6, 5.2, 6).
1. Detect the block step from the data: a transfer function averaged over a single block has a
   coherence of exactly one, so the dwell at which coh < 1 first appears gives the step.
2. Hann window, 50 % overlap: correlation between adjacent periodograms (Welch 1967) and the
   equivalent number of independent averages n_eff.
3. Monte Carlo of the transfer-function estimator (white noise, set coherence, Hann, 50 % overlap)
   against Eq. (11) with n = n_eff.
4. Reference-pair scatter per dwell quintile against Eq. (11) x sqrt(2) with n_eff.
5. Sweep-speed rule of Sec. 6 with n_eff.
Input: results/room_pipeline_<f>Hz.npz (fields dwell [recorded samples/2048], coh, coh_r1r2, ref_ratio)."""
import sys, numpy as np
F=int(sys.argv[1]) if len(sys.argv)>1 else 750
Z=np.load(f'results/room_pipeline_{F}Hz.npz'); d=Z['dwell']; coh=Z['coh']
B=2048; FS=48000.0
print(f"== 1. block step from the data, {F} Hz ({len(d)} cells)")
for lo,hi in [(1.0,1.45),(1.45,1.50),(1.50,1.55),(1.55,2.0),(2.0,2.5)]:
    m=(d>=lo)&(d<hi); print(f"  dwell {lo:.2f}-{hi:.2f} blocks: {m.sum():5d} cells, coherence exactly 1: {100*np.mean(np.abs(coh[m]-1)<1e-5):5.1f} %")
step=0.5
print(f"  -> second average starts at 1.5 blocks: step {int(step*B)} samples, i.e. 50 % overlap")
def K_of(dw): return np.floor((dw*B-B)/(step*B)+1e-9).astype(int)+1
def neff(K,rho):  # Welch: var ratio = (1/K)[1+2 (K-1)/K rho]
    K=np.asarray(K,float); return K/(1+2*(K-1)/K*rho)
n=np.arange(B); hann=0.5-0.5*np.cos(2*np.pi*n/B)
def rho_of(w,D): return (np.sum(w[:-D]*w[D:])**2)/(np.sum(w*w)**2)
rH=rho_of(hann,B//2); rR=rho_of(np.ones(B),B//2)
print(f"== 2. correlation of adjacent periodograms at 50 % overlap: Hann {rH:.4f}, rectangular {rR:.4f}")
K=K_of(d); ne=neff(K,rH)
print(f"  averages per cell: median K = {np.median(K):.0f} (no-overlap count {np.median(d):.2f}); n_eff (Hann) median {np.median(ne):.2f}")
print(f"  cells with a single average (K=1): {100*np.mean(K==1):.1f} %")
# 3. Monte Carlo
rng=np.random.default_rng(1); Bs=256; hs=0.5-0.5*np.cos(2*np.pi*np.arange(Bs)/Bs)
def est(L,g2,win,stp,trials=400):
    errs=[]
    for _ in range(trials):
        x=rng.standard_normal(L); s2=(1-g2)/g2; y=x+np.sqrt(s2)*rng.standard_normal(L)
        starts=range(0,L-Bs+1,stp); Sxy=0; Sxx=0
        for s in starts:
            X=np.fft.rfft(win*x[s:s+Bs]); Y=np.fft.rfft(win*y[s:s+Bs]); Sxy=Sxy+Y*np.conj(X); Sxx=Sxx+np.abs(X)**2
        H=Sxy/Sxx; errs.append(np.abs(H[5:-5]-1))
    return np.median(np.concatenate(errs))
bp=lambda n,g2: 1.177*np.sqrt((1-g2)/(2*n*g2))
print("== 3. Monte Carlo of the estimator (Hann, 50 % overlap) vs Eq. (11) with n_eff")
print(f"  {'L/B':>5} {'K':>3} {'n_eff':>6} {'g2':>4} {'sim':>6} {'Eq11(n_eff)':>11} {'Eq11(L/B)':>9}")
for LB in (2.0,3.0,4.0,6.0,10.0):
    L=int(LB*Bs); Kk=(L-Bs)//(Bs//2)+1; nn=float(neff(Kk,rH))
    for g2 in (0.7,0.9):
        sim=est(L,g2,hs,Bs//2)
        print(f"  {LB:5.1f} {Kk:3d} {nn:6.2f} {g2:4.1f} {sim:6.3f} {bp(nn,g2):11.3f} {bp(LB,g2):9.3f}")
# 4. reference pair
g2r=np.clip(Z['coh_r1r2'],1e-3,0.9999); q=Z['ref_ratio']; qm=np.median(q.real)+1j*np.median(q.imag); e=np.abs(q/qm-1)
edges=np.percentile(d,[0,20,40,60,80,100])
print("== 4. reference-pair scatter per dwell quintile vs sqrt(2) x Eq. (11)")
print(f"  {'blocks':>11} {'K med':>5} {'n_eff':>6} {'obs':>6} {'pred n_eff':>10} {'pred L/B':>8} {'coh':>6}")
for a,b in zip(edges[:-1],edges[1:]):
    m=(d>=a)&(d<=b); gg=np.median(g2r[m]); nn=np.median(ne[m])
    print(f"  {a:4.1f}-{b:5.1f} {np.median(K[m]):5.0f} {nn:6.2f} {np.median(e[m]):6.3f} {np.sqrt(2)*bp(nn,gg):10.3f} {np.sqrt(2)*bp(np.median(d[m]),gg):8.3f} {gg:6.3f}")
qs=[(a,b) for a,b in zip(edges[1:-1],edges[2:])]
nn_=[np.median(ne[(d>=a)&(d<=b)]) for a,b in qs]; oo_=[np.median(e[(d>=a)&(d<=b)]) for a,b in qs]
print(f"  log-log slope of the scatter vs n_eff, quintiles 2-5 (K>1): {np.polyfit(np.log(nn_),np.log(oo_),1)[0]:+.2f} (pure averaging noise: -0.50)")
# 4b. probe pressure: point error vs Eq. (11) with n_eff and the probe-reference coherence
pe=Z['node_err']; g2p=np.clip(coh,1e-3,0.999); ok=K>1
pred=bp(ne,g2p)
print("== 4b. probe-pressure point error (leave-one-out) vs Eq. (11), cells with K>1")
print(f"  median observed {np.median(pe[ok]):.3f}, median Eq.(11) {np.median(pred[ok]):.3f}, ratio {np.median(pe[ok])/np.median(pred[ok]):.2f}")
print(f"  corr(log observed, log predicted) over cells: {np.corrcoef(np.log(pe[ok]),np.log(pred[ok]))[0,1]:+.2f}")
mids=[];meds=[]
for a,b in zip(edges[:-1],edges[1:]):
    m=(d>=a)&(d<=b); mids.append(np.median(ne[m])); meds.append(np.median(pe[m]))
x=1/np.array(mids); y=np.array(meds)**2; A=np.vstack([x,np.ones_like(x)]).T; (aa,bb),*_=np.linalg.lstsq(A,y,rcond=None)
print("  quintile medians:", " ".join(f"{v:.2f}" for v in meds))
print(f"  fit of median^2 = a/n_eff + b over quintiles: a={aa:.3f}, b={bb:.3f} -> dwell-independent floor {np.sqrt(max(bb,0)):.2f}")
# 5. sweep-speed rule
print("== 5. sweep-speed rule, target median error 10 %")
for g2 in (0.9,0.7):
    nreq=1.177**2*(1-g2)/(2*g2*0.1**2)
    Kreq=int(np.ceil(nreq*(1+2*rH)))  # large-K limit of n_eff = K/(1+2 rho)
    while neff(Kreq,rH)<nreq: Kreq+=1
    Lb=1+(Kreq-1)*step; t=Lb*B/FS
    print(f"  g2={g2}: n_eff {nreq:.1f} -> K={Kreq} averages, record {Lb:.1f} blocks = {t:.2f} s per cell, "
          f"{2/t:.1f} cm/s for 2 cm cells; cells reaching it: {100*np.mean(K>=Kreq):.1f} %; 9190 cells: {9190*t/3600:.1f} h")
print(f"  scan: median {np.median(d)*B/FS:.3f} s per cell -> {2/(np.median(d)*B/FS):.0f} cm/s")
