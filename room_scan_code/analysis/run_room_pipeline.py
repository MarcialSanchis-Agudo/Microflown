#!/usr/bin/env python3
"""
run_room_pipeline.py F0 — the full room pipeline at one frequency, with every quality metric.

Stage 1  raw transfer functions (ref1) from the .psd, rotated into the supplied frame
         (origin = node of maximum |p| at 750 Hz; rotation fitted once, exact to 3e-12 m).
Stage 2a four-feature anomaly score, top 5 % per channel flagged       (room_stage2.score)
Stage 2b dwell-weighted local regular Helmholtz reconstruction, 96 nb  (room_stage2.local_recon)
Stage 3  energy flux I = -1/2 Re(p conj(u_stored)); flux-line intersection from nodes r > 0.2 m.
Metrics  raw node error (leave-one-out over all nodes, 23-24 nb); split-half random error (raw nearest node / recon
         96 per half); Euler consistency (median corr of u_stored with grad p/(i w rho0), 8 cm
         linear fit, 1500 sampled nodes); outward flux; |p| ~ r^s; localisation vs random directions.
Writes results/room_pipeline_<f>Hz.npz and prints one summary block.
"""
import sys, json, numpy as np
sys.path.insert(0, 'lib')
from room_raw import raw, to_frame
from room_stage2 import score, local_recon, local_fill, dwell_blocks
from scipy_free_interp import knn

F0 = float(sys.argv[1]) if len(sys.argv) > 1 else 750.0
R = raw(F0); f = R['f']; k = 2 * np.pi * f / 343.0; om = 2 * np.pi * f; RHO = 1.204
X = R['pos']; pos, Rm = to_frame(X); P = R['p_ref1']; U = (Rm @ R['u_ref1'].T).T
nb = dwell_blocks(X); r = np.linalg.norm(pos, axis=1)
out = dict(f=f, coh_median=float(np.median(R['coh_p_ref1'])))

# Stage 2a
s, fl, _ = score(pos, P, U)
anyf = fl['p'] | fl['ux'] | fl['uy'] | fl['uz']
out['flag_any_pct'] = float(anyf.mean() * 100)
out['coh_flagged'] = float(np.median(R['coh_p_ref1'][fl['p']])); out['coh_kept'] = float(np.median(R['coh_p_ref1'][~fl['p']]))
# Stage 2b
q = np.arange(len(P)); rec = {}
for n, x, fln in (('p', P, fl['p']), ('ux', U[:, 0], fl['ux']), ('uy', U[:, 1], fl['uy']), ('uz', U[:, 2], fl['uz'])):
    rec[n] = local_recon(pos, x, nb * (~fln), k, q, n_nb=96)
Pr = rec['p']; Ur = np.column_stack([rec['ux'], rec['uy'], rec['uz']])

# metrics: held-out raw node error
pr = local_fill(pos, P, np.zeros(len(P), bool), k, query=np.arange(len(P)))   # leave-one-out, all nodes
node_err = np.abs(pr - P) / np.abs(pr); out['node_err_raw'] = float(np.median(node_err))
# split-half
rng = np.random.default_rng(4); qq = rng.choice(len(P), 800, replace=False); half = rng.random(len(P)) < 0.5
Aa, Bb = np.flatnonzero(half), np.flatnonzero(~half)
ia, _ = knn(pos[qq], pos[Aa], 1); ib, _ = knn(pos[qq], pos[Bb], 1); pa, pb = P[Aa[ia[:, 0]]], P[Bb[ib[:, 0]]]
out['split_raw'] = float(np.median(np.abs(pa - pb) / np.sqrt(2) / np.abs(0.5 * (pa + pb))))
w = nb * (~fl['p'])
pa = local_recon(pos, P, w, k, qq, n_nb=96, src=Aa); pb = local_recon(pos, P, w, k, qq, n_nb=96, src=Bb)
out['split_rec'] = float(np.median(np.abs(pa - pb) / np.sqrt(2) / np.abs(0.5 * (pa + pb))))
# Euler
from numpy.linalg import lstsq
sel = np.random.default_rng(0).choice(len(pos), 1500, replace=False)
for lab, p_, u_ in (('raw', P, U), ('rec', Pr, Ur)):
    cs = []
    for i in sel:
        d = np.linalg.norm(pos - pos[i], axis=1); nbh = np.where((d > 0) & (d < 0.08))[0]
        if len(nbh) < 8: continue
        Xm = np.column_stack([np.ones(len(nbh)), pos[nbh] - pos[i]]); g = lstsq(Xm, p_[nbh], rcond=None)[0][1:]
        ue = g / (1j * om * RHO); cs.append(np.real(np.vdot(ue, u_[i])) / (np.linalg.norm(ue) * np.linalg.norm(u_[i]) + 1e-30))
    out['euler_' + lab] = float(np.median(cs)); out['euler_n'] = len(cs)
# flux, decay, localisation
def xstar(Pp, D, ww):
    M = np.eye(3)[None] - D[:, :, None] * D[:, None, :]; Am = (ww[:, None, None] * M).sum(0); b = (ww[:, None, None] * M @ Pp[:, :, None]).sum(0)[:, 0]
    return np.linalg.solve(Am, b)
m = (r > 0.15) & (r < 2.6); far = r > 0.2
for lab, p_, u_ in (('raw', P, U), ('rec', Pr, Ur)):
    I = -0.5 * np.real(p_[:, None] * np.conj(u_)); Im = np.linalg.norm(I, axis=1); D = I / np.maximum(Im, 1e-30)[:, None]
    ob = np.sum(D * pos / np.maximum(r, 1e-9)[:, None], 1) > 0
    out['outward_' + lab] = float(ob.mean() * 100)
    out['slope_' + lab] = float(np.polyfit(np.log(r[m]), np.log(np.abs(p_[m])), 1)[0])
    out['loc_' + lab + '_cm'] = float(np.linalg.norm(xstar(pos[far], D[far], Im[far])) * 100)
rng = np.random.default_rng(0); nul = []
Im_rec = np.linalg.norm(-0.5 * np.real(Pr[:, None] * np.conj(Ur)), axis=1)
for _ in range(20):
    Rd = rng.standard_normal((far.sum(), 3)); Rd /= np.linalg.norm(Rd, axis=1)[:, None]
    nul.append(np.linalg.norm(xstar(pos[far], Rd, Im_rec[far])) * 100)
out['loc_null_cm_median'] = float(np.median(nul)); out['loc_null_cm_p5'] = float(np.percentile(nul, 5))
np.savez(f'results/room_pipeline_{int(round(f))}Hz.npz', pos=pos, p_raw=P, u_raw=U, p=Pr, u=Ur, dwell=nb,
         coh=R['coh_p_ref1'], node_err=node_err, coh_r1r2=R['coh_r1r2'], ref_ratio=R['p_ref1']/R['p_ref2'], flag_p=fl['p'], flag_ux=fl['ux'], flag_uy=fl['uy'], flag_uz=fl['uz'], score_p=s['p'])
print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in out.items()}))
