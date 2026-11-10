#!/usr/bin/env python3
"""
verify_room_polar.py F0 — measured polar patterns of the room scan and the far-field extrapolation test.

Measured patterns (raw nodes; the 16 cm reconstruction neighbourhood spans ~30 deg at 0.3 m, coarser
than the bins).  Planes: horizontal through the source (|elevation| < 20 deg) and vertical through
the x axis (|y|/r < sin 20 deg, angle atan2(z, x)).  Shells: inner 0.2-0.4 m, outer 0.8-1.2 m.
15-deg bins with >= 5 nodes; value = median of |p| r; 90 % bootstrap interval (200 resamples of the
bin's nodes); dB re the shell's largest bin.  Coverage = fraction of the 24 bins with >= 5 nodes.

Extrapolation: exterior SH (Eq. 2, lambda = 1e-3) fitted to the raw nodes 0.15-0.5 m (conj(p): the
data are e^{+iwt}, h_l^(1) is outgoing for e^{-iwt}), evaluated at the outer-shell nodes, binned the
same way.  Reported: RMS dB difference between predicted and measured outer patterns over bins covered
by both, for L = 0, 2, 4, and between the measured inner and outer patterns; plus the node-level
median complex error and median |dB| error of the prediction at the 0.8-1.2 m nodes (trivial p = 0: 1).
"""
import sys, json, numpy as np
sys.path.insert(0, 'lib')
from mf_pipeline.mf_directivity import angles_from_pos, fit_complex_SH_nearfield, SHFitOptions, eval_complex_SH_points
F0 = float(sys.argv[1]) if len(sys.argv) > 1 else 750.0
Z = np.load(f'results/room_pipeline_{int(round(F0))}Hz.npz'); pos = Z['pos']; k = 2 * np.pi * F0 / 343.0; P = Z['p_raw']
r = np.linalg.norm(pos, axis=1)
el = np.degrees(np.arcsin(np.clip(pos[:, 2] / np.maximum(r, 1e-9), -1, 1))); az = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]))
PL = {'horizontal': (np.abs(el) < 20, az),
      'vertical': (np.abs(pos[:, 1]) / np.maximum(r, 1e-9) < np.sin(np.radians(20)), np.degrees(np.arctan2(pos[:, 2], pos[:, 0])))}
SH = {'inner': (0.2, 0.4), 'outer': (0.8, 1.2)}
edges = np.arange(-180, 181, 15); cen = 0.5 * (edges[:-1] + edges[1:]); nb_ = len(cen)
rng = np.random.default_rng(0)

def pattern(val, sel, ang, boot=True):
    v = np.full(nb_, np.nan); lo = np.full(nb_, np.nan); hi = np.full(nb_, np.nan)
    for j in range(nb_):
        mm = np.flatnonzero(sel & (ang >= edges[j]) & (ang < edges[j + 1]))
        if len(mm) >= 5:
            v[j] = np.median(val[mm])
            if boot:
                b = [np.median(val[rng.choice(mm, len(mm))]) for _ in range(200)]; lo[j], hi[j] = np.percentile(b, [5, 95])
    ref = np.nanmax(v)
    return 20 * np.log10(v / ref), 20 * np.log10(lo / ref), 20 * np.log10(hi / ref)

# SH fits on inner raw nodes
rr, tt, pp = angles_from_pos(pos, np.zeros(3)); inn = (r >= 0.15) & (r <= 0.5); outn = (r >= 0.8) & (r <= 1.2)
pred = {}
for L in (0, 2, 4):
    a = fit_complex_SH_nearfield(np.conj(P[inn]), rr[inn], tt[inn], pp[inn], k, SHFitOptions(lmax=L, lam=1e-3))
    pred[L] = np.conj(eval_complex_SH_points(a, rr, tt, pp, k, L))
res = {'f': F0, 'bins_deg': cen.tolist(), 'n_inner_fit': int(inn.sum()), 'n_outer': int(outn.sum())}
for plane, (sel0, ang) in PL.items():
    pats = {}
    for sh, (a_, b_) in SH.items():
        sel = sel0 & (r >= a_) & (r < b_)
        pats[sh] = pattern(np.abs(P) * r, sel, ang); res[f'{plane}_{sh}'] = [np.where(np.isfinite(x), np.round(x, 2), None).tolist() for x in pats[sh]]
        res[f'{plane}_{sh}_n'] = int(sel.sum()); res[f'{plane}_{sh}_coverage'] = float(np.isfinite(pats[sh][0]).mean())
    sel = sel0 & (r >= 0.8) & (r < 1.2)
    for L in (0, 2, 4):
        pl = pattern(np.abs(pred[L]) * r, sel, ang, boot=False)[0]; res[f'{plane}_pred_L{L}'] = np.where(np.isfinite(pl), np.round(pl, 2), None).tolist()
        ok = np.isfinite(pl) & np.isfinite(pats['outer'][0]); res[f'{plane}_pred_L{L}_rms_dB'] = float(np.sqrt(np.mean((pl[ok] - pats['outer'][0][ok]) ** 2)))
    ok = np.isfinite(pats['inner'][0]) & np.isfinite(pats['outer'][0])
    res[f'{plane}_inner_vs_outer_rms_dB'] = float(np.sqrt(np.mean((pats['inner'][0][ok] - pats['outer'][0][ok]) ** 2))) if ok.sum() else None
    res[f'{plane}_inner_vs_outer_nbins'] = int(ok.sum())
    ci = np.nanmedian(pats['outer'][2] - pats['outer'][1])
    print(f"{plane:10s}: inner {res[f'{plane}_inner_n']} nodes, coverage {res[f'{plane}_inner_coverage']:.2f}, range {np.nanmax(pats['inner'][0])-np.nanmin(pats['inner'][0]):.1f} dB | "
          f"outer {res[f'{plane}_outer_n']} nodes, coverage {res[f'{plane}_outer_coverage']:.2f}, range {np.nanmax(pats['outer'][0])-np.nanmin(pats['outer'][0]):.1f} dB, median 90% CI width {ci:.1f} dB | "
          f"inner vs outer RMS {res[f'{plane}_inner_vs_outer_rms_dB']} dB over {ok.sum()} bins | SH->outer RMS dB: L0 {res[f'{plane}_pred_L0_rms_dB']:.1f}, L2 {res[f'{plane}_pred_L2_rms_dB']:.1f}, L4 {res[f'{plane}_pred_L4_rms_dB']:.1f}")
for L in (0, 2, 4):
    e = np.abs(pred[L][outn] - P[outn]) / np.abs(P[outn]); edb = np.abs(20 * np.log10(np.abs(pred[L][outn]) / np.abs(P[outn])))
    res[f'node_L{L}'] = (float(np.median(e)), float(np.median(edb)))
    print(f"node level, SH L={L} fitted on {inn.sum()} nodes at 0.15-0.5 m -> {outn.sum()} nodes at 0.8-1.2 m: median complex error {np.median(e):.2f} (p=0: 1.00), median |dB| {np.median(edb):.1f} dB")
json.dump(res, open(f'results/room_polar_{int(round(F0))}Hz.json', 'w'))
