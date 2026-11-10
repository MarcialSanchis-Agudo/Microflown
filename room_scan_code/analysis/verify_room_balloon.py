#!/usr/bin/env python3
"""Directivity balloons of the room scan, after Fernandez Comesana et al. (ICSV 2012, Eq. 3):
each raw point at distance r from the origin is projected onto a sphere by the spreading factor,
|p| r; points in the shell 0.2-0.4 m are binned on a BIN x BIN deg azimuth-elevation grid (BIN = argv[1], default 30); value per
bin = median of |p| r over its points (>= 3 points), in dB re the largest bin.  Coverage = solid-angle
fraction of bins with data.  Also the fraction of covered solid angle within 6 dB of the maximum."""
import sys, json, numpy as np
BIN = int(sys.argv[1]) if len(sys.argv) > 1 else 30
out = {}
for F0 in (492.1875, 750.0, 1007.8125, 1242.1875):
    Z = np.load(f'results/room_pipeline_{int(round(F0))}Hz.npz'); pos = Z['pos']; P = Z['p_raw']
    r = np.linalg.norm(pos, axis=1); m = (r >= 0.2) & (r < 0.4)
    az = np.degrees(np.arctan2(pos[:, 1], pos[:, 0])); el = np.degrees(np.arcsin(np.clip(pos[:, 2] / np.maximum(r, 1e-9), -1, 1)))
    ae = np.arange(-180, 181, BIN); ee = np.arange(-90, 91, BIN)
    V = np.full((len(ee) - 1, len(ae) - 1), np.nan); N = np.zeros_like(V)
    for i in range(len(ee) - 1):
        for j in range(len(ae) - 1):
            s = m & (el >= ee[i]) & (el < ee[i + 1]) & (az >= ae[j]) & (az < ae[j + 1]); N[i, j] = s.sum()
            if s.sum() >= 3: V[i, j] = np.median(np.abs(P[s]) * r[s])
    dB = 20 * np.log10(V / np.nanmax(V))
    dO = (np.sin(np.radians(ee[1:])) - np.sin(np.radians(ee[:-1])))[:, None] * np.radians(BIN) * np.ones((1, len(ae) - 1))
    cov = float(dO[np.isfinite(dB)].sum() / (4 * np.pi)); w6 = float(dO[np.isfinite(dB) & (dB > -6)].sum() / dO[np.isfinite(dB)].sum())
    imax = np.unravel_index(np.nanargmax(V), V.shape)
    out[int(round(F0))] = dict(dB=np.where(np.isfinite(dB), np.round(dB, 2), None).tolist(), n_points=int(m.sum()), coverage=cov,
                               within6dB=w6, range_dB=float(np.nanmax(dB) - np.nanmin(dB)), max_dir=(float(0.5 * (ae[imax[1]] + ae[imax[1] + 1])), float(0.5 * (ee[imax[0]] + ee[imax[0] + 1]))))
    print(f"{F0:7.1f} Hz: {m.sum()} points; solid-angle coverage {cov * 100:.0f}%; range {out[int(round(F0))]['range_dB']:.1f} dB; "
          f"covered area within 6 dB of max {w6 * 100:.0f}%; max at az {out[int(round(F0))]['max_dir'][0]:.0f}, el {out[int(round(F0))]['max_dir'][1]:.0f} deg")
out['az_edges'] = ae.tolist(); out['el_edges'] = ee.tolist()
json.dump(out, open('results/room_balloon.json', 'w'))
