#!/usr/bin/env python3
"""Joint p-u GP (750 Hz, hyperparameters of room_gp_750Hz.json) predicted on horizontal planes at four
heights and saved for the 3-D slice-stack figure and the wave animation."""
import sys, json, numpy as np; sys.path.insert(0, 'lib')
from room_gp import gp_predict
Z = np.load('results/room_pipeline_750Hz.npz'); pos = Z['pos']; k = 2 * np.pi * 750 / 343; om = 2 * np.pi * 750
lvp = np.abs(Z['p']); lvu = np.linalg.norm(Z['u'], axis=1) / np.sqrt(3); hp = json.load(open('results/room_gp_750Hz.json'))
lo = pos.min(0) - 0.05; hi = pos.max(0) + 0.05; h = 0.025
xs, ys = np.arange(lo[0], hi[0] + h, h), np.arange(lo[1], hi[1] + h, h); X, Y = np.meshgrid(xs, ys, indexing='ij')
ZS = [float(v) for v in sys.argv[1:]] if len(sys.argv) > 1 else [-0.45, -0.2, 0.05, 0.3]
for zz in ZS:
    Q = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, zz)])
    m, sd, sig, mu = gp_predict(pos, Z['p_raw'], Z['u_raw'], Z['dwell'], lvp, lvu, k, om, Q, beta_p=hp['beta_p'], beta_u=hp['beta_u'], alpha=hp['alpha'], B=0.10, margin=0.195, with_u=True)
    I = -0.5 * np.real(m[:, None] * np.conj(mu)); s = X.shape
    np.savez(f'results/gp_plane_z{zz:+.2f}.npz', xs=xs, ys=ys, z=zz, m=m.reshape(s), sd=sd.reshape(s), sig=sig.reshape(s), Ix=I[:, 0].reshape(s), Iy=I[:, 1].reshape(s), Iz=I[:, 2].reshape(s))
    rr = (sd / sig)[np.isfinite(sd)]; print(f"z = {zz:+.2f} m: {np.isfinite(sd).sum()} cells predicted, {np.mean(rr < 0.6) * 100:.0f}% with sd < 0.6 prior", flush=True)
