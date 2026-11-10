#!/usr/bin/env python3
"""
make_fig_validation.py — Fig. 3, validation on canonical sources, house style.

(a) directivity error vs truncation order Lmax (monopole, x-dipole);
(b) ratio of fitted to analytical amplitude vs evaluation radius;
(c) horizontal-plane dipole overlay at 750 Hz.
N = 3000 nodes in r in [0.10, 0.88] m, 40 dB SNR, lambda = 1e-3.
Run from the repo root with SciPy, or with PYTHONPATH=results/scipy_shim.
"""
import sys
import numpy as np
sys.path.insert(0, 'lib')
import paperstyle as S
from mf_pipeline.mf_directivity import (angles_from_pos, fit_complex_SH_nearfield, SHFitOptions,
                                        eval_on_lonlat_map, eval_complex_SH_at_radius)

C, F0, SNR, N, LAM = 343.0, 750.0, 40.0, 3000, 1e-3
k = 2 * np.pi * F0 / C
rng = np.random.default_rng(42)
r = rng.uniform(.10, .88, N); th = np.arccos(rng.uniform(-1, 1, N)); ph = rng.uniform(-np.pi, np.pi, N)
pos = np.column_stack([r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)])
rr, tt, pp = angles_from_pos(pos, np.zeros(3))
mono = np.exp(1j * k * rr) / rr
dip = pos[:, 0] * np.exp(1j * k * rr) / rr ** 2 * (1j * k - 1 / rr)
g = np.random.default_rng(7)
def noisy(s):
    sd = np.mean(np.abs(s)) / 10 ** (SNR / 20)
    return s + sd * (g.standard_normal(s.shape) + 1j * g.standard_normal(s.shape)) / np.sqrt(2)
data = {'mono': noisy(mono), 'dip': noisy(dip)}
def fit(s, L): return fit_complex_SH_nearfield(data[s], rr, tt, pp, k, SHFitOptions(lmax=L, lam=LAM))
def err(V, E): return float(np.sqrt(np.mean((V / V.max() - E / E.max()) ** 2))) * 100

lon, lat, _ = eval_on_lonlat_map(np.zeros(1), k, 1.0, 0, 181, 91)
LO, LA = np.meshgrid(np.radians(lon), np.radians(lat))
exact = {'mono': np.ones_like(LA), 'dip': np.abs(np.cos(LA) * np.cos(LO))}
Ls = list(range(9)); E = {s: [] for s in data}
for L in Ls:
    for s in data:
        E[s].append(err(np.abs(eval_on_lonlat_map(fit(s, L), k, 1.0, L, 181, 91)[2]), exact[s]))
print("error vs Lmax [%]  mono:", np.round(E['mono'], 4), "\n                   dip :", np.round(E['dip'], 4))

a8 = {s: fit(s, 8) for s in data}
Rv = np.logspace(np.log10(0.10), np.log10(6.0), 50); ratio = {s: [] for s in data}
for R_ in Rv:
    vm = abs(eval_complex_SH_at_radius(a8['mono'], np.array([np.pi / 2]), np.array([0.0]), k, R_, 8)[0])
    vd = abs(eval_complex_SH_at_radius(a8['dip'], np.array([np.pi / 2]), np.array([0.0]), k, R_, 8)[0])
    ratio['mono'].append(vm * R_); ratio['dip'].append(vd / (np.sqrt(k ** 2 + 1 / R_ ** 2) / R_))
print("radial ratio range: mono [%.4f, %.4f]  dip [%.4f, %.4f]" % (min(ratio['mono']), max(ratio['mono']),
                                                                   min(ratio['dip']), max(ratio['dip'])))

R = S.Row(W=S.TEXTW, H=2.3, left=0.58, bottom=0.55)
a = R.add_plot(1.85, 1.55)
a.semilogy(Ls, np.maximum(E['mono'], 3e-3), 'o-', color=S.BLUE, ms=3.5, lw=1.2, label=r'monopole ($\ell=0$)')
a.semilogy(Ls, E['dip'], 's-', color=S.ORANGE, ms=3.5, lw=1.2, label=r'dipole ($\ell=1$)')
a.axvspan(-0.5, 0.5, color=S.BLUE, alpha=0.08)
a.set_xlabel(r'$L_{\max}$'); a.set_ylabel('directivity error [%]'); a.set_xticks(Ls); a.set_ylim(2e-3, 3e2)
a.grid(color='0.9', lw=0.5); a.legend(frameon=False, loc='center right'); S.tag(a, 'a')
R.gap(0.72)
b = R.add_plot(1.85, 1.55)
b.semilogx(Rv, ratio['mono'], color=S.BLUE, lw=1.4, label='monopole')
b.semilogx(Rv, ratio['dip'], color=S.ORANGE, lw=1.4, ls='--', label='dipole lobe')
b.axvspan(0.10, 0.88, color='0.5', alpha=0.10); b.text(0.30, 0.9933, 'sampled shell', ha='center', va='bottom', fontsize=7, color='0.35')
b.axhline(1, color='0.5', lw=0.6, ls=':'); b.set_ylim(0.993, 1.003)
b.set_xlabel('evaluation radius $r$ [m]'); b.set_ylabel(r'$|\hat p|\,/\,|p_{\rm exact}|$')
b.grid(color='0.9', lw=0.5, which='both'); b.legend(frameon=False, loc='upper right'); S.tag(b, 'b')
R.gap(0.45)
c = R.fig.add_axes([(R.x + 0.10) / R.W, (R.y0 + 0.02) / R.H, 1.38 / R.W, 1.38 / R.H], projection='polar')
i0 = np.argmin(np.abs(lat)); Vd = np.abs(eval_on_lonlat_map(a8['dip'], k, 1.0, 8, 181, 91)[2])[i0]
Ed = exact['dip'][i0]
todb = lambda x: np.maximum(20 * np.log10(np.maximum(x / x.max(), 1e-12)), -30)
c.plot(np.radians(lon), (todb(Ed) + 30) / 30, color=S.BLUE, lw=1.6, label='analytical')
c.plot(np.radians(lon), (todb(Vd) + 30) / 30, color=S.ORANGE, lw=1.1, ls='--', label=r'SH, $L_{\max}=8$')
c.set_ylim(0, 1); c.set_yticks([1 / 3, 2 / 3, 1]); c.set_yticklabels(['$-20$', '$-10$', '0 dB'], fontsize=6.5); c.set_rlabel_position(112)
c.legend(frameon=False, fontsize=6.3, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, handlelength=1.2)
c.set_xticks(np.radians([0, 90, 180, 270])); c.set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=7)
c.grid(color='0.85', lw=0.5)
S.tag(c, 'c', x=-0.12, y=1.05)
R.save('paperfigs/fig_validation_canonical.pdf', 'results/fig_validation_preview.png')
print("wrote paperfigs/fig_validation_canonical.pdf")
