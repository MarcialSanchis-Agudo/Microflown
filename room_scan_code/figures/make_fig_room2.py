#!/usr/bin/env python3
"""
make_fig_room2.py — figures of the full room paper that make_fig_room.py does not draw.

  fig_setup.pdf        photographs, floor plan with scan footprint and furniture
  fig_pipeline.pdf     the processing chain as implemented, with its checks
  fig_interior_val.pdf interior reconstruction on known fields; neighbourhood trade-off
  fig_raw.pdf          raw field at 750 Hz: level, phase, coherence per frequency
  fig_flags.pdf        Stage-2a anomaly score: where the flags are, score distribution, coherence
  fig_recon2.pdf       raw vs reconstructed level and phase; Euler consistency
  fig_localise.pdf     flux-line localisation vs distance cut and frequency, against random directions
  fig_polar.pdf        measured polar patterns (inner / outer shell), SH prediction, angular coverage
  fig_multifreq.pdf    pipeline metrics at the five frequencies

Inputs: results/room_pipeline_<f>Hz.npz (run_room_pipeline.py), room_polar_<f>Hz.json
Prints every number used.
"""
import sys, json, numpy as np
sys.path.insert(0, 'lib')
import paperstyle as S
import matplotlib, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from room_stage2 import local_fill, local_recon
from scipy_free_interp import idw_grid, knn

FR = (257.8125, 492.1875, 750.0, 1007.8125, 1242.1875); FL = ['258', '492', '750', '1008', '1242']
D = {int(round(f)): np.load(f'results/room_pipeline_{int(round(f))}Hz.npz') for f in FR}
M = {}
for line in open('results/room_pipeline_allf.log'):
    j = json.loads(line); M[int(round(j['f']))] = j
Z = D[750]; pos = Z['pos']; r = np.linalg.norm(pos, axis=1); K = 2 * np.pi * 750 / 343.0
zc = pos[:, 2]; hsl = np.abs(zc - np.median(zc)) < 0.18 * np.ptp(zc)
XL, YL = (-2.80, 0.55), (-0.62, 1.32); FLOOR = -1.369
V = np.load('results/room_mesh_vertices.npy').astype(float)
Vx, Vy, Vh = V[:, 0], V[:, 2], V[:, 1] - FLOOR


def photo(ax, path, aspect=0.60, rot=0):
    im = plt.imread(path)
    if rot: im = np.rot90(im, rot)
    h, w = im.shape[:2]; tw = int(h * aspect)
    if tw < w: x0 = (w - tw) // 2; im = im[:, x0:x0 + tw]
    ax.imshow(im); ax.set_axis_off()


# (setup figure: see make_fig_setup3d.py)
# ════════════════════════════════════════════════════════════ pipeline
fig = plt.figure(figsize=(S.TEXTW, 2.55)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.set_axis_off()
def bx(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.0', fc=fc, ec='0.25', lw=0.8))
    ax.text(x + w / 2, y + h - 1.6, title, ha='center', va='top', fontsize=8, fontweight='bold')
    ax.text(x + w / 2, y + h - 5.0, body, ha='center', va='top', fontsize=6.6, linespacing=1.35)
def arr(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=1.0, color='0.2'))
C1, C2, C3 = '#e8eef9', '#fbeee6', '#eaf4ee'
bx(1, 17, 18, 18, '1  Acquisition', 'hand-held probe, tracked\n$p,\\,u_x,u_y,u_z$ per 2 cm cell\ntransfer function\nto a fixed reference\ncoherence, dwell time', C1)
bx(22, 17, 18, 18, '2a  Noise detection', 'four features vs the\n11 nearest neighbours:\nmagnitude, phase,\ncorrelation, impedance\ntop 5 % per channel', C1)
bx(43, 17, 18, 18, '2b  Reconstruction', 'local $j_\\ell Y_\\ell^m$ fit, 96 points,\ndwell weights (Eq. 4); or\njoint $p$, $\\mathbf{u}$ Gaussian process\nwith error bars (Eqs. 5-7)', C1)
bx(64, 17, 16, 18, '3  Energy flux', '$\\mathbf{I}=\\frac{1}{2}\\mathrm{Re}(p\\,\\mathbf{u}^*)$\n$\\mathbf{u}=-\\mathbf{u}_{\\rm stored}$\nflux-line intersection\npoints $>0.2$ m from origin\n(Eq. 8)', C1)
bx(83, 17, 16, 18, '4  Outputs', 'maps of $p$, $\\mathbf{I}$\nsource position\ndirectivity\n(balloons, polar cuts)', C1)
for x0 in (19.5, 40.5, 61.5, 80.5): arr(x0, 26, x0 + 2.0, 26)
bx(1, 1.5, 30, 12.5, 'Validation (Sec. 4)', 'exterior $h^{(1)}_\\ell Y_\\ell^m$: monopole, dipole\ninterior fit and joint process:\nexact fields at the real positions', C3)
bx(34, 1.5, 32, 12.5, 'Checks on the data (Sec. 5)', 'reference-pair scatter (1); split-half error (2b)\nEuler consistency of $p$ and $\\mathbf{u}$ (2b)\nrandom-direction null (3);\nheld-out calibration of error bars (2b)', C2)
bx(69, 1.5, 30, 12.5, 'Design rule for acquisition (Sec. 6)', 'averages per cell for a target error\n(Eq. 9)  $\\rightarrow$  maximum sweep speed', C2)
fig.savefig('paperfigs/fig_pipeline.pdf'); fig.savefig('results/fig_pipeline_preview.png', dpi=130); plt.close(fig)

# ════════════════════════════════════════════════════════════ interior validation
Gf = lambda s: (lambda d: np.exp(-1j * K * d) / d)(np.linalg.norm(pos - np.array(s), axis=1))
F1 = Gf([0, 0, 0]); F2 = F1 + 0.6 * Gf([0, 0, -1.9]) + 0.5 * Gf([0.9, 0, 0]) + 0.4 * Gf([0, 2.6, 0])
rng = np.random.default_rng(0); q = rng.choice(len(pos), 600, replace=False); bad = np.zeros(len(pos), bool); bad[q] = True
good = np.flatnonzero(~bad); idx, dd = knn(pos[q], pos[good], 8); ww = 1 / np.maximum(dd, 1e-9) ** 2
E = {}
for nm, Fk in (('monopole', F1), ('room-like', F2)):
    E[nm + ' Helmholtz'] = np.abs(local_fill(pos, Fk, bad, K, query=q) - Fk[q]) / np.abs(Fk[q])
    E[nm + ' IDW'] = np.abs((ww * Fk[good][idx]).sum(1) / ww.sum(1) - Fk[q]) / np.abs(Fk[q])
for kk, v in E.items(): print(f"[interior] {kk}: median {np.median(v) * 100:.2f}%")
w = Z['dwell'] * (~Z['flag_p']); rs = np.random.default_rng(4); qq = rs.choice(len(pos), 800, replace=False); half = rs.random(len(pos)) < 0.5   # same draw as run_room_pipeline.py
Aa, Bb = np.flatnonzero(half), np.flatnonzero(~half); nn = [12, 24, 48, 96, 192]; bias, noise = [], []
for n in nn:
    bias.append(np.median(np.abs(local_recon(pos, F2, w, K, qq, n_nb=n) - F2[qq]) / np.abs(F2[qq])))
    pa = local_recon(pos, Z['p_raw'], w, K, qq, n_nb=n, src=Aa); pb = local_recon(pos, Z['p_raw'], w, K, qq, n_nb=n, src=Bb)
    noise.append(np.median(np.abs(pa - pb) / np.sqrt(2) / np.abs(0.5 * (pa + pb))))
_, dn = knn(pos[::10], pos, max(nn)); rad = [np.median(dn[:, n - 1]) for n in nn]
print("[interior] n_nb", nn, "radius [cm]", np.round(np.array(rad) * 100, 1), "bias %", np.round(np.array(bias) * 100, 1), "split-half noise %", np.round(np.array(noise) * 100, 1))
R = S.Row(W=S.TEXTW, H=2.2, left=1.25, bottom=0.45)
a = R.add_plot(2.35, 1.6)
labs = list(E.keys()); bp = a.boxplot([E[k] * 100 for k in labs], vert=False, showfliers=False, whis=(5, 95), widths=0.55, patch_artist=True, medianprops=dict(color='k'))
for pch, k_ in zip(bp['boxes'], labs): pch.set_facecolor(S.BLUE if 'Helmholtz' in k_ else S.GREY); pch.set_alpha(0.6)
a.set_yticks(range(1, 5)); a.set_yticklabels([k_.replace(' Helmholtz', ',\nlocal Helmholtz').replace(' IDW', ',\ninverse distance') for k_ in labs], fontsize=7.5)
a.set_xscale('log'); a.set_xlim(0.01, 100); a.set_xlabel('leave-one-out error [%]'); a.grid(color='0.9', lw=0.5, axis='x'); S.tag(a, 'a')
R.gap(0.75); b = R.add_plot(R.W - R.x - 0.1, 1.6)
b.plot(nn, np.array(bias) * 100, 'o-', color=S.BLUE, ms=4, label='bias, known field')
b.plot(nn, np.array(noise) * 100, 's-', color=S.ORANGE, ms=4, label='random error, data (split-half)')
b.axvline(96, color='0.5', lw=0.7, ls=':'); b.set_xscale('log'); b.set_xticks(nn); b.set_xticklabels([str(n) for n in nn])
b.set_xlabel('neighbours $n_{\\rm nb}$'); b.set_ylabel('median relative error [%]'); b.grid(color='0.9', lw=0.5)
b.legend(frameon=False, fontsize=7, loc='center right', bbox_to_anchor=(1.0, 0.33)); S.tag(b, 'b', x=0.40)
R.save('paperfigs/fig_interior_val.pdf', 'results/fig_interior_val_preview.png')

# ════════════════════════════════════════════════════════════ raw data
def phase_map(p): return np.angle(p * r * np.exp(1j * K * r))
R = S.Row(W=S.TEXTW, H=2.25, left=0.58, bottom=0.45)
a = R.add_map(XL, YL, 2.0); o = np.argsort(np.abs(Z['p_raw'][hsl]))
Lr = np.log10(np.abs(Z['p_raw'])); vmin, vmax = np.percentile(Lr[hsl], [2, 99.5])
sc = a.scatter(pos[hsl][o, 0], pos[hsl][o, 1], c=Lr[hsl][o], cmap=S.SEQ, vmin=vmin, vmax=vmax, s=1.3, lw=0, rasterized=True)
S.source(a); a.set_xlabel('$x$ [m]'); a.set_ylabel('$y$ [m]'); S.tag(a, 'a')
S.cbar(R.fig, sc, R.add_cbar(), r'$\log_{10}|p|$'); R.gap(0.42)
b = R.add_map(XL, YL, 2.0)
sp = b.scatter(pos[hsl, 0], pos[hsl, 1], c=phase_map(Z['p_raw'])[hsl], cmap='twilight', vmin=-np.pi, vmax=np.pi, s=1.3, lw=0, rasterized=True)
S.source(b); b.set_xlabel('$x$ [m]'); b.set_yticklabels([]); S.tag(b, 'b')
S.cbar(R.fig, sp, R.add_cbar(), r'arg$(p\,r\,e^{\mathrm{i}kr})$', ticks=[-np.pi, 0, np.pi]).set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
R.gap(0.85); c = R.add_plot(R.W - R.x - 0.08)
bpx = c.boxplot([D[int(round(f))]['coh'] for f in FR], showfliers=False, whis=(5, 95), widths=0.55, patch_artist=True, medianprops=dict(color='k'))
for pch in bpx['boxes']: pch.set_facecolor(S.GREY)
c.set_xticklabels(FL, fontsize=6.5, rotation=45); c.set_xlabel('$f$ [Hz]'); c.set_ylabel(r'$\gamma^2_{p,\mathrm{ref}}$'); c.set_ylim(0, 1.18); c.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); S.tag(c, 'c')
R.save('paperfigs/fig_raw.pdf', 'results/fig_raw_preview.png')
coh_raw = abs(np.mean(np.exp(1j * phase_map(Z['p_raw'])[(r > 0.3) & (r < 2.6)]))); coh_rec = abs(np.mean(np.exp(1j * phase_map(Z['p'])[(r > 0.3) & (r < 2.6)])))
print(f"[raw] phase coherence of p r e^(ikr), 0.3-2.6 m: raw {coh_raw:.2f}, reconstructed {coh_rec:.2f}")

# ════════════════════════════════════════════════════════════ flags
anyf = Z['flag_p'] | Z['flag_ux'] | Z['flag_uy'] | Z['flag_uz']
R = S.Row(W=S.TEXTW, H=2.25, left=0.58, bottom=0.48)
a = R.add_map(XL, YL, 2.3)
a.scatter(pos[:, 0], pos[:, 1], s=0.6, c=S.GREY, lw=0, rasterized=True)
a.scatter(pos[Z['flag_p'], 0], pos[Z['flag_p'], 1], s=2.5, c=S.RED, lw=0, rasterized=True, label='$p$ flagged')
S.source(a); a.set_xlabel('$x$ [m]'); a.set_ylabel('$y$ [m]'); S.tag(a, 'a')
R.gap(0.62); b = R.add_plot(1.75)
sp_ = Z['score_p']; thr = np.quantile(sp_, 0.95)
b.hist(sp_, bins=np.linspace(0, np.percentile(sp_, 99.8), 60), color=S.GREY, log=True); b.axvline(thr, color=S.RED, lw=1.0)
b.text(thr * 1.03, b.get_ylim()[1] * 0.5, 'top 5 %\nflagged', color=S.RED, fontsize=6.5, va='top')
b.set_xlabel('anomaly score, $p$ channel'); b.set_ylabel('number of points'); S.tag(b, 'b', x=0.03)
R.gap(0.58); c = R.add_plot(R.W - R.x - 0.08)
cp, fl = Z['coh'], Z['flag_p']; bins = np.linspace(0, 1, 21)
c.hist(cp[~fl], bins=bins, density=True, color=S.GREY, label='kept'); c.hist(cp[fl], bins=bins, density=True, histtype='step', color=S.RED, lw=1.3, label='flagged')
c.set_xlabel(r'coherence $\gamma^2_{p,\mathrm{ref}}$'); c.set_ylabel('density'); c.set_xlim(0, 1); c.legend(frameon=False, loc='upper center', fontsize=7); S.tag(c, 'c', x=0.80)
R.save('paperfigs/fig_flags.pdf', 'results/fig_flags_preview.png')
dflag = np.median(np.linalg.norm(pos[Z['flag_p']], axis=1)); dkept = np.median(np.linalg.norm(pos[~Z['flag_p']], axis=1))
print(f"[flags] p flagged {fl.sum()}; any {anyf.sum()} ({anyf.mean() * 100:.1f}%); median distance from source flagged {dflag:.2f} m vs kept {dkept:.2f} m; "
      f"median dwell flagged {np.median(Z['dwell'][fl]):.1f} vs kept {np.median(Z['dwell'][~fl]):.1f} blocks")

# ════════════════════════════════════════════════════════════ reconstruction: level, phase, Euler
from numpy.linalg import lstsq
om = 2 * np.pi * 750; sel = np.random.default_rng(0).choice(len(pos), 1500, replace=False); EU = {}
for lab, p_, u_ in (('raw', Z['p_raw'], Z['u_raw']), ('rec', Z['p'], Z['u'])):
    cs = []
    for i in sel:
        d_ = np.linalg.norm(pos - pos[i], axis=1); nbh = np.where((d_ > 0) & (d_ < 0.08))[0]
        if len(nbh) < 8: continue
        Xm = np.column_stack([np.ones(len(nbh)), pos[nbh] - pos[i]]); g = lstsq(Xm, p_[nbh], rcond=None)[0][1:]
        ue = g / (1j * om * 1.204); cs.append(np.real(np.vdot(ue, u_[i])) / (np.linalg.norm(ue) * np.linalg.norm(u_[i]) + 1e-30))
    EU[lab] = np.array(cs)
print(f"[recon] Euler median raw {np.median(EU['raw']):.2f}, rec {np.median(EU['rec']):.2f} ({len(EU['rec'])} nodes)")
W = S.TEXTW; H = 4.1; fig = plt.figure(figsize=(W, H)); mw = 2.0; mh = mw * (YL[1] - YL[0]) / (XL[1] - XL[0])
lref = np.log10(np.abs(Z['p'][hsl]).max()); vmin, vmax = -35, 0
for row, (kind, cm, vr, ttl) in enumerate(((None, S.SEQ, (vmin, vmax), r'$|p|$ [dB re max]'), ('ph', 'twilight', (-np.pi, np.pi), r'arg$(p\,r\,e^{\mathrm{i}kr})$'))):
    yb = 0.42 + (1 - row) * (mh + 0.42)
    for col, (p_, t) in enumerate(((Z['p_raw'], 'a' if row == 0 else 'c'), (Z['p'], 'b' if row == 0 else 'd'))):
        a = fig.add_axes([(0.58 + col * (mw + 0.14)) / W, yb / H, mw / W, mh / H])
        val = 20 * (np.log10(np.abs(p_)) - lref) if kind is None else phase_map(p_)
        o = np.argsort(val[hsl]) if kind is None else np.arange(hsl.sum())
        m_ = a.scatter(pos[hsl][o, 0], pos[hsl][o, 1], c=val[hsl][o], cmap=cm, vmin=vr[0], vmax=vr[1], s=1.2, lw=0, rasterized=True)
        S.source(a); a.set_xlim(*XL); a.set_ylim(*YL); a.set_aspect('equal'); S.tag(a, t)
        if col == 0: a.set_ylabel('$y$ [m]')
        else: a.set_yticklabels([])
        if row == 1: a.set_xlabel('$x$ [m]')
        else: a.set_xticklabels([])
    cax = fig.add_axes([(0.58 + 2 * mw + 0.14 + 0.06) / W, yb / H, 0.09 / W, mh / H])
    cbx = S.cbar(fig, m_, cax, ttl, ticks=None if kind is None else [-np.pi, 0, np.pi])
    if kind: cbx.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
xc = 0.58 + 2 * mw + 0.14 + 0.06 + 0.09 + 0.95; c = fig.add_axes([xc / W, 0.42 / H, (W - xc - 0.08) / W, (2 * mh + 0.42) / H])
bins = np.linspace(-1, 1, 41)
c.hist(EU['raw'], bins=bins, density=True, color=S.GREY, label='raw'); c.hist(EU['rec'], bins=bins, density=True, histtype='step', color=S.ORANGE, lw=1.4, label='reconstructed')
c.set_xlabel('Euler consistency'); c.set_ylabel('density'); c.set_xlim(-1, 1); c.legend(frameon=False, loc='upper left', fontsize=7, bbox_to_anchor=(0, 0.93)); S.tag(c, 'e')
fig.savefig('paperfigs/fig_recon2.pdf'); fig.savefig('results/fig_recon2_preview.png', dpi=130); plt.close(fig)

# ════════════════════════════════════════════════════════════ localisation
def xstar(Pp, Dd, ww):
    Mm = np.eye(3)[None] - Dd[:, :, None] * Dd[:, None, :]; Am = (ww[:, None, None] * Mm).sum(0); bb = (ww[:, None, None] * Mm @ Pp[:, :, None]).sum(0)[:, 0]
    return np.linalg.solve(Am, bb)
cuts = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]; LC = {'raw': [], 'rec': [], 'null_raw': [], 'null_rec': [], 'nlo_rec': [], 'nhi_rec': [], 'nlo_raw': [], 'nhi_raw': []}
rng = np.random.default_rng(0)
for lab, p_, u_ in (('raw', Z['p_raw'], Z['u_raw']), ('rec', Z['p'], Z['u'])):
    I = -0.5 * np.real(p_[:, None] * np.conj(u_)); Im = np.linalg.norm(I, axis=1); Dd = I / np.maximum(Im, 1e-30)[:, None]
    for c_ in cuts:
        s_ = r > c_; LC[lab].append(np.linalg.norm(xstar(pos[s_], Dd[s_], Im[s_])) * 100)
        if True:
            nu = []
            for _ in range(20):
                Rd = rng.standard_normal((s_.sum(), 3)); Rd /= np.linalg.norm(Rd, axis=1)[:, None]; nu.append(np.linalg.norm(xstar(pos[s_], Rd, Im[s_])) * 100)
            LC['null_' + lab].append(np.median(nu)); LC['nlo_' + lab].append(np.percentile(nu, 5)); LC['nhi_' + lab].append(np.percentile(nu, 95))
print("[localise] cuts", cuts, "raw", np.round(LC['raw'], 1), "null(raw w)", np.round(LC['null_raw'], 1), "rec", np.round(LC['rec'], 1), "null(rec w)", np.round(LC['null_rec'], 1))
R = S.Row(W=S.TEXTW, H=2.2, left=0.58, bottom=0.45)
a = R.add_plot(2.9, 1.6)
a.fill_between(cuts, LC['nlo_rec'], LC['nhi_rec'], color=S.ORANGE, alpha=0.30, lw=0, label='random, 5–95 %'); a.plot(cuts, LC['null_rec'], color=S.ORANGE, lw=1.0, ls='--', label='random, reconstructed weights')
a.fill_between(cuts, LC['nlo_raw'], LC['nhi_raw'], color='0.6', alpha=0.35, lw=0); a.plot(cuts, LC['null_raw'], color='0.45', lw=1.0, ls='--', label='random, raw weights')
a.plot(cuts, LC['raw'], 's-', color='0.45', ms=4, label='raw'); a.plot(cuts, LC['rec'], 'o-', color=S.ORANGE, ms=4, label='reconstructed')
a.set_xlabel('minimum distance of the points used [m]'); a.set_ylabel(r'$|\mathbf{x}^*|$ [cm]'); a.set_yscale('log'); a.grid(color='0.9', lw=0.5, which='both')
a.legend(frameon=False, fontsize=6.5, loc='lower right'); S.tag(a, 'a')
R.gap(0.85); b = R.add_plot(R.W - R.x - 0.1, 1.6); xf = np.arange(5)
b.bar(xf - 0.25, [M[int(round(f))]['loc_raw_cm'] for f in FR], 0.25, color='0.6', label='raw')
b.bar(xf, [M[int(round(f))]['loc_rec_cm'] for f in FR], 0.25, color=S.ORANGE, label='reconstructed')
b.bar(xf + 0.25, [M[int(round(f))]['loc_null_cm_median'] for f in FR], 0.25, color='white', edgecolor='0.4', hatch='///', lw=0.6, label='random (rec. weights)')
b.set_xticks(xf); b.set_xticklabels(FL); b.set_xlabel('$f$ [Hz]'); b.set_ylabel(r'$|\mathbf{x}^*|$ [cm], $r>0.2$ m'); b.set_yscale('log')
b.legend(frameon=False, fontsize=6.5, loc='upper right', ncol=1); b.set_ylim(3, 3000); b.grid(color='0.9', lw=0.5, axis='y'); S.tag(b, 'b')
R.save('paperfigs/fig_localise.pdf', 'results/fig_localise_preview.png')

# ════════════════════════════════════════════════════════════ multi-frequency summary
R = S.Row(W=S.TEXTW, H=2.3, left=0.52, bottom=0.62); xf = np.arange(5)
z0 = R.add_plot(1.2, 1.5)
bpx = z0.boxplot([D[int(round(f))]['coh'] for f in FR], positions=xf, showfliers=False, whis=(5, 95), widths=0.55, patch_artist=True, medianprops=dict(color='k'))
for pch in bpx['boxes']: pch.set_facecolor(S.GREY)
z0.set_xticks(xf); z0.set_xticklabels(FL, fontsize=6.5, rotation=45); z0.set_xlabel('$f$ [Hz]'); z0.set_ylabel(r'coherence $\gamma^2_{p,\mathrm{ref}}$'); z0.set_ylim(0, 1.2); z0.set_yticks([0, 0.5, 1.0]); S.tag(z0, 'a')
R.gap(0.55); a = R.add_plot(1.45, 1.5)
a.plot(xf, [M[int(round(f))]['node_err_raw'] * 100 for f in FR], 'D-', color='0.3', ms=4, label='raw, leave-one-out')
a.plot(xf, [M[int(round(f))]['split_raw'] * 100 for f in FR], 's-', color='0.6', ms=4, label='raw, split-half')
a.plot(xf, [M[int(round(f))]['split_rec'] * 100 for f in FR], 'o-', color=S.ORANGE, ms=4, label='reconstructed, split-half')
a.set_xticks(xf); a.set_xticklabels(FL, fontsize=6.5, rotation=45); a.set_xlabel('$f$ [Hz]'); a.set_ylabel('median error [%]'); a.set_ylim(0, 215)
a.legend(frameon=False, fontsize=5.8, loc='upper right'); a.grid(color='0.9', lw=0.5); S.tag(a, 'b', x=0.03, y=0.97)
R.gap(0.55); b = R.add_plot(1.35, 1.5)
b.plot(xf, [M[int(round(f))]['euler_raw'] for f in FR], 's-', color='0.5', ms=4, label='raw')
b.plot(xf, [M[int(round(f))]['euler_rec'] for f in FR], 'o-', color=S.ORANGE, ms=4, label='reconstructed')
b.set_xticks(xf); b.set_xticklabels(FL, fontsize=6.5, rotation=45); b.set_xlabel('$f$ [Hz]'); b.set_ylabel('Euler consistency'); b.set_ylim(0, 1)
b.legend(frameon=False, fontsize=6.5, loc='lower right'); b.grid(color='0.9', lw=0.5); S.tag(b, 'c')
R.gap(0.55); c = R.add_plot(R.W - R.x - 0.08, 1.5)
c.plot(xf, [M[int(round(f))]['outward_raw'] for f in FR], 's-', color='0.5', ms=4, label='raw')
c.plot(xf, [M[int(round(f))]['outward_rec'] for f in FR], 'o-', color=S.ORANGE, ms=4, label='reconstructed')
c.set_xticks(xf); c.set_xticklabels(FL, fontsize=6.5, rotation=45); c.set_xlabel('$f$ [Hz]'); c.set_ylabel('outward flux [% of points]'); c.set_ylim(20, 100)
c.legend(frameon=False, fontsize=6.5, loc='lower right'); c.grid(color='0.9', lw=0.5); S.tag(c, 'd')
R.save('paperfigs/fig_multifreq.pdf', 'results/fig_multifreq_preview.png')
print("wrote setup, pipeline, interior_val, raw, flags, recon2, localise, polar, multifreq")
