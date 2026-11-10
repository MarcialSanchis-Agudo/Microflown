#!/usr/bin/env python3
"""make_viewer3d.py — self-contained interactive 3-D viewer (single HTML file, no external libraries):
scan points coloured by reconstructed pressure level (dB re max) and energy-flux streamlines coloured by
|I| (dB re max), 750 Hz, acoustic frame.  Drag to rotate, wheel to zoom, buttons to toggle layers."""
import json, numpy as np, matplotlib
Z = np.load('results/room_pipeline_750Hz.npz'); pos = Z['pos']
L = 20 * np.log10(np.abs(Z['p']) / np.abs(Z['p']).max())
F = np.load('results/flux3d_lines.npz'); lines = [F[k] for k in F.files]
lut = lambda name: [matplotlib.colors.to_hex(matplotlib.colormaps[name](i / 63)) for i in range(64)]
data = dict(pts=np.round(np.column_stack([pos, np.clip(L, -40, 0)]), 3).tolist(),
            lines=[np.round(l_, 3).tolist() for l_ in lines], plasma=lut('plasma'), viridis=lut('viridis'))
html = """<div style="font-family:Georgia,serif;max-width:980px;margin:0 auto">
<h2 style="margin:8px 0 2px">Walk-around scan of a furnished room: reconstructed field at 750 Hz</h2>
<p style="margin:0 0 8px;color:#444;font-size:14px">Scan points coloured by pressure level; energy-flux streamlines coloured by intensity.
Star: origin (pressure maximum, marking the loudspeaker). Drag to rotate, scroll to zoom.</p>
<div style="margin-bottom:6px">
<button id="bp">scan points</button> <button id="bl">streamlines</button> <button id="br">reset view</button>
<span style="font-size:13px;color:#555;margin-left:12px">points: |p| dB re max (−40…0, plasma) &nbsp; lines: |I| dB re max (−30…0, viridis)</span></div>
<canvas id="c" width="960" height="600" style="border:1px solid #ddd;background:#fff;cursor:grab;width:100%"></canvas>
</div>
<script>
const D = __DATA__;
const cv = document.getElementById('c'), cx = cv.getContext('2d');
let yaw = -0.9, pitch = 0.45, zoom = 1.0, showP = true, showL = true;
const cen = [-1.1, 0.35, -0.1];
function col(lut, v, lo, hi){ let t = Math.min(1, Math.max(0, (v - lo) / (hi - lo))); return lut[Math.round(t * 63)]; }
function proj(x, y, z){
  let X = x - cen[0], Y = y - cen[1], Zz = z - cen[2];
  let cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  let x1 = cy * X - sy * Y, y1 = sy * X + cy * Y;
  let y2 = cp * y1 - sp * Zz, z2 = sp * y1 + cp * Zz;
  let d = 6 - y2, f = 1000 * zoom / d;
  return [cv.width / 2 + f * x1, cv.height / 2 - f * z2, d];
}
function draw(){
  cx.clearRect(0, 0, cv.width, cv.height);
  // axes
  [[[0,0,0],[0.5,0,0],'x'],[[0,0,0],[0,0.5,0],'y'],[[0,0,0],[0,0,0.5],'z']].forEach(a => {
    let p0 = proj(-2.9 + a[0][0], -0.6 + a[0][1], -0.9 + a[0][2]), p1 = proj(-2.9 + a[1][0], -0.6 + a[1][1], -0.9 + a[1][2]);
    cx.strokeStyle = '#333'; cx.lineWidth = 1.5; cx.beginPath(); cx.moveTo(p0[0], p0[1]); cx.lineTo(p1[0], p1[1]); cx.stroke();
    cx.fillStyle = '#333'; cx.font = '14px Georgia'; cx.fillText(a[2] + ' (0.5 m)', p1[0] + 4, p1[1]); });
  let items = [];
  if (showP) D.pts.forEach(p => { let q = proj(p[0], p[1], p[2]); items.push([q[2], 0, q, p[3]]); });
  if (showL) D.lines.forEach(l => { for (let i = 1; i < l.length; i++){ let a = proj(l[i-1][0], l[i-1][1], l[i-1][2]), b = proj(l[i][0], l[i][1], l[i][2]);
      items.push([(a[2] + b[2]) / 2, 1, a, b, l[i][3]]); } });
  items.sort((u, v) => v[0] - u[0]);
  items.forEach(it => {
    if (it[1] == 0){ cx.fillStyle = col(D.plasma, it[3], -40, 0); cx.globalAlpha = 0.55; cx.fillRect(it[2][0] - 1.2, it[2][1] - 1.2, 2.4, 2.4); }
    else { cx.globalAlpha = 0.95; cx.strokeStyle = col(D.viridis, it[4], -30, 0); cx.lineWidth = 2; cx.beginPath(); cx.moveTo(it[2][0], it[2][1]); cx.lineTo(it[3][0], it[3][1]); cx.stroke(); }
  });
  cx.globalAlpha = 1; let s = proj(0, 0, 0); cx.font = '26px serif'; cx.fillStyle = '#fff'; cx.strokeStyle = '#000'; cx.lineWidth = 1.2;
  cx.strokeText('\\u2605', s[0] - 11, s[1] + 9); cx.fillText('\\u2605', s[0] - 11, s[1] + 9);
}
let drag = null;
cv.addEventListener('mousedown', e => { drag = [e.clientX, e.clientY]; cv.style.cursor = 'grabbing'; });
window.addEventListener('mouseup', () => { drag = null; cv.style.cursor = 'grab'; });
window.addEventListener('mousemove', e => { if (!drag) return; yaw += (e.clientX - drag[0]) * 0.008; pitch = Math.max(-1.5, Math.min(1.5, pitch + (e.clientY - drag[1]) * 0.008)); drag = [e.clientX, e.clientY]; draw(); });
cv.addEventListener('wheel', e => { e.preventDefault(); zoom *= e.deltaY < 0 ? 1.1 : 0.9; draw(); }, {passive: false});
document.getElementById('bp').onclick = () => { showP = !showP; draw(); };
document.getElementById('bl').onclick = () => { showL = !showL; draw(); };
document.getElementById('br').onclick = () => { yaw = -0.9; pitch = 0.45; zoom = 1; draw(); };
draw();
</script>""".replace('__DATA__', json.dumps(data, separators=(',', ':')))
open('room_flux_3d_viewer.html', 'w').write('<!DOCTYPE html><html><head><meta charset="utf-8"><title>Room scan 3-D viewer</title></head><body>' + html + '</body></html>')
print('wrote room_flux_3d_viewer.html', f"{len(html)/1e6:.1f} MB")
