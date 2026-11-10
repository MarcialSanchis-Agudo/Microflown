"""Raw room-scan channels at one frequency from the .psd (HDF5) export."""
import sys, numpy as np; sys.path.insert(0,'lib')
from mini_h5 import H5
PSD='data/Martial measurements.psd'; G='/Process1/'
_h=None
def h5():
    global _h
    if _h is None: _h=H5(PSD)
    return _h
def freqs(): return h5().dataset(G+'FreqVar').ravel()
def raw(f0):
    h=h5(); f=freqs(); kb=int(np.argmin(abs(f-f0)))
    g=lambda n: h.dataset(G+n)
    out=dict(f=float(f[kb]), pos=g('datPostition').T.astype(float))
    for r in ('ref1','ref2'):
        out['p_'+r]=(g(f'TFp{r}_i')[:,0,kb]+1j*g(f'TFp{r}_j')[:,0,kb]).astype(complex)
        out['u_'+r]=(g(f'TFxyz{r}_i')[:,:,kb]+1j*g(f'TFxyz{r}_j')[:,:,kb]).astype(complex)
        out['coh_p_'+r]=g(f'COHp{r}')[:,0,kb].astype(float)
        out['coh_u_'+r]=g(f'COHxyz{r}')[:,:,kb].astype(float)
    out['coh_r1r2']=g('COHref1ref2')[:,0,kb].astype(float)
    return out


def frame_transform():
    """R, t of the analysis frame (lib/frame.json): x_frame = R x_raw + t."""
    import json, os
    J = json.load(open(os.path.join(os.path.dirname(__file__), 'frame.json')))
    return np.array(J['R']), np.array(J['t'])


def to_frame(X):
    """Probe positions in the analysis frame; also returns R (to rotate vectors such as the velocity)."""
    R, t = frame_transform()
    return (R @ np.asarray(X, float).T).T + t, R
