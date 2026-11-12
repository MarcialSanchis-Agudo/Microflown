#!/usr/bin/env python3
"""Verify the closed-form GP cross-covariances quoted in Sec. 2.4 against the code (_cov_blocks) and
finite differences of sigma^2 j0(k|x-x'|):
 Cov(p(x),u_j(x')) = conj(a) s2 k j1(kr) rhat_j ,  rhat=(x-x')/r
 Cov(u_i(x),u_j(x')) = |a|^2 s2 k^2 [ j1(kr)/(kr) delta_ij - j2(kr) rhat_i rhat_j ]
 prior var of u_j = |a|^2 s2 k^2 / 3 ;  90% radius of circular complex Gaussian = sqrt(ln 10) sd."""
import sys, numpy as np; sys.path.insert(0, 'lib')
from room_gp import _cov_blocks
def jn(l, x):
    if l == 0: return np.sin(x)/x
    if l == 1: return np.sin(x)/x**2 - np.cos(x)/x
    return (3/x**2 - 1)*np.sin(x)/x - 3*np.cos(x)/x**2
k = 2*np.pi*750/343; s2 = 1.7; a = 0.3/(1j*2*np.pi*750*1.204)
rng = np.random.default_rng(0); err = []
for _ in range(20):
    x1, x2 = rng.normal(size=3)*0.2, rng.normal(size=3)*0.2
    C = _cov_blocks(x1[None], x2[None], k, s2, a)
    d = x1-x2; r = np.linalg.norm(d); e = d/r; kr = k*r
    Cpu = np.conj(a)*s2*k*jn(1, kr)*e
    Cuu = abs(a)**2*s2*k**2*(jn(1, kr)/kr*np.eye(3) - jn(2, kr)*np.outer(e, e))
    # finite differences: Cov(p(x1),u_j(x2)) = conj(a) d/dx2_j s2 j0 ; Cov(u_i(x1),u_j(x2)) = |a|^2 d2/dx1_i dx2_j
    f = lambda y1, y2: s2*jn(0, k*np.linalg.norm(y1-y2)); h = 1e-5; E = np.eye(3)
    fd_pu = np.array([np.conj(a)*(f(x1, x2+h*E[j])-f(x1, x2-h*E[j]))/(2*h) for j in range(3)])
    fd_uu = np.array([[abs(a)**2*(f(x1+h*E[i], x2+h*E[j])-f(x1+h*E[i], x2-h*E[j])-f(x1-h*E[i], x2+h*E[j])+f(x1-h*E[i], x2-h*E[j]))/(4*h*h) for j in range(3)] for i in range(3)])
    err += [np.max(np.abs(C[0, 1:]-Cpu))/np.max(np.abs(Cpu)), np.max(np.abs(C[1:, 1:]-Cuu))/np.max(np.abs(Cuu)),
            np.max(np.abs(fd_pu-Cpu))/np.max(np.abs(Cpu)), np.max(np.abs(fd_uu-Cuu))/np.max(np.abs(Cuu))]
C0 = _cov_blocks(np.zeros((1, 3)), np.zeros((1, 3)), k, s2, a)
print(f"max relative mismatch (code vs formula, FD vs formula): {max(err):.1e}")
print(f"prior velocity variance code {C0[1,1].real:.6e} formula {abs(a)**2*s2*k**2/3:.6e}")
print(f"90% radius sqrt(ln10) = {np.sqrt(np.log(10)):.4f}")
