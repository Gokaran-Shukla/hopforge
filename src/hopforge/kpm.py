"""Kernel polynomial method (KPM) for large sparse Hamiltonians.

Weisse, Wellein, Alvermann & Fehske, Rev. Mod. Phys. 78, 275 (2006);
Kubo-Bastin conductivity: Garcia, Covaci & Rappoport, PRL 114, 116602 (2015).

All functions take a scipy.sparse (or dense) Hermitian matrix H, e.g.
``model.supercell([60, 60]).hamiltonian_sparse()`` (periodic) or
``system.hamiltonian()`` (finite transport region).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def spectral_bounds(H, margin=0.01):
    """Estimate (Emin, Emax) with a sparse eigensolver (Gershgorin fallback)."""
    N = H.shape[0]
    try:
        if N < 400:
            w = np.linalg.eigvalsh(H.toarray() if sp.issparse(H) else H)
            lo, hi = w[0], w[-1]
        else:
            hi = spla.eigsh(H, k=1, which="LA", tol=1e-4, return_eigenvectors=False)[0]
            lo = spla.eigsh(H, k=1, which="SA", tol=1e-4, return_eigenvectors=False)[0]
    except Exception:
        Hc = sp.csr_matrix(H)
        r = np.asarray(abs(Hc).sum(1)).ravel()
        d = Hc.diagonal().real
        lo, hi = (d - (r - abs(d))).min(), (d + (r - abs(d))).max()
    w = hi - lo
    return float(lo - margin * w - 1e-9), float(hi + margin * w + 1e-9)


def jackson_kernel(N):
    n = np.arange(N)
    q = np.pi / (N + 1)
    return ((N - n + 1) * np.cos(q * n) + np.sin(q * n) / np.tan(q)) / (N + 1)


def lorentz_kernel(N, lam=4.0):
    n = np.arange(N)
    return np.sinh(lam * (1 - n / N)) / np.sinh(lam)


def _kernel(N, kernel):
    if kernel == "jackson":
        return jackson_kernel(N)
    if kernel == "lorentz":
        return lorentz_kernel(N)
    if kernel is None:
        return np.ones(N)
    raise ValueError("kernel must be 'jackson', 'lorentz' or None")


class _Scaled:
    def __init__(self, H, bounds=None):
        self.H = sp.csr_matrix(H)
        lo, hi = bounds or spectral_bounds(self.H)
        self.a = (hi - lo) / 2.0
        self.b = (hi + lo) / 2.0
        self.N = self.H.shape[0]

    def __call__(self, v):
        return (self.H @ v - self.b * v) / self.a

    def eps(self, E):
        return (np.asarray(E, float) - self.b) / self.a


def _random_vectors(N, R, rng):
    return np.exp(2j * np.pi * rng.random((N, R)))   # random phases, <r|r> = N


def moments(H, num_moments=200, vectors=None, num_vectors=10, operator=None, seed=0,
            bounds=None):
    """Chebyshev moments mu_n = <v| O T_n(H~) |v> (averaged over vectors)."""
    S = _Scaled(H, bounds)
    rng = np.random.default_rng(seed)
    V = _random_vectors(S.N, num_vectors, rng) if vectors is None else np.asarray(vectors, complex)
    if V.ndim == 1:
        V = V[:, None]
    O = (lambda x: x) if operator is None else (lambda x: operator @ x)
    left = O(V) if operator is not None else V
    mu = np.zeros((num_moments, V.shape[1]), complex)
    a0, a1 = V, S(V)
    mu[0] = np.einsum("ir,ir->r", left.conj(), a0)
    if num_moments > 1:
        mu[1] = np.einsum("ir,ir->r", left.conj(), a1)
    for n in range(2, num_moments):
        a0, a1 = a1, 2 * S(a1) - a0
        mu[n] = np.einsum("ir,ir->r", left.conj(), a1)
    return S, mu


def _reconstruct(S, mu, energies, kernel):
    Nm = mu.shape[0]
    g = _kernel(Nm, kernel)
    eps = S.eps(energies)
    inside = np.abs(eps) < 1
    out = np.zeros(len(eps))
    e = eps[inside]
    T = np.cos(np.outer(np.arange(Nm), np.arccos(e)))  # T_n(e)
    c = g * mu
    c[1:] *= 2
    out[inside] = (c @ T).real / (np.pi * np.sqrt(1 - e ** 2) * S.a)
    return out


def dos(H, energies=None, num_moments=300, num_vectors=10, seed=0, kernel="jackson",
        operator=None, bounds=None):
    """KPM density of states (states per energy, integrates to the matrix
    dimension). ``operator`` gives e.g. spin- or orbital-projected DOS."""
    S, mu = moments(H, num_moments, None, num_vectors, operator, seed, bounds)
    if energies is None:
        energies = np.linspace(S.b - 0.98 * S.a, S.b + 0.98 * S.a, 800)
    energies = np.asarray(energies, float)
    return energies, _reconstruct(S, mu.mean(1), energies, kernel)


def ldos(H, sites, energies, num_moments=400, kernel="jackson", ns=1, bounds=None):
    """KPM local DOS on the given basis indices (exact unit vectors, no
    stochastic error). With ``ns`` > 1, ``sites`` are site indices and the
    internal (spin) components are summed. Returns array (len(sites), nE)."""
    N = H.shape[0]
    idx = np.concatenate([[s * ns + a for a in range(ns)] for s in np.atleast_1d(sites)])
    V = np.zeros((N, len(idx)), complex)
    V[idx, np.arange(len(idx))] = 1
    S, mu = moments(H, num_moments, V, bounds=bounds)
    out = np.array([_reconstruct(S, mu[:, r], energies, kernel) for r in range(len(idx))])
    return out.reshape(-1, ns, len(np.atleast_1d(energies))).sum(1)


def conductivity(H, vx, vy, area, energies, num_moments=200, num_vectors=10, seed=0,
                 kernel="jackson", kT=0.0, bounds=None, n_eps=4000, hall=None):
    """Kubo-Bastin conductivity sigma_xy(mu) by KPM (Garcia et al. 2015).

    H, vx, vy : sparse Hamiltonian and velocity operators (dH/dk_x, dH/dk_y),
                e.g. from ``model.supercell(P).hamiltonian_sparse()`` and
                ``velocity_sparse('x')``.
    area      : area (2D) or volume (3D) of the whole system.
    Returns sigma in units of e^2/h * (length)^(2-d), with the same sign
    convention as ``hopforge.hall_conductivity``. For vx = vy it gives the
    longitudinal (Kubo-Greenwood) conductivity.
    """
    S = _Scaled(H, bounds)
    if hall is None:
        hall = (vx is not vy) and (sp.csr_matrix(vx) != sp.csr_matrix(vy)).nnz > 0
    rng = np.random.default_rng(seed)
    R = _random_vectors(S.N, num_vectors, rng)
    M = num_moments
    vx = sp.csr_matrix(vx) / S.a
    vy = sp.csr_matrix(vy) / S.a

    def cheb_all(start):
        out = np.empty((M,) + start.shape, complex)
        out[0] = start
        if M > 1:
            out[1] = S(start)
        for n in range(2, M):
            out[n] = 2 * S(out[n - 1]) - out[n - 2]
        return out

    L = cheb_all(vx @ R)                         # T_m(H) vx r
    Tn = cheb_all(R)                             # T_n(H) r
    Rv = np.stack([vy @ Tn[n] for n in range(M)])  # vy T_n(H) r
    # MU[n, m] = Tr[vx T_n vy T_m] = sum_r <T_n vx r | vy T_m r>  (verified
    # against exact diagonalisation + Kubo formula in tests/test_transport.py)
    MU = np.einsum("nir,mir->nm", L.conj(), Rv) / num_vectors
    # exact identities: MU is Hermitian, and for a Hall response (vx != vy)
    # only the antisymmetric part survives (sigma_yx = -sigma_xy uses MU^T).
    # Projecting the stochastic estimate onto them removes most of the noise.
    MU = 0.5 * (MU + MU.conj().T)
    if hall:
        MU = 1j * MU.imag
    g = _kernel(M, kernel)
    delta = np.ones(M)
    delta[0] = 2
    MU = MU * np.outer(g / delta, g / delta)
    theta = np.linspace(0, np.pi, n_eps + 2)[1:-1]
    eps = np.cos(theta)
    s = np.sin(theta)
    n = np.arange(M)
    T = np.cos(np.outer(n, theta))                      # T_n(eps)
    A = (eps[None] - 1j * n[:, None] * s[None]) * np.exp(1j * n[:, None] * theta[None])
    B = (eps[None] + 1j * n[:, None] * s[None]) * np.exp(-1j * n[:, None] * theta[None])
    integrand = np.einsum("ne,nm,me->e", A, MU, T) + np.einsum("me,nm,ne->e", B, MU, T)
    integrand = integrand.real / (1 - eps ** 2) ** 2
    dE = s * (np.pi / (n_eps + 1))                       # d eps = sin(theta) d theta
    energies = np.atleast_1d(np.asarray(energies, float))
    out = []
    for mu_ in energies:
        e_mu = S.eps(mu_)
        if kT > 0:
            f = 1 / (np.exp(np.clip((eps - e_mu) * S.a / kT, -200, 200)) + 1)
        else:
            f = (eps < e_mu).astype(float)
        val = 4 / (np.pi * area) * np.sum(f * integrand * dE)
        out.append(val)
    # e^2/hbar -> e^2/h and the hopforge (Wannier90) sign convention
    return -2 * np.pi * np.array(out)
