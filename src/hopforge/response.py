"""Linear and nonlinear response beyond the Hall conductivity.

Natural units e = hbar = k_B = 1 (energies e.g. eV, lengths Angstrom) unless
a ``units`` option says otherwise; every function documents its formula.
Electron charge is -e; signs follow Xiao, Chang & Niu, RMP 82, 1959 (2010)
for Berry-phase quantities. All k-sums run in parallel with ``n_jobs``.

Functions
---------
boltzmann(...)            sigma/tau, Seebeck, electronic kappa, Lorenz number
edelstein(...)            current-induced spin / orbital polarisation (extrinsic
                          Boltzmann + intrinsic interband terms)
berry_curvature_dipole(...)  D_ab for the nonlinear Hall effect (Sodemann-Fu)
orbital_moment(...)       band-resolved orbital magnetic moment m_n(k)
orbital_magnetization(...) modern theory M = sum f [m + (mu - E) Omega]
optical_conductivity(...) interband (+ Drude) Kubo-Greenwood sigma_ab(omega)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parallel import pmap, split, chunks_for
from .solvers import kmesh

MU_B_PER_UNIT = 0.26250  # (e/hbar) eV A^2 in Bohr magnetons
SIGMA_3D_PER_FS = 3.6966e6   # sigma[S/m] = this * tau[fs] * X[eV/A]
SIGMA_2D_PER_FS = 3.6966e-4  # sheet sigma[S] = this * tau[fs] * X[eV]
_AX = {"x": 0, "y": 1, "z": 2}


def _ax(a):
    return _AX[a] if isinstance(a, str) else int(a)


def _fermi(E, mu, kT):
    if kT <= 0:
        return (E < mu).astype(float)
    return 0.5 * (1 - np.tanh(0.5 * (E - mu) / kT))


def _mdfde(E, mu, kT):
    """-df/dE (normalised peak 1/(4 kT))."""
    x = 0.5 * (E - mu) / kT
    return 0.25 / kT / np.cosh(np.clip(x, -300, 300)) ** 2


def _kloop(model, nk, func, n_jobs):
    """Run func(E, V, vel) on k-chunks in parallel and sum the results."""
    k = kmesh(model, nk)
    parts = split(k, chunks_for(len(k), n_jobs))

    def one(kk):
        E, V = model.eig(kk)
        vel = model.velocity(kk)
        vb = np.einsum("kin,akij,kjm->aknm", V.conj(), vel, V, optimize=True)  # band basis
        return func(E, V, vb)
    res = pmap(one, parts, n_jobs)
    tot = res[0]
    for r in res[1:]:
        tot = tot + r if not isinstance(tot, tuple) else tuple(a + b for a, b in zip(tot, r))
    return tot, len(k)


# ---------------------------------------------------------------------------
# Boltzmann transport
# ---------------------------------------------------------------------------
@dataclass
class BoltzmannResult:
    mu: np.ndarray
    sigma_over_tau: np.ndarray   # (nmu, D, D) natural units (e^2/hbar^2 * energy/length^(d-2))
    seebeck: np.ndarray          # (nmu, D, D) in k_B/e  (multiply by 86.17 for uV/K)
    kappa_over_tau: np.ndarray   # electronic, natural units
    lorenz: np.ndarray           # (nmu,) L = kappa_xx / (sigma_xx T) in (k_B/e)^2
    kT: float
    dim: int

    def sigma_SI(self, tau_fs):
        """sigma in S/m (3D) or S (2D sheet) for a relaxation time in fs."""
        c = SIGMA_3D_PER_FS if self.dim == 3 else SIGMA_2D_PER_FS
        return c * tau_fs * self.sigma_over_tau

    @property
    def seebeck_uV_per_K(self):
        return 86.17333 * self.seebeck


def boltzmann(model, mu, kT=0.025, nk=80, n_jobs=None):
    """Constant-relaxation-time Boltzmann transport.

    L^(n)_ab = (1/(V N_k)) sum_nk (-df/dE) (E - mu)^n v_a v_b   (per tau)
    sigma = L0,  S = -(1/kT) L0^-1 L1  [k_B/e],  kappa = (L2 - L1 L0^-1 L1) / T,
    Lorenz number L = kappa_xx / (sigma_xx T) -> pi^2/3 for a degenerate metal.
    Band velocities are the diagonal of the velocity operator.
    """
    mus = np.atleast_1d(np.asarray(mu, float))
    D = model.lat.D

    def f(E, V, vb):
        v = np.real(np.einsum("akn n->akn".replace(" ", ""), vb))   # (D, nk, nb) diagonal
        out = np.zeros((len(mus), 3, D, D))
        for i, m in enumerate(mus):
            w = _mdfde(E, m, kT)
            for nn in range(3):
                ww = w * (E - m) ** nn
                out[i, nn] = np.einsum("akn,bkn,kn->ab", v, v, ww)
        return out
    tot, Nk = _kloop(model, nk, f, n_jobs)
    L = tot / (Nk * model.lat.cell_volume)
    L0, L1, L2 = L[:, 0], L[:, 1], L[:, 2]
    dims = list(range(model.dim))
    S = np.zeros_like(L0)
    K = np.zeros_like(L0)
    Lor = np.zeros(len(mus))
    for i in range(len(mus)):
        sub = np.ix_(dims, dims)
        L0i = np.linalg.pinv(L0[i][sub])
        S[i][sub] = -(L0i @ L1[i][sub]) / kT
        K[i][sub] = (L2[i][sub] - L1[i][sub] @ L0i @ L1[i][sub]) / kT
        Lor[i] = K[i][0, 0] / (L0[i][0, 0] * kT) if L0[i][0, 0] > 0 else np.nan
    return BoltzmannResult(mus, L0, S, K, Lor, kT, model.dim)


def drude_weight(model, mu, kT=0.01, nk=80, n_jobs=None):
    """Intraband (Drude) weight sigma/tau at chemical potential mu (natural units)."""
    return boltzmann(model, mu, kT, nk, n_jobs).sigma_over_tau


# ---------------------------------------------------------------------------
# Edelstein effect
# ---------------------------------------------------------------------------
def edelstein(model, operator, field="x", mu=0.0, kT=0.01, tau=1.0, nk=100, eta=1e-4,
              part="both", n_jobs=None):
    """Current-induced polarisation of ``operator`` (spin s_a, orbital L_a, ...)
    per unit electric field along ``field``, per unit cell.

    extrinsic (Boltzmann, TRS-even):  chi = tau * (1/N_k) sum f'(E_n) v^b_nn O_nn * (-1)
        [delta f = e tau E . v df/dE with electron charge -e, e = 1]
    intrinsic (interband, needs broken TRS, Freimuth/Bluegel/Mokrousov form):
        chi = -2 (1/N_k) sum_n f_n sum_{m!=n} Im[O_nm v^b_mn] / ((E_n-E_m)^2 + eta^2)
    Returns dict with 'extrinsic', 'intrinsic', 'total' (arrays over mu)."""
    mus = np.atleast_1d(np.asarray(mu, float))
    b = _ax(field)
    O = np.asarray(operator)

    def f(E, V, vb):
        Ob = np.einsum("kin,ij,kjm->knm", V.conj(), O, V, optimize=True)
        vbb = vb[b]
        ext = np.zeros(len(mus)); intr = np.zeros(len(mus))
        Odiag, vdiag = np.real(np.einsum("knn->kn", Ob)), np.real(np.einsum("knn->kn", vbb))
        dE = E[:, :, None] - E[:, None, :]
        nb = E.shape[1]
        off = ~np.eye(nb, dtype=bool)[None]
        term = np.where(off, np.imag(Ob * vbb.transpose(0, 2, 1)) / (dE ** 2 + eta ** 2), 0.0).sum(2)
        for i, m in enumerate(mus):
            ext[i] = tau * np.sum(_mdfde(E, m, kT) * vdiag * Odiag)
            intr[i] = -2 * np.sum(_fermi(E, m, kT) * term)
        return np.stack([ext, intr])
    tot, Nk = _kloop(model, nk, f, n_jobs)
    ext, intr = tot / Nk
    if part == "extrinsic":
        intr = intr * 0
    elif part == "intrinsic":
        ext = ext * 0
    return dict(mu=mus, extrinsic=ext, intrinsic=intr, total=ext + intr)


# ---------------------------------------------------------------------------
# Berry curvature dipole and orbital moment
# ---------------------------------------------------------------------------
def _omega_and_m(E, vb, a, b, eta=0.0):
    dE = E[:, :, None] - E[:, None, :]
    nb = E.shape[1]
    off = ~np.eye(nb, dtype=bool)[None]
    X = vb[a] * vb[b].transpose(0, 2, 1)          # v^a_nm v^b_mn
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = off & (np.abs(dE) > 1e-9)
        Om = np.where(ok, -2 * X.imag / (dE ** 2 + eta ** 2), 0).sum(2)
        m = np.where(ok, -X.imag * dE / (dE ** 2 + eta ** 2), 0).sum(2)
    return Om, m


def berry_curvature_dipole(model, mu=0.0, kT=0.01, nk=200, component="xy", n_jobs=None):
    """Berry curvature dipole (Sodemann & Fu, PRL 115, 216806 (2015)):

        D_c = int dk/(2pi)^d sum_n f_n d_{k_c} Omega_n^{ab}
            = int dk/(2pi)^d sum_n (-df/dE) v^c_nn Omega_n^{ab}

    Returns (nmu, D) with components c = x, y(, z); dimensionless in 2D.
    The nonlinear Hall current is j ~ (e^3 tau / hbar^2) D E E."""
    mus = np.atleast_1d(np.asarray(mu, float))
    a, b = _ax(component[0]), _ax(component[1])
    D = model.lat.D

    def f(E, V, vb):
        Om, _ = _omega_and_m(E, vb, a, b)
        v = np.real(np.einsum("aknn->akn", vb))
        out = np.zeros((len(mus), D))
        for i, m in enumerate(mus):
            w = _mdfde(E, m, kT)
            out[i] = np.einsum("akn,kn,kn->a", v, w, Om)
        return out
    tot, Nk = _kloop(model, nk, f, n_jobs)
    # int dk/(2pi)^d F = mean_k F / V_cell ; D = int f dOmega = int (-f') v Omega
    return tot / (Nk * model.lat.cell_volume)


def orbital_moment(model, k, component="xy", eta=0.0):
    """Band-resolved orbital moment m_n^{ab}(k) = -Im sum_{m!=n} v^a_nm v^b_mn
    / (E_n - E_m) (units e/hbar * energy * length^2; multiply by
    MU_B_PER_UNIT for Bohr magnetons with eV and Angstrom). For two bands
    m_n = (E_n - E_m) Omega_n / 2. Returns (E, Omega, m) arrays (nk, nb)."""
    a, b = _ax(component[0]), _ax(component[1])
    k = np.atleast_2d(np.asarray(k, float))
    E, V = model.eig(k)
    vel = model.velocity(k)
    vb = np.einsum("kin,akij,kjm->aknm", V.conj(), vel, V, optimize=True)
    Om, m = _omega_and_m(E, vb, a, b, eta)
    return E, Om, m


def orbital_magnetization(model, mu=0.0, kT=0.0, nk=100, component="xy", n_jobs=None):
    """Modern theory of orbital magnetisation (Xiao 2005, Thonhauser 2005,
    Ceresoli 2006) per unit cell:
        M = (1/N_k) sum_nk f_n [ m_n + (mu - E_n) Omega_n ]
    (units e/hbar * energy * length^2; x MU_B_PER_UNIT -> mu_B per cell).
    In a Chern-insulator gap dM/dmu = C * A_cell / (2 pi) (Streda)."""
    mus = np.atleast_1d(np.asarray(mu, float))
    a, b = _ax(component[0]), _ax(component[1])

    def f(E, V, vb):
        Om, m = _omega_and_m(E, vb, a, b)
        return np.array([np.sum(_fermi(E, u, kT) * (m + (u - E) * Om)) for u in mus])
    tot, Nk = _kloop(model, nk, f, n_jobs)
    return tot / Nk


# ---------------------------------------------------------------------------
# optical conductivity
# ---------------------------------------------------------------------------
def optical_conductivity(model, omegas, mu=0.0, kT=0.0, nk=100, eta=0.02, component="xx",
                         drude_tau=None, n_jobs=None):
    """Kubo-Greenwood optical conductivity sigma_ab(omega) (complex, units
    e^2/hbar * length^(2-d); 2D: divide by 2 pi for e^2/h):

        sigma_ab = (i / (V N_k)) sum_{n != m} (f_n - f_m) / (E_m - E_n)
                   * v^a_nm v^b_mn / (omega + E_n - E_m + i eta)

    plus an optional Drude term D_ab / (1/tau - i omega) with D from Boltzmann.
    Graphene (two valleys, spinless): Re sigma_xx -> e^2/(8 hbar) = 0.125."""
    oms = np.atleast_1d(np.asarray(omegas, float))
    a, b = _ax(component[0]), _ax(component[1])

    def f(E, V, vb):
        fE = _fermi(E, mu, kT)
        nbands = E.shape[1]
        out = np.zeros(len(oms), complex)
        for n in range(nbands):
            for mm in range(nbands):
                if n == mm:
                    continue
                dEmn = E[:, mm] - E[:, n]
                df = fE[:, n] - fE[:, mm]
                ok = (np.abs(df) > 1e-12) & (np.abs(dEmn) > 1e-9)
                if not ok.any():
                    continue
                w = (df[ok] / dEmn[ok]) * vb[a][ok, n, mm] * vb[b][ok, mm, n]
                out += 1j * np.sum(w[None, :] / (oms[:, None] - dEmn[ok][None, :] + 1j * eta), axis=1)
        return out
    tot, Nk = _kloop(model, nk, f, n_jobs)
    sigma = tot / (Nk * model.lat.cell_volume)
    if drude_tau is not None:
        Dw = boltzmann(model, mu, max(kT, 1e-3), nk, n_jobs).sigma_over_tau[0][a, b]
        sigma = sigma + Dw / (1.0 / drude_tau - 1j * oms)
    return sigma
