"""Non-equilibrium Green's functions (NEGF) at finite bias.

For a scattering region with leads held at chemical potentials mu_L, the
lesser Green's function is G^< = i sum_L f_L G Gamma_L G^dag, which gives

* the non-equilibrium density  n_i = int dE/2pi sum_L f_L [G Gamma_L G^dag]_ii
  (equilibrium part on a complex contour, so bound states are included;
  bias window on the real axis),
* the current (Landauer-Buettiker at the self-consistent potential)
  I = (e/h) int dE T(E) [f_L - f_R],
* bond currents  J_{i->j} = 2 Im(H_ji rho_ij)  from the window part,
* a self-consistent electrostatic potential (Hartree level):
  phi_i = U_H (n_i - n_i^0) + phi_ext,i (local charging) or a full
  Coulomb matrix, solved with linear mixing.

Units: energies in the model's units, current in e/h x energy (multiply by
G0 = e^2/h = 3.874e-5 A/V when energies are in eV: ``result.current_A``).

>>> res = tr.negf.solve(sys, bias=0.5, mu=0.0, kT=0.01, hartree=1.0)
>>> res.current, res.density, res.potential, res.bond_currents
>>> V, I = tr.negf.iv(sys, np.linspace(-1, 1, 21), hartree=1.0, n_jobs=40)

References: Datta, *Quantum Transport: Atom to Transistor* (2005);
Brandbyge et al., PRB 65, 165401 (2002) (TranSIESTA contour scheme).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..parallel import pmap
from .solver import G0_SIEMENS, bias_potential


def _fermi(E, mu, kT):
    if kT <= 0:
        return (E < mu).astype(float) + 0.5 * (E == mu)
    return 0.5 * (1 - np.tanh((E - mu) / (2 * kT)))


@dataclass
class NEGFResult:
    bias: float
    mu_leads: np.ndarray
    current: float                  # e/h x energy
    energies: np.ndarray            # window grid
    transmission: np.ndarray        # T(E) on the window grid
    density: np.ndarray             # per basis state
    density_eq: np.ndarray          # zero-bias reference density per site (self-consistent runs)
    potential: np.ndarray           # total electrostatic potential energy per basis state
    bonds: np.ndarray               # (nb, 2) site pairs
    bond_currents: np.ndarray       # particle current a -> b (same units as current)
    iterations: int
    converged: bool
    ns: int = 1

    @property
    def current_A(self):
        """Current in amperes, assuming energies in eV."""
        return self.current * G0_SIEMENS

    def site_density(self):
        return self.density.reshape(-1, self.ns).sum(1)

    def site_potential(self):
        return self.potential.reshape(-1, self.ns).mean(1)


class _Device:
    """G(z) machinery for a System at fixed potential and lead shifts."""

    def __init__(self, system, params, potential, shifts, eta):
        self.sys, self.params, self.eta = system, params, eta
        self.H = (system.hamiltonian(params) + sp.diags(np.asarray(potential, complex))).tocsc()
        self.N = self.H.shape[0]
        self.shifts = shifts
        self.iface = [L.matrices(params)[3] for L in system.leads]

    def _A(self, z):
        N = self.N
        A = z * sp.identity(N, dtype=complex, format="csc") - self.H
        rows, cols, vals, Gam = [], [], [], []
        for li, L in enumerate(self.sys.leads):
            S = L.self_energy(z - self.shifts[li], self.params, eta=self.eta)
            idx = self.iface[li]
            r, c = np.meshgrid(idx, idx, indexing="ij")
            rows.append(r.ravel()); cols.append(c.ravel()); vals.append(S.ravel())
            Gam.append(1j * (S - S.conj().T))
        Sig = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
        return (A - Sig).tocsc(), Gam

    def diag_G(self, z, chunk=256):
        """Diagonal of G(z) (complex z, upper half plane)."""
        A, _ = self._A(z)
        lu = spla.splu(A)
        N = self.N
        out = np.empty(N, complex)
        for s in range(0, N, chunk):
            idx = np.arange(s, min(N, s + chunk))
            B = np.zeros((N, len(idx)), complex)
            B[idx, np.arange(len(idx))] = 1
            out[idx] = lu.solve(B)[idx, np.arange(len(idx))]
        return out

    def window(self, E):
        """Per-lead spectral columns at real E: returns (rho_L (nlead, N) = diag G Gamma_L G^dag,
        list of psi_L = G[:, iface] sqrt(Gamma_L), T matrix)."""
        A, Gam = self._A(E + 1j * self.eta)
        lu = spla.splu(A)
        rho, psis = [], []
        for li, idx in enumerate(self.iface):
            B = np.zeros((self.N, len(idx)), complex)
            B[idx, np.arange(len(idx))] = 1
            Gc = lu.solve(B)
            g, U = np.linalg.eigh(Gam[li])
            keep = g > 1e-14 * max(g.max(), 1e-300)
            psi = Gc @ (U[:, keep] * np.sqrt(g[keep]))
            psis.append(psi)
            rho.append((np.abs(psi) ** 2).sum(1))
        nl = len(self.iface)
        T = np.zeros((nl, nl))
        for i in range(nl):
            for j in range(nl):
                if i != j:
                    # T_ij = Tr[Gamma_i G Gamma_j G^dag] = || sqrt(Gamma_i) G_{ij} sqrt(Gamma_j) ||^2
                    T[i, j] = np.sum(np.abs(psis[j][self.iface[i]].conj().T @ _sqrtm(Gam[i])) ** 2)
        return np.array(rho), psis, T


def _sqrtm(M):
    g, U = np.linalg.eigh(M)
    return (U * np.sqrt(np.clip(g, 0, None))) @ U.conj().T


def _contour_density(dev, E_lo, E_c, npts=40):
    """-(1/pi) Im int_{E_lo}^{E_c} G(E) dE along a semicircle in the upper half plane
    (exact for T = 0 occupation below E_c, includes bound states)."""
    R = (E_c - E_lo) / 2
    c = (E_c + E_lo) / 2
    x, w = np.polynomial.legendre.leggauss(npts)
    th = np.pi * (1 - x) / 2                 # pi -> 0
    z = c + R * np.exp(1j * th)
    dz = 1j * R * np.exp(1j * th) * (-np.pi / 2)
    acc = np.zeros(dev.N, complex)
    for zi, dzi, wi in zip(z, dz, w):
        acc += wi * dzi * dev.diag_G(zi)
    # integral along the arc from E_lo to E_c equals the real-axis integral (no poles inside)
    return -acc.imag / np.pi


def _lower_bound(system, params, potential, shifts):
    H = system.hamiltonian(params)
    r = np.asarray(abs(H).sum(1)).ravel()
    d = H.diagonal().real + np.asarray(potential).real
    lo = (d - (r - np.abs(H.diagonal()))).min()
    for li, L in enumerate(system.leads):
        H0, Vm = L.matrices(params)[:2]
        rr = np.abs(H0).sum(1) + 2 * np.abs(Vm).sum(1)
        lo = min(lo, (np.diag(H0).real - rr + np.abs(np.diag(H0))).min() + shifts[li])
    return lo - 0.5


def _bond_list(system, params):
    H = system.hamiltonian(params).tocoo()
    ns = system.ns
    a, b = H.row // ns, H.col // ns
    m = a < b
    return np.unique(np.stack([a[m], b[m]], 1), axis=0)


def _bond_currents(system, params, potential, rho_win, bonds):
    """J_{a->b} = 2 Im Tr[H_ba rho_ab] for the window density matrix rho (N x N, dense)."""
    H = (system.hamiltonian(params) + sp.diags(np.asarray(potential, complex))).tocsr()
    ns = system.ns
    J = np.zeros(len(bonds))
    for n, (a, b) in enumerate(bonds):
        sa, sb = slice(a * ns, (a + 1) * ns), slice(b * ns, (b + 1) * ns)
        Hba = H[sb, sa].toarray()
        J[n] = 2 * np.imag(np.trace(Hba @ rho_win[sa, sb]))
    return J


def solve(system, bias=0.0, mu=0.0, kT=0.0, params=None, profile="linear", axis=0, lead_voltages=None,
          hartree=None, coulomb=None, mixing=0.3, tol=1e-5, maxiter=60, n_energy=301, n_contour=40,
          eta=1e-8, from_lead=0, to_lead=1, currents=True, verbose=False):
    """Self-consistent (or fixed-potential) NEGF solution at finite bias.

    bias          : V; lead ``from_lead`` at mu + V/2, ``to_lead`` at mu - V/2
                    (other leads at mu unless ``lead_voltages`` gives all of them)
    profile       : initial / external potential: 'linear' ramp along ``axis``,
                    'flat', an array (per basis state) or a function(system, bias)
    hartree       : local charging energy U_H: phi_i = U_H (n_i - n_i^0) (per site)
    coulomb       : full interaction matrix (nsites x nsites) instead of U_H
    Returns an :class:`NEGFResult`.
    """
    nl = len(system.leads)
    if nl < 2:
        raise ValueError("NEGF needs at least two leads")
    if lead_voltages is None:
        lead_voltages = np.zeros(nl)
        lead_voltages[from_lead], lead_voltages[to_lead] = bias / 2, -bias / 2
    lead_voltages = np.asarray(lead_voltages, float)
    mus = mu + lead_voltages
    shifts = list(lead_voltages)
    ns = system.ns
    nsite = system.nsites
    if callable(profile):
        phi_ext = np.asarray(profile(system, bias), float)
    elif isinstance(profile, str):
        phi_ext = bias_potential(system, bias, axis, profile)
    else:
        phi_ext = np.asarray(profile, float)

    # zero-bias reference density (charge neutrality) for the Hartree term
    selfcons = hartree is not None or coulomb is not None
    if selfcons:
        if coulomb is None:
            coulomb = hartree * np.eye(nsite)
        coulomb = np.asarray(coulomb, float)
        n0 = _density(system, params, np.zeros(system.nbasis), [0.0] * nl, np.full(nl, mu), kT,
                      n_energy, n_contour, eta)[0].reshape(-1, ns).sum(1)
    phi = phi_ext.copy()
    it, converged = 0, not selfcons
    while True:
        n, E, w, rho_win_parts, T = _density(system, params, phi, shifts, mus, kT, n_energy, n_contour, eta,
                                             want_window=True, pair=(from_lead, to_lead))
        if not selfcons:
            break
        nsites_dens = n.reshape(-1, ns).sum(1)
        phi_new = phi_ext + np.repeat(coulomb @ (nsites_dens - n0), ns)
        err = np.abs(phi_new - phi).max()
        if verbose:
            print(f"iter {it}: max |dphi| = {err:.2e}")
        phi = (1 - mixing) * phi + mixing * phi_new
        it += 1
        if err < tol:
            converged = True
            n, E, w, rho_win_parts, T = _density(system, params, phi, shifts, mus, kT, n_energy, n_contour,
                                                 eta, want_window=True, pair=(from_lead, to_lead))
            break
        if it >= maxiter:
            break
    fL, fR = _fermi(E, mus[from_lead], kT), _fermi(E, mus[to_lead], kT)
    I = float(np.sum(w * T * (fL - fR)))       # e/h x energy (Landauer at the converged potential)
    bonds = _bond_list(system, params) if currents else np.zeros((0, 2), int)
    J = np.zeros(len(bonds))
    if currents and len(bonds):
        J = _bond_currents(system, params, phi, rho_win_parts, bonds) * 2 * np.pi
    return NEGFResult(bias, mus, float(I), E, T, n, n0 if selfcons else None, phi, bonds, J, it, converged, ns)


def _density(system, params, phi, shifts, mus, kT, n_energy, n_contour, eta, want_window=False, pair=(0, 1)):
    """Density per basis state; optionally the window density matrix for currents
    (lead pair[0] minus pair[1] occupations) and T(E) between the pair.
    Returns (n, E, weights, rho_win, T). At kT = 0 the bias window [mu_min, mu_max]
    is integrated with the midpoint rule (exact for the step occupations); at
    kT > 0 a trapezoid rule over the window widened by 12 kT."""
    dev = _Device(system, params, phi, shifts, eta)
    mu_lo, mu_hi = mus.min(), mus.max()
    if kT > 0:
        E_c, E_hi = mu_lo - 12 * kT, mu_hi + 12 * kT
        E = np.linspace(E_c, E_hi, n_energy)
        w = np.full(len(E), E[1] - E[0]); w[[0, -1]] *= 0.5
    else:
        E_c, E_hi = mu_lo, mu_hi
        if E_hi - E_c < 1e-15:
            E, w = np.zeros(0), np.zeros(0)
        else:
            edges = np.linspace(E_c, E_hi, n_energy + 1)
            E, w = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    E_lo = _lower_bound(system, params, phi, shifts)
    n = _contour_density(dev, E_lo, E_c, n_contour)
    T = np.zeros(len(E))
    rho_win = np.zeros((dev.N, dev.N), complex) if want_window else None
    i0, i1 = pair
    for k, e in enumerate(E):
        rho, psis, Tm = dev.window(e)
        f = np.array([_fermi(np.array([e]), m, kT)[0] for m in mus])
        n = n + w[k] * (f[:, None] * rho).sum(0) / (2 * np.pi)
        T[k] = Tm[i1, i0]
        if want_window:
            df = f[i0] - f[i1]
            if abs(df) > 1e-14:
                psi = psis[i0]
                rho_win += w[k] * df * (psi @ psi.conj().T) / (2 * np.pi)
    return n, E, w, rho_win, T


def iv(system, biases, mu=0.0, kT=0.0, n_jobs=None, **kw):
    """Self-consistent I(V) over many biases (parallel over biases).
    Returns (biases, currents in e/h x energy)."""
    biases = np.atleast_1d(np.asarray(biases, float))
    kw.setdefault("currents", False)
    res = pmap(lambda V: solve(system, V, mu, kT, **kw).current, list(biases), n_jobs)
    return biases, np.array(res)


def differential_conductance(system, bias, dV=1e-3, **kw):
    """dI/dV (e^2/h) by a centred difference of self-consistent currents."""
    Ip = solve(system, bias + dV / 2, currents=False, **kw).current
    Im = solve(system, bias - dV / 2, currents=False, **kw).current
    return (Ip - Im) / dV
