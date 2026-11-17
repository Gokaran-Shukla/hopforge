"""Scattering solver: Green's functions, transmissions (Caroli / Fisher-Lee),
transmission eigenvalues and shot noise, multi-terminal conductance matrices,
LDOS, scattering wavefunctions and bond currents.

    G^r(E) = [ (E + i eta) - H_S - sum_l Sigma_l(E) ]^{-1}
    Gamma_l = i (Sigma_l - Sigma_l^dag)
    T_{ij}  = Tr[ Gamma_i G^r Gamma_j G^a ]        (from lead j to lead i)

Transmissions are in units of e^2/h per spin-resolved channel (no factor 2).
Sweeps over energies, parameters and disorder realisations run in parallel
with ``n_jobs`` worker processes (see :mod:`hopforge.parallel`).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..parallel import pmap


class SMatrix:
    """Scattering quantities at one energy. Create with ``system.smatrix(E, params)``."""

    def __init__(self, system, energy, params=None, eta=1e-10, lead_eta=1e-10, potential=None,
                 lead_shifts=None):
        if not system.leads:
            raise ValueError("attach at least one lead first")
        self.sys = system
        self.energy = float(energy)
        self.params = params
        H = system.hamiltonian(params)
        N = system.nbasis
        if potential is not None:
            # electrostatic potential energy per basis state (e.g. a bias ramp)
            H = H + sp.diags(np.asarray(potential, complex))
        self.lead_shifts = lead_shifts or [0.0] * len(system.leads)
        A = (energy + 1j * eta) * sp.identity(N, dtype=complex, format="csr") - H
        self.Sigma, self.Gamma, self.iface = [], [], []
        rows, cols, vals = [], [], []
        for li, L in enumerate(system.leads):
            idx = L.matrices(params)[3]
            S = L.self_energy(energy - self.lead_shifts[li], params, eta=lead_eta)
            self.Sigma.append(S)
            self.Gamma.append(1j * (S - S.conj().T))
            self.iface.append(idx)
            r, c = np.meshgrid(idx, idx, indexing="ij")
            rows.append(r.ravel()); cols.append(c.ravel()); vals.append(S.ravel())
        Sig = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(N, N))
        self.A = (A - Sig).tocsc()
        self.lu = spla.splu(self.A)
        self._cols = {}

    # -------------------------------------------------------------------
    def _G_cols(self, lead):
        if lead not in self._cols:
            idx = self.iface[lead]
            B = np.zeros((self.A.shape[0], len(idx)), complex)
            B[idx, np.arange(len(idx))] = 1
            self._cols[lead] = self.lu.solve(B)
        return self._cols[lead]

    def greens(self, to_lead, from_lead):
        """Block of G^r between the interface states of two leads."""
        return self._G_cols(from_lead)[self.iface[to_lead]]

    def _block(self, lead, block):
        if block is None:
            return None
        return self.sys.mask(**block)[self.iface[lead]]

    def num_modes(self, lead, block=None):
        P = None
        if block is not None:
            P = np.diag(self.sys.lead_mask(lead, **block).astype(float))
        return self.sys.leads[lead].num_modes(self.energy, P, self.params)

    def lead_modes(self, lead):
        """(k, velocity, wavefunctions) of the propagating modes of a lead."""
        return self.sys.leads[lead].modes(self.energy, None, self.params)

    def transmission(self, to_lead, from_lead, to_block=None, from_block=None):
        """Transmission probability from ``from_lead`` to ``to_lead``.

        ``*_block`` select a conserved sub-block, e.g. ``dict(spin='up')`` or
        ``dict(orbitals=['A_h'])`` (holes of a BdG model). For the same lead and
        same block the reflection probability is returned (from unitarity).
        """
        if to_lead == from_lead and to_block == from_block:
            Nf = self.num_modes(from_lead, from_block)
            out = 0.0
            for i in range(len(self.sys.leads)):
                if i != from_lead:
                    out += self.transmission(i, from_lead, None, from_block)
            if from_block is not None:
                comp = ~self.sys.mask(**from_block)
                out += self._caroli(from_lead, from_lead, comp[self.iface[from_lead]],
                                    self._block(from_lead, from_block))
            return Nf - out
        return self._caroli(to_lead, from_lead, self._block(to_lead, to_block),
                            self._block(from_lead, from_block))

    def _caroli(self, i, j, mi=None, mj=None):
        G = self.greens(i, j)
        Gi, Gj = self.Gamma[i], self.Gamma[j]
        if mi is not None:
            Gi = Gi * np.outer(mi, mi)
        if mj is not None:
            Gj = Gj * np.outer(mj, mj)
        return float(np.real(np.trace(Gi @ G @ Gj @ G.conj().T)))

    # -------------------------------------------------------------------
    @staticmethod
    def _sqrt_psd(M, tol=1e-12):
        w, U = np.linalg.eigh(0.5 * (M + M.conj().T))
        w = np.clip(w, 0, None)
        keep = w > tol * max(w.max(), 1e-300)
        return U[:, keep] * np.sqrt(w[keep])

    def transmission_matrix(self, to_lead, from_lead):
        """t = Gamma_to^{1/2} G^r Gamma_from^{1/2} in the channel spaces of the
        two leads (unitarily equivalent to the mode-basis block of S: same
        singular values, T = Tr t t^dag)."""
        Ai = self._sqrt_psd(self.Gamma[to_lead])
        Aj = self._sqrt_psd(self.Gamma[from_lead])
        return Ai.conj().T @ self.greens(to_lead, from_lead) @ Aj

    def transmission_eigenvalues(self, to_lead=1, from_lead=0):
        """Transmission eigenvalues tau_n in [0, 1], descending."""
        s = np.linalg.svd(self.transmission_matrix(to_lead, from_lead), compute_uv=False)
        return np.clip(s ** 2, 0, 1)

    def shot_noise(self, to_lead=1, from_lead=0):
        """Zero-temperature shot noise (S, F): S = sum tau (1 - tau) in units of
        2 e^3 |V| / h and the Fano factor F = S / sum tau."""
        tau = self.transmission_eigenvalues(to_lead, from_lead)
        S, T = float(np.sum(tau * (1 - tau))), float(np.sum(tau))
        return S, (S / T if T > 1e-14 else np.nan)

    # -------------------------------------------------------------------
    def conductance_matrix(self):
        """Multi-terminal conductance matrix (e^2/h): I_i = sum_j G_ij V_j."""
        n = len(self.sys.leads)
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    G[i, j] = -self.transmission(i, j)
        for i in range(n):
            G[i, i] = -G[:, i].sum()
        return G

    def four_terminal_resistance(self, current_in, current_out, v_plus, v_minus):
        """R = (V_plus - V_minus) / I, current from ``current_in`` to
        ``current_out`` (units of h/e^2)."""
        G = self.conductance_matrix()
        n = G.shape[0]
        keep = [k for k in range(n) if k != current_out]
        I = np.zeros(n)
        I[current_in], I[current_out] = 1.0, -1.0
        V = np.zeros(n)
        V[keep] = np.linalg.lstsq(G[np.ix_(keep, keep)], I[keep], rcond=None)[0]
        return V[v_plus] - V[v_minus]

    # -------------------------------------------------------------------
    def ldos(self, chunk=512):
        """Local density of states -Im G_aa / pi on every basis state."""
        N = self.A.shape[0]
        out = np.empty(N)
        for s in range(0, N, chunk):
            idx = np.arange(s, min(N, s + chunk))
            B = np.zeros((N, len(idx)), complex)
            B[idx, np.arange(len(idx))] = 1
            X = self.lu.solve(B)
            out[idx] = -X[idx, np.arange(len(idx))].imag / np.pi
        return out

    def wavefunctions(self, lead, tol=1e-10):
        """Scattering states injected from ``lead``:
        sum_n psi_n psi_n^dag = G Gamma_lead G^dag."""
        g, U = np.linalg.eigh(self.Gamma[lead])
        keep = g > tol * max(g.max(), 1e-300)
        return self._G_cols(lead) @ (U[:, keep] * np.sqrt(g[keep]))

    def density(self, lead, per_site=True, operator=None):
        """Density injected from ``lead`` per unit energy; ``operator`` (ns x ns)
        gives e.g. a spin density."""
        psi = self.wavefunctions(lead)
        ns = self.sys.ns
        if operator is None:
            rho = (np.abs(psi) ** 2).sum(1) / (2 * np.pi)
            return rho.reshape(-1, ns).sum(1) if per_site else rho
        P = psi.reshape(-1, ns, psi.shape[1])
        return np.einsum("sim,ij,sjm->s", P.conj(), np.asarray(operator), P).real / (2 * np.pi)

    def current(self, lead, operator=None, where=None):
        """Bond currents from ``lead`` (sum over any cross-section = transmission).
        Returns (bonds (nb, 2), J). ``operator`` (ns x ns) for spin currents,
        ``where(pos_a, pos_b) -> bool`` to restrict bonds."""
        psi = self.wavefunctions(lead)
        Hd = self.sys.hamiltonian(self.params).tocsr()
        H = Hd.tocoo()
        ns = self.sys.ns
        Op = np.eye(ns) if operator is None else np.asarray(operator)
        a_site, b_site = H.row // ns, H.col // ns
        mask = a_site < b_site
        pairs = np.unique(np.stack([a_site[mask], b_site[mask]], 1), axis=0)
        if where is not None:
            P = self.sys.positions()
            pairs = np.array([p for p in pairs if where(P[p[0]], P[p[1]])], int).reshape(-1, 2)
        J = np.zeros(len(pairs))
        for n, (a, b) in enumerate(pairs):
            Hba = Hd[a * ns:(a + 1) * ns, b * ns:(b + 1) * ns].toarray().conj().T
            pa, pb = psi[a * ns:(a + 1) * ns], psi[b * ns:(b + 1) * ns]
            M = 0.5 * (Op @ Hba + Hba @ Op)
            J[n] = np.sum(2 * np.imag(np.einsum("im,ij,jm->m", pb.conj(), M, pa)))
        return pairs, J


def smatrix(system, energy, params=None, **kw):
    """Solve the scattering problem at one energy."""
    return SMatrix(system, energy, params=params, **kw)


def conductance(system, energies, to_lead=1, from_lead=0, params=None, n_jobs=None, **kw):
    """T(E) between two leads, parallel over energies."""
    energies = np.atleast_1d(energies)
    f = lambda E: SMatrix(system, E, params=params).transmission(to_lead, from_lead, **kw)
    return np.array(pmap(f, energies, n_jobs))


def sweep(system, func, energies=None, params_list=None, n_jobs=None):
    """Parallel sweep of ``func(smatrix)`` over energies and/or parameter sets.
    Returns a flat list (one parameter set) or a list per parameter set."""
    energies = [0.0] if energies is None else list(np.atleast_1d(energies))
    params_list = [None] if params_list is None else list(params_list)
    jobs = [(E, p) for p in params_list for E in energies]
    out = pmap(lambda job: func(SMatrix(system, job[0], params=job[1])), jobs, n_jobs)
    if len(params_list) == 1:
        return out
    n = len(energies)
    return [out[i * n:(i + 1) * n] for i in range(len(params_list))]


def disorder_average(make_system, seeds, energies, to_lead=1, from_lead=0, n_jobs=None, log=False):
    """Mean and std of T(E) (or ln T) over disorder realisations, in parallel.
    ``make_system(seed)`` returns a System with leads attached."""
    energies = np.atleast_1d(energies)

    def one(seed):
        s = make_system(seed)
        T = np.array([SMatrix(s, E).transmission(to_lead, from_lead) for E in energies])
        return np.log(np.clip(T, 1e-300, None)) if log else T
    res = np.array(pmap(one, list(seeds), n_jobs))
    return res.mean(0), res.std(0)


def andreev_conductance(system, energies, normal_lead=0, electron=None, hole=None, params=None,
                        n_jobs=None):
    """NS-junction conductance (BdG), BTK form G = N_e - R_ee + R_he (e^2/h);
    parallel over energies. ``electron``/``hole`` are block specs such as
    dict(orbitals=['A_e']) / dict(orbitals=['A_h'])."""
    energies = np.atleast_1d(energies)

    def one(E):
        S = SMatrix(system, E, params=params)
        Ree = S.transmission(normal_lead, normal_lead, electron, electron)
        Rhe = S.transmission(normal_lead, normal_lead, hole, electron)
        return S.num_modes(normal_lead, electron) - Ree + Rhe
    return np.array(pmap(one, energies, n_jobs))


def spectrum(system, params_list=None, k=10, sigma=0.0, n_jobs=None, dense=None):
    """Eigenvalues of the closed scattering region (leads ignored) for each
    parameter set (like kwant.plotter.spectrum). Dense for <= 2000 states,
    otherwise the ``k`` eigenvalues closest to ``sigma``. Parallel."""
    params_list = [None] if params_list is None else list(params_list)

    def one(p):
        H = system.hamiltonian(p)
        if (dense if dense is not None else H.shape[0] <= 2000):
            return np.linalg.eigvalsh(H.toarray())
        return np.sort(spla.eigsh(H, k=k, sigma=sigma, return_eigenvectors=False).real)
    return pmap(one, params_list, n_jobs)


G0_SIEMENS = 3.874045864e-5   # e^2 / h


def bias_potential(system, bias, axis=0, profile="linear"):
    """Potential energy per basis state for a bias V between lead 0 (left,
    at +V/2 ... electron energy -eV/2) and lead 1: a linear drop along ``axis``
    across the scattering region (non-self-consistent), or 'flat' (no drop)."""
    P = system.positions()[:, axis]
    lo, hi = P.min(), P.max()
    if profile == "flat" or hi - lo < 1e-12:
        u = np.zeros(len(P))
    else:
        u = bias * (0.5 - (P - lo) / (hi - lo))
    return np.repeat(u, system.ns)


def current(system, bias, mu=0.0, kT=0.0259, n_energy=201, params=None, axis=0, profile="linear",
            from_lead=0, to_lead=1):
    """Landauer current I(V) = (e/h) int dE T(E, V) [f_L(E) - f_R(E)] in amperes
    (energies in eV). Lead ``from_lead`` sits at mu + V/2, ``to_lead`` at mu - V/2.
    The transmission is recomputed at the bias (linear potential drop)."""
    if abs(bias) < 1e-14:
        return 0.0
    window = abs(bias) / 2 + 8 * kT
    E = np.linspace(mu - window, mu + window, n_energy)
    pot = bias_potential(system, bias, axis, profile)
    shifts = [0.0] * len(system.leads)
    shifts[from_lead], shifts[to_lead] = +bias / 2, -bias / 2
    T = np.array([SMatrix(system, e, params=params, potential=pot, lead_shifts=shifts).transmission(to_lead, from_lead)
                  for e in E])
    fL = 1 / (np.exp(np.clip((E - mu - bias / 2) / kT, -200, 200)) + 1)
    fR = 1 / (np.exp(np.clip((E - mu + bias / 2) / kT, -200, 200)) + 1)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(G0_SIEMENS * trapz(T * (fL - fR), E))


def iv_curve(system, biases, mu=0.0, kT=0.0259, n_energy=201, params=None, axis=0, profile="linear",
             n_jobs=None):
    """I(V) for many biases (parallel over biases). Returns amperes."""
    return np.array(pmap(lambda V: current(system, V, mu, kT, n_energy, params, axis, profile),
                         list(np.atleast_1d(biases)), n_jobs))
