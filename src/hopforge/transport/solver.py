"""Scattering solver: Green's functions, transmissions (Caroli / Fisher-Lee),
multi-terminal conductance matrices, LDOS, scattering wavefunctions and
bond currents.

    G^r(E) = [ (E + i eta) - H_S - sum_l Sigma_l(E) ]^{-1}
    Gamma_l = i (Sigma_l - Sigma_l^dag)
    T_{ij}  = Tr[ Gamma_i G^r Gamma_j G^a ]        (from lead j to lead i)

Transmissions are in units of e^2/h per spin-resolved channel (no factor 2).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _pad(W, n):
    if W.shape[0] == n:
        return W
    return sp.vstack([W, sp.csr_matrix((n - W.shape[0], W.shape[1]))]).tocsr()


class SMatrix:
    """Scattering quantities at one energy. Create with ``system.smatrix(E)``."""

    def __init__(self, system, energy, eta=1e-10, lead_eta=1e-10):
        if not system.leads:
            raise ValueError("attach at least one lead first")
        self.sys = system
        self.energy = float(energy)
        H = system.hamiltonian()
        N = system.nbasis
        A = (energy + 1j * eta) * sp.identity(N, dtype=complex, format="csr") - H
        self.Sigma, self.Gamma, self.iface = [], [], []
        Sig_total = sp.csr_matrix((N, N), dtype=complex)
        for L in system.leads:
            L.W = _pad(L.W, N)
            S = L.self_energy(energy, eta=lead_eta)
            idx = L.iface
            self.Sigma.append(S)
            self.Gamma.append(1j * (S - S.conj().T))
            self.iface.append(idx)
            r, c = np.meshgrid(idx, idx, indexing="ij")
            Sig_total = Sig_total + sp.csr_matrix((S.ravel(), (r.ravel(), c.ravel())), shape=(N, N))
        self.A = (A - Sig_total).tocsc()
        self.lu = spla.splu(self.A)
        self._cols = {}

    # -------------------------------------------------------------------
    def _G_cols(self, lead):
        """Columns of G^r on the interface of ``lead``: shape (N, n_iface)."""
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
        m = self.sys.mask(**block)
        return m[self.iface[lead]]

    def num_modes(self, lead, block=None):
        P = None
        if block is not None:
            P = np.diag(self.sys.lead_mask(lead, **block).astype(float))
        return self.sys.leads[lead].num_modes(self.energy, P)

    def transmission(self, to_lead, from_lead, to_block=None, from_block=None):
        """Transmission probability from ``from_lead`` to ``to_lead``.

        ``*_block`` select a conserved sub-block, e.g. ``dict(spin='up')`` or
        ``dict(orbitals=['h'])`` (holes of a BdG model). For the same lead and
        same block the reflection probability is returned (from unitarity).
        """
        same = to_lead == from_lead and (to_block == from_block)
        if same:
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

    def conductance_matrix(self):
        """Multi-terminal conductance matrix (e^2/h): I_i = sum_j G_ij V_j with
        G_ij = -T_ij (i != j), G_ii = N_i - R_ii. Rows/columns sum to zero."""
        n = len(self.sys.leads)
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    G[i, j] = -self.transmission(i, j)
        for i in range(n):
            G[i, i] = -G[:, i].sum() + 0.0
        return G

    def four_terminal_resistance(self, current_in, current_out, v_plus, v_minus):
        """R = (V_plus - V_minus) / I with current I injected at ``current_in``
        and extracted at ``current_out`` (units of h/e^2)."""
        G = self.conductance_matrix()
        n = G.shape[0]
        ground = current_out
        keep = [k for k in range(n) if k != ground]
        I = np.zeros(n)
        I[current_in] = 1.0
        I[current_out] = -1.0
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
        """Scattering states injected from ``lead``: columns psi_n with
        sum_n psi_n psi_n^dag = G Gamma_lead G^dag (the lead spectral function)."""
        g, U = np.linalg.eigh(self.Gamma[lead])
        keep = g > tol * max(g.max(), 1e-300)
        return self._G_cols(lead) @ (U[:, keep] * np.sqrt(g[keep]))

    def density(self, lead, per_site=True):
        """Electron density injected from ``lead`` per unit energy:
        diag(G Gamma G^dag) / 2 pi (summed over internal dof if per_site)."""
        psi = self.wavefunctions(lead)
        rho = (np.abs(psi) ** 2).sum(1) / (2 * np.pi)
        if per_site:
            return rho.reshape(-1, self.sys.ns).sum(1)
        return rho

    def current(self, lead, operator=None):
        """Bond currents carried by the states injected from ``lead``,
        normalised so that the total current through any cross-section equals
        the transmission out of that lead (i.e. units of e/h per unit energy).
        Returns (bonds, J): bonds is an (nb, 2) array of site indices (a, b)
        and J the current from a to b. ``operator`` (ns x ns) gives e.g. spin
        currents (default: charge)."""
        psi = self.wavefunctions(lead)
        H = self.sys.hamiltonian().tocoo()
        ns = self.sys.ns
        Op = np.eye(ns) if operator is None else np.asarray(operator)
        a_site, b_site = H.row // ns, H.col // ns
        mask = a_site < b_site
        pairs = np.unique(np.stack([a_site[mask], b_site[mask]], 1), axis=0)
        Hd = self.sys.hamiltonian().tocsr()
        J = np.zeros(len(pairs))
        for n, (a, b) in enumerate(pairs):
            Hab = Hd[a * ns:(a + 1) * ns, b * ns:(b + 1) * ns].toarray()
            pa, pb = psi[a * ns:(a + 1) * ns], psi[b * ns:(b + 1) * ns]
            # J_{a->b} = sum_modes 2 Im( psi_b^dag  (O H_ba + H_ba O)/2  psi_a ) / (2 pi)
            Hba = Hab.conj().T
            M = 0.5 * (Op @ Hba + Hba @ Op)
            J[n] = np.sum(2 * np.imag(np.einsum("im,ij,jm->m", pb.conj(), M, pa)))
        return pairs, J


def smatrix(system, energy, **kw):
    """Solve the scattering problem at one energy."""
    return SMatrix(system, energy, **kw)


def conductance(system, energies, to_lead=1, from_lead=0, **kw):
    """Transmission T(E) between two leads for an array of energies."""
    energies = np.atleast_1d(energies)
    return np.array([SMatrix(system, E).transmission(to_lead, from_lead, **kw) for E in energies])


def andreev_conductance(system, energies, normal_lead=0, electron=None, hole=None):
    """Conductance of a normal-superconductor junction (BdG model), Blonder-
    Tinkham-Klapwijk form G = N_e - R_ee + R_he = T_(S<-e) + 2 R_he (e^2/h).

    ``electron`` / ``hole`` are block specs, e.g. dict(orbitals=['e']) and
    dict(orbitals=['h'])."""
    energies = np.atleast_1d(energies)
    out = []
    for E in energies:
        S = SMatrix(system, E)
        Ree = S.transmission(normal_lead, normal_lead, electron, electron)
        Rhe = S.transmission(normal_lead, normal_lead, hole, electron)
        Ne = S.num_modes(normal_lead, electron)
        out.append(Ne - Ree + Rhe)
    return np.array(out)
