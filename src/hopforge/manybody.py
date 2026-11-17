"""Strongly correlated electrons and spins by exact diagonalisation (ED).

Fermions (spin-1/2 electrons) on a cluster with N_up, N_dn conservation:

    H = sum_{ij,s} t_ij c^dag_is c_js                (from any hopforge model)
      + U  sum_a n_a,up n_a,dn                        (Hubbard)
      + V  sum_<ij> n_i n_j                           (extended Hubbard)
      + Kanamori multi-orbital terms per atom:        U, U' = U - 2J, J (Hund),
        spin-flip and pair-hopping (rotationally invariant for U' = U - 2J)

Spin-1/2 models with S^z_total conservation:

    H = sum_ij J_ij [ S^z_i S^z_j Delta + (S^+_i S^-_j + h.c.)/2 ] - h sum_i S^z_i

Ground states via Lanczos (scipy ARPACK), Green's functions via the Lanczos
continued fraction, and parallel parameter scans (``scan``).

>>> cl = hf.models.honeycomb().finite(2)            # or a PBC supercell (pbc=True)
>>> ed = hf.manybody.Hubbard(cl, U=4.0)
>>> E0, psi = ed.ground_state(n_up=4, n_dn=4)
>>> ed.charge_gap(4, 4), ed.spin_correlation(psi, 0, 1, 4, 4)
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .parallel import pmap


# =============================================================================
# bit-string bases
# =============================================================================
def _popcount(x):
    x = np.asarray(x, np.uint64)
    c = np.zeros(x.shape, np.int64)
    y = x.copy()
    while np.any(y):
        c += (y & np.uint64(1)).astype(np.int64)
        y = y >> np.uint64(1)
    return c


@lru_cache(maxsize=64)
def sector_basis(L, N):
    """Sorted array of L-bit integers with N bits set."""
    if N < 0 or N > L:
        return np.zeros(0, np.uint64)
    states = [sum(1 << i for i in c) for c in itertools.combinations(range(L), N)]
    return np.array(sorted(states), np.uint64)


def _bit(states, i):
    return ((states >> np.uint64(i)) & np.uint64(1)).astype(bool)


def _hop_matrix(L, N, t):
    """Sparse matrix of sum_ij t_ij c^dag_i c_j on the N-particle sector of one
    spin species (Jordan-Wigner signs with orbital order 0..L-1)."""
    basis = sector_basis(L, N)
    D = len(basis)
    rows, cols, vals = [], [], []
    t = np.asarray(t)
    idx = np.arange(D)
    for i in range(L):
        for j in range(L):
            tij = t[i, j]
            if tij == 0:
                continue
            if i == j:
                occ = _bit(basis, i)
                rows.append(idx[occ]); cols.append(idx[occ]); vals.append(np.full(occ.sum(), tij))
                continue
            ok = _bit(basis, j) & ~_bit(basis, i)
            if not ok.any():
                continue
            s = basis[ok]
            new = (s & ~np.uint64(1 << j)) | np.uint64(1 << i)
            lo, hi = min(i, j), max(i, j)
            mask = np.uint64(((1 << hi) - 1) & ~((1 << (lo + 1)) - 1))
            sign = 1 - 2 * (_popcount(s & mask) % 2)
            rows.append(np.searchsorted(basis, new)); cols.append(idx[ok]); vals.append(tij * sign)
    if not rows:
        return sp.csr_matrix((D, D), dtype=complex)
    return sp.csr_matrix((np.concatenate(vals).astype(complex), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(D, D))


def _cdag_c(L, N, i, j):
    """Single operator c^dag_i c_j on a one-species sector (sparse)."""
    t = np.zeros((L, L))
    t[i, j] = 1.0
    return _hop_matrix(L, N, t)


# =============================================================================
# fermionic Hubbard-type models
# =============================================================================
def _single_particle(model, pbc, twist):
    """(L x L) spin-diagonal hopping matrix of a cluster; requires no SOC."""
    if model.dim == 0:
        H = model.hamiltonian()
    else:
        if not pbc:
            raise ValueError("a periodic model needs pbc=True (or use model.finite(n))")
        k = np.zeros(model.dim) if twist is None else np.asarray(twist, float)
        H = model.hamiltonian(k, gauge="periodic")
    if model.spinful:
        up, dn = H[0::2, 0::2], H[1::2, 1::2]
        if np.abs(H[0::2, 1::2]).max() > 1e-12 or not np.allclose(up, dn):
            raise ValueError("ED with N_up/N_dn conservation needs spin-diagonal, spin-independent hopping "
                             "(use the spinless template)")
        H = up
    return np.asarray(H)


@dataclass
class EDResult:
    energies: np.ndarray
    states: np.ndarray
    sector: tuple


class Hubbard:
    """Hubbard / extended Hubbard / multi-orbital Kanamori model on a cluster.

    model  : spinless hopforge Model of the cluster: 0D (``m.finite(...)``) or a
             periodic supercell with ``pbc=True`` (uses H(k = twist)).
    U      : Hubbard U (scalar, or per-orbital array)
    V      : nearest-neighbour density interaction (applied to the bonds of ``V_shell``)
    atoms  : list of orbital-index groups forming multi-orbital atoms for the
             Kanamori terms with ``Up`` (U') and ``J`` (Hund); default Up = U - 2J
    mu     : chemical potential (shifts every orbital)
    """

    def __init__(self, model, U=0.0, V=0.0, V_shell=1, atoms=None, J=0.0, Up=None, mu=0.0,
                 pbc=False, twist=None):
        self.model = model
        self.t = _single_particle(model, pbc, twist) - mu * np.eye(model.norb)
        self.L = self.t.shape[0]
        self.U = np.broadcast_to(np.asarray(U, float), (self.L,)).copy()
        self.V = V
        self.Vbonds = []
        if V:
            if model.dim == 0:
                bonds = model.neighbors(V_shell)
            else:
                bonds = model.neighbors(V_shell)
            self.Vbonds = [(i, j) for i, j, R, d in bonds if i != j]
        self.atoms = atoms or []
        self.J = J
        self.Up = (np.mean(self.U) - 2 * J) if Up is None else Up
        self._cache = {}

    # ---------------------------------------------------------------
    def dimension(self, n_up, n_dn):
        return len(sector_basis(self.L, n_up)) * len(sector_basis(self.L, n_dn))

    def hamiltonian(self, n_up, n_dn):
        """Sparse many-body Hamiltonian in the (N_up, N_dn) sector.
        Basis index = i_up * D_dn + i_dn."""
        key = (n_up, n_dn)
        if key in self._cache:
            return self._cache[key]
        L = self.L
        bu, bd = sector_basis(L, n_up), sector_basis(L, n_dn)
        Du, Dd = len(bu), len(bd)
        Tu, Td = _hop_matrix(L, n_up, self.t), _hop_matrix(L, n_dn, self.t)
        H = sp.kron(Tu, sp.identity(Dd), format="csr") + sp.kron(sp.identity(Du), Td, format="csr")
        # diagonal interactions
        nu = np.array([_bit(bu, i) for i in range(L)], float)    # (L, Du)
        nd = np.array([_bit(bd, i) for i in range(L)], float)    # (L, Dd)
        diag = np.einsum("i,iu,id->ud", self.U, nu, nd)
        if self.V:
            for i, j in self.Vbonds:
                ni = nu[i][:, None] + nd[i][None, :]
                nj = nu[j][:, None] + nd[j][None, :]
                diag += self.V * ni * nj
        offd = []
        for atom in self.atoms:
            for a in atom:
                for b in atom:
                    if a == b:
                        continue
                    # U' n_a,up n_b,dn
                    diag += self.Up * nu[a][:, None] * nd[b][None, :]
                    if a < b:
                        # (U' - J) n_a,s n_b,s for equal spins
                        diag += (self.Up - self.J) * (nu[a] * nu[b])[:, None]
                        diag += (self.Up - self.J) * (nd[a] * nd[b])[None, :]
                    if self.J:
                        # spin flip  -J c+a,up c a,dn c+b,dn c b,up = +J (c+a,up c b,up)(c+b,dn c a,dn)
                        offd.append(self.J * sp.kron(_cdag_c(L, n_up, a, b), _cdag_c(L, n_dn, b, a), format="csr"))
                        # pair hopping  J c+a,up c+a,dn c b,dn c b,up = J (c+a,up c b,up)(c+a,dn c b,dn)
                        offd.append(self.J * sp.kron(_cdag_c(L, n_up, a, b), _cdag_c(L, n_dn, a, b), format="csr"))
        H = H + sp.diags(diag.ravel().astype(complex))
        for M in offd:
            H = H + M
        H = H.tocsr()
        self._cache[key] = H
        return H

    def eigs(self, n_up, n_dn, k=1, dense_limit=1500):
        """Lowest ``k`` eigenpairs in a sector."""
        H = self.hamiltonian(n_up, n_dn)
        D = H.shape[0]
        if D == 0:
            return EDResult(np.array([np.inf]), np.zeros((0, 1)), (n_up, n_dn))
        if D <= dense_limit:
            w, v = np.linalg.eigh(H.toarray())
            return EDResult(w[:k], v[:, :k], (n_up, n_dn))
        w, v = spla.eigsh(H, k=k, which="SA", tol=1e-10)
        o = np.argsort(w)
        return EDResult(w[o], v[:, o], (n_up, n_dn))

    def ground_state(self, n_up, n_dn):
        r = self.eigs(n_up, n_dn, 1)
        return float(r.energies[0]), r.states[:, 0]

    def ground_state_all_sectors(self, n_total, n_jobs=None):
        """Lowest energy over all S_z sectors with fixed total N: (E0, (n_up, n_dn))."""
        secs = [(nu, n_total - nu) for nu in range(0, n_total + 1)
                if 0 <= n_total - nu <= self.L and nu <= self.L and nu >= n_total - nu]
        E = pmap(lambda s: self.ground_state(*s)[0], secs, n_jobs)
        i = int(np.argmin(E))
        return E[i], secs[i]

    # ---------------------------------------------------------------
    # observables
    # ---------------------------------------------------------------
    def _occ(self, n_up, n_dn):
        L = self.L
        bu, bd = sector_basis(L, n_up), sector_basis(L, n_dn)
        nu = np.array([_bit(bu, i) for i in range(L)], float)
        nd = np.array([_bit(bd, i) for i in range(L)], float)
        return nu, nd

    def density(self, psi, n_up, n_dn):
        """<n_i,up>, <n_i,dn> per orbital."""
        nu, nd = self._occ(n_up, n_dn)
        P = np.abs(psi.reshape(nu.shape[1], nd.shape[1])) ** 2
        return P.sum(1) @ nu.T, P.sum(0) @ nd.T

    def double_occupancy(self, psi, n_up, n_dn):
        nu, nd = self._occ(n_up, n_dn)
        P = np.abs(psi.reshape(nu.shape[1], nd.shape[1])) ** 2
        return np.einsum("ud,iu,id->i", P, nu, nd)

    def szsz(self, psi, n_up, n_dn):
        """Matrix <S^z_i S^z_j>."""
        nu, nd = self._occ(n_up, n_dn)
        P = np.abs(psi.reshape(nu.shape[1], nd.shape[1])) ** 2
        Sz = 0.5 * (nu[:, :, None] - nd[:, None, :])           # (L, Du, Dd)
        return np.einsum("ud,iud,jud->ij", P, Sz, Sz)

    def spin_correlation(self, psi, i, j, n_up, n_dn):
        """<S_i . S_j> (full SU(2) correlation incl. spin-flip parts)."""
        L = self.L
        Du, Dd = len(sector_basis(L, n_up)), len(sector_basis(L, n_dn))
        zz = self.szsz(psi, n_up, n_dn)[i, j]
        if i == j:
            # S^2_i = 3/4 (n - 2 n_up n_dn)
            nu, nd = self._occ(n_up, n_dn)
            P = np.abs(psi.reshape(Du, Dd)) ** 2
            return float(0.75 * np.einsum("ud,ud->", P, nu[i][:, None] + nd[i][None, :] - 2 * nu[i][:, None] * nd[i][None, :]))
        # S+_i S-_j = c+_i,up c_i,dn c+_j,dn c_j,up = -(c+_i,up c_j,up)(c+_j,dn c_i,dn)
        op = -sp.kron(_cdag_c(L, n_up, i, j), _cdag_c(L, n_dn, j, i), format="csr")
        pm = np.vdot(psi, op @ psi)
        return float(zz + np.real(pm))           # S+S- + S-S+ = 2 Re <S+S->, times 1/2

    def total_spin(self, psi, n_up, n_dn):
        """S from <S^2> = S(S+1)."""
        S2 = sum(self.spin_correlation(psi, i, j, n_up, n_dn) for i in range(self.L) for j in range(self.L))
        return 0.5 * (-1 + np.sqrt(1 + 4 * max(S2, 0)))

    def charge_gap(self, n_up, n_dn):
        """E(N+1) + E(N-1) - 2 E(N) (adding/removing an up electron)."""
        E0 = self.ground_state(n_up, n_dn)[0]
        Ep = self.ground_state(n_up + 1, n_dn)[0]
        Em = self.ground_state(n_up - 1, n_dn)[0]
        return Ep + Em - 2 * E0

    def spin_gap(self, n_up, n_dn):
        """E(S_z + 1) - E(S_z) at fixed N."""
        return self.ground_state(n_up + 1, n_dn - 1)[0] - self.ground_state(n_up, n_dn)[0]

    def structure_factor(self, psi, n_up, n_dn, qs):
        """Spin structure factor S(q) = (1/L) sum_ij e^{iq(r_i - r_j)} <S^z_i S^z_j>
        (Cartesian q)."""
        zz = self.szsz(psi, n_up, n_dn)
        P = self.model.lat.positions
        out = []
        for q in np.atleast_2d(qs):
            ph = np.exp(1j * P[:, :len(q)] @ q)
            out.append(np.real(ph.conj() @ zz @ ph) / self.L)
        return np.array(out)

    # ---------------------------------------------------------------
    # Green's function (Lanczos continued fraction)
    # ---------------------------------------------------------------
    def _cdag_up(self, i, n_up, n_dn):
        """Sparse c^dag_{i,up}: sector (n_up, n_dn) -> (n_up + 1, n_dn)."""
        L = self.L
        b0, b1 = sector_basis(L, n_up), sector_basis(L, n_up + 1)
        ok = ~_bit(b0, i)
        s = b0[ok]
        new = s | np.uint64(1 << i)
        sign = 1 - 2 * (_popcount(s & np.uint64((1 << i) - 1)) % 2)
        C = sp.csr_matrix((sign.astype(complex), (np.searchsorted(b1, new), np.flatnonzero(ok))),
                          shape=(len(b1), len(b0)))
        Dd = len(sector_basis(L, n_dn))
        return sp.kron(C, sp.identity(Dd), format="csr")

    @staticmethod
    def _lanczos_cf(H, v0, E0, omegas, eta, sign, nsteps=200):
        nrm = np.linalg.norm(v0)
        if nrm < 1e-14:
            return np.zeros(len(omegas), complex)
        v = v0 / nrm
        a, b = [], []
        vprev = np.zeros_like(v)
        beta = 0.0
        for _ in range(min(nsteps, H.shape[0])):
            w = H @ v - beta * vprev
            alpha = np.real(np.vdot(v, w))
            w = w - alpha * v
            a.append(alpha)
            beta = np.linalg.norm(w)
            if beta < 1e-12:
                break
            b.append(beta)
            vprev, v = v, w / beta
        z = omegas + 1j * eta
        g = np.zeros(len(z), complex)
        for n in range(len(a) - 1, -1, -1):
            bn2 = b[n] ** 2 if n < len(b) else 0.0
            if sign > 0:   # particle: z + E0 - H
                g = 1.0 / (z + E0 - a[n] - (bn2 * g if n < len(a) - 1 else 0))
            else:          # hole: z - E0 + H
                g = 1.0 / (z - E0 + a[n] - (bn2 * g if n < len(a) - 1 else 0))
        return nrm ** 2 * g

    def greens_function(self, i, n_up, n_dn, omegas, eta=0.05, nsteps=200):
        """Local retarded G_ii,up(omega) at T = 0 (particle + hole parts).
        Spectral function A(omega) = -Im G / pi."""
        omegas = np.asarray(omegas, float)
        E0, psi = self.ground_state(n_up, n_dn)
        G = np.zeros(len(omegas), complex)
        if n_up < self.L:
            Cd = self._cdag_up(i, n_up, n_dn)
            G += self._lanczos_cf(self.hamiltonian(n_up + 1, n_dn), Cd @ psi, E0, omegas, eta, +1, nsteps)
        if n_up > 0:
            Cd = self._cdag_up(i, n_up - 1, n_dn)
            G += self._lanczos_cf(self.hamiltonian(n_up - 1, n_dn), Cd.conj().T @ psi, E0, omegas, eta, -1, nsteps)
        return G


# =============================================================================
# spin-1/2 models
# =============================================================================
class SpinModel:
    """Spin-1/2 XXZ / Heisenberg model with arbitrary couplings.

    bonds : list of (i, j, J) (each bond once), or build from a hopforge model
            with :meth:`from_model` (neighbour shells -> J1, J2, ...)
    Delta : XXZ anisotropy (1 = Heisenberg); h : field along z
    """

    def __init__(self, L, bonds, Delta=1.0, h=0.0, positions=None):
        self.L = L
        self.bonds = [(int(i), int(j), float(J)) for i, j, J in bonds]
        self.Delta, self.h = Delta, h
        self.positions = positions
        self._cache = {}

    @classmethod
    def from_model(cls, model, J=(1.0,), pbc=True, **kw):
        """Bonds from the neighbour shells of a cluster model (periodic
        supercell -> periodic boundary conditions)."""
        bonds = {}
        for shell, Js in enumerate(J, start=1):
            if not Js:
                continue
            for i, j, R, d in model.neighbors(shell):
                if model.dim and not pbc and np.any(R != 0):
                    continue
                if i == j:
                    continue
                key = (min(i, j), max(i, j))
                bonds[key] = bonds.get(key, 0.0) + Js
        return cls(model.norb, [(i, j, Jv) for (i, j), Jv in bonds.items()],
                   positions=model.lat.positions, **kw)

    @classmethod
    def chain(cls, L, J1=1.0, J2=0.0, pbc=True, **kw):
        bonds = [(i, (i + 1) % L, J1) for i in range(L if pbc else L - 1)]
        if J2:
            bonds += [(i, (i + 2) % L, J2) for i in range(L if pbc else L - 2)]
        return cls(L, bonds, positions=np.arange(L, dtype=float)[:, None], **kw)

    def hamiltonian(self, n_up):
        """Sector with n_up up spins (S^z = n_up - L/2)."""
        if n_up in self._cache:
            return self._cache[n_up]
        L = self.L
        basis = sector_basis(L, n_up)
        D = len(basis)
        idx = np.arange(D)
        sz = np.array([_bit(basis, i) for i in range(L)], float) - 0.5      # (L, D)
        diag = -self.h * sz.sum(0)
        rows, cols, vals = [], [], []
        for i, j, J in self.bonds:
            diag += J * self.Delta * sz[i] * sz[j]
            anti = _bit(basis, i) ^ _bit(basis, j)
            s = basis[anti]
            flipped = s ^ np.uint64((1 << i) | (1 << j))
            rows.append(np.searchsorted(basis, flipped)); cols.append(idx[anti]); vals.append(np.full(anti.sum(), 0.5 * J))
        H = sp.csr_matrix((np.concatenate(vals) if vals else [], (np.concatenate(rows) if rows else [],
                           np.concatenate(cols) if cols else [])), shape=(D, D)) + sp.diags(diag)
        H = H.tocsr()
        self._cache[n_up] = H
        return H

    def ground_state(self, n_up=None, k=1):
        n_up = self.L // 2 if n_up is None else n_up
        H = self.hamiltonian(n_up)
        if H.shape[0] <= 1500:
            w, v = np.linalg.eigh(H.toarray())
        else:
            w, v = spla.eigsh(H, k=max(k, 1), which="SA", tol=1e-12)
            o = np.argsort(w)
            w, v = w[o], v[:, o]
        return (float(w[0]), v[:, 0]) if k == 1 else (w[:k], v[:, :k])

    def lowest_per_sector(self, n_jobs=None):
        """Lowest energy in every S^z sector: dict {S^z: E}."""
        ns = list(range(self.L + 1))
        E = pmap(lambda n: self.ground_state(n)[0], ns, n_jobs)
        return {n - self.L / 2: e for n, e in zip(ns, E)}

    def szsz(self, psi, n_up=None):
        n_up = self.L // 2 if n_up is None else n_up
        basis = sector_basis(self.L, n_up)
        sz = np.array([_bit(basis, i) for i in range(self.L)], float) - 0.5
        P = np.abs(psi) ** 2
        return np.einsum("d,id,jd->ij", P, sz, sz)

    def structure_factor(self, psi, qs, n_up=None):
        zz = self.szsz(psi, n_up)
        P = self.positions
        out = []
        for q in np.atleast_2d(qs):
            ph = np.exp(1j * P[:, :len(q)] @ q)
            out.append(np.real(ph.conj() @ zz @ ph) / self.L)
        return np.array(out)


# =============================================================================
# parameter scans
# =============================================================================
def scan(builder, params_list, measure, n_jobs=None):
    """Solve a correlated model for many parameter sets in parallel.

    builder(**params) -> solver object (Hubbard / SpinModel)
    measure(solver, params) -> any result (energy, gap, correlation, ...)
    """
    return pmap(lambda p: measure(builder(**p), p), list(params_list), n_jobs)


def grid(**axes):
    """Cartesian product of parameter axes: grid(U=[1, 2], J=[0, 0.5])."""
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*[np.atleast_1d(axes[k]) for k in keys])]


# =============================================================================
# spinless fermions (complex hoppings, density-density interactions)
# =============================================================================
class Fermions:
    """Interacting spinless fermions on a cluster (e.g. fractional Chern
    insulators, t-V models):

        H = sum_ij t_ij c^dag_i c_j + sum_{shells} V_s sum_<ij>_s n_i n_j

    model : spinless hopforge Model (0D, or periodic supercell with pbc=True and
            optional twisted boundary ``twist`` in reduced units)
    V     : list of interaction strengths per neighbour shell, e.g. [1.0, 0.5]
    """

    def __init__(self, model, V=(0.0,), pbc=True, twist=None, mu=0.0, bonds=None):
        if model.spinful:
            raise ValueError("Fermions is for spinless models")
        self.model = model
        self.t = _single_particle(model, pbc if model.dim else False, twist) - mu * np.eye(model.norb)
        self.L = self.t.shape[0]
        self.Vbonds = []
        if bonds is not None:
            self.Vbonds = list(bonds)
        else:
            for shell, Vs in enumerate(np.atleast_1d(V), start=1):
                if not Vs:
                    continue
                seen = set()
                for i, j, R, d in model.neighbors(shell):
                    key = (min(i, j), max(i, j))
                    if i == j or key in seen:
                        continue
                    seen.add(key)
                    self.Vbonds.append((key[0], key[1], float(Vs)))
        self._cache = {}

    def hamiltonian(self, N):
        if N in self._cache:
            return self._cache[N]
        basis = sector_basis(self.L, N)
        H = _hop_matrix(self.L, N, self.t)
        occ = np.array([_bit(basis, i) for i in range(self.L)], float)
        diag = np.zeros(len(basis))
        for i, j, Vij in self.Vbonds:
            diag += Vij * occ[i] * occ[j]
        H = (H + sp.diags(diag.astype(complex))).tocsr()
        self._cache[N] = H
        return H

    def spectrum(self, N, k=6):
        """Lowest ``k`` eigenvalues in the N-particle sector."""
        H = self.hamiltonian(N)
        if H.shape[0] <= 2000:
            return np.linalg.eigvalsh(H.toarray())[:k]
        return np.sort(spla.eigsh(H, k=k, which="SA", tol=1e-10, return_eigenvectors=False).real)

    def ground_state(self, N):
        H = self.hamiltonian(N)
        if H.shape[0] <= 2000:
            w, v = np.linalg.eigh(H.toarray())
        else:
            w, v = spla.eigsh(H, k=1, which="SA", tol=1e-10)
        return float(w[0]), v[:, 0]

    def density(self, psi, N):
        basis = sector_basis(self.L, N)
        occ = np.array([_bit(basis, i) for i in range(self.L)], float)
        return occ @ (np.abs(psi) ** 2)


# =============================================================================
# general spin-1/2 Hamiltonians from Pauli strings (no symmetry assumed)
# =============================================================================
_PAULI = {"I": sp.identity(2, format="csr", dtype=complex),
          "X": sp.csr_matrix(np.array([[0, 1], [1, 0]], complex)),
          "Y": sp.csr_matrix(np.array([[0, -1j], [1j, 0]], complex)),
          "Z": sp.csr_matrix(np.array([[1, 0], [0, -1]], complex))}


class PauliModel:
    """H = sum_k c_k P_k with P_k a product of Pauli matrices, e.g.
    ``[(J, {0: 'Z', 1: 'Z'}), (h, {0: 'X'})]``. Full 2^L Hilbert space
    (L <= ~22). Covers transverse-field Ising, Kitaev honeycomb (bond-dependent
    XX/YY/ZZ), DM interactions, XYZ, ring exchange ..."""

    def __init__(self, L, terms):
        self.L = L
        self.terms = [(complex(c), dict(ops)) for c, ops in terms]
        self._H = None

    def _string(self, ops):
        # qubit 0 is the most significant factor
        M = sp.identity(1, format="csr", dtype=complex)
        for q in range(self.L):
            M = sp.kron(M, _PAULI[ops.get(q, "I")], format="csr")
        return M

    def hamiltonian(self):
        if self._H is None:
            D = 2 ** self.L
            H = sp.csr_matrix((D, D), dtype=complex)
            for c, ops in self.terms:
                H = H + c * self._string(ops)
            self._H = H.tocsr()
        return self._H

    def ground_state(self, k=1):
        H = self.hamiltonian()
        if H.shape[0] <= 256:
            w, v = np.linalg.eigh(H.toarray())
        else:
            w, v = spla.eigsh(H, k=max(k, 2), which="SA", tol=1e-12)
            o = np.argsort(w)
            w, v = w[o], v[:, o]
        return (float(w[0]), v[:, 0]) if k == 1 else (w[:k], v[:, :k])

    def expectation(self, psi, ops):
        return complex(np.vdot(psi, self._string(ops) @ psi))

    @classmethod
    def transverse_field_ising(cls, L, J=1.0, h=1.0, pbc=True):
        """H = -J sum Z_i Z_{i+1} - h sum X_i (Sachdev, Quantum Phase
        Transitions; critical point h = J)."""
        terms = [(-J, {i: "Z", (i + 1) % L: "Z"}) for i in range(L if pbc else L - 1)]
        terms += [(-h, {i: "X"}) for i in range(L)]
        return cls(L, terms)

    @classmethod
    def kitaev_honeycomb(cls, model_cluster, Jx=1.0, Jy=1.0, Jz=1.0):
        """Kitaev's honeycomb model on a honeycomb cluster (Ann. Phys. 321, 2
        (2006)): bonds classified by direction into x, y, z types,
        H = -sum J_a sigma^a_i sigma^a_j. ``model_cluster`` is a honeycomb
        supercell (periodic) or flake."""
        terms = []
        for i, j, R, d in model_cluster.neighbors(1):
            ang = np.degrees(np.arctan2(d[1], d[0])) % 180
            a = "Z" if abs(ang - 90) < 1 else ("X" if abs(ang - 30) < 1 else "Y")
            Ja = {"X": Jx, "Y": Jy, "Z": Jz}[a]
            terms.append((-Ja, {i: a, j: a}))
        return cls(model_cluster.norb, terms)


def tfim_exact_energy(L, J=1.0, h=1.0):
    """Exact ground-state energy of the periodic transverse-field Ising chain
    (Pfeuty, Ann. Phys. 57, 79 (1970)): free fermions in the even-parity
    (antiperiodic, Neveu-Schwarz) sector, E0 = -sum_k eps_k / 2,
    eps_k = 2 sqrt(J^2 + h^2 - 2 J h cos k), k = +-pi(2n+1)/L."""
    k = np.pi * (2 * np.arange(L) + 1) / L
    eps = 2 * np.sqrt(J ** 2 + h ** 2 - 2 * J * h * np.cos(k))
    return -0.5 * eps.sum()


# =============================================================================
# Sachdev-Ye-Kitaev model
# =============================================================================
def majoranas(N):
    """Jordan-Wigner Majorana operators chi_1..chi_N (N even), {chi_a, chi_b} = delta_ab
    (normalisation chi^2 = 1/2), as sparse 2^(N/2) matrices."""
    n = N // 2
    X, Y, Z, I = _PAULI["X"], _PAULI["Y"], _PAULI["Z"], _PAULI["I"]
    ops = []
    for j in range(n):
        for P in (X, Y):
            M = sp.identity(1, format="csr", dtype=complex)
            for q in range(n):
                M = sp.kron(M, Z if q < j else (P if q == j else I), format="csr")
            ops.append(M / np.sqrt(2))
    return ops


class SYK:
    """Majorana Sachdev-Ye-Kitaev model (Sachdev & Ye, PRL 70, 3339 (1993);
    Kitaev 2015; Maldacena & Stanford, PRD 94, 106002 (2016)):

        H = sum_{i<j<k<l} J_ijkl chi_i chi_j chi_k chi_l,
        <J_ijkl^2> = 3! J^2 / N^3,   {chi_a, chi_b} = delta_ab

    Large-N ground-state energy (q = 4): E0 / N = -0.0406 J.
    Built with signed-permutation algebra (every Majorana string flips a fixed
    bit mask), so N = 28-30 is fast.
    """

    def __init__(self, N, J=1.0, q=4, seed=0):
        import math
        if N % 2 or q % 2:
            raise ValueError("N and q must be even")
        self.N, self.J, self.q = N, J, q
        n = N // 2
        D = 2 ** n
        states = np.arange(D, dtype=np.int64)
        rng = np.random.default_rng(seed)
        var = math.factorial(q - 1) * J ** 2 / N ** (q - 1)
        # Majorana a acts on qubit j = a // 2: X_j (even a) or Y_j (odd a) with a Z string
        lower = [(1 << j) - 1 for j in range(n)]
        acc = {}
        for idx in itertools.combinations(range(N), q):
            Jc = rng.normal(0, np.sqrt(var))
            cur = states.copy()
            ph = np.ones(D, complex)
            for a in reversed(idx):            # rightmost operator acts first
                j = a // 2
                bit = (cur >> j) & 1
                ph *= 1 - 2 * (_popcount((cur & lower[j]).astype(np.uint64)) % 2)
                if a % 2:
                    ph *= 1j * (1 - 2 * bit)   # Y|0> = i|1>, Y|1> = -i|0>
                cur = cur ^ (1 << j)
            mask = int(cur[0] ^ states[0])
            acc[mask] = acc.get(mask, 0) + Jc * ph * (0.5 ** (q / 2))
        rows, cols, vals = [], [], []
        for mask, v in acc.items():
            rows.append(states ^ mask); cols.append(states); vals.append(v)
        H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(D, D))
        if q % 4 == 2:
            H = 1j * H
        self.H = ((H + H.conj().T) / 2).tocsr()
        self.parity = 1 - 2 * (_popcount(states.astype(np.uint64)) % 2)

    def spectrum(self, sector=None):
        """All eigenvalues (dense), optionally in a parity sector (+1 / -1)."""
        H = self.H
        if sector is not None:
            idx = np.flatnonzero(self.parity == sector)
            H = H[idx][:, idx]
        return np.linalg.eigvalsh(H.toarray())

    def ground_energy(self):
        if self.H.shape[0] <= 1024:
            return float(self.spectrum()[0])
        return float(spla.eigsh(self.H, k=1, which="SA", tol=1e-10, return_eigenvectors=False)[0])


def level_spacing_ratio(E):
    """Mean adjacent-gap ratio <r> (Oganesyan-Huse): 0.386 Poisson, 0.5307 GOE,
    0.5996 GUE, 0.6744 GSE (Atas et al., PRL 110, 084101 (2013))."""
    E = np.sort(np.asarray(E))
    E = E[len(E) // 10: -len(E) // 10 or None]       # drop spectral edges
    s = np.diff(E)
    s = s[s > 1e-12 * np.abs(E).max()] if np.any(s > 0) else s
    r = np.minimum(s[1:], s[:-1]) / np.maximum(s[1:], s[:-1])
    return float(np.mean(r))


def syk_level_statistics(N, realizations=20, seed=0, n_jobs=None):
    """<r> of the Majorana SYK model in one parity sector; random-matrix class
    set by N mod 8 (You, Ludwig & Xu, PRB 95, 115150 (2017); Garcia-Garcia &
    Verbaarschot, PRD 94, 126010 (2016)): 0 -> GOE, 2, 6 -> GUE, 4 -> GSE
    (GSE spectra are Kramers degenerate: degeneracies are removed)."""
    def one(s):
        E = SYK(N, seed=seed + s).spectrum(sector=+1)
        if N % 8 == 4:
            E = np.unique(np.round(E, 10))
        return level_spacing_ratio(E)
    return float(np.mean(pmap(one, range(realizations), n_jobs)))


# =============================================================================
# Bose-Hubbard model: site-decoupled mean field (Fisher et al. 1989)
# =============================================================================
def bose_hubbard_meanfield(U, zt, mu, nmax=8, tol=1e-10, maxiter=2000, psi0=0.3):
    """Self-consistent site-decoupled mean field of the Bose-Hubbard model
    H = -t sum b^dag b + U/2 sum n(n-1) - mu sum n (z neighbours):
    H_MF = -zt (psi* b + psi b^dag) + zt |psi|^2 + U/2 n(n-1) - mu n.
    Returns (psi, <n>, E). psi = 0 in a Mott lobe, psi > 0 in the superfluid."""
    n = np.arange(nmax + 1)
    b = np.diag(np.sqrt(n[1:]), 1)
    psi = psi0
    for _ in range(maxiter):
        H = -zt * psi * (b + b.T) + zt * psi ** 2 + np.diag(0.5 * U * n * (n - 1) - mu * n)
        w, v = np.linalg.eigh(H)
        g = v[:, 0]
        new = float(g @ b @ g)
        if abs(new - psi) < tol:
            psi = new
            break
        psi = 0.5 * psi + 0.5 * new
    return abs(psi), float(g @ (n * g)), float(w[0])


def mott_lobe_boundary(n, U=1.0, mu=None):
    """Analytic mean-field Mott-lobe boundary for filling n:
    zt_c(mu) = (n U - mu)(mu - (n-1) U) / (mu + U); lobe tip at
    (zt/U)_c = 1 / ((2n+1) + 2 sqrt(n(n+1)))  -> n = 1: (U/zt)_c = 3 + 2 sqrt(2)."""
    if mu is None:
        return 1.0 / ((2 * n + 1) + 2 * np.sqrt(n * (n + 1)))
    mu = np.asarray(mu, float)
    return (n * U - mu) * (mu - (n - 1) * U) / (mu + U)
