"""Finite scattering regions with semi-infinite leads (Kwant-style builder).

A :class:`System` is cut out of any periodic hopforge :class:`~hopforge.Model`
(the *template*) by a shape function, flood-filled from a starting point. Leads
are semi-infinite continuations of a template along a lattice vector.

>>> sq = hf.models.square()
>>> sys = hf.transport.System(sq, lambda r: 0 <= r[0] < 20 and 0 <= r[1] < 10)
>>> sys.attach_lead((-1, 0)); sys.attach_lead((1, 0))
>>> sys.transmission(1, 0, energy=-3.0)      # = number of open channels

Sites are labelled (orbital, R) with R the integer cell vector. The basis of
the region is site-major with the spin index inner (as in Model).
"""
from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..model import Model


def _neighbor_table(model: Model):
    """For every orbital i: list of (j, R, T) with <i,0|H|j,R> = T (both
    orientations of every stored hopping)."""
    nb = [[] for _ in range(model.norb)]
    for t in model._terms:
        nb[t.i].append((t.j, tuple(t.R), t.T))
        nb[t.j].append((t.i, tuple(-t.R), t.T.conj().T))
    return nb


def _pos(model, site):
    o, R = site
    return model.lat.positions[o] + (np.array(R) @ model.lat.vectors if model.dim else 0.0)


@dataclass
class Lead:
    """A semi-infinite lead. Cell 0 touches the scattering region; cell j+1 is
    cell j translated by ``vector``."""
    vector: tuple
    model: Model
    sites0: list                      # sites (orbital, R) of cell 0
    H0: np.ndarray                    # intra-cell Hamiltonian
    V: np.ndarray                     # coupling cell j -> cell j+1
    W: sp.csr_matrix                  # coupling region -> cell 0  (n_region x n_lead)
    iface: np.ndarray = None          # region basis indices coupled to the lead
    Wi: np.ndarray = None             # dense W rows on iface
    name: str = ""
    _cache: dict = field(default_factory=dict)

    @property
    def nb(self):
        return self.H0.shape[0]

    # -------------------------------------------------------------------
    def surface_green(self, energy, eta=1e-10, maxiter=200, tol=1e-13):
        """Retarded surface Green's function of the isolated semi-infinite lead
        (Lopez Sancho - Rubio decimation)."""
        key = ("g", complex(energy), eta)
        if key in self._cache:
            return self._cache[key]
        z = energy + 1j * eta
        I = np.eye(self.nb)
        es, e = self.H0.astype(complex).copy(), self.H0.astype(complex).copy()
        a, b = self.V.astype(complex).copy(), self.V.conj().T.astype(complex).copy()
        for _ in range(maxiter):
            g = np.linalg.solve(z * I - e, np.hstack([a, b]))
            ga, gb = g[:, :self.nb], g[:, self.nb:]
            agb, bga = a @ gb, b @ ga
            es = es + agb
            e = e + agb + bga
            a, b = a @ ga, b @ gb
            if np.abs(a).max() < tol and np.abs(b).max() < tol:
                break
        else:
            warnings.warn("surface Green's function did not converge; increase eta")
        gs = np.linalg.inv(z * I - es)
        self._cache[key] = gs
        return gs

    def self_energy(self, energy, **kw):
        """Self-energy on the interface basis states (dense, len(iface)^2)."""
        gs = self.surface_green(energy, **kw)
        return self.Wi @ gs @ self.Wi.conj().T

    def bands(self, k):
        """Lead band structure E_n(k), k in units of 1/period (k in [-pi, pi])."""
        k = np.atleast_1d(k)
        H = self.H0[None] + self.V[None] * np.exp(1j * k)[:, None, None] \
            + self.V.conj().T[None] * np.exp(-1j * k)[:, None, None]
        return np.linalg.eigvalsh(H)

    def modes(self, energy, projector=None, tol=1e-6):
        """Propagating modes at ``energy`` from the generalised eigenproblem
        (E - H0 - lambda V - V^dag / lambda) phi = 0.

        Returns (k, velocity, phi) for the propagating modes (|lambda| = 1);
        within degenerate subspaces the velocity matrix is diagonalised, so
        band crossings are handled exactly."""
        import scipy.linalg as sla
        H0, V = self.H0, self.V
        if projector is not None:
            idx = np.flatnonzero(np.diag(projector) > 0.5)
            H0, V = H0[np.ix_(idx, idx)], V[np.ix_(idx, idx)]
        n = H0.shape[0]
        I, Z = np.eye(n), np.zeros((n, n))
        A = np.block([[Z, I], [-V.conj().T, energy * I - H0]])
        B = np.block([[I, Z], [Z, V]])
        lam, vec = sla.eig(A, B)
        ok = np.isfinite(lam) & (np.abs(np.abs(lam) - 1) < tol)
        lam, vec = lam[ok], vec[:n, ok]
        ks, vs, phis = [], [], []
        used = np.zeros(len(lam), bool)
        for a in range(len(lam)):
            if used[a]:
                continue
            grp = np.flatnonzero(~used & (np.abs(lam - lam[a]) < 1e-5))
            used[grp] = True
            k = float(np.angle(lam[a]))
            Hk = H0 + V * np.exp(1j * k) + V.conj().T * np.exp(-1j * k)
            Phi, _ = np.linalg.qr(vec[:, grp])
            # refine: project onto the exact eigenspace of H(k) at this energy
            w, U = np.linalg.eigh(Hk)
            sub = U[:, np.abs(w - energy) < 1e-5 * max(1, np.abs(w).max())]
            if sub.shape[1] >= Phi.shape[1]:
                Phi = sub
            dH = 1j * (V * np.exp(1j * k) - V.conj().T * np.exp(-1j * k))
            vel, Uv = np.linalg.eigh(Phi.conj().T @ dH @ Phi)
            for m in range(len(vel)):
                ks.append(k); vs.append(vel[m]); phis.append(Phi @ Uv[:, m])
        order = np.argsort(ks)
        return (np.array(ks)[order], np.array(vs)[order],
                np.array(phis).T[:, order] if phis else np.zeros((n, 0)))

    def num_modes(self, energy, projector=None):
        """Number of propagating channels (right movers) at ``energy``; with
        ``projector`` (on the lead cell basis) only within that conserved block."""
        k, v, _ = self.modes(energy, projector)
        if len(k) == 0:
            return 0
        # modes with identical k from different degenerate subspaces are merged above
        vscale = max(np.abs(self.V).max(), 1e-12)
        return int((v > 1e-6 * vscale).sum())


class System:
    """Finite scattering region built from a template model and a shape.

    Parameters
    ----------
    model : periodic hopforge Model (any dimension >= 1, or a 0D model)
    shape : callable ``shape(pos) -> bool`` on Cartesian positions
    start : a position inside the shape (flood fill seed); default: first
            orbital of the home cell
    onsite : optional callable ``onsite(site_pos, orbital_name) -> scalar or
             (ns x ns)`` added to the template on-site term (gates, disorder,
             superconducting pairing profiles, ...)
    peierls : optional callable ``phase(r_from, r_to)`` (magnetic field). The
             same function is applied in the leads, so use a gauge that is
             invariant along each lead (see :func:`landau_gauge`).
    max_sites : safety limit for the flood fill
    """

    def __init__(self, model: Model, shape, start=None, onsite=None, peierls=None,
                 max_sites=2_000_000):
        if model.dim == 0:
            raise ValueError("use a periodic template model; for a 0D model call model.finite()")
        self.model = model
        self.shape = shape
        self.onsite_fn = onsite
        self.peierls = peierls
        self.ns = model.ns
        self._nbt = _neighbor_table(model)
        self.sites = self._flood(shape, start, max_sites)
        self.index = {s: n for n, s in enumerate(self.sites)}
        self.leads: list[Lead] = []
        self._H = None

    # -------------------------------------------------------------------
    # construction
    # -------------------------------------------------------------------
    def _flood(self, shape, start, max_sites):
        m = self.model
        if start is None:
            seeds = [(o, (0,) * m.dim) for o in range(m.norb)]
        else:
            start = np.asarray(start, float)
            frac = np.floor(m.lat.fractional(start[: m.lat.D]) if m.dim else []).astype(int)
            cands = []
            for o in range(m.norb):
                for dR in np.ndindex(*(3,) * m.dim):
                    R = tuple(frac + np.array(dR) - 1)
                    cands.append((np.linalg.norm(_pos(m, (o, R)) - start[: m.lat.D]), (o, R)))
            cands.sort(key=lambda x: x[0])
            seeds = [c[1] for c in cands]
        seeds = [s for s in seeds if shape(_pos(m, s))]
        if not seeds:
            raise ValueError("no site of the template lies inside the shape near `start`")
        seen = set(seeds)
        order = list(seeds)
        q = deque(seeds)
        while q:
            o, R = q.popleft()
            for j, dR, _ in self._nbt[o]:
                s = (j, tuple(np.add(R, dR)))
                if s in seen:
                    continue
                if shape(_pos(m, s)):
                    seen.add(s)
                    order.append(s)
                    q.append(s)
                    if len(order) > max_sites:
                        raise RuntimeError("flood fill exceeded max_sites: is the shape finite?")
        # sites at the same position but not reached by hoppings (e.g. other
        # orbitals of an atom) are added too
        extra = []
        for o, R in list(order):
            for o2 in range(m.norb):
                s = (o2, R)
                if s not in seen and np.allclose(m.lat.positions[o2], m.lat.positions[o]) and shape(_pos(m, s)):
                    seen.add(s)
                    extra.append(s)
        return order + extra

    def add_sites(self, sites):
        for s in sites:
            if s not in self.index:
                self.index[s] = len(self.sites)
                self.sites.append(s)
        self._H = None

    def remove_sites(self, sites):
        rm = set(sites)
        if self.leads:
            raise RuntimeError("remove sites before attaching leads")
        self.sites = [s for s in self.sites if s not in rm]
        self.index = {s: n for n, s in enumerate(self.sites)}
        self._H = None

    # -------------------------------------------------------------------
    @property
    def nsites(self):
        return len(self.sites)

    @property
    def nbasis(self):
        return len(self.sites) * self.ns

    def positions(self):
        return np.array([_pos(self.model, s) for s in self.sites])

    def _onsite_block(self, model, site):
        o, R = site
        blk = model._onsite[o].copy()
        if self.onsite_fn is not None:
            v = self.onsite_fn(_pos(model, site), model.lat.names[o])
            blk = blk + model._block(v)
        return blk

    def _hop(self, model, a, b, T):
        if self.peierls is None:
            return T
        return T * np.exp(1j * self.peierls(_pos(model, a), _pos(model, b)))

    def hamiltonian(self, sparse=True):
        """Hamiltonian of the isolated scattering region."""
        if self._H is None:
            ns, m = self.ns, self.model
            rows, cols, vals = [], [], []
            for n, s in enumerate(self.sites):
                blk = self._onsite_block(m, s)
                for a in range(ns):
                    for b in range(ns):
                        if blk[a, b] != 0:
                            rows.append(n * ns + a); cols.append(n * ns + b); vals.append(blk[a, b])
                o, R = s
                for j, dR, T in self._nbt[o]:
                    t = (j, tuple(np.add(R, dR)))
                    nn = self.index.get(t)
                    if nn is None:
                        continue
                    T = self._hop(m, s, t, T)
                    for a in range(ns):
                        for b in range(ns):
                            if T[a, b] != 0:
                                rows.append(n * ns + a); cols.append(nn * ns + b); vals.append(T[a, b])
            N = self.nbasis
            self._H = sp.csr_matrix((vals, (rows, cols)), shape=(N, N), dtype=complex)
            self._H.sum_duplicates()
        return self._H if sparse else self._H.toarray()

    # -------------------------------------------------------------------
    # leads
    # -------------------------------------------------------------------
    def attach_lead(self, vector, shape=None, model=None, onsite=None, peierls="region",
                    fill=True, name=None):
        """Attach a semi-infinite lead going off along ``vector`` (an integer
        lattice vector with a single nonzero entry +-1, e.g. (-1, 0) = left).

        shape  : cross-section ``shape(pos) -> bool`` (default: every row of
                 the region, i.e. the lead is as wide as the region)
        model  : template of the lead (default: the region template); must share
                 the lattice and orbitals - e.g. a superconducting BdG lead
        onsite : on-site function for the lead (default None: clean template;
                 ``"region"`` reuses the region's function, e.g. a uniform gate)
        peierls: magnetic phase in the lead: ``"region"`` (default) uses the
                 system's gauge, ``None`` makes a field-free lead (e.g. the side
                 probes of a Hall bar), or give a callable
        fill   : add lead sites into the region so the interface is complete
                 (like Kwant), needed for irregular shapes
        Returns the lead index.
        """
        m = model or self.model
        if m.norb != self.model.norb or m.spinful != self.model.spinful:
            raise ValueError("lead template must have the same orbitals as the region")
        vec = np.asarray(vector, int)
        nz = np.flatnonzero(vec)
        if len(nz) != 1 or abs(vec[nz[0]]) != 1:
            raise ValueError("vector must be +-1 along exactly one lattice direction; "
                             "build a supercell template for other directions")
        d, sgn = int(nz[0]), int(vec[nz[0]])
        rng = max((abs(t.R[d]) for t in m._terms), default=0)
        if rng > 1:
            raise ValueError(f"hoppings reach {rng} cells along the lead; use a supercell "
                             "template so that hoppings connect neighbouring cells only")
        perp = [a for a in range(self.model.dim) if a != d]
        row = lambda s: (s[0], tuple(s[1][a] for a in perp))
        layer = lambda s: s[1][d]

        # rows of the cross-section and the first region layer of each row
        first = {}
        for s in self.sites:
            r, L = row(s), layer(s)
            if r not in first or (sgn < 0 and L < first[r]) or (sgn > 0 and L > first[r]):
                first[r] = L
        if shape is None:
            rows = sorted(first)          # every row of the region (filled up to the lead)
        else:
            # scan rows near the region's transverse extent
            rows = set()
            Ls = [layer(s) for s in self.sites]
            Lmid = min(Ls) if sgn < 0 else max(Ls)
            cand = {row(s) for s in self.sites}
            # extend the candidate rows by neighbours transversally (lead may be wider)
            grow = True
            while grow:
                grow = False
                for (o, Rp) in list(cand):
                    for j, dR, _ in self._nbt[o]:
                        r2 = (j, tuple(np.add(Rp, [dR[a] for a in perp])))
                        if r2 in cand:
                            continue
                        R = [0] * self.model.dim
                        for a, v in zip(perp, r2[1]):
                            R[a] = v
                        R[d] = Lmid
                        if shape(_pos(m, (r2[0], tuple(R)))):
                            cand.add(r2)
                            grow = True
            for r in cand:
                R = [0] * self.model.dim
                for a, v in zip(perp, r[1]):
                    R[a] = v
                R[d] = Lmid
                if shape(_pos(m, (r[0], tuple(R)))):
                    rows.add(r)
            rows = sorted(rows)
        rows_in = [r for r in rows if r in first]
        if not rows_in:
            raise ValueError("lead cross-section does not touch the scattering region")
        start = (min(first[r] for r in rows_in) - 1) if sgn < 0 else (max(first[r] for r in rows_in) + 1)

        def mksite(r, L):
            R = [0] * self.model.dim
            for a, v in zip(perp, r[1]):
                R[a] = v
            R[d] = L
            return (r[0], tuple(R))

        if fill:
            add = []
            for r in rows_in:
                f = first[r]
                Ls = range(start + 1, f) if sgn < 0 else range(f + 1, start)
                add += [mksite(r, L) for L in Ls]
            # also rows of the cross-section absent from the region are not filled
            self.add_sites(add)

        cell0 = [mksite(r, start) for r in rows]
        cell1 = [mksite(r, start + sgn) for r in rows]
        cell2 = {mksite(r, start + 2 * sgn) for r in rows}
        i0 = {s: n for n, s in enumerate(cell0)}
        i1 = {s: n for n, s in enumerate(cell1)}
        ns = self.ns
        nl = len(cell0) * ns
        nbt = _neighbor_table(m)
        old_onsite, old_peierls = self.onsite_fn, self.peierls
        if not (isinstance(onsite, str) and onsite == "region"):
            self.onsite_fn = onsite
        if not (isinstance(peierls, str) and peierls == "region"):
            self.peierls = peierls
        H0 = np.zeros((nl, nl), complex)
        V = np.zeros((nl, nl), complex)
        H1 = np.zeros((nl, nl), complex)
        Wr, Wc, Wv = [], [], []
        for n, s in enumerate(cell0):
            H0[n * ns:(n + 1) * ns, n * ns:(n + 1) * ns] += self._onsite_block(m, s)
            H1[n * ns:(n + 1) * ns, n * ns:(n + 1) * ns] += self._onsite_block(m, cell1[n])
            o, R = s
            for j, dR, T in nbt[o]:
                t = (j, tuple(np.add(R, dR)))
                Tp = self._hop(m, s, t, T)
                if t in i0:
                    H0[n * ns:(n + 1) * ns, i0[t] * ns:(i0[t] + 1) * ns] += Tp
                elif t in i1:
                    V[n * ns:(n + 1) * ns, i1[t] * ns:(i1[t] + 1) * ns] += Tp
                elif t in self.index:
                    # region -> lead coupling (store as W[region, lead] = <region|H|lead>)
                    # the interface bond uses the region's gauge
                    self.peierls = old_peierls
                    Tr = self._hop(m, t, s, T.conj().T)
                    if not (isinstance(peierls, str) and peierls == "region"):
                        self.peierls = peierls
                    rn = self.index[t]
                    for a in range(ns):
                        for b in range(ns):
                            if Tr[a, b] != 0:
                                Wr.append(rn * ns + a); Wc.append(n * ns + b); Wv.append(Tr[a, b])
            # gauge / translation check with cell 1
            s1 = cell1[n]
            o1, R1 = s1
            for j, dR, T in nbt[o1]:
                t = (j, tuple(np.add(R1, dR)))
                if t in i1:
                    H1[n * ns:(n + 1) * ns, i1[t] * ns:(i1[t] + 1) * ns] += self._hop(m, s1, t, T)
        self.onsite_fn, self.peierls = old_onsite, old_peierls
        if not np.allclose(H0, H1, atol=1e-9):
            warnings.warn("lead Hamiltonian is not translation invariant along the lead "
                          "(check the gauge of the magnetic field / the on-site function)")
        W = sp.csr_matrix((Wv, (Wr, Wc)), shape=(self.nbasis, nl), dtype=complex)
        W.sum_duplicates()
        iface = np.unique(W.nonzero()[0])
        lead = Lead(tuple(vec), m, cell0, H0, V, W, iface, W[iface].toarray(),
                    name or f"lead{len(self.leads)}")
        self.leads.append(lead)
        self._H = None
        return len(self.leads) - 1

    # -------------------------------------------------------------------
    # masks / projectors for conservation laws
    # -------------------------------------------------------------------
    def mask(self, spin=None, orbitals=None):
        """Boolean mask over the region basis selecting a spin (``'up'``,
        ``'down'``) and/or a set of template orbitals (names or indices)."""
        m = self.model
        sel_orb = None
        if orbitals is not None:
            sel_orb = set(m._orb_list(orbitals))
        out = np.zeros(self.nbasis, bool)
        for n, (o, R) in enumerate(self.sites):
            if sel_orb is not None and o not in sel_orb:
                continue
            for a in range(self.ns):
                if spin is None or (spin == "up" and a == 0) or (spin == "down" and a == 1):
                    out[n * self.ns + a] = True
        return out

    def lead_mask(self, lead, spin=None, orbitals=None):
        L = self.leads[lead]
        m = self.model
        sel_orb = set(m._orb_list(orbitals)) if orbitals is not None else None
        out = np.zeros(L.nb, bool)
        for n, (o, R) in enumerate(L.sites0):
            if sel_orb is not None and o not in sel_orb:
                continue
            for a in range(self.ns):
                if spin is None or (spin == "up" and a == 0) or (spin == "down" and a == 1):
                    out[n * self.ns + a] = True
        return out

    # -------------------------------------------------------------------
    # convenience wrappers to the solver
    # -------------------------------------------------------------------
    def smatrix(self, energy, **kw):
        from .solver import smatrix
        return smatrix(self, energy, **kw)

    def transmission(self, to_lead, from_lead, energy, **kw):
        return self.smatrix(energy).transmission(to_lead, from_lead, **kw)

    def conductance(self, energies, to_lead=1, from_lead=0, **kw):
        """Two-terminal conductance G(E) in units of e^2/h."""
        energies = np.atleast_1d(energies)
        return np.array([self.smatrix(E).transmission(to_lead, from_lead, **kw) for E in energies])

    def plot(self, **kw):
        from .plot import plot_system
        return plot_system(self, **kw)

    def __repr__(self):
        return (f"<transport.System: {self.nsites} sites, {self.nbasis} basis states, "
                f"{len(self.leads)} leads>")


def landau_gauge(B, lead_axis="x"):
    """Peierls phase for a uniform field B (phase per unit area, i.e.
    B = 2 pi * flux / (flux quantum * area)) in a Landau gauge that is
    translation invariant along ``lead_axis`` (``'x'``: A = (-B y, 0))."""
    if lead_axis == "x":
        return lambda ri, rj: -B * 0.5 * (ri[1] + rj[1]) * (rj[0] - ri[0])
    return lambda ri, rj: B * 0.5 * (ri[0] + rj[0]) * (rj[1] - ri[1])


def anderson(W, seed=None):
    """On-site function for Anderson disorder of strength W (uniform in
    [-W/2, W/2]); reproducible per site position."""
    import zlib
    rng_seed = 0 if seed is None else int(seed)

    def f(pos, name):
        key = f"{rng_seed}|{name}|" + ",".join(f"{x:.6f}" for x in np.ravel(pos))
        h = zlib.crc32(key.encode())
        return np.random.default_rng([rng_seed, h]).uniform(-W / 2, W / 2)
    return f
