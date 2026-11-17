"""Finite scattering regions with semi-infinite leads (Kwant-style builder).

A :class:`System` is cut out of any periodic hopforge :class:`~hopforge.Model`
(the *template*) by a shape function, flood-filled from a starting point. Leads
are semi-infinite continuations of a template along a lattice vector.

>>> sq = hf.models.square()
>>> sys = hf.transport.System(sq, lambda r: 0 <= r[0] < 20 and 0 <= r[1] < 10)
>>> sys.attach_lead((-1, 0)); sys.attach_lead((1, 0))
>>> sys.transmission(1, 0, energy=-3.0)      # = number of open channels

Value functions and parameters (as in Kwant)
--------------------------------------------
``onsite(pos, name, **params)``      added to the template on-site block
``hopping(r_i, r_j, T, **params)``   returns the new hopping block (e.g. strain)
``peierls(r_i, r_j, **params)``      magnetic phase added to every hopping
Each function receives only the parameters it declares; pass them at solve
time: ``sys.smatrix(E, params=dict(V=0.3, B=0.1))``.

Sites are labelled (orbital, R) with R the integer cell vector. The basis of
the region is site-major with the spin index inner (as in Model).
"""
from __future__ import annotations

import inspect
import warnings
from collections import deque

import numpy as np
import scipy.sparse as sp

from ..model import Model


def _neighbor_table(model: Model):
    """For every orbital i: list of (j, R, T) with <i,0|H|j,R> = T (both
    orientations of every stored hopping)."""
    nb = [[] for _ in range(model.norb)]
    for t in model._terms:
        nb[t.i].append((t.j, tuple(int(x) for x in t.R), t.T))
        nb[t.j].append((t.i, tuple(int(-x) for x in t.R), t.T.conj().T))
    return nb


def _pos(model, site):
    o, R = site
    return model.lat.positions[o] + (np.array(R) @ model.lat.vectors if model.dim else 0.0)


def _call(f, *args, params=None):
    """Call a value function with only the parameters it accepts."""
    if f is None:
        return None
    params = params or {}
    try:
        sig = inspect.signature(f)
    except (TypeError, ValueError):
        return f(*args, **params)
    ps = sig.parameters
    if any(p.kind == p.VAR_KEYWORD for p in ps.values()):
        return f(*args, **params)
    kw = {k: v for k, v in params.items() if k in ps}
    missing = [n for n, p in list(ps.items())[len(args):]
               if p.default is p.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
               and n not in kw]
    if missing:
        raise TypeError(f"value function {getattr(f, '__name__', f)} needs parameters {missing}; "
                        "pass them with params={...}")
    return f(*args, **kw)


def _key(params):
    if not params:
        return ()
    try:
        return tuple(sorted((k, v if np.isscalar(v) else id(v)) for k, v in params.items()))
    except TypeError:
        return None


class _Values:
    """On-site / hopping evaluation with the value functions of a region or lead."""

    def __init__(self, model, onsite=None, hopping=None, peierls=None):
        self.model, self.onsite, self.hopping, self.peierls = model, onsite, hopping, peierls

    def onsite_block(self, site, params):
        o, R = site
        blk = self.model._onsite[o].copy()
        if self.onsite is not None:
            v = _call(self.onsite, _pos(self.model, site), self.model.lat.names[o], params=params)
            blk = blk + self.model._block(v)
        return blk

    def hop(self, a, b, T, params):
        ra, rb = _pos(self.model, a), _pos(self.model, b)
        if self.hopping is not None:
            T = self.model._block(_call(self.hopping, ra, rb, T, params=params))
        if self.peierls is not None:
            T = T * np.exp(1j * _call(self.peierls, ra, rb, params=params))
        return T


class Lead:
    """A semi-infinite lead. Cell 0 touches the scattering region; cell j+1 is
    cell j translated by ``vector``. Matrices are rebuilt for every parameter
    set (``matrices(params)``)."""

    def __init__(self, system, vector, model, cell0, cell1, values, region_values, name):
        self.system = system
        self.vector = tuple(vector)
        self.model = model
        self.sites0 = cell0
        self._cell1 = cell1
        self.values = values
        self.region_values = region_values
        self.name = name
        self._nbt = _neighbor_table(model)
        self._mats = {}
        self._cache = {}
        self.ns = model.ns

    # ---------------------------------------------------------------------
    def matrices(self, params=None):
        """(H0, V, W, iface, Wi) for the given parameters."""
        key = _key(params)
        if key is not None and key in self._mats and self._mats[key][2].shape[0] == self.system.nbasis:
            return self._mats[key]
        ns, sysm = self.ns, self.system
        cell0, cell1 = self.sites0, self._cell1
        i0 = {s: n for n, s in enumerate(cell0)}
        i1 = {s: n for n, s in enumerate(cell1)}
        nl = len(cell0) * ns
        H0 = np.zeros((nl, nl), complex)
        V = np.zeros((nl, nl), complex)
        H1 = np.zeros((nl, nl), complex)
        Wr, Wc, Wv = [], [], []
        for n, s in enumerate(cell0):
            H0[n * ns:(n + 1) * ns, n * ns:(n + 1) * ns] += self.values.onsite_block(s, params)
            H1[n * ns:(n + 1) * ns, n * ns:(n + 1) * ns] += self.values.onsite_block(cell1[n], params)
            o, R = s
            for j, dR, T in self._nbt[o]:
                t = (j, tuple(np.add(R, dR)))
                if t in i0:
                    H0[n * ns:(n + 1) * ns, i0[t] * ns:(i0[t] + 1) * ns] += self.values.hop(s, t, T, params)
                elif t in i1:
                    V[n * ns:(n + 1) * ns, i1[t] * ns:(i1[t] + 1) * ns] += self.values.hop(s, t, T, params)
                elif t in sysm.index:
                    Tr = self.region_values.hop(t, s, T.conj().T, params)   # <region|H|lead>
                    rn = sysm.index[t]
                    for a in range(ns):
                        for b in range(ns):
                            if Tr[a, b] != 0:
                                Wr.append(rn * ns + a); Wc.append(n * ns + b); Wv.append(Tr[a, b])
            s1 = cell1[n]
            o1, R1 = s1
            for j, dR, T in self._nbt[o1]:
                t = (j, tuple(np.add(R1, dR)))
                if t in i1:
                    H1[n * ns:(n + 1) * ns, i1[t] * ns:(i1[t] + 1) * ns] += self.values.hop(s1, t, T, params)
        if not np.allclose(H0, H1, atol=1e-9):
            warnings.warn(f"{self.name}: Hamiltonian is not translation invariant along the lead "
                          "(check the gauge of the magnetic field / the on-site function)")
        W = sp.csr_matrix((Wv, (Wr, Wc)), shape=(sysm.nbasis, nl), dtype=complex)
        W.sum_duplicates()
        iface = np.unique(W.nonzero()[0])
        out = (H0, V, W, iface, W[iface].toarray())
        if key is not None:
            self._mats[key] = out
        return out

    @property
    def H0(self):
        return self.matrices(self.system._last_params)[0]

    @property
    def V(self):
        return self.matrices(self.system._last_params)[1]

    @property
    def nb(self):
        return len(self.sites0) * self.ns

    # ---------------------------------------------------------------------
    def surface_green(self, energy, params=None, eta=1e-10, maxiter=200, tol=1e-13):
        """Retarded surface Green's function of the isolated semi-infinite lead
        (Lopez Sancho - Rubio decimation)."""
        key = ("g", complex(energy), eta, _key(params))
        if key[3] is not None and key in self._cache:
            return self._cache[key]
        H0, Vm = self.matrices(params)[:2]
        n = H0.shape[0]
        z = energy + 1j * eta
        I = np.eye(n)
        es, e = H0.astype(complex).copy(), H0.astype(complex).copy()
        a, b = Vm.astype(complex).copy(), Vm.conj().T.astype(complex).copy()
        for _ in range(maxiter):
            g = np.linalg.solve(z * I - e, np.hstack([a, b]))
            ga, gb = g[:, :n], g[:, n:]
            agb, bga = a @ gb, b @ ga
            es = es + agb
            e = e + agb + bga
            a, b = a @ ga, b @ gb
            if np.abs(a).max() < tol and np.abs(b).max() < tol:
                break
        else:
            warnings.warn("surface Green's function did not converge; increase eta")
        gs = np.linalg.inv(z * I - es)
        if key[3] is not None:
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[key] = gs
        return gs

    def self_energy(self, energy, params=None, **kw):
        """Self-energy on the interface basis states (dense, len(iface)^2)."""
        _, _, _, iface, Wi = self.matrices(params)
        gs = self.surface_green(energy, params, **kw)
        return Wi @ gs @ Wi.conj().T

    def bands(self, k, params=None):
        """Lead band structure E_n(k), k in units of 1/period (k in [-pi, pi])."""
        H0, V = self.matrices(params)[:2]
        k = np.atleast_1d(k)
        H = H0[None] + V[None] * np.exp(1j * k)[:, None, None] + V.conj().T[None] * np.exp(-1j * k)[:, None, None]
        return np.linalg.eigvalsh(H)

    def modes(self, energy, projector=None, params=None, tol=1e-6):
        """Propagating modes at ``energy`` from the generalised eigenproblem
        (E - H0 - lambda V - V^dag / lambda) phi = 0.

        Returns (k, velocity, phi) for the propagating modes (|lambda| = 1);
        within degenerate subspaces the velocity matrix is diagonalised, so band
        crossings are handled exactly. Velocities > 0 move away from the region."""
        import scipy.linalg as sla
        H0, V = self.matrices(params)[:2]
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

    def num_modes(self, energy, projector=None, params=None):
        """Number of propagating channels (right movers) at ``energy``; with
        ``projector`` (on the lead cell basis) only within that conserved block."""
        k, v, _ = self.modes(energy, projector, params)
        if len(k) == 0:
            return 0
        vscale = max(np.abs(self.matrices(params)[1]).max(), 1e-12)
        return int((v > 1e-6 * vscale).sum())


class System:
    """Finite scattering region built from a template model and a shape.

    Parameters
    ----------
    model : periodic hopforge Model (any dimension >= 1)
    shape : callable ``shape(pos) -> bool`` on Cartesian positions
    start : a position inside the shape (flood fill seed); default: first
            orbital of the home cell
    onsite : ``onsite(pos, orbital_name, **params) -> scalar or (ns x ns)``
             added to the template on-site term (gates, disorder, pairing, ...)
    hopping : ``hopping(r_i, r_j, T, **params) -> new block`` (strain, spatially
             varying SOC, ...)
    peierls : ``phase(r_from, r_to, **params)`` (magnetic field); see
             :func:`landau_gauge` and :func:`field_gauge`.
    max_sites : safety limit for the flood fill
    """

    def __init__(self, model: Model, shape, start=None, onsite=None, peierls=None, hopping=None,
                 max_sites=2_000_000):
        if model.dim == 0:
            raise ValueError("use a periodic template model; for a 0D model call model.finite()")
        self.model = model
        self.shape = shape
        self.ns = model.ns
        self.values = _Values(model, onsite, hopping, peierls)
        self._nbt = _neighbor_table(model)
        self.sites = self._flood(shape, start, max_sites)
        self.index = {s: n for n, s in enumerate(self.sites)}
        self.leads: list[Lead] = []
        self._H = {}
        self._last_params = None

    # backwards-compatible attributes
    @property
    def onsite_fn(self):
        return self.values.onsite

    @property
    def peierls(self):
        return self.values.peierls

    # -------------------------------------------------------------------
    # construction
    # -------------------------------------------------------------------
    def _flood(self, shape, start, max_sites):
        m = self.model
        if start is None:
            seeds = [(o, (0,) * m.dim) for o in range(m.norb)]
        else:
            start = np.asarray(start, float)
            frac = np.floor(m.lat.fractional(start[: m.lat.D])).astype(int)
            cands = []
            for o in range(m.norb):
                for dR in np.ndindex(*(3,) * m.dim):
                    R = tuple(int(x) for x in (frac + np.array(dR) - 1))
                    cands.append((np.linalg.norm(_pos(m, (o, R)) - start[: m.lat.D]), (o, R)))
            cands.sort(key=lambda x: x[0])
            seeds = [c[1] for c in cands]
        seeds = [s for s in seeds if shape(_pos(m, s))]
        if not seeds:
            raise ValueError("no site of the template lies inside the shape near `start`")
        seeds = seeds[:1] if start is not None else seeds
        seen = set(seeds)
        order = list(seeds)
        q = deque(seeds)
        while q:
            o, R = q.popleft()
            for j, dR, _ in self._nbt[o]:
                s = (j, tuple(int(x) for x in np.add(R, dR)))
                if s in seen:
                    continue
                if shape(_pos(m, s)):
                    seen.add(s)
                    order.append(s)
                    q.append(s)
                    if len(order) > max_sites:
                        raise RuntimeError("flood fill exceeded max_sites: is the shape finite?")
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
        self._H = {}

    def remove_sites(self, sites):
        if self.leads:
            raise RuntimeError("remove sites before attaching leads")
        rm = set(sites)
        self.sites = [s for s in self.sites if s not in rm]
        self.index = {s: n for n, s in enumerate(self.sites)}
        self._H = {}

    # -------------------------------------------------------------------
    @property
    def nsites(self):
        return len(self.sites)

    @property
    def nbasis(self):
        return len(self.sites) * self.ns

    def positions(self):
        return np.array([_pos(self.model, s) for s in self.sites])

    def hamiltonian(self, params=None, sparse=True):
        """Hamiltonian of the isolated scattering region for given parameters."""
        key = _key(params)
        self._last_params = params
        if key is None or key not in self._H:
            ns, m = self.ns, self.model
            rows, cols, vals = [], [], []
            for n, s in enumerate(self.sites):
                blk = self.values.onsite_block(s, params)
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
                    T = self.values.hop(s, t, T, params)
                    for a in range(ns):
                        for b in range(ns):
                            if T[a, b] != 0:
                                rows.append(n * ns + a); cols.append(nn * ns + b); vals.append(T[a, b])
            N = self.nbasis
            H = sp.csr_matrix((vals, (rows, cols)), shape=(N, N), dtype=complex)
            H.sum_duplicates()
            if key is None:
                return H if sparse else H.toarray()
            if len(self._H) > 64:
                self._H.clear()
            self._H[key] = H
        H = self._H[key]
        return H if sparse else H.toarray()

    # -------------------------------------------------------------------
    # leads
    # -------------------------------------------------------------------
    def attach_lead(self, vector, shape=None, model=None, onsite=None, peierls="region",
                    hopping=None, fill=True, name=None):
        """Attach a semi-infinite lead going off along ``vector`` (an integer
        lattice vector with a single nonzero entry +-1, e.g. (-1, 0) = left).

        shape  : cross-section ``shape(pos) -> bool`` (default: every row of
                 the region, i.e. the lead is as wide as the region)
        model  : template of the lead (default: the region template); must share
                 the orbitals - e.g. a superconducting BdG lead
        onsite : lead on-site function (default None: clean template;
                 ``"region"`` reuses the region's function, e.g. a uniform gate)
        peierls: ``"region"`` (default) uses the system's gauge, ``None`` makes a
                 field-free lead (e.g. the side probes of a Hall bar), or a callable
        hopping: lead hopping function (default None; ``"region"`` reuses it)
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
        rng = max((abs(int(t.R[d])) for t in m._terms), default=0)
        if rng > 1:
            raise ValueError(f"hoppings reach {rng} cells along the lead; use a supercell "
                             "template so that hoppings connect neighbouring cells only")
        perp = [a for a in range(self.model.dim) if a != d]
        row = lambda s: (s[0], tuple(s[1][a] for a in perp))
        layer = lambda s: s[1][d]
        first = {}
        for s in self.sites:
            r, L = row(s), layer(s)
            if r not in first or (sgn < 0 and L < first[r]) or (sgn > 0 and L > first[r]):
                first[r] = L

        def mksite(r, L):
            R = [0] * self.model.dim
            for a, v in zip(perp, r[1]):
                R[a] = int(v)
            R[d] = int(L)
            return (r[0], tuple(R))

        if shape is None:
            rows = sorted(first)
        else:
            Ls = [layer(s) for s in self.sites]
            Lmid = min(Ls) if sgn < 0 else max(Ls)
            cand = {row(s) for s in self.sites}
            grow = True
            while grow:
                grow = False
                for (o, Rp) in list(cand):
                    for j, dR, _ in self._nbt[o]:
                        r2 = (j, tuple(int(x) for x in np.add(Rp, [dR[a] for a in perp])))
                        if r2 not in cand and shape(_pos(m, mksite(r2, Lmid))):
                            cand.add(r2)
                            grow = True
            rows = sorted(r for r in cand if shape(_pos(m, mksite(r, Lmid))))
        rows_in = [r for r in rows if r in first]
        if not rows_in:
            raise ValueError("lead cross-section does not touch the scattering region")
        start = (min(first[r] for r in rows_in) - 1) if sgn < 0 else (max(first[r] for r in rows_in) + 1)
        if fill:
            add = []
            for r in rows_in:
                f = first[r]
                Ls = range(start + 1, f) if sgn < 0 else range(f + 1, start)
                add += [mksite(r, L) for L in Ls]
            self.add_sites(add)
        cell0 = [mksite(r, start) for r in rows]
        cell1 = [mksite(r, start + sgn) for r in rows]
        reg = self.values
        lv = _Values(m,
                     reg.onsite if (isinstance(onsite, str) and onsite == "region") else onsite,
                     reg.hopping if (isinstance(hopping, str) and hopping == "region") else hopping,
                     reg.peierls if (isinstance(peierls, str) and peierls == "region") else peierls)
        lead = Lead(self, vec, m, cell0, cell1, lv, reg, name or f"lead{len(self.leads)}")
        self.leads.append(lead)
        self._H = {}
        return len(self.leads) - 1

    # -------------------------------------------------------------------
    # masks / projectors for conservation laws
    # -------------------------------------------------------------------
    def mask(self, spin=None, orbitals=None):
        """Boolean mask over the region basis selecting a spin (``'up'``,
        ``'down'``) and/or a set of template orbitals (names or indices)."""
        m = self.model
        sel_orb = set(m._orb_list(orbitals)) if orbitals is not None else None
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
    def smatrix(self, energy, params=None, **kw):
        from .solver import smatrix
        return smatrix(self, energy, params=params, **kw)

    def transmission(self, to_lead, from_lead, energy, params=None, **kw):
        return self.smatrix(energy, params).transmission(to_lead, from_lead, **kw)

    def conductance(self, energies, to_lead=1, from_lead=0, params=None, n_jobs=None, **kw):
        """Two-terminal conductance G(E) in units of e^2/h (parallel over energies)."""
        from .solver import conductance
        return conductance(self, energies, to_lead, from_lead, params=params, n_jobs=n_jobs, **kw)

    def plot(self, **kw):
        from .plot import plot_system
        return plot_system(self, **kw)

    def __repr__(self):
        return (f"<transport.System: {self.nsites} sites, {self.nbasis} basis states, "
                f"{len(self.leads)} leads>")


# ---------------------------------------------------------------------------
# gauges and disorder
# ---------------------------------------------------------------------------
def landau_gauge(B, lead_axis="x"):
    """Peierls phase for a uniform field B (phase per unit area, i.e.
    B = 2 pi * flux / (flux quantum * area)) in a Landau gauge that is
    translation invariant along ``lead_axis`` (``'x'``: A = (-B y, 0))."""
    if lead_axis == "x":
        return lambda ri, rj: -B * 0.5 * (ri[1] + rj[1]) * (rj[0] - ri[0])
    return lambda ri, rj: B * 0.5 * (ri[0] + rj[0]) * (rj[1] - ri[1])


def field_gauge(B, lead_axis="x", origin=0.0, nquad=8):
    """Peierls phases for an arbitrary 2D field profile ``B(x, y, **params)``
    (phase per unit area), like Kwant's ``magnetic_gauge`` for planar systems.

    lead_axis 'x': A = (-int_origin^y B(x, y') dy', 0); leads along x must sit
    where B does not depend on x. lead_axis 'y': A = (0, int_origin^x B dx').
    The returned function accepts the same parameters as ``B``."""
    gx, gw = np.polynomial.legendre.leggauss(nquad)

    def line_int(f, a, b):
        if a == b:
            return 0.0
        t = 0.5 * (gx + 1)
        return 0.5 * (b - a) * sum(w * f(a + ti * (b - a)) for ti, w in zip(t, gw))

    def phase(ri, rj, **params):
        ri, rj = np.asarray(ri, float), np.asarray(rj, float)

        def A(r):
            x, y = r[0], r[1]
            if lead_axis == "x":
                return np.array([-line_int(lambda yy: _call(B, x, yy, params=params), origin, y), 0.0])
            return np.array([0.0, line_int(lambda xx: _call(B, xx, y, params=params), origin, x)])
        d = (rj - ri)[:2]
        # integral of A . dl along the straight bond (Gauss-Legendre)
        t = 0.5 * (gx + 1)
        return 0.5 * sum(w * A(ri[:2] + ti * d) @ d for ti, w in zip(t, gw))

    try:
        bp = list(inspect.signature(B).parameters)[2:]
    except (TypeError, ValueError):
        bp = []
    if bp:
        # expose B's parameters so that _call passes them through
        ns = {}
        exec("def gauge(ri, rj, " + ", ".join(bp) + "):\n"
             "    return _phase(ri, rj, " + ", ".join(f"{p}={p}" for p in bp) + ")", {"_phase": phase}, ns)
        return ns["gauge"]
    return phase


def anderson(W, seed=None):
    """On-site function for Anderson disorder of strength W (uniform in
    [-W/2, W/2]); reproducible per site position and seed."""
    import zlib
    rng_seed = 0 if seed is None else int(seed)

    def f(pos, name):
        key = f"{rng_seed}|{name}|" + ",".join(f"{x:.6f}" for x in np.ravel(pos))
        h = zlib.crc32(key.encode())
        return np.random.default_rng([rng_seed, h]).uniform(-W / 2, W / 2)
    return f
