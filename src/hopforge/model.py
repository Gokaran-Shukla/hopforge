"""Lattice and tight-binding Model: the heart of hopforge.

Conventions
-----------
* ``dim``  : number of periodic directions (0, 1, 2 or 3).
* ``D``    : dimension of real space the orbitals live in (>= dim).
* k-points are given in *reduced* coordinates (units of reciprocal vectors)
  unless ``cartesian=True`` is passed.
* Bloch Hamiltonian (atomic / "convention II" gauge, the default)::

      H_ij(k) = sum_R <i,0|H|j,R> exp(i k . (R + r_j - r_i))

  ``gauge="periodic"`` drops the orbital positions (convention I). Spectra are
  identical; Berry phases and polarisation are physical only in the atomic gauge.
* Spinful basis is orbital-major: index = orbital * 2 + spin (up=0, down=1).
* hbar = 1, energies and lengths in whatever units you choose (eV and Angstrom
  are typical).
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field

import numpy as np

from . import operators as ops


# ==========================================================================
# Lattice
# ==========================================================================
class Lattice:
    """A Bravais lattice plus a list of orbitals.

    Parameters
    ----------
    vectors : array (dim, D)
        Lattice vectors as rows. Use ``[]`` or ``None`` for a finite (0D) system.
    orbitals : list
        Each entry is ``(name, position)`` or just ``position``. Names may carry
        an orbital kind after a colon, e.g. ``"Mo:dxy"`` (site ``Mo``, kind
        ``dxy``); without a colon the kind is ``s``.
    coords : "fractional" | "cartesian"
        How the orbital positions are given. Fractional needs dim == D.
    """

    def __init__(self, vectors, orbitals, coords="fractional"):
        orbs = []
        for n, o in enumerate(orbitals):
            if isinstance(o, (tuple, list)) and len(o) == 2 and isinstance(o[0], str):
                orbs.append((o[0], np.atleast_1d(np.asarray(o[1], float))))
            else:
                orbs.append((f"o{n}", np.atleast_1d(np.asarray(o, float))))
        if vectors is None or len(vectors) == 0:
            D = max(len(p) for _, p in orbs)
            A = np.zeros((0, D))
            if coords == "fractional":
                coords = "cartesian"
        else:
            A = np.atleast_2d(np.asarray(vectors, float))
        self.vectors = A
        dim, D = A.shape
        pos = []
        for name, p in orbs:
            if coords == "fractional":
                if dim != D:
                    raise ValueError("fractional coordinates need square lattice vectors; use coords='cartesian'")
                if len(p) != dim:
                    raise ValueError(f"orbital {name}: expected {dim} fractional coords")
                pos.append(p @ A)
            else:
                q = np.zeros(D)
                q[: len(p)] = p
                pos.append(q)
        self.names = [n for n, _ in orbs]
        if len(set(self.names)) != len(self.names):
            raise ValueError("orbital names must be unique")
        self.positions = np.array(pos, float).reshape(len(orbs), D)

    # ---- basic properties -------------------------------------------------
    @property
    def dim(self) -> int:
        return self.vectors.shape[0]

    @property
    def D(self) -> int:
        return self.vectors.shape[1]

    @property
    def norb(self) -> int:
        return len(self.names)

    @property
    def reciprocal(self) -> np.ndarray:
        """Reciprocal vectors as rows, a_i . b_j = 2 pi delta_ij."""
        A = self.vectors
        if self.dim == 0:
            return np.zeros((0, self.D))
        return 2 * np.pi * np.linalg.solve(A @ A.T, A)

    @property
    def cell_volume(self) -> float:
        """Length / area / volume of the unit cell (in the periodic subspace)."""
        A = self.vectors
        if self.dim == 0:
            return 1.0
        return float(np.sqrt(abs(np.linalg.det(A @ A.T))))

    def site(self, i) -> str:
        return self.names[i].split(":")[0]

    def kind(self, i) -> str:
        n = self.names[i]
        return ops.normalize_kind(n.split(":", 1)[1]) if ":" in n else "s"

    def fractional(self, cart):
        """Cartesian -> fractional (only along the periodic directions)."""
        A = self.vectors
        return np.linalg.solve(A @ A.T, A @ np.asarray(cart, float).T).T

    def __repr__(self):
        return f"Lattice(dim={self.dim}, D={self.D}, norb={self.norb})"


# ==========================================================================
# Model
# ==========================================================================
@dataclass
class _Term:
    i: int
    j: int
    R: np.ndarray
    T: np.ndarray  # (ns, ns)


class Model:
    """A tight-binding model on a :class:`Lattice`.

    Parameters
    ----------
    lattice : Lattice
    spinful : bool
        If True every orbital carries spin 1/2 and amplitudes may be 2x2 spin
        matrices (scalars mean spin-independent).
    name : str, optional
    """

    def __init__(self, lattice: Lattice, spinful: bool = False, name: str = "model"):
        self.lat = lattice
        self.spinful = bool(spinful)
        self.name = name
        self._onsite = np.zeros((lattice.norb, self.ns, self.ns), complex)
        self._terms: list[_Term] = []
        self._cache = None
        self.special_points: dict[str, np.ndarray] = {"G": np.zeros(lattice.dim)}
        self.info: dict = {}

    # ------------------------------------------------------------------
    # sizes and helpers
    # ------------------------------------------------------------------
    @property
    def ns(self) -> int:
        return 2 if self.spinful else 1

    @property
    def norb(self) -> int:
        return self.lat.norb

    @property
    def nbands(self) -> int:
        return self.lat.norb * self.ns

    @property
    def dim(self) -> int:
        return self.lat.dim

    def copy(self) -> "Model":
        return copy.deepcopy(self)

    def orb(self, o) -> int:
        """Orbital index from an index or a name."""
        if isinstance(o, (int, np.integer)):
            if not 0 <= o < self.norb:
                raise IndexError(f"orbital index {o} out of range")
            return int(o)
        try:
            return self.lat.names.index(o)
        except ValueError:
            raise KeyError(f"no orbital named {o!r}; have {self.lat.names}") from None

    def orbitals_of(self, site: str) -> list[int]:
        """All orbital indices whose site label is ``site``."""
        return [i for i in range(self.norb) if self.lat.site(i) == site]

    def _orb_list(self, sel):
        if sel is None or (isinstance(sel, str) and sel == "all"):
            return list(range(self.norb))
        if isinstance(sel, (str, int, np.integer)):
            if isinstance(sel, str) and sel not in self.lat.names:
                idx = self.orbitals_of(sel)
                if idx:
                    return idx
            return [self.orb(sel)]
        return [self.orb(s) for s in sel]

    def _block(self, t) -> np.ndarray:
        t = np.asarray(t, complex)
        if t.ndim == 0:
            return t * np.eye(self.ns)
        if t.shape == (4,) and self.spinful:
            return ops.spin_block(*t)
        if t.shape != (self.ns, self.ns):
            raise ValueError(f"amplitude must be scalar or {self.ns}x{self.ns}")
        return t

    def _R(self, R):
        if R is None:
            return np.zeros(self.dim, int)
        R = np.atleast_1d(np.asarray(R, int))
        if R.shape != (self.dim,):
            raise ValueError(f"R must have {self.dim} integer components")
        return R

    def _changed(self):
        self._cache = None

    # ------------------------------------------------------------------
    # building the Hamiltonian
    # ------------------------------------------------------------------
    def add_onsite(self, e, orbitals="all"):
        """Add on-site energy ``e`` (scalar or spin matrix) to orbitals."""
        B = self._block(e)
        if not np.allclose(B, B.conj().T):
            raise ValueError("on-site term must be Hermitian")
        for i in self._orb_list(orbitals):
            self._onsite[i] += B
        self._changed()
        return self

    def set_onsite(self, e, orbitals="all"):
        """Replace the on-site term of orbitals."""
        for i in self._orb_list(orbitals):
            self._onsite[i] = 0
        return self.add_onsite(e, orbitals)

    def add_hopping(self, t, i, j, R=None):
        """Add the hopping ``<i, 0| H | j, R> = t`` and its Hermitian conjugate.

        Add every bond once; the reverse bond (j, i, -R) is generated
        automatically. ``t`` may be a scalar, a 2x2 spin matrix, or
        ``[t0, tx, ty, tz]`` (Pauli decomposition) for spinful models.
        """
        i, j, R = self.orb(i), self.orb(j), self._R(R)
        B = self._block(t)
        if i == j and not R.any():
            raise ValueError("i == j with R = 0 is an on-site term; use add_onsite")
        self._terms.append(_Term(i, j, R, B))
        self._changed()
        return self

    def add_onsite_matrix(self, M, orbitals):
        """Hermitian matrix acting inside a group of orbitals of one cell.

        ``M`` has shape (n*ns, n*ns) in the orbital-major basis of ``orbitals``.
        """
        idx = self._orb_list(orbitals)
        M = np.asarray(M, complex)
        ns = self.ns
        if M.shape != (len(idx) * ns,) * 2:
            raise ValueError("matrix size does not match the orbital group")
        if not np.allclose(M, M.conj().T):
            raise ValueError("on-site matrix must be Hermitian")
        for a, ia in enumerate(idx):
            for b, ib in enumerate(idx):
                blk = M[a * ns:(a + 1) * ns, b * ns:(b + 1) * ns]
                if a == b:
                    self._onsite[ia] += blk
                elif a < b and np.abs(blk).max() > 0:
                    self.add_hopping(blk, ia, ib, None)
        self._changed()
        return self

    def add_hopping_matrix(self, T, orbitals_i, orbitals_j, R):
        """Matrix-valued hopping from a group of orbitals (home cell) to a group
        in cell ``R``; shape (n_i*ns, n_j*ns). The h.c. is added automatically."""
        ii, jj = self._orb_list(orbitals_i), self._orb_list(orbitals_j)
        R = self._R(R)
        if not R.any():
            raise ValueError("R = 0 matrices belong in add_onsite_matrix")
        T = np.asarray(T, complex)
        ns = self.ns
        for a, ia in enumerate(ii):
            for b, jb in enumerate(jj):
                blk = T[a * ns:(a + 1) * ns, b * ns:(b + 1) * ns]
                if np.abs(blk).max() > 0:
                    self.add_hopping(blk, ia, jb, R)
        return self

    # ---- neighbours -------------------------------------------------------
    def neighbors(self, shell=1, orbitals_i=None, orbitals_j=None, tol=1e-5, search=3,
                  by_site=True):
        """List the bonds of a neighbour shell as ``(i, j, R, d)`` tuples.

        Each unordered bond appears once. Shells are ranked by distance between
        *different* positions (orbitals sitting at the same place are skipped).
        ``shell`` can be an int or a list of ints.
        """
        shells = [shell] if np.isscalar(shell) else list(shell)
        bonds = self._all_bonds(search, tol)
        dists = sorted({round(b[3], 5) for b in bonds})
        wanted = {dists[s - 1] for s in shells if s - 1 < len(dists)}
        si = set(self._orb_list(orbitals_i)) if orbitals_i is not None else None
        sj = set(self._orb_list(orbitals_j)) if orbitals_j is not None else None
        out = []
        for i, j, R, dist, d in bonds:
            if round(dist, 5) not in wanted:
                continue
            if si is not None and sj is not None:
                if not ((i in si and j in sj) or (j in si and i in sj)):
                    continue
                if not (i in si and j in sj):  # orient the bond i -> j
                    i, j, R, d = j, i, -R, -d
            out.append((i, j, R, d))
        return out

    def _all_bonds(self, search, tol):
        A, P = self.lat.vectors, self.lat.positions
        rng = range(-search, search + 1)
        Rs = [np.array(R, int) for R in itertools.product(rng, repeat=self.dim)] or [np.zeros(0, int)]
        bonds = []
        for R in Rs:
            shift = R @ A if self.dim else 0.0
            for i in range(self.norb):
                for j in range(self.norb):
                    d = P[j] + shift - P[i]
                    dist = np.linalg.norm(d)
                    if dist < tol:
                        continue
                    # keep one orientation of each unordered bond
                    if i > j or (i == j and tuple(R) < tuple(-R)):
                        continue
                    bonds.append((i, j, R, dist, d))
        return bonds

    def add_neighbors(self, t, shell=1, orbitals_i=None, orbitals_j=None, **kw):
        """Add hopping ``t`` (scalar/spin matrix or a callable ``t(d)``) on all
        bonds of the given neighbour shell(s). Returns the number of bonds."""
        bonds = self.neighbors(shell, orbitals_i, orbitals_j, **kw)
        for i, j, R, d in bonds:
            amp = t(d) if callable(t) else t
            self.add_hopping(amp, i, j, R)
        return len(bonds)

    # ---- spin-dependent helpers ---------------------------------------------
    def add_exchange(self, J, direction=(0, 0, 1), orbitals="all"):
        """On-site exchange / Zeeman term ``J * (m . sigma)``."""
        self._need_spin()
        m = np.asarray(direction, float)
        m = m / np.linalg.norm(m)
        return self.add_onsite(J * (m[0] * ops.sx + m[1] * ops.sy + m[2] * ops.sz), orbitals)

    add_zeeman = add_exchange

    def add_rashba(self, lam, shell=1, orbitals_i=None, orbitals_j=None, normal=(0, 0, 1)):
        """Rashba hopping ``i lam (sigma x d_hat) . n`` on a neighbour shell."""
        self._need_spin()
        n = np.asarray(normal, float)

        def amp(d):
            d3 = np.zeros(3)
            d3[: len(d)] = d
            d3 /= np.linalg.norm(d3)
            sv = [ops.sx, ops.sy, ops.sz]
            c = np.cross(np.eye(3), d3)  # rows: e_a x d
            M = sum(n[a] * sum(c[b][a] * sv[b] for b in range(3)) for a in range(3))
            return 1j * lam * M
        return self.add_neighbors(amp, shell, orbitals_i, orbitals_j)

    def add_slater_koster(self, params, shell=1, sites=None, **kw):
        """Slater-Koster hoppings between orbitals of different sites.

        ``params`` is a dict (``sss, sps, pps, ppp, sds, pds, pdp, dds, ddp, ddd``)
        or a callable ``params(site_i, site_j, distance) -> dict``. Orbital kinds
        come from the names (``"Mo:dxy"``). ``sites`` optionally restricts the
        bonds to a pair of site labels, e.g. ``("Mo", "S")``.
        """
        bonds = self.neighbors(shell, **kw)
        n = 0
        for i, j, R, d in bonds:
            si, sj = self.lat.site(i), self.lat.site(j)
            if sites is not None and {si, sj} != set(sites) and (si, sj) != tuple(sites):
                continue
            p = params(si, sj, np.linalg.norm(d)) if callable(params) else params
            v = ops.slater_koster(self.lat.kind(i), self.lat.kind(j), d, p)
            if abs(v) > 1e-14:
                self.add_hopping(v, i, j, R)
                n += 1
        return n

    def apply_peierls(self, phase):
        """Multiply every hopping by ``exp(i * phase(r_i, r_j))`` where r are
        Cartesian positions of the two ends (use for magnetic fields)."""
        A, P = self.lat.vectors, self.lat.positions
        for term in self._terms:
            ri = P[term.i]
            rj = P[term.j] + (term.R @ A if self.dim else 0.0)
            term.T = term.T * np.exp(1j * phase(ri, rj))
        self._changed()
        return self

    def _need_spin(self):
        if not self.spinful:
            raise ValueError("this needs a spinful model (Model(..., spinful=True))")

    # ------------------------------------------------------------------
    # compiled arrays
    # ------------------------------------------------------------------
    def _compile(self):
        if self._cache is not None:
            return self._cache
        A, P = self.lat.vectors, self.lat.positions
        nt = len(self._terms)
        I = np.array([t.i for t in self._terms], int)
        J = np.array([t.j for t in self._terms], int)
        R = np.array([t.R for t in self._terms], float).reshape(nt, self.dim)
        T = np.array([t.T for t in self._terms], complex).reshape(nt, self.ns, self.ns)
        Rc = R @ A if self.dim else np.zeros((nt, self.lat.D))
        dc = Rc + P[J] - P[I] if nt else np.zeros((0, self.lat.D))
        self._cache = dict(I=I, J=J, R=R, T=T, Rc=Rc, dc=dc)
        return self._cache

    def _kcart(self, k, cartesian=False):
        """Normalise k input.  Accepted shapes: one point (dim,), many points
        (nk, dim); for 1D models a scalar is one point and a 1D array is a list
        of points."""
        k = np.asarray(k, float)
        if self.dim == 0:
            n = 1 if k.ndim <= 1 else len(k)
            return np.zeros((n, self.lat.D)), k.ndim <= 1
        ncomp = self.lat.D if cartesian else self.dim
        if k.ndim == 0:
            single, k = True, k.reshape(1, 1)
        elif k.ndim == 1:
            if self.dim == 1 and not cartesian:
                single, k = False, k.reshape(-1, 1)
            else:
                single, k = True, k.reshape(1, -1)
        else:
            single = False
        if k.shape[-1] != ncomp:
            raise ValueError(f"k needs {ncomp} components, got shape {k.shape}")
        kc = k if cartesian else k @ self.lat.reciprocal
        return kc, single

    def hamiltonian(self, k=None, gauge="atomic", cartesian=False):
        """Bloch Hamiltonian H(k), shape (nk, nb, nb) or (nb, nb) for one k."""
        if k is None:
            k = np.zeros(self.dim)
        kc, single = self._kcart(k, cartesian)
        c = self._compile()
        nk, no, ns = len(kc), self.norb, self.ns
        H = np.zeros((nk, no, ns, no, ns), complex)
        if len(c["I"]):
            disp = c["dc"] if gauge == "atomic" else c["Rc"]
            ph = np.exp(1j * kc @ disp.T)  # (nk, nt)
            vals = ph[:, :, None, None] * c["T"][None]  # (nk, nt, ns, ns)
            np.add.at(H, (slice(None), c["I"], slice(None), c["J"], slice(None)),
                      vals.transpose(1, 0, 2, 3))
        H = H.reshape(nk, no * ns, no * ns)
        H = H + H.conj().transpose(0, 2, 1)
        for i in range(no):
            H[:, i * ns:(i + 1) * ns, i * ns:(i + 1) * ns] += self._onsite[i]
        return H[0] if single else H

    hk = hamiltonian

    def hamiltonian_sparse(self, k=None, gauge="atomic"):
        """Bloch Hamiltonian at a single k as a scipy.sparse CSR matrix
        (use for large supercells, KPM, sparse eigensolvers)."""
        import scipy.sparse as sp
        if k is None:
            k = np.zeros(self.dim)
        kc, _ = self._kcart(k)
        kc = kc[0]
        c = self._compile()
        ns, no = self.ns, self.norb
        rows, cols, vals = [], [], []
        if len(c["I"]):
            disp = c["dc"] if gauge == "atomic" else c["Rc"]
            ph = np.exp(1j * disp @ kc)
            a, b = np.meshgrid(np.arange(ns), np.arange(ns), indexing="ij")
            for I, J, p, T in zip(c["I"], c["J"], ph, c["T"]):
                v = (p * T).ravel()
                r = (I * ns + a).ravel(); cc = (J * ns + b).ravel()
                rows += [r, cc]; cols += [cc, r]; vals += [v, v.conj()]
        for i in range(no):
            blk = self._onsite[i]
            a, b = np.meshgrid(np.arange(ns), np.arange(ns), indexing="ij")
            rows.append((i * ns + a).ravel()); cols.append((i * ns + b).ravel()); vals.append(blk.ravel())
        N = no * ns
        if not rows:
            return sp.csr_matrix((N, N), dtype=complex)
        H = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(N, N)).tocsr()
        H.sum_duplicates()
        return H

    def velocity_sparse(self, axis, k=None, gauge="atomic"):
        """Sparse velocity operator dH/dk_axis (axis 0/1/2 or 'x'/'y'/'z')."""
        import scipy.sparse as sp
        ax = ops._axis(axis)
        if k is None:
            k = np.zeros(self.dim)
        kc, _ = self._kcart(k)
        kc = kc[0]
        c = self._compile()
        ns, no = self.ns, self.norb
        disp = c["dc"] if gauge == "atomic" else c["Rc"]
        rows, cols, vals = [], [], []
        a, b = np.meshgrid(np.arange(ns), np.arange(ns), indexing="ij")
        for I, J, d, T in zip(c["I"], c["J"], disp, c["T"]):
            M = 1j * d[ax] * np.exp(1j * d @ kc) * T
            r = (I * ns + a).ravel(); cc = (J * ns + b).ravel()
            rows += [r, cc]; cols += [cc, r]; vals += [M.ravel(), M.conj().ravel()]
        N = no * ns
        if not rows:
            return sp.csr_matrix((N, N), dtype=complex)
        V = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(N, N)).tocsr()
        V.sum_duplicates()
        return V

    def velocity(self, k=None, gauge="atomic", cartesian=False):
        """Velocity operators dH/dk_a (a = Cartesian x, y, z, ...),
        shape (D, nk, nb, nb) (or (D, nb, nb) for one k)."""
        if k is None:
            k = np.zeros(self.dim)
        kc, single = self._kcart(k, cartesian)
        c = self._compile()
        nk, no, ns, D = len(kc), self.norb, self.ns, self.lat.D
        V = np.zeros((D, nk, no * ns, no * ns), complex)
        if len(c["I"]):
            disp = c["dc"] if gauge == "atomic" else c["Rc"]
            ph = np.exp(1j * kc @ disp.T)
            for a in range(D):
                Va = np.zeros((nk, no, ns, no, ns), complex)
                vals = (1j * disp[:, a])[None, :, None, None] * ph[:, :, None, None] * c["T"][None]
                np.add.at(Va, (slice(None), c["I"], slice(None), c["J"], slice(None)),
                          vals.transpose(1, 0, 2, 3))
                Va = Va.reshape(nk, no * ns, no * ns)
                V[a] = Va + Va.conj().transpose(0, 2, 1)
        return V[:, 0] if single else V

    def eig(self, k=None, vectors=True, gauge="atomic"):
        """Eigenvalues (and eigenvectors as columns) at k-points."""
        H = self.hamiltonian(k, gauge=gauge)
        if vectors:
            return np.linalg.eigh(H)
        return np.linalg.eigvalsh(H)

    def energies(self, k=None):
        return self.eig(k, vectors=False)

    # ------------------------------------------------------------------
    # operators
    # ------------------------------------------------------------------
    def spin_operator(self, axis="z"):
        """Spin s_a = sigma_a / 2 on the full basis (hbar = 1)."""
        self._need_spin()
        s = ops.pauli(axis) / 2
        return np.kron(np.eye(self.norb), s)

    def orbital_operator(self, axis="z"):
        """Atomic orbital angular momentum L_a built from orbital kinds
        (p and d orbitals on the same site)."""
        a = ops._axis(axis)
        no = self.norb
        L = np.zeros((no, no), complex)
        P = self.lat.positions
        for i in range(no):
            for j in range(no):
                if self.lat.site(i) == self.lat.site(j) and np.allclose(P[i], P[j]):
                    L[i, j] = ops.orbital_angular_momentum(self.lat.kind(i), self.lat.kind(j), a)
        return np.kron(L, np.eye(self.ns))

    def projector(self, orbitals):
        """Diagonal projector onto a set of orbitals (all spins)."""
        p = np.zeros(self.nbands)
        for i in self._orb_list(orbitals):
            p[i * self.ns:(i + 1) * self.ns] = 1
        return np.diag(p)

    # ------------------------------------------------------------------
    # geometry transformations (implemented in geometry.py)
    # ------------------------------------------------------------------
    def supercell(self, P):
        from .geometry import supercell
        return supercell(self, P)

    def slab(self, direction, n, periodic=False):
        from .geometry import cut
        return cut(self, direction, n, periodic=periodic)

    ribbon = slab

    def finite(self, n):
        from .geometry import finite
        return finite(self, n)

    def remove_orbitals(self, orbitals):
        from .geometry import remove_orbitals
        return remove_orbitals(self, orbitals)

    # ------------------------------------------------------------------
    # convenience wrappers
    # ------------------------------------------------------------------
    def bands(self, path=None, n=200):
        from .solvers import bands
        return bands(self, path, n)

    def plot_bands(self, path=None, n=200, **kw):
        from .plotting import plot_bands
        return plot_bands(self, path, n, **kw)

    def plot(self, **kw):
        from .plotting import plot_lattice
        return plot_lattice(self, **kw)

    def summary(self) -> str:
        nterms = len(self._terms)
        lines = [f"hopforge Model '{self.name}'",
                 f"  periodic dims : {self.dim}   (space dim {self.lat.D})",
                 f"  orbitals      : {self.norb}  {self.lat.names}",
                 f"  spinful       : {self.spinful}  -> {self.nbands} bands",
                 f"  hopping terms : {nterms} (+ h.c.)",
                 f"  special points: {list(self.special_points)}"]
        return "\n".join(lines)

    def __repr__(self):
        return (f"<Model '{self.name}': dim={self.dim}, orbitals={self.norb}, "
                f"spinful={self.spinful}, bands={self.nbands}>")

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def hoppings(self):
        """List of (i, j, R, T) for inspection."""
        return [(t.i, t.j, t.R.copy(), t.T.copy()) for t in self._terms]

    def onsite(self):
        return self._onsite.copy()

    def to_wannier90_hr(self, filename):
        """Write a wannier90_hr.dat style file (spin index inner)."""
        c = self._compile()
        Rset = {tuple(r) for r in c["R"].astype(int)} | {tuple(-r) for r in c["R"].astype(int)}
        Rset.add(tuple([0] * self.dim))
        Rs = sorted(Rset)
        nb = self.nbands
        mats = {R: np.zeros((nb, nb), complex) for R in Rs}
        ns = self.ns
        for t in self._terms:
            R = tuple(t.R)
            mats[R][t.i * ns:(t.i + 1) * ns, t.j * ns:(t.j + 1) * ns] += t.T
            mR = tuple(-t.R)
            mats[mR][t.j * ns:(t.j + 1) * ns, t.i * ns:(t.i + 1) * ns] += t.T.conj().T
        for i in range(self.norb):
            mats[tuple([0] * self.dim)][i * ns:(i + 1) * ns, i * ns:(i + 1) * ns] += self._onsite[i]
        with open(filename, "w") as f:
            f.write(f"written by hopforge ({self.name})\n{nb}\n{len(Rs)}\n")
            for n in range(len(Rs)):
                f.write("    1" + ("\n" if (n + 1) % 15 == 0 else ""))
            f.write("\n")
            for R in Rs:
                R3 = list(R) + [0] * (3 - len(R))
                for b in range(nb):
                    for a in range(nb):
                        v = mats[R][a, b]
                        f.write(f"{R3[0]:5d}{R3[1]:5d}{R3[2]:5d}{a + 1:5d}{b + 1:5d}"
                                f"{v.real:12.6f}{v.imag:12.6f}\n")
        return filename
