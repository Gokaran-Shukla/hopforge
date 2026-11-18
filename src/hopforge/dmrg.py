"""Matrix-product states and DMRG for 1D (and narrow quasi-1D) interacting systems.

Reaches system sizes far beyond exact diagonalisation (hundreds of sites) for
gapped and critical chains, with controlled truncation error.

    >>> import hopforge as hf
    >>> H = hf.dmrg.heisenberg(100)                       # MPO
    >>> res = hf.dmrg.dmrg(H, chi=[20, 50, 100, 100], sweeps=6)
    >>> res.energy, res.entropy                            # E0, S(bond)
    >>> res.mps.expect(hf.dmrg.SPIN["Sz"], site=50)
    >>> res.mps.correlation(hf.dmrg.SPIN["Sz"], hf.dmrg.SPIN["Sz"], 10, 30)

Building blocks
---------------
* ``Sites``: local operator sets (``SPIN`` for S = 1/2, ``FERMION`` for
  spinless fermions, ``ELECTRON`` for spinful electrons). Fermionic
  operators are flagged; Jordan-Wigner strings are inserted automatically.
* ``build_mpo(L, sites, terms)``: any Hamiltonian written as a list of
  ``(coefficient, [(op_name, site), ...])`` terms (operator products in
  order), compressed to the optimal bond dimension by SVD.
* Ready-made MPOs: ``heisenberg`` (XXZ, J1-J2, field), ``tfim``,
  ``hubbard`` (Hubbard / extended Hubbard), ``spinless_fermions`` (t-V),
  ``from_model`` (any 1D hopforge Model + Hubbard U / nearest-neighbour V).
* ``dmrg(H, chi, sweeps, ...)``: two-site DMRG with a bond-dimension schedule,
  truncation cutoff, optional penalty projection for excited states.
* ``MPS``: norm, overlap, ``expect``, ``correlation``, ``entanglement``.

References: White, PRL 69, 2863 (1992); Schollwoeck, Ann. Phys. 326, 96 (2011).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

# =============================================================================
# local operators
# =============================================================================


class Sites:
    """Local Hilbert space: operators (d x d) and which of them are fermionic."""

    def __init__(self, ops, fermionic=(), parity=None, name=""):
        self.ops = {k: np.asarray(v, complex) for k, v in ops.items()}
        self.d = next(iter(self.ops.values())).shape[0]
        self.ops.setdefault("Id", np.eye(self.d))
        self.fermionic = set(fermionic)
        self.parity = None if parity is None else np.asarray(parity, complex)
        self.name = name

    def __getitem__(self, k):
        return self.ops[k]


_sp = np.array([[0, 1], [0, 0]], complex)
SPIN = Sites({"Sz": np.diag([0.5, -0.5]), "Sp": _sp, "Sm": _sp.T,
              "Sx": 0.5 * np.array([[0, 1], [1, 0]]), "Sy": 0.5 * np.array([[0, -1j], [1j, 0]]),
              "X": np.array([[0, 1], [1, 0]]), "Z": np.diag([1.0, -1.0])}, name="spin-1/2")

# spin-1: basis |+1>, |0>, |-1>
_s1p = np.sqrt(2) * np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], complex)
SPIN1 = Sites({"Sz": np.diag([1.0, 0, -1]), "Sp": _s1p, "Sm": _s1p.T,
               "string": np.diag([-1.0, 1, -1])}, name="spin-1")

# spinless fermion: basis |0>, |1>
_c = np.array([[0, 1], [0, 0]], complex)
FERMION = Sites({"c": _c, "cdag": _c.T, "n": np.diag([0.0, 1.0]), "F": np.diag([1.0, -1.0])},
                fermionic=("c", "cdag"), parity=np.diag([1.0, -1.0]), name="spinless fermion")

# electron: basis |0>, |up>, |dn>, |updn> with |updn> = c+_up c+_dn |0>
_cu = np.zeros((4, 4), complex); _cu[0, 1] = 1; _cu[2, 3] = 1
_cd = np.zeros((4, 4), complex); _cd[0, 2] = 1; _cd[1, 3] = -1
_nu = np.diag([0.0, 1, 0, 1]); _nd = np.diag([0.0, 0, 1, 1])
ELECTRON = Sites({"cup": _cu, "cdagup": _cu.conj().T, "cdn": _cd, "cdagdn": _cd.conj().T,
                  "nup": _nu, "ndn": _nd, "n": _nu + _nd, "nupndn": _nu @ _nd,
                  "Sz": 0.5 * (_nu - _nd), "Sp": _cu.conj().T @ _cd, "Sm": _cd.conj().T @ _cu,
                  "F": np.diag([1.0, -1, -1, 1])},
                 fermionic=("cup", "cdagup", "cdn", "cdagdn"), parity=np.diag([1.0, -1, -1, 1]),
                 name="electron")


# =============================================================================
# MPO construction
# =============================================================================


class MPO:
    """Matrix-product operator; W[i] has shape (wL, wR, d, d) = (left, right, out, in)."""

    def __init__(self, W, sites=None, name=""):
        self.W = [np.asarray(w, complex) for w in W]
        self.L = len(W)
        self.sites = sites
        self.name = name

    @property
    def bond_dims(self):
        return [w.shape[1] for w in self.W[:-1]]

    def to_dense(self):
        """Full matrix (small L only)."""
        T = self.W[0][0]                           # (wR, d, d)
        for w in self.W[1:]:
            T = np.einsum("aij,abkl->bikjl", T, w)
            b, i, k, j, l = T.shape
            T = T.reshape(b, i * k, j * l)
        return T[0]

    def __add__(self, other):
        return _mpo_sum([self, other])

    def __mul__(self, c):
        W = [w.copy() for w in self.W]
        W[0] = W[0] * c
        return MPO(W, self.sites, self.name)

    __rmul__ = __mul__


def _product_mpo(L, sites, coef, ops):
    """Product operator coef * O1 O2 ... with Jordan-Wigner strings."""
    d = sites.d
    mats = [np.eye(d, dtype=complex) for _ in range(L)]
    for name, s in ops:
        o = sites[name]
        ferm = name in sites.fermionic
        for i in range(L):
            if i == s:
                mats[i] = mats[i] @ o
            elif ferm and i < s:
                mats[i] = mats[i] @ sites.parity
    mats[0] = mats[0] * coef
    return [m[None, None] for m in mats]


def _mpo_sum(mpos):
    L = mpos[0].L
    W = []
    for i in range(L):
        blocks = [m.W[i] for m in mpos]
        d = blocks[0].shape[2]
        if i == 0:
            W.append(np.concatenate(blocks, axis=1))
        elif i == L - 1:
            W.append(np.concatenate(blocks, axis=0))
        else:
            wl = sum(b.shape[0] for b in blocks); wr = sum(b.shape[1] for b in blocks)
            w = np.zeros((wl, wr, d, d), complex)
            a = b_ = 0
            for blk in blocks:
                w[a:a + blk.shape[0], b_:b_ + blk.shape[1]] = blk
                a += blk.shape[0]; b_ += blk.shape[1]
            W.append(w)
    return MPO(W, mpos[0].sites)


def compress_mpo(mpo, tol=1e-12):
    """Exact SVD compression of an MPO (left-to-right then right-to-left)."""
    W = [w.copy() for w in mpo.W]
    L = len(W)
    for sweep in (range(L - 1), range(L - 1, 0, -1)):
        for i in sweep:
            if sweep.step is None or sweep.step > 0:
                wl, wr, d, _ = W[i].shape
                M = W[i].transpose(0, 2, 3, 1).reshape(wl * d * d, wr)
                U, S, Vh = np.linalg.svd(M, full_matrices=False)
                keep = max(1, int(np.sum(S > tol * S[0])))
                W[i] = U[:, :keep].reshape(wl, d, d, keep).transpose(0, 3, 1, 2)
                R = S[:keep, None] * Vh[:keep]
                W[i + 1] = np.einsum("ab,bcij->acij", R, W[i + 1])
            else:
                wl, wr, d, _ = W[i].shape
                M = W[i].reshape(wl, wr * d * d)
                U, S, Vh = np.linalg.svd(M, full_matrices=False)
                keep = max(1, int(np.sum(S > tol * S[0])))
                W[i] = Vh[:keep].reshape(keep, wr, d, d)
                Lm = U[:, :keep] * S[:keep]
                W[i - 1] = np.einsum("abij,bc->acij", W[i - 1], Lm)
    return MPO(W, mpo.sites, mpo.name)


def build_mpo(L, sites, terms, tol=1e-12, chunk=60):
    """MPO of H = sum_terms coef * O_1 O_2 ... from a list
    ``[(coef, [(op, site), ...]), ...]`` (operators applied right to left as
    written: the product is O_1 @ O_2 @ ...). Fermionic operators get
    Jordan-Wigner strings. Compressed exactly to the minimal bond dimension."""
    terms = [t for t in terms if abs(t[0]) > 0]
    if not terms:
        raise ValueError("no terms")
    acc = None
    for s in range(0, len(terms), chunk):
        part = _mpo_sum([MPO(_product_mpo(L, sites, c, ops), sites) for c, ops in terms[s:s + chunk]])
        part = compress_mpo(part, tol)
        acc = part if acc is None else compress_mpo(_mpo_sum([acc, part]), tol)
    acc.sites = sites
    return acc


# ---------------------------------------------------------------- model MPOs
def heisenberg(L, J=1.0, Jz=None, J2=0.0, h=0.0, pbc=False, spin=0.5):
    """XXZ chain (spin 1/2 or 1): J (SxSx + SySy) + Jz SzSz on nearest neighbours,
    J2 (isotropic) on next-nearest neighbours, field -h Sz."""
    Jz = J if Jz is None else Jz
    sites = SPIN if spin == 0.5 else SPIN1
    T = []
    bonds = [(i, i + 1) for i in range(L - 1)] + ([(L - 1, 0)] if pbc else [])
    for i, j in bonds:
        T += [(J / 2, [("Sp", i), ("Sm", j)]), (J / 2, [("Sm", i), ("Sp", j)]), (Jz, [("Sz", i), ("Sz", j)])]
    if J2:
        for i in range(L - 2):
            T += [(J2 / 2, [("Sp", i), ("Sm", i + 2)]), (J2 / 2, [("Sm", i), ("Sp", i + 2)]),
                  (J2, [("Sz", i), ("Sz", i + 2)])]
    if h:
        T += [(-h, [("Sz", i)]) for i in range(L)]
    m = build_mpo(L, sites, T); m.name = f"Heisenberg S={spin} L={L}"
    return m


def aklt(L, beta=1.0 / 3):
    """Spin-1 bilinear-biquadratic chain sum [S.S + beta (S.S)^2]; beta = 1/3 is
    the AKLT point with exact energy -2/3 per bond and string order -4/9
    (Affleck, Kennedy, Lieb & Tasaki, PRL 59, 799 (1987))."""
    pairs = [(1.0, "Sz", "Sz"), (0.5, "Sp", "Sm"), (0.5, "Sm", "Sp")]
    T = []
    for i in range(L - 1):
        for c, a, b in pairs:
            T.append((c, [(a, i), (b, i + 1)]))
        for c1, a1, b1 in pairs:
            for c2, a2, b2 in pairs:
                T.append((beta * c1 * c2, [(a1, i), (a2, i), (b1, i + 1), (b2, i + 1)]))
    m = build_mpo(L, SPIN1, T); m.name = f"AKLT L={L}"
    return m


def tfim(L, J=1.0, h=1.0, pbc=False, hz=0.0):
    """Transverse-field Ising chain H = -J sum Z_i Z_{i+1} - h sum X_i - hz sum Z_i
    (Pauli matrices). A tiny ``hz`` selects a symmetry-broken ground state in the
    ordered phase instead of the (exponentially close) cat state."""
    T = [(-J, [("Z", i), ("Z", i + 1)]) for i in range(L - 1)] + [(-h, [("X", i)]) for i in range(L)]
    if hz:
        T += [(-hz, [("Z", i)]) for i in range(L)]
    if pbc:
        T.append((-J, [("Z", L - 1), ("Z", 0)]))
    m = build_mpo(L, SPIN, T); m.name = f"TFIM L={L}"
    return m


def spinless_fermions(L, t=1.0, V=0.0, mu=0.0):
    """t-V chain: -t (c+_i c_{i+1} + h.c.) + V n_i n_{i+1} - mu n_i."""
    T = []
    for i in range(L - 1):
        T += [(-t, [("cdag", i), ("c", i + 1)]), (-t, [("cdag", i + 1), ("c", i)])]
        if V:
            T.append((V, [("n", i), ("n", i + 1)]))
    if mu:
        T += [(-mu, [("n", i)]) for i in range(L)]
    m = build_mpo(L, FERMION, T); m.name = f"t-V chain L={L}"
    return m


def hubbard(L, t=1.0, U=4.0, V=0.0, mu=None, t2=0.0):
    """1D (extended) Hubbard chain with open boundaries:
    -t sum (c+_{i s} c_{i+1 s} + h.c.) - t2 (next neighbours) + U sum n_up n_dn
    + V sum n_i n_{i+1} - mu sum n_i. ``mu=None`` uses the particle-hole
    symmetric value U/2 + V (half filling)."""
    mu = U / 2 + V if mu is None else mu
    T = []
    for r, tt in ((1, t), (2, t2)):
        if not tt:
            continue
        for i in range(L - r):
            for s in ("up", "dn"):
                T += [(-tt, [("cdag" + s, i), ("c" + s, i + r)]), (-tt, [("cdag" + s, i + r), ("c" + s, i)])]
    T += [(U, [("nupndn", i)]) for i in range(L)]
    T += [(-mu, [("n", i)]) for i in range(L)]
    if V:
        T += [(V, [("n", i), ("n", i + 1)]) for i in range(L - 1)]
    m = build_mpo(L, ELECTRON, T); m.name = f"Hubbard L={L}"
    return m


def from_model(model, L, U=0.0, V=0.0, mu=0.0):
    """MPO for L unit cells of a 1D hopforge Model (open boundaries), with
    on-site Hubbard U (spinful models: U n_up n_dn on every orbital) and a
    density interaction V between the same orbital in neighbouring cells.
    Orbitals are ordered cell by cell; spinful models use ELECTRON sites,
    spinless models FERMION sites.

    Hopping terms follow hopforge's convention <i, 0|H|j, R> = T (h.c. added)."""
    if model.dim != 1:
        raise ValueError("from_model needs a 1D model (make quasi-1D systems with model.ribbon/supercell)")
    norb, spinful = model.norb, model.spinful
    sites = ELECTRON if spinful else FERMION
    lab = ("up", "dn")
    N = L * norb

    def cd(g, a):
        return ("cdag" + lab[a], g) if spinful else ("cdag", g)

    def c_(g, b):
        return ("c" + lab[b], g) if spinful else ("c", g)

    T = []
    for c in range(L):
        for i in range(norb):
            O = np.atleast_2d(model._onsite[i])
            g = c * norb + i
            for a in range(O.shape[0]):
                for b in range(O.shape[1]):
                    if abs(O[a, b]) > 1e-14:
                        T.append((O[a, b], [cd(g, a), c_(g, b)]))
    for tm in model._terms:
        R = int(np.asarray(tm.R).ravel()[0])
        M = np.atleast_2d(tm.T)
        for c in range(L):
            if not 0 <= c + R < L:
                continue
            gi, gj = c * norb + tm.i, (c + R) * norb + tm.j
            for a in range(M.shape[0]):
                for b in range(M.shape[1]):
                    if abs(M[a, b]) > 1e-14:
                        T.append((M[a, b], [cd(gi, a), c_(gj, b)]))
                        T.append((np.conj(M[a, b]), [cd(gj, b), c_(gi, a)]))
    if U and spinful:
        T += [(U, [("nupndn", g)]) for g in range(N)]
    if V:
        T += [(V, [("n", g), ("n", g + norb)]) for g in range(N - norb)]
    if mu:
        T += [(-mu, [("n", g)]) for g in range(N)]
    m = build_mpo(N, sites, T); m.name = f"{model.name} x {L}"
    return m


def number_penalty(mpo, N0, weight=1.0):
    """H + weight (N - N0)^2 to target a particle-number sector."""
    L, sites = mpo.L, mpo.sites
    T = [(1.0, [("n", i), ("n", j)]) for i in range(L) for j in range(L)]
    T += [(-2 * N0, [("n", i)]) for i in range(L)] + [(N0 ** 2 / L, [("Id", i)]) for i in range(L)]
    pen = build_mpo(L, sites, T)
    return compress_mpo(_mpo_sum([mpo, pen * weight]))


# =============================================================================
# MPS
# =============================================================================


class MPS:
    """Matrix-product state; A[i] has shape (chiL, d, chiR)."""

    def __init__(self, A):
        self.A = [np.asarray(a, complex) for a in A]
        self.L = len(A)

    @classmethod
    def random(cls, L, d, chi=8, seed=0):
        rng = np.random.default_rng(seed)
        dims = [1] + [min(chi, d ** min(i, L - i)) for i in range(1, L)] + [1]
        A = [rng.normal(size=(dims[i], d, dims[i + 1])) + 0j for i in range(L)]
        m = cls(A)
        m.right_canonicalize()
        return m

    @classmethod
    def product(cls, states):
        """Product state from a list of local vectors."""
        return cls([np.asarray(s, complex)[None, :, None] for s in states])

    @property
    def bond_dims(self):
        return [a.shape[2] for a in self.A[:-1]]

    def right_canonicalize(self):
        for i in range(self.L - 1, 0, -1):
            a, d, b = self.A[i].shape
            Q, R = np.linalg.qr(self.A[i].reshape(a, d * b).T)
            self.A[i] = Q.T.reshape(-1, d, b)
            self.A[i - 1] = np.einsum("xdy,yz->xdz", self.A[i - 1], R.T)
        self.A[0] /= np.linalg.norm(self.A[0])

    def overlap(self, other):
        E = np.ones((1, 1))
        for a, b in zip(self.A, other.A):
            E = np.tensordot(a.conj(), np.tensordot(E, b, (1, 0)), ([0, 1], [0, 1]))
        return E[0, 0]

    def norm(self):
        return float(np.sqrt(abs(self.overlap(self))))

    def _sandwich(self, opmap):
        """<psi| prod_i O_i |psi> / <psi|psi> for a dict site -> matrix."""
        E = np.ones((1, 1))
        for i, a in enumerate(self.A):
            o = opmap.get(i)
            b = a if o is None else np.einsum("ij,xjz->xiz", o, a)
            E = np.tensordot(a.conj(), np.tensordot(E, b, (1, 0)), ([0, 1], [0, 1]))
        return E[0, 0] / self.overlap(self)

    def expect(self, op, site=None):
        """<O_site> (or the array over all sites if site is None)."""
        op = np.asarray(op, complex)
        if site is None:
            return np.array([self._sandwich({i: op}).real for i in range(self.L)])
        return self._sandwich({site: op})

    def correlation(self, opA, opB, i, j, string=None):
        """<A_i B_j> (i < j); ``string`` (e.g. the fermion parity) is inserted on
        the sites strictly between i and j for fermionic correlators."""
        if i == j:
            return self._sandwich({i: np.asarray(opA) @ np.asarray(opB)})
        m = {i: np.asarray(opA, complex), j: np.asarray(opB, complex)}
        if string is not None:
            for k in range(min(i, j) + 1, max(i, j)):
                m[k] = np.asarray(string, complex)
        return self._sandwich(m)

    def entanglement(self):
        """Von Neumann entropies S(l) of all bonds (left block = sites < l)."""
        A = [a.copy() for a in self.A]
        # left-canonicalise and read Schmidt values from the SVD of the remainder
        S_out = []
        for i in range(self.L - 1):
            a, d, b = A[i].shape
            U, S, Vh = np.linalg.svd(A[i].reshape(a * d, b), full_matrices=False)
            A[i] = U.reshape(a, d, -1)
            A[i + 1] = np.einsum("xy,ydz->xdz", S[:, None] * Vh, A[i + 1])
        # now sweep back computing true Schmidt values
        for i in range(self.L - 1, 0, -1):
            a, d, b = A[i].shape
            U, S, Vh = np.linalg.svd(A[i].reshape(a, d * b), full_matrices=False)
            A[i] = Vh.reshape(-1, d, b)
            A[i - 1] = np.einsum("xdy,yz->xdz", A[i - 1], U * S)
            p = S ** 2 / np.sum(S ** 2)
            p = p[p > 1e-16]
            S_out.append(float(-np.sum(p * np.log(p))))
        return np.array(S_out[::-1])

    def energy(self, mpo):
        E = np.ones((1, 1, 1))
        for a, w in zip(self.A, mpo.W):
            E = np.einsum("xwy,xsz,wvst,ytu->zvu", E, a.conj(), w, a, optimize=True)
        return (E[0, 0, 0] / self.overlap(self)).real


# =============================================================================
# DMRG
# =============================================================================


@dataclass
class DMRGResult:
    energy: float
    mps: MPS
    energies: list = field(default_factory=list)      # energy after each sweep
    truncation: list = field(default_factory=list)    # max discarded weight per sweep
    entropy: np.ndarray = None                        # S(bond) at the end
    chi: list = field(default_factory=list)

    def __repr__(self):
        return (f"DMRGResult(E0 = {self.energy:.10f}, sweeps = {len(self.energies)}, "
                f"max chi = {max(self.mps.bond_dims) if self.mps.L > 1 else 1}, "
                f"trunc = {self.truncation[-1] if self.truncation else 0:.1e})")


def _left_env(E, A, W):
    T = np.tensordot(E, A, (2, 0))                      # x w t u
    T = np.tensordot(T, W, ([1, 2], [0, 3]))            # x u v s
    T = np.tensordot(T, A.conj(), ([0, 3], [0, 1]))     # u v z
    return T.transpose(2, 1, 0)


def _right_env(E, B, W):
    T = np.tensordot(B, E, (2, 2))                      # y t z v
    T = np.tensordot(T, W, ([1, 3], [3, 1]))            # y z w s
    T = np.tensordot(T, B.conj(), ([1, 3], [2, 1]))     # y w x
    return T.transpose(2, 1, 0)


def _ov_left(E, A, P):
    return np.tensordot(A.conj(), np.tensordot(E, P, (1, 0)), ([0, 1], [0, 1]))


def _ov_right(E, B, P):
    return np.tensordot(B.conj(), np.tensordot(P, E, (2, 1)), ([1, 2], [1, 2]))


def dmrg(mpo, chi=64, sweeps=8, cutoff=1e-10, tol=1e-9, psi0=None, orthogonal_to=(), weight=None,
         seed=0, lanczos_tol=1e-10, verbose=False):
    """Two-site DMRG ground state of an MPO.

    chi       : int or a list (bond dimension per sweep; the last value is reused)
    sweeps    : maximum number of full (left->right->left) sweeps
    cutoff    : discarded weight per SVD
    tol       : stop when the energy changes by less than tol between sweeps
    orthogonal_to : previously found MPS (excited states via a penalty
                    weight * sum |phi><phi|; weight defaults to 10 x |E| scale)
    """
    L = mpo.L
    d = mpo.W[0].shape[2]
    sched = list(np.atleast_1d(chi))
    psi = MPS.random(L, d, min(8, sched[0]), seed) if psi0 is None else MPS([a.copy() for a in psi0.A])
    real = all(np.abs(w.imag).max() < 1e-14 for w in mpo.W) and all(np.abs(o.A[0].imag).max() < 1e-14 for o in orthogonal_to)
    dtype = float if real else complex
    W_use = [w.real.copy() if real else w for w in mpo.W]
    psi.A = [a.real.copy() if real else a for a in psi.A]
    psi.right_canonicalize()
    ortho = [MPS([a.copy() for a in o.A]) for o in orthogonal_to]
    if real:
        for o in ortho:
            o.A = [a.real.copy() for a in o.A]
    if ortho and weight is None:
        weight = 10.0 * (1 + abs(sum(np.abs(w).max() for w in mpo.W)))
    A, W = psi.A, W_use
    LE = [None] * (L + 1); RE = [None] * (L + 1)
    LE[0] = np.ones((1, 1, 1)); RE[L] = np.ones((1, 1, 1))
    LO = [[None] * (L + 1) for _ in ortho]; RO = [[None] * (L + 1) for _ in ortho]
    for k in range(len(ortho)):
        LO[k][0] = np.ones((1, 1)); RO[k][L] = np.ones((1, 1))
    for i in range(L - 1, 0, -1):
        RE[i] = _right_env(RE[i + 1], A[i], W[i])
        for k, o in enumerate(ortho):
            RO[k][i] = _ov_right(RO[k][i + 1], A[i], o.A[i])

    energies, truncs, chis = [], [], []
    E_old = np.inf
    entropy = np.zeros(max(L - 1, 0))

    def solve(i, chimax):
        Le, Re, W1, W2 = LE[i], RE[i + 2], W[i], W[i + 1]
        theta0 = np.einsum("xsy,ytz->xstz", A[i], A[i + 1])
        shape = theta0.shape
        projs = []
        for k, o in enumerate(ortho):
            P = np.einsum("xy,ysz,ztw,uw->xstu", LO[k][i], o.A[i], o.A[i + 1], RO[k][i + 2])
            projs.append(P.ravel())

        def mv(v):
            t = v.reshape(shape)
            r = np.tensordot(Le, t, (2, 0))                     # A w s t c
            r = np.tensordot(r, W1, ([1, 2], [0, 3]))           # A t c v S
            r = np.tensordot(r, W2, ([3, 1], [0, 3]))           # A c S u T
            r = np.tensordot(r, Re, ([1, 3], [2, 1]))           # A S T C
            r = r.ravel()
            for p in projs:
                r = r + weight * p * np.vdot(p, v)
            return r

        n = theta0.size
        if n <= 64:
            Hm = np.array([mv(e) for e in np.eye(n)]).T
            e, v = np.linalg.eigh(0.5 * (Hm + Hm.conj().T))
            E0, vec = e[0], v[:, 0]
        else:
            op = LinearOperator((n, n), matvec=mv, dtype=dtype)
            e, v = eigsh(op, k=1, which="SA", v0=theta0.ravel(), tol=lanczos_tol, ncv=min(n, 20))
            E0, vec = e[0], v[:, 0]
        theta = vec.reshape(shape[0] * shape[1], shape[2] * shape[3])
        U, S, Vh = np.linalg.svd(theta, full_matrices=False)
        S = S / np.linalg.norm(S)
        w = S ** 2
        tail = np.cumsum(w[::-1])[::-1]       # tail[k] = sum_{j >= k} w_j
        keep = min(chimax, len(S))
        while keep > 1 and tail[keep - 1] < cutoff:
            keep -= 1
        disc = float(tail[keep]) if keep < len(S) else 0.0
        p = w[:keep] / w[:keep].sum()
        p = p[p > 1e-16]
        ent = float(-np.sum(p * np.log(p)))
        return E0, U[:, :keep], S[:keep], Vh[:keep], disc, ent, shape

    for sw in range(sweeps):
        chimax = int(sched[min(sw, len(sched) - 1)])
        tmax = 0.0
        # left -> right
        for i in range(L - 1):
            E0, U, S, Vh, disc, ent, sh = solve(i, chimax)
            A[i] = U.reshape(sh[0], sh[1], -1)
            A[i + 1] = (S[:, None] * Vh).reshape(-1, sh[2], sh[3])
            LE[i + 1] = _left_env(LE[i], A[i], W[i])
            for k, o in enumerate(ortho):
                LO[k][i + 1] = _ov_left(LO[k][i], A[i], o.A[i])
            tmax = max(tmax, disc)
        # right -> left
        for i in range(L - 2, -1, -1):
            E0, U, S, Vh, disc, ent, sh = solve(i, chimax)
            A[i] = (U * S).reshape(sh[0], sh[1], -1)
            A[i + 1] = Vh.reshape(-1, sh[2], sh[3])
            RE[i + 1] = _right_env(RE[i + 2], A[i + 1], W[i + 1])
            for k, o in enumerate(ortho):
                RO[k][i + 1] = _ov_right(RO[k][i + 2], A[i + 1], o.A[i + 1])
            entropy[i] = ent
            tmax = max(tmax, disc)
        E = psi.energy(mpo) if ortho else float(np.real(E0))
        energies.append(E); truncs.append(tmax); chis.append(max(psi.bond_dims) if L > 1 else 1)
        if verbose:
            print(f"sweep {sw + 1}: E = {E:.12f}  chi = {chis[-1]}  trunc = {tmax:.1e}")
        if abs(E - E_old) < tol and sw >= min(len(sched), sweeps) - 1:
            break
        E_old = E
    return DMRGResult(energies[-1], psi, energies, truncs, entropy.copy(), chis)


def excited_states(mpo, n=2, **kw):
    """Lowest ``n`` states by successive DMRG runs with orthogonality penalties."""
    out = []
    for k in range(n):
        r = dmrg(mpo, orthogonal_to=[o.mps for o in out], seed=k, **kw)
        r.energy = r.mps.energy(mpo)
        out.append(r)
    return out


# =============================================================================
# exact references and analysis
# =============================================================================


def tfim_exact_obc(L, J=1.0, h=1.0):
    """Exact ground-state energy of the open chain H = -J sum Z_i Z_{i+1} - h sum X_i.

    Jordan-Wigner fermions give H = sum A_ij c+_i c_j + 1/2 sum (B_ij c+_i c+_j + h.c.) - hL
    with A_ii = 2h, A_{i,i+1} = -J, B_{i,i+1} = -J = -B_{i+1,i}; then
    E0 = -hL + Tr(A)/2 - (1/2) sum_k Lambda_k = -(1/2) sum_k Lambda_k with
    Lambda_k^2 the eigenvalues of (A - B)(A + B) (Lieb, Schultz & Mattis 1961)."""
    A = np.diag(np.full(L, 2.0 * h)); B = np.zeros((L, L))
    for i in range(L - 1):
        A[i, i + 1] = A[i + 1, i] = -J
        B[i, i + 1], B[i + 1, i] = -J, J
    lam = np.sqrt(np.abs(np.linalg.eigvalsh((A - B) @ (A + B))))
    return float(-0.5 * lam.sum())


def calabrese_cardy_obc(S, L, fit_range=(0.2, 0.8)):
    """Central charge from S(l) of an open chain: S = c/6 ln[(2L/pi) sin(pi l/L)] + const."""
    l = np.arange(1, L)
    x = np.log(2 * L / np.pi * np.sin(np.pi * l / L))
    sel = (l >= fit_range[0] * L) & (l <= fit_range[1] * L)
    c6, _ = np.polyfit(x[sel], np.asarray(S)[sel], 1)
    return 6 * c6
