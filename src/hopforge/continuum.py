"""Discretise continuum (k.p) Hamiltonians into tight-binding models,
in the spirit of ``kwant.continuum``.

>>> m = hf.discretize("A*(k_x*sigma_x + k_y*sigma_y) + (M - B*(k_x**2 + k_y**2))*sigma_z",
...                   coords="xy", grid=1.0, params=dict(A=1, B=1, M=0.5))

Rules (symmetric finite differences on a (hyper)cubic grid of spacing a):
    k_i      -> -i (psi(x+a) - psi(x-a)) / (2a)
    k_i^2    -> -(psi(x+a) - 2 psi(x) + psi(x-a)) / a^2
    k_i^n    -> products of the above (odd n: one first derivative)
Matrix symbols ``sigma_0, sigma_x, sigma_y, sigma_z`` and ``tau_*`` are
recognised as Pauli matrices acting on two spaces (``tau_z*sigma_x`` means
tau_z (x) sigma_x, tau outer); scalar terms get the identity automatically.
Parameters may be left symbolic in the string and given numerically in
``params``.
"""
from __future__ import annotations

import itertools

import numpy as np

from .model import Lattice, Model

_P = {
    "0": np.eye(2), "x": np.array([[0, 1], [1, 0]]), "y": np.array([[0, -1j], [1j, 0]]),
    "z": np.array([[1, 0], [0, -1]]),
}


def _sympy():
    try:
        import sympy
        return sympy
    except ImportError as e:  # pragma: no cover
        raise ImportError("hf.discretize needs sympy (pip install sympy)") from e


def _parse(expr, coords, params):
    """Return (sympy, k symbols, sympy Matrix H(k)).

    sigma_* and tau_* are treated as non-commuting symbols acting on two
    different spaces (tau outer, sigma inner); scalar terms get the identity."""
    sy = _sympy()
    ks = sy.symbols([f"k_{c}" for c in coords], commutative=True)
    if not isinstance(expr, str):
        return sy, ks, sy.Matrix(expr)
    ns = {f"k_{c}": k for c, k in zip(coords, ks)}
    nc = {}
    for space in ("tau", "sigma"):
        for c in _P:
            nc[f"{space}_{c}"] = sy.Symbol(f"{space}_{c}", commutative=False)
    ns.update(nc)
    ns["I"] = sy.I
    for kname, v in (params or {}).items():
        ns[kname] = v
    e = sy.expand(sy.sympify(expr, locals=ns))
    used = {sp_ for sp_ in ("tau", "sigma") if any(str(f).startswith(sp_) for f in e.free_symbols)}
    spaces = [sp_ for sp_ in ("tau", "sigma") if sp_ in used]
    dim = 2 ** len(spaces)
    H = sy.zeros(dim, dim)
    for term in sy.Add.make_args(e):
        c, ncs = term.args_cnc()
        prods = {sp_: np.eye(2, dtype=complex) for sp_ in spaces}
        for f in ncs:
            base, exp_ = (f.base, int(f.exp)) if f.is_Pow else (f, 1)
            space, comp = str(base).split("_")
            for _ in range(exp_):
                prods[space] = prods[space] @ _P[comp]
        M = np.ones((1, 1), complex)
        for sp_ in spaces:
            M = np.kron(M, prods[sp_])
        H += sy.Mul(*c) * sy.Matrix(M)
    return sy, ks, H


def _first(a, axis, dim):
    e = np.zeros(dim, int)
    e[axis] = 1
    return {tuple(e): -1j / (2 * a), tuple(-e): 1j / (2 * a)}


def _second(a, axis, dim):
    e = np.zeros(dim, int)
    e[axis] = 1
    return {tuple(e): -1 / a ** 2, tuple(np.zeros(dim, int)): 2 / a ** 2, tuple(-e): -1 / a ** 2}


def _compose(A, B):
    out = {}
    for s1, c1 in A.items():
        for s2, c2 in B.items():
            s = tuple(np.add(s1, s2))
            out[s] = out.get(s, 0) + c1 * c2
    return out


def _monomial_operator(powers, a, dim):
    op = {tuple(np.zeros(dim, int)): 1.0}
    for axis, n in enumerate(powers):
        for _ in range(n // 2):
            op = _compose(op, _second(a, axis, dim))
        if n % 2:
            op = _compose(op, _first(a, axis, dim))
    return op


def discretize(expr, coords="xy", grid=1.0, params=None, name=None):
    """Turn a k.p Hamiltonian (string or sympy Matrix) into a hopforge Model on
    a square/cubic grid of spacing ``grid``. Returns a periodic Model whose
    orbitals ``o0, o1, ...`` are the components of the matrix."""
    sy, ks, H = _parse(expr, coords, params)
    dim = len(coords)
    nmat = H.shape[0]
    terms = {}
    for i in range(nmat):
        for j in range(nmat):
            elem = sy.expand(H[i, j])
            if elem == 0:
                continue
            poly = sy.Poly(elem, *ks)
            for powers, coeff in poly.terms():
                c = complex(sy.N(coeff))
                if c == 0:
                    continue
                for s, v in _monomial_operator(powers, grid, dim).items():
                    M = terms.setdefault(s, np.zeros((nmat, nmat), complex))
                    M[i, j] += c * v
    lat = Lattice(np.eye(dim) * grid, [(f"o{n}", [0.0] * dim) for n in range(nmat)])
    m = Model(lat, name=name or "discretized k.p")
    zero = tuple([0] * dim)
    orbs = [f"o{n}" for n in range(nmat)]
    if zero in terms:
        M0 = terms.pop(zero)
        if not np.allclose(M0, M0.conj().T):
            raise ValueError("discretised on-site term is not Hermitian; check the expression")
        m.add_onsite_matrix(M0, orbs)
    done = set()
    for s, T in terms.items():
        if s in done or np.abs(T).max() < 1e-14:
            continue
        ms = tuple(-np.array(s))
        if ms in terms and not np.allclose(terms[ms], T.conj().T):
            raise ValueError("expression is not Hermitian (symmetrise products of k and "
                             "position-independent matrices)")
        # keep one orientation (the h.c. is added by the Model)
        if tuple(s) > ms:
            m.add_hopping_matrix(T, orbs, orbs, list(s))
        done.update({s, ms})
    pts = {"G": [0.0] * dim}
    lab = "XYZ"
    for a in range(dim):
        p = [0.0] * dim
        p[a] = 0.5
        pts[lab[a]] = p
    if dim >= 2:
        pts["M"] = [0.5, 0.5] + [0.0] * (dim - 2)
    if dim == 3:
        pts["R"] = [0.5, 0.5, 0.5]
    m.special_points = {k: np.array(v) for k, v in pts.items()}
    m.info["default_path"] = ["X", "G", "M"] if dim >= 2 else ["X", "G"]
    m.info["continuum"] = str(expr)
    return m


def lambdify_continuum(expr, coords="xy", params=None):
    """Numerical H(k) of the continuum model (Cartesian k) for comparisons."""
    sy, ks, H = _parse(expr, coords, params)
    f = sy.lambdify(ks, H, "numpy")
    return lambda *k: np.array(f(*k), dtype=complex)
