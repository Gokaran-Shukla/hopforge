"""Supercells, ribbons / slabs, finite flakes and vacancies."""
from __future__ import annotations

import itertools

import numpy as np

from .model import Lattice, Model, _Term


def _new_like(model, lattice, name):
    m = Model(lattice, spinful=model.spinful, name=name)
    m.info = dict(model.info)
    return m


def supercell(model: Model, P) -> Model:
    """Supercell with new lattice vectors ``A' = P @ A``.

    ``P`` is an integer matrix (dim x dim), or a list of integers meaning a
    diagonal matrix, e.g. ``supercell(m, [3, 1])``.
    """
    dim = model.dim
    P = np.asarray(P, int)
    if P.ndim == 1:
        P = np.diag(P)
    if P.shape != (dim, dim):
        raise ValueError(f"P must be {dim}x{dim}")
    det = int(round(abs(np.linalg.det(P))))
    if det == 0:
        raise ValueError("P is singular")
    A = model.lat.vectors
    Pinv = np.linalg.inv(P)
    corners = np.array([sum(c) if c else np.zeros(dim, int)
                        for r in range(dim + 1) for c in itertools.combinations(list(P), r)])
    lo, hi = corners.min(0), corners.max(0)
    eps = 1e-9
    cells = []
    for R in itertools.product(*[range(lo[a], hi[a] + 1) for a in range(dim)]):
        f = np.array(R) @ Pinv
        if np.all(f > -eps) and np.all(f < 1 - eps):
            cells.append(np.array(R, int))
    assert len(cells) == det, "supercell construction failed"
    index = {tuple(R): c for c, R in enumerate(cells)}
    no = model.norb
    names, pos = [], []
    for c, R in enumerate(cells):
        for o in range(no):
            nm = model.lat.names[o]
            site, _, kind = nm.partition(":")
            new = f"{site}_{c}" + (f":{kind}" if kind else "")
            names.append(new)
            pos.append(model.lat.positions[o] + R @ A)
    lat = Lattice(P @ A, list(zip(names, pos)), coords="cartesian")
    sc = _new_like(model, lat, f"{model.name} supercell")
    for c in range(len(cells)):
        sc._onsite[c * no:(c + 1) * no] = model._onsite
    for t in model._terms:
        for c, Rc in enumerate(cells):
            target = Rc + t.R
            Rs = np.floor(target @ Pinv + eps).astype(int)
            rem = target - Rs @ P
            c2 = index[tuple(rem)]
            sc._terms.append(_Term(c * no + t.i, c2 * no + t.j, Rs, t.T.copy()))
    sc._changed()
    return sc


def cut(model: Model, direction: int, n: int, periodic: bool = False) -> Model:
    """Make the model finite along lattice direction ``direction`` with ``n``
    unit cells (open boundaries). Returns a model with one fewer periodic
    dimension: a ribbon from a 2D model, a slab from a 3D model, a chain
    segment (0D) from a 1D model.

    If ``periodic=True`` the hoppings that wrap around are kept, which simply
    gives an n-fold supercell (useful for comparison).
    """
    dim = model.dim
    if not 0 <= direction < dim:
        raise ValueError(f"direction must be in 0..{dim - 1}")
    diag = [1] * dim
    diag[direction] = n
    sc = supercell(model, diag)
    if periodic:
        return sc
    keep = [a for a in range(dim) if a != direction]
    A = sc.lat.vectors[keep]
    lat = Lattice(A, list(zip(sc.lat.names, sc.lat.positions)), coords="cartesian")
    out = _new_like(model, lat, f"{model.name} cut{direction}x{n}")
    out._onsite = sc._onsite.copy()
    for t in sc._terms:
        if t.R[direction] != 0:
            continue
        out._terms.append(_Term(t.i, t.j, t.R[keep].copy(), t.T.copy()))
    if out.dim == 1:
        out.special_points = {"-X": np.array([-0.5]), "G": np.array([0.0]), "X": np.array([0.5])}
    elif out.dim == 2:
        out.special_points = {"G": np.zeros(2), "X": np.array([0.5, 0]), "Y": np.array([0, 0.5]),
                              "M": np.array([0.5, 0.5])}
    out._changed()
    return out


def finite(model: Model, n) -> Model:
    """Finite (0D) flake with ``n`` cells along every periodic direction
    (``n`` may be an int or a list)."""
    dim = model.dim
    ns = [n] * dim if np.isscalar(n) else list(n)
    m = model
    for a in reversed(range(dim)):
        m = cut(m, a, ns[a])
    m.name = f"{model.name} flake"
    return m


def remove_orbitals(model: Model, orbitals) -> Model:
    """Remove orbitals (vacancies). Bonds touching them are deleted."""
    rm = set(model._orb_list(orbitals))
    keep = [i for i in range(model.norb) if i not in rm]
    new_index = {o: n for n, o in enumerate(keep)}
    lat = Lattice(model.lat.vectors, [(model.lat.names[i], model.lat.positions[i]) for i in keep],
                  coords="cartesian")
    out = _new_like(model, lat, model.name)
    out.special_points = dict(model.special_points)
    out._onsite = model._onsite[keep].copy()
    for t in model._terms:
        if t.i in new_index and t.j in new_index:
            out._terms.append(_Term(new_index[t.i], new_index[t.j], t.R.copy(), t.T.copy()))
    out._changed()
    return out


def add_disorder(model: Model, W: float, seed=None) -> Model:
    """Anderson disorder: random on-site energies uniform in [-W/2, W/2]."""
    rng = np.random.default_rng(seed)
    out = model.copy()
    for i in range(out.norb):
        out.add_onsite(rng.uniform(-W / 2, W / 2), i)
    return out
