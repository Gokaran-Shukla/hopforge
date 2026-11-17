"""k-paths, band structures, k-meshes, density of states and Fermi levels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# k-space utilities
# ---------------------------------------------------------------------------
def kmesh(model, n, shift=0.0, endpoint=False):
    """Uniform Monkhorst-Pack-like mesh in reduced coordinates.

    ``n`` is an int (same along all directions) or a list. Returns (nk, dim).
    """
    dim = model.dim
    if dim == 0:
        return np.zeros((1, 0))
    ns = [n] * dim if np.isscalar(n) else list(n)
    axes = [np.linspace(0, 1, m, endpoint=endpoint) + shift / m for m in ns]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=-1)


@dataclass
class KPath:
    k: np.ndarray        # (nk, dim) reduced
    x: np.ndarray        # (nk,) cumulative Cartesian distance
    ticks: list          # x positions of the vertices
    labels: list         # vertex labels


def _label_tex(s):
    return {"G": r"$\Gamma$", "Gamma": r"$\Gamma$", "-X": r"$-X$"}.get(s, s)


def kpath(model, path=None, n=200):
    """Piecewise-linear path through special points.

    ``path`` may be a list of labels (``["G", "K", "M", "G"]``), a list of
    reduced coordinates, or a mixture. Labels are looked up in
    ``model.special_points``. Use ``"|"`` to break the path. ``n`` is the
    total number of points (distributed by length).
    """
    if model.dim == 0:
        return KPath(np.zeros((1, 0)), np.zeros(1), [0.0], [""])
    if path is None:
        path = model.info.get("default_path") or list(model.special_points)
        if len(path) == 1:
            path = ["G"]
    B = model.lat.reciprocal
    segs, cur = [], []
    for p in path:
        if isinstance(p, str) and p == "|":
            if cur:
                segs.append(cur)
            cur = []
            continue
        if isinstance(p, str):
            if p not in model.special_points:
                raise KeyError(f"unknown point {p!r}; known: {list(model.special_points)}")
            cur.append((p, np.atleast_1d(np.asarray(model.special_points[p], float))))
        else:
            v = np.atleast_1d(np.asarray(p, float))
            cur.append((str(np.round(v, 3).tolist()), v))
    if cur:
        segs.append(cur)
    lengths = []
    for seg in segs:
        for (_, a), (_, b) in zip(seg[:-1], seg[1:]):
            lengths.append(np.linalg.norm((b - a) @ B))
    total = sum(lengths) or 1.0
    ks, xs, ticks, labels = [], [], [], []
    x0, li = 0.0, 0
    for s, seg in enumerate(segs):
        if len(seg) == 1:
            ks.append(seg[0][1][None])
            xs.append(np.array([x0]))
            ticks.append(x0)
            labels.append(_label_tex(seg[0][0]))
            continue
        for v, ((la, a), (lb, b)) in enumerate(zip(seg[:-1], seg[1:])):
            L = lengths[li]
            li += 1
            m = max(2, int(round(n * L / total)))
            t = np.linspace(0, 1, m)
            if v > 0:
                t = t[1:]
            ks.append(a + t[:, None] * (b - a))
            xs.append(x0 + t * L)
            if v == 0:
                if s > 0 and ticks and abs(ticks[-1] - x0) < 1e-12:
                    labels[-1] = labels[-1] + "|" + _label_tex(la)
                else:
                    ticks.append(x0)
                    labels.append(_label_tex(la))
            x0 += L
            ticks.append(x0)
            labels.append(_label_tex(lb))
    return KPath(np.concatenate(ks), np.concatenate(xs), ticks, labels)


@dataclass
class Bands:
    path: KPath
    energies: np.ndarray            # (nk, nb)
    vectors: np.ndarray | None = None

    @property
    def x(self):
        return self.path.x

    def gap(self, n_occ):
        """Band gap between band n_occ-1 and n_occ along the path."""
        E = self.energies
        return float(E[:, n_occ].min() - E[:, n_occ - 1].max())

    def weights(self, P):
        """Expectation value of a Hermitian operator P for every state."""
        V = self.vectors
        return np.einsum("kin,ij,kjn->kn", V.conj(), P, V).real


def bands(model, path=None, n=200, vectors=False):
    """Band structure along a path. Returns a :class:`Bands` object."""
    kp = path if isinstance(path, KPath) else kpath(model, path, n)
    if vectors:
        E, V = model.eig(kp.k)
        return Bands(kp, E, V)
    return Bands(kp, model.energies(kp.k))


# ---------------------------------------------------------------------------
# density of states
# ---------------------------------------------------------------------------
@dataclass
class DOS:
    energies: np.ndarray
    dos: np.ndarray
    pdos: dict


def dos(model, nk=60, energies=None, sigma=0.05, projections=None, ne=600):
    """Gaussian-broadened density of states per unit cell (states / energy).

    ``projections`` : dict ``{label: orbitals}`` for projected DOS.
    """
    k = kmesh(model, nk)
    E, V = model.eig(k)
    if energies is None:
        lo, hi = E.min(), E.max()
        pad = 5 * sigma + 0.05 * (hi - lo + 1e-9)
        energies = np.linspace(lo - pad, hi + pad, ne)
    energies = np.asarray(energies, float)
    nk_tot = len(k)

    def broaden(eps, w=None):
        out = np.zeros_like(energies)
        eps = eps.ravel()
        w = np.ones_like(eps) if w is None else w.ravel()
        for chunk in range(0, len(eps), 20000):
            e = eps[chunk:chunk + 20000]
            ww = w[chunk:chunk + 20000]
            g = np.exp(-0.5 * ((energies[:, None] - e[None]) / sigma) ** 2)
            out += (g * ww[None]).sum(1)
        return out / (nk_tot * sigma * np.sqrt(2 * np.pi))

    total = broaden(E)
    pd = {}
    for label, orbs in (projections or {}).items():
        P = np.diag(model.projector(orbs))
        w = np.einsum("kin,i->kn", np.abs(V) ** 2, P)
        pd[label] = broaden(E, w)
    return DOS(energies, total, pd)


def fermi_level(model, n_electrons, nk=40):
    """Chemical potential (zero temperature) for ``n_electrons`` per unit cell."""
    E = np.sort(model.energies(kmesh(model, nk)).ravel())
    nk_tot = len(kmesh(model, nk))
    idx = int(round(n_electrons * nk_tot))
    if idx <= 0:
        return float(E[0])
    if idx >= len(E):
        return float(E[-1])
    return float(0.5 * (E[idx - 1] + E[idx]))


def band_gap(model, n_occ, nk=60):
    """Indirect gap between band n_occ-1 and n_occ on a uniform mesh."""
    E = model.energies(kmesh(model, nk))
    return float(E[:, n_occ].min() - E[:, n_occ - 1].max())
