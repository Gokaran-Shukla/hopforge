"""Surface (and bulk) spectral functions of semi-infinite crystals, in the
spirit of WannierTools' surface-state module.

    A_surf(k_par, E) = -1/pi Im Tr [ O G_s(k_par, E + i eta) ]

G_s is the surface Green's function of the semi-infinite stack of principal
layers along lattice direction ``direction`` (Lopez Sancho-Rubio decimation,
vectorised over energies; parallel over k_par with ``n_jobs``).

>>> E, A = hf.surface.spectral(m, direction=2, kpath=["X", "G", "M"], energies=np.linspace(-1, 1, 200))
>>> hf.surface.plot_spectral(...)
>>> K1, K2, A = hf.surface.fermi_arc_map(weyl, direction=0, energy=0.0, nk=101)
"""
from __future__ import annotations

import numpy as np

from .parallel import pmap


def _layer_model(model, direction):
    rng = max((abs(int(t.R[direction])) for t in model._terms), default=0)
    if rng <= 1:
        return model, 1
    P = np.eye(model.dim, dtype=int)
    P[direction, direction] = rng
    return model.supercell(P), rng


def _layer_blocks(model, kpar, direction):
    """H0(k_par) and V(k_par) (coupling layer j -> j+1) in the periodic gauge."""
    def Hk(kd):
        k = np.insert(np.asarray(kpar, float), direction, kd)
        return model.hamiltonian(k, gauge="periodic")
    H0p, Hpi, Hhalf = Hk(0.0), Hk(0.5), Hk(0.25)
    H0 = 0.5 * (H0p + Hpi)
    Asum = 0.5 * (H0p - Hpi)          # V + V^dag
    Bm = Hhalf - H0                   # i (V - V^dag)
    V = 0.5 * (Asum - 1j * Bm)
    return H0, V


def _decimate(H0, V, energies, eta, maxiter=200, tol=1e-12):
    """Batched Sancho-Rubio over energies: surface G (side coupled through V)
    and bulk G. Returns (Gs, Gb) with shape (nE, n, n)."""
    nE, n = len(energies), H0.shape[0]
    z = (np.asarray(energies) + 1j * eta)[:, None, None] * np.eye(n)[None]
    es = np.broadcast_to(H0, (nE, n, n)).astype(complex).copy()
    e = es.copy()
    a = np.broadcast_to(V, (nE, n, n)).astype(complex).copy()
    b = np.broadcast_to(V.conj().T, (nE, n, n)).astype(complex).copy()
    for _ in range(maxiter):
        g = np.linalg.solve(z - e, np.concatenate([a, b], axis=2))
        ga, gb = g[:, :, :n], g[:, :, n:]
        agb, bga = a @ gb, b @ ga
        es = es + agb
        e = e + agb + bga
        a, b = a @ ga, b @ gb
        if max(np.abs(a).max(), np.abs(b).max()) < tol:
            break
    return np.linalg.inv(z - es), np.linalg.inv(z - e)


def spectral_at(model, kpar, energies, direction, side="top", eta=1e-3, operator=None,
                bulk=False):
    """Surface (or bulk) spectral function at one k_par for many energies."""
    H0, V = _layer_blocks(model, kpar, direction)
    # 'bottom' surface: the crystal extends towards +direction (couples via V);
    # 'top' surface: it extends towards -direction (couples via V^dag).
    Vs = V if side == "bottom" else V.conj().T
    Gs, Gb = _decimate(H0, Vs, energies, eta)
    G = Gb if bulk else Gs
    if operator is None:
        return -np.trace(G, axis1=1, axis2=2).imag / np.pi
    return -np.einsum("ij,eji->e", operator, G).imag / np.pi


def spectral(model, direction, kpath=None, energies=None, n=150, side="top", eta=1e-3,
             operator=None, bulk=False, n_jobs=None):
    """Surface spectral function along a k-path in the surface Brillouin zone.

    direction : lattice direction normal to the surface (index)
    kpath     : labels/points of the *bulk* model (the component along
                ``direction`` is ignored), e.g. ["X", "G", "M"]
    Returns (path, energies, A) with A of shape (nk, nE).
    """
    from .solvers import kpath as _kpath
    m, _ = _layer_model(model, direction)
    if energies is None:
        energies = np.linspace(-1, 1, 200)
    energies = np.asarray(energies, float)
    p = _kpath(model, kpath, n)
    # map bulk reduced k to the layer model: same reduced coordinates except the
    # stacking direction (supercell along it only rescales that component)
    kpars = [np.delete(k, direction) for k in p.k]
    A = pmap(lambda kp: spectral_at(m, kp, energies, direction, side, eta, operator, bulk),
             kpars, n_jobs)
    return p, energies, np.array(A)


def fermi_arc_map(model, direction, energy=0.0, nk=81, side="top", eta=2e-3, operator=None,
                  krange=(-0.5, 0.5), n_jobs=None):
    """Surface spectral weight at fixed energy on a 2D grid of the surface BZ
    (3D models): Fermi arcs, surface Fermi surfaces, spin textures.
    Returns (K1, K2, A) with reduced in-plane coordinates."""
    if model.dim != 3:
        raise ValueError("fermi_arc_map needs a 3D model")
    m, _ = _layer_model(model, direction)
    g = np.linspace(krange[0], krange[1], nk)
    K1, K2 = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([K1.ravel(), K2.ravel()], 1)
    vals = pmap(lambda kp: spectral_at(m, kp, [energy], direction, side, eta, operator)[0], pts,
                n_jobs, chunksize=max(1, len(pts) // 256))
    return K1, K2, np.array(vals).reshape(nk, nk)


def plot_spectral(path, energies, A, ax=None, cmap="inferno", log=True, title="surface spectral function"):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    Z = np.log10(A.T + 1e-3) if log else A.T
    ax.pcolormesh(path.x, energies, Z, shading="auto", cmap=cmap)
    ax.set_xticks(path.ticks)
    ax.set_xticklabels(path.labels)
    ax.set_ylabel("Energy")
    ax.set_title(title)
    return ax
