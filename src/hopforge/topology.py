"""Topological invariants and Berry-phase quantities.

All functions take ``occ`` = number of occupied bands (lowest ``occ`` bands)
or an explicit list of band indices.

Sign conventions (Xiao, Chang & Niu, RMP 82, 1959 (2010)):
    A_n = i <u_n | grad_k u_n>,   Omega_n = curl A_n
    Omega_n^{ab} = -2 Im sum_{m != n} <n|v_a|m><m|v_b|n> / (E_n - E_m)^2
    C = (1 / 2 pi) * integral over the BZ of Omega^{xy}
Berry phases are returned in (-pi, pi]; Wannier charge centres (WCC) are in
reduced units of the lattice vector along the Wilson loop, in [0, 1).
"""
from __future__ import annotations

import numpy as np

from .solvers import kmesh, _eig_parallel
from .parallel import pmap


def _bands_sel(model, occ):
    if occ is None:
        raise ValueError("specify occ (number of occupied bands or list of indices)")
    if np.isscalar(occ):
        return list(range(int(occ)))
    return list(occ)


def _basis_positions(model):
    return np.repeat(model.lat.positions, model.ns, axis=0)


def _loop_states(model, occ, kstart, G, nk):
    """Occupied eigenvectors on the closed loop kstart -> kstart + G
    (atomic gauge, nk points, closing with the exact boundary phase)."""
    sel = _bands_sel(model, occ)
    t = np.linspace(0, 1, nk, endpoint=False)
    ks = kstart[None] + t[:, None] * G[None]
    _, V = model.eig(ks)
    V = V[:, :, sel]
    Gc = G @ model.lat.reciprocal
    phase = np.exp(-1j * _basis_positions(model) @ Gc)
    Vend = phase[:, None] * V[0]
    return V, Vend


def _wilson_matrix(V, Vend):
    W = np.eye(V.shape[2], dtype=complex)
    for i in range(len(V)):
        nxt = V[i + 1] if i + 1 < len(V) else Vend
        M = V[i].conj().T @ nxt
        u, _, vh = np.linalg.svd(M)  # unitary part: removes discretisation error
        W = W @ (u @ vh)
    return W


def berry_phase(model, occ, kstart=None, direction=0, nk=100, gauge="atomic"):
    """Berry phase (sum over occupied bands) along the closed loop
    ``kstart -> kstart + b_direction``. Returns a value in (-pi, pi]."""
    dim = model.dim
    kstart = np.zeros(dim) if kstart is None else np.asarray(kstart, float)
    G = np.zeros(dim)
    G[direction] = 1
    if gauge == "atomic":
        V, Vend = _loop_states(model, occ, kstart, G, nk)
    else:
        sel = _bands_sel(model, occ)
        t = np.linspace(0, 1, nk, endpoint=False)
        _, V = model.eig(kstart[None] + t[:, None] * G[None], gauge="periodic")
        V = V[:, :, sel]
        Vend = V[0]
    W = _wilson_matrix(V, Vend)
    return float(-np.angle(np.linalg.det(W)))


def zak_phase(model, occ=None, nk=200, gauge="atomic"):
    """Zak phase of a 1D model (occupied bands = lower half by default)."""
    if model.dim != 1:
        raise ValueError("zak_phase is for 1D models; use berry_phase otherwise")
    occ = model.nbands // 2 if occ is None else occ
    return berry_phase(model, occ, None, 0, nk, gauge)


def polarization(model, occ=None, nk=200):
    """Electronic polarisation (in units of e * lattice constant, mod 1) of a 1D
    model = -Zak / 2 pi  (electron charge -e)."""
    return (-zak_phase(model, occ, nk) / (2 * np.pi)) % 1.0


def wilson_loop(model, occ, direction=0, scan=None, nk=100, nscan=51, fixed=None,
                scan_range=(0.0, 1.0), n_jobs=None):
    """Hybrid Wannier charge centres (WCC) along ``direction`` as a function of
    the momentum along ``scan``.

    Parameters
    ----------
    direction : reciprocal direction of the Wilson loop.
    scan      : reciprocal direction that is scanned (default: the other one in 2D).
    fixed     : dict ``{axis: value}`` for remaining directions in 3D.
    Returns (kscan, wcc) with wcc of shape (nscan, nocc), sorted, in [0, 1).
    """
    dim = model.dim
    if dim < 2:
        raise ValueError("wilson_loop needs dim >= 2 (use berry_phase in 1D)")
    if scan is None:
        scan = [a for a in range(dim) if a != direction][0]
    fixed = fixed or {}
    G = np.zeros(dim)
    G[direction] = 1
    kscan = np.linspace(scan_range[0], scan_range[1], nscan)

    def one(ks):
        k0 = np.zeros(dim)
        k0[scan] = ks
        for ax, val in fixed.items():
            k0[ax] = val
        V, Vend = _loop_states(model, occ, k0, G, nk)
        W = _wilson_matrix(V, Vend)
        ph = -np.angle(np.linalg.eigvals(W)) / (2 * np.pi)
        return np.sort(ph % 1.0)
    out = pmap(one, kscan, n_jobs, chunksize=max(1, nscan // 64))
    return kscan, np.array(out)


def chern_number(model, occ, nk=60, plane=(0, 1), fixed=None, n_jobs=None):
    """Chern number of the occupied bands by the Fukui-Hatsugai-Suzuki
    lattice method (exactly integer for a fine enough mesh)."""
    dim = model.dim
    if dim < 2:
        raise ValueError("Chern number needs dim >= 2")
    sel = _bands_sel(model, occ)
    n1, n2 = (nk, nk) if np.isscalar(nk) else nk
    a, b = plane
    k = np.zeros((n1, n2, dim))
    g1, g2 = np.meshgrid(np.arange(n1) / n1, np.arange(n2) / n2, indexing="ij")
    k[..., a], k[..., b] = g1, g2
    for ax, val in (fixed or {}).items():
        k[..., ax] = val
    kf = k.reshape(-1, dim)
    if n_jobs is None or n_jobs == 1:
        _, V = model.eig(kf, gauge="periodic")
    else:
        from .parallel import split, chunks_for
        parts = split(kf, chunks_for(len(kf), n_jobs))
        V = np.concatenate(pmap(lambda kk: model.eig(kk, gauge="periodic")[1], parts, n_jobs))
    V = V[:, :, sel].reshape(n1, n2, model.nbands, len(sel))

    def link(Va, Vb):
        d = np.linalg.det(np.einsum("xyia,xyib->xyab", Va.conj(), Vb))
        return d / np.abs(d)

    V1 = np.roll(V, -1, axis=0)
    V2 = np.roll(V, -1, axis=1)
    U1 = link(V, V1)
    U2 = link(V, V2)
    F = np.angle(U1 * np.roll(U2, -1, axis=0) * np.roll(U1, -1, axis=1).conj() * U2.conj())
    return float(-F.sum() / (2 * np.pi))


def berry_curvature(model, k, occ=None, mu=None, component="xy", eta=0.0, per_band=False,
                    n_jobs=None):
    """Berry curvature Omega^{ab} at k-points (Kubo formula).

    Returns the sum over occupied bands (``occ`` or all states below ``mu``)
    with shape (nk,), or the per-band values (nk, nb) if ``per_band=True``.
    """
    from .kubo import _band_resolved_parallel
    a, b = _comp(component)
    k = np.atleast_2d(np.asarray(k, float)) if model.dim > 1 else np.asarray(k, float).reshape(-1, 1)
    E, Om = _band_resolved_parallel(model, k, a, b, None, eta, n_jobs)
    if per_band:
        return Om
    if mu is not None:
        return (Om * (E < mu)).sum(1)
    sel = _bands_sel(model, model.nbands // 2 if occ is None else occ)
    return Om[:, sel].sum(1)


def berry_curvature_map(model, occ=None, nk=60, plane=(0, 1), fixed=None, mu=None, component="xy",
                        n_jobs=None):
    """Berry curvature on an nk x nk grid of the reduced plane; returns
    (k1, k2, Omega) ready for plotting.plot_kmap."""
    dim = model.dim
    g = np.linspace(0, 1, nk, endpoint=False)
    G1, G2 = np.meshgrid(g, g, indexing="ij")
    k = np.zeros((nk, nk, dim))
    k[..., plane[0]], k[..., plane[1]] = G1, G2
    for ax, val in (fixed or {}).items():
        k[..., ax] = val
    Om = berry_curvature(model, k.reshape(-1, dim), occ=occ, mu=mu, component=component,
                         n_jobs=n_jobs)
    return G1, G2, Om.reshape(nk, nk)


def _comp(component):
    m = {"x": 0, "y": 1, "z": 2}
    if isinstance(component, str):
        return m[component[0]], m[component[1]]
    return component


# ----------------------------------------------------------------------------
# Z2 invariants
# ----------------------------------------------------------------------------
def _largest_gap_mid(x):
    x = np.sort(np.asarray(x) % 1.0)
    gaps = np.diff(np.concatenate([x, [x[0] + 1]]))
    i = int(np.argmax(gaps))
    return (x[i] + gaps[i] / 2) % 1.0


def z2_from_wcc(wcc):
    """Z2 index from WCC over half the BZ (k_scan in [0, 0.5]) using the
    largest-gap method of Soluyanov & Vanderbilt, PRB 83, 235401 (2011)."""
    count = 0
    for i in range(len(wcc) - 1):
        z0, z1 = _largest_gap_mid(wcc[i]), _largest_gap_mid(wcc[i + 1])
        d = (z1 - z0 + 0.5) % 1.0 - 0.5
        for x in wcc[i + 1]:
            if d > 0 and ((x - z0) % 1.0) < d:
                count += 1
            elif d < 0 and ((z0 - x) % 1.0) < -d:
                count += 1
    return count % 2


def z2_invariant(model, occ, direction=0, nk=100, nscan=101, fixed=None, scan=None, n_jobs=None):
    """Z2 invariant of a time-reversal-invariant 2D model (or of one
    TRS-invariant plane of a 3D model, chosen with ``fixed``)."""
    _, wcc = wilson_loop(model, occ, direction, scan, nk, nscan, fixed, (0.0, 0.5), n_jobs)
    return z2_from_wcc(wcc)


def z2_3d(model, occ, nk=60, nscan=61, n_jobs=None):
    """Strong and weak Z2 indices (nu0; nu1 nu2 nu3) of a 3D TRS model from
    the six TRS-invariant planes k_i = 0 and k_i = 1/2."""
    if model.dim != 3:
        raise ValueError("z2_3d needs a 3D model")
    planes = [(i, val) for i in range(3) for val in (0.0, 0.5)]

    def one(pl):
        i, val = pl
        others = [a for a in range(3) if a != i]
        return z2_invariant(model, occ, direction=others[0], scan=others[1], nk=nk,
                            nscan=nscan, fixed={i: val})
    z = dict(zip(planes, pmap(one, planes, n_jobs)))
    nu0 = (z[(0, 0.0)] + z[(0, 0.5)]) % 2
    weak = tuple(z[(i, 0.5)] for i in range(3))
    return (nu0,) + weak


def fu_kane_parity(model, occ, parity_op, trim=None):
    """Z2 from parity eigenvalues at TRIM (Fu & Kane 2007) for models with
    inversion symmetry. ``parity_op`` is the inversion matrix acting on the
    basis (in the periodic gauge). Returns (z2, products) where products holds
    delta_i for each TRIM. Kramers pairs are counted once."""
    import itertools
    dim = model.dim
    trim = trim or [np.array(t, float) for t in itertools.product([0, 0.5], repeat=dim)]
    sel = _bands_sel(model, occ)
    deltas = []
    for t in trim:
        E, V = model.eig(np.asarray(t), gauge="periodic")
        Vo = V[:, sel]
        P = Vo.conj().T @ parity_op @ Vo
        ev = np.linalg.eigvals(P).real
        prod = np.prod(np.sign(ev))
        deltas.append(np.sign(prod) if model.spinful is False else
                      np.prod(np.sign(np.sort(ev)[::2])))
    deltas = [int(d) for d in deltas]
    z2 = int(round((1 - np.prod(deltas)) / 2))
    return z2, deltas


# ----------------------------------------------------------------------------
# 1D chiral winding
# ----------------------------------------------------------------------------
def winding_number(model, chiral, nk=400):
    """Winding number of a 1D chiral-symmetric model.

    ``chiral`` is the chiral operator Gamma (Gamma H Gamma^-1 = -H). The model
    is evaluated in the periodic gauge, as is standard for winding numbers.
    """
    if model.dim != 1:
        raise ValueError("winding_number needs a 1D model")
    Gm = np.asarray(chiral, complex)
    w, U = np.linalg.eigh(Gm)
    order = np.argsort(-w)
    U = U[:, order]
    n_plus = int((w > 0).sum())
    k = np.linspace(0, 1, nk, endpoint=False)
    H = model.hamiltonian(k, gauge="periodic")
    Hc = U.conj().T @ H @ U
    if np.abs(Hc[:, :n_plus, :n_plus]).max() > 1e-8:
        raise ValueError("the model does not anticommute with the given chiral operator")
    q = Hc[:, :n_plus, n_plus:]
    d = np.linalg.det(q)
    ph = np.unwrap(np.angle(np.concatenate([d, d[:1]])))
    return float((ph[-1] - ph[0]) / (2 * np.pi))
