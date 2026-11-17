"""Self-consistent mean-field (Hartree-Fock) Hubbard model on any spinful
hopforge model: collinear and noncollinear magnetism.

    H_U = U sum_i n_i,up n_i,dn
    -> sum_i c_i^dag [ U (Tr rho_i - rho_i) ] c_i - U (n_up n_dn - |rho_updn|^2)

rho_i is the 2x2 on-site spin density matrix, rho_ss' = <c_s'^dag c_s>.
Magnetic orders larger than the unit cell need a supercell template
(e.g. ``model.supercell([2, 2])``) and an initial moment pattern.

>>> m = hf.models.honeycomb().make_spinful()
>>> res = hf.meanfield.hubbard(m, U=3.0, filling=0.5, init={"A": (0, 0, 1), "B": (0, 0, -1)})
>>> res.moments           # per orbital (mx, my, mz)
>>> res.model.plot_bands()
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import operators as ops
from .parallel import pmap, split, chunks_for
from .solvers import kmesh


@dataclass
class MeanFieldResult:
    model: object               # mean-field Model (template + U terms)
    rho: np.ndarray             # (norb, 2, 2) on-site spin density matrices
    moments: np.ndarray         # (norb, 3) spin moments <sigma> (n_up - n_dn for z)
    occupations: np.ndarray     # (norb,)
    mu: float
    energy: float               # total mean-field energy per cell
    converged: bool
    iterations: int
    history: list

    def __repr__(self):
        return (f"<MeanFieldResult converged={self.converged} it={self.iterations} "
                f"E={self.energy:.6f} |m|max={np.abs(self.moments).max():.4f}>")


def _fermi(E, mu, kT):
    if kT <= 0:
        return (E < mu).astype(float)
    return 0.5 * (1 - np.tanh(0.5 * (E - mu) / kT))


def _find_mu(E, n_target, kT):
    """Chemical potential giving n_target electrons per k-point (bisection)."""
    lo, hi = E.min() - 1, E.max() + 1
    nk = E.shape[0]
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _fermi(E, mid, kT).sum() / nk > n_target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _mf_model(model, U, rho, Uorb):
    m = model.copy()
    m.name = f"MF[{model.name}]"
    for i in range(model.norb):
        Ui = U * Uorb[i]
        if Ui:
            r = rho[i]
            m._onsite[i] = m._onsite[i] + Ui * (np.trace(r) * np.eye(2) - r)
    m._changed()
    return m


def hubbard(model, U, filling=None, n_electrons=None, init=None, nk=24, kT=0.01, mixing=0.5,
            tol=1e-6, maxiter=300, orbitals=None, seed=0, n_jobs=None, verbose=False):
    """Self-consistent Hubbard mean field.

    U          : on-site interaction (applied to ``orbitals``, default all)
    filling    : fraction of the total number of bands that is filled
                 (0.5 = half filling), or give ``n_electrons`` per cell
    init       : dict {orbital name/index or site label: (mx, my, mz)} initial
                 moments, or 'ferro', 'random'; default small random moments
    nk, kT     : k-mesh and Fermi smearing; mixing: linear density mixing
    Returns a :class:`MeanFieldResult`.
    """
    if not model.spinful:
        raise ValueError("hubbard() needs a spinful model; use model.make_spinful()")
    no = model.norb
    Uorb = np.zeros(no)
    for i in (model._orb_list(orbitals) if orbitals is not None else range(no)):
        Uorb[i] = 1.0
    nb = model.nbands
    if n_electrons is None:
        if filling is None:
            raise ValueError("give filling or n_electrons")
        n_electrons = filling * nb
    rng = np.random.default_rng(seed)
    nav = n_electrons / no
    mvec = np.zeros((no, 3))
    if init is None or (isinstance(init, str) and init == "random"):
        mvec = 0.1 * rng.normal(size=(no, 3))
    elif isinstance(init, str) and init == "ferro":
        mvec[:, 2] = 0.5
    else:
        for key, v in init.items():
            for i in model._orb_list(key):
                mvec[i] = v
    sig = [ops.sx, ops.sy, ops.sz]
    rho = np.array([0.5 * (nav * np.eye(2) + sum(mvec[i, a] * sig[a] for a in range(3)))
                    for i in range(no)])
    k = kmesh(model, nk)
    parts = split(k, chunks_for(len(k), n_jobs))
    history = []
    converged = False
    mu = 0.0
    for it in range(1, maxiter + 1):
        mf = _mf_model(model, U, rho, Uorb)

        def eig_part(kk):
            return mf.eig(kk)
        res = pmap(eig_part, parts, n_jobs)
        E = np.concatenate([r[0] for r in res])
        V = np.concatenate([r[1] for r in res])
        mu = _find_mu(E, n_electrons, kT)
        f = _fermi(E, mu, kT)
        Vr = V.reshape(len(E), no, 2, nb)
        rho_new = np.einsum("kisn,kitn,kn->ist", Vr, Vr.conj(), f) / len(E)
        diff = np.abs(rho_new - rho).max()
        history.append(diff)
        if verbose:
            print(f"it {it:3d}  d_rho = {diff:.2e}  mu = {mu:.5f}")
        rho = (1 - mixing) * rho + mixing * rho_new
        if diff < tol:
            converged = True
            break
    mf = _mf_model(model, U, rho, Uorb)
    E = np.concatenate(pmap(lambda kk: mf.energies(kk), parts, n_jobs))
    f = _fermi(E, mu, kT)
    band = np.sum(f * E) / len(E)
    dc = sum(U * Uorb[i] * (rho[i, 0, 0].real * rho[i, 1, 1].real - abs(rho[i, 0, 1]) ** 2)
             for i in range(no))
    moments = np.array([[np.real(np.trace(rho[i] @ s)) for s in sig] for i in range(no)])
    occ = np.real(np.trace(rho, axis1=1, axis2=2))
    return MeanFieldResult(mf, rho, moments, occ, float(mu), float(band - dc), converged, it, history)


def phase_diagram(model, Us, **kw):
    """Run :func:`hubbard` for a list of U (warm start from the previous
    solution). Returns list of results."""
    out, init = [], kw.pop("init", None)
    for U in Us:
        r = hubbard(model, U, init=init, **kw)
        out.append(r)
        if np.abs(r.moments).max() > 1e-3:
            init = {i: r.moments[i] for i in range(model.norb)}
    return out
