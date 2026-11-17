"""Intrinsic (Berry-curvature-like) Kubo Hall responses.

    sigma_{ab}^{O} = -(e^2 / hbar) * integral dk/(2 pi)^d  sum_n f_n Omega^{O}_{n,ab}

    Omega^{O}_{n,ab} = -2 Im sum_{m != n} <n|J^O_a|m><m|v_b|n> / ((E_n - E_m)^2 + eta^2)

with J^O_a = {O, v_a}/2 (O = 1 gives the anomalous Hall conductivity, O = s_z
the spin Hall conductivity, O = L_z the orbital Hall conductivity). The sign
convention matches Wannier90 (berry.F90): sigma_xy = -(e^2/h) C in 2D.

Units of the returned numbers (hbar = 1, lengths in Angstrom):
    2D : e^2/h                (for O != 1: (hbar/e) e^2/h, with s = sigma/2)
    3D : S/cm                 (for O != 1: (hbar/e) S/cm)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solvers import kmesh

E2_HBAR_SI = 2.434134807e-4  # e^2 / hbar in Siemens


def _anticomm(O, v):
    return 0.5 * (O @ v + v @ O)


def _band_resolved(model, k, a, b, O=None, eta=0.0, chunk=2000, degeneracy_tol=1e-8):
    """Energies (nk, nb) and band-resolved Omega^O_{n,ab} (nk, nb)."""
    Es, Oms = [], []
    for s in range(0, len(k), chunk):
        kk = k[s:s + chunk]
        E, V = model.eig(kk)
        vel = model.velocity(kk)
        va, vb = vel[a], vel[b]
        Ja = va if O is None else _anticomm(O, va)
        A = np.einsum("kin,kij,kjm->knm", V.conj(), Ja, V, optimize=True)
        B = np.einsum("kin,kij,kjm->knm", V.conj(), vb, V, optimize=True)
        dE = E[:, :, None] - E[:, None, :]
        denom = dE ** 2 + eta ** 2
        mask = np.abs(dE) > degeneracy_tol if eta == 0 else np.ones_like(dE, bool)
        nb = E.shape[1]
        mask &= ~np.eye(nb, dtype=bool)[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(mask, (A * B.transpose(0, 2, 1)) / np.where(mask, denom, 1), 0)
        Oms.append(-2 * term.sum(2).imag)
        Es.append(E)
    return np.concatenate(Es), np.concatenate(Oms)


def _fermi(E, mu, kT):
    if kT <= 0:
        return (E < mu).astype(float)
    x = np.clip((E - mu) / kT, -200, 200)
    return 1.0 / (np.exp(x) + 1.0)


@dataclass
class HallResult:
    mu: np.ndarray
    sigma: np.ndarray
    units: str
    component: str
    operator: str

    def __float__(self):
        return float(self.sigma[0]) if np.ndim(self.sigma) else float(self.sigma)

    def __repr__(self):
        if self.sigma.size == 1:
            return f"sigma^{self.operator}_{self.component} = {float(self.sigma[0]):.6g} [{self.units}]"
        return f"HallResult({self.operator}, {self.component}, {self.sigma.size} chemical potentials, [{self.units}])"


def hall_conductivity(model, mu=0.0, nk=100, operator=None, component="xy", eta=0.0, kT=0.0,
                      op_name=None, length_unit_angstrom=True):
    """Intrinsic Hall-type conductivity on a uniform k mesh.

    Parameters
    ----------
    mu : float or array of chemical potentials (one diagonalisation for all)
    nk : mesh size (int or list per direction)
    operator : None for charge (AHC); a matrix (e.g. model.spin_operator('z')
               or model.orbital_operator('z')) for spin / orbital Hall.
    component : 'xy', 'yz', ... -> current along the first, field along the second.
    eta : broadening in the energy denominators (0 = exact, degeneracies skipped)
    kT  : temperature (energy units) in the Fermi function
    """
    dim = model.dim
    if dim < 2:
        raise ValueError("Hall conductivity needs dim >= 2")
    m = {"x": 0, "y": 1, "z": 2}
    a, b = m[component[0]], m[component[1]]
    k = kmesh(model, nk)
    E, Om = _band_resolved(model, k, a, b, operator, eta)
    mus = np.atleast_1d(np.asarray(mu, float))
    vol = model.lat.cell_volume
    vals = np.array([(Om * _fermi(E, u, kT)).sum(1).mean() for u in mus])
    if dim == 2:
        sigma = -2 * np.pi * vals / vol
        units = "e^2/h"
    else:
        sigma = -vals / vol * E2_HBAR_SI * (1e8 if length_unit_angstrom else 1.0)
        units = "S/cm"
    name = "charge" if operator is None else (op_name or "O")
    if operator is not None:
        units = "(hbar/e) " + units
    return HallResult(mus, sigma, units, component, name)


def anomalous_hall(model, mu=0.0, nk=100, component="xy", **kw):
    """Anomalous Hall conductivity (Berry curvature of occupied states)."""
    return hall_conductivity(model, mu, nk, None, component, op_name="charge", **kw)


def spin_hall(model, mu=0.0, nk=100, spin="z", component="xy", **kw):
    """Intrinsic spin Hall conductivity sigma^{s_spin}_{component}."""
    return hall_conductivity(model, mu, nk, model.spin_operator(spin), component,
                             op_name=f"S{spin}", **kw)


def orbital_hall(model, mu=0.0, nk=100, L="z", component="xy", **kw):
    """Intrinsic orbital Hall conductivity with the atomic L operator
    (atom-centred approximation), built from the orbital kinds (p, d)."""
    return hall_conductivity(model, mu, nk, model.orbital_operator(L), component,
                             op_name=f"L{L}", **kw)
