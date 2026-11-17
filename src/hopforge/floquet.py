"""Floquet engineering of periodically driven tight-binding models.

A time-periodic vector potential A(t) enters every hopping as a Peierls phase
exp(i A(t) . d_ij) (e = hbar = 1, d_ij the Cartesian bond). The harmonics
H_m(k) = (1/T) int dt H(k, t) e^{i m w t} are themselves tight-binding models.

* :func:`harmonics`   -> {m: Model} (exact Fourier components, FFT in time)
* :class:`FloquetModel` -> Sambe-space quasienergies (truncated harmonics)
* :class:`EffectiveModel` -> high-frequency (van Vleck, first order in 1/w)
  H_eff = H_0 + sum_{m>0} [H_m, H_-m] / (m w)
  Both behave like a Model for bands and topology (chern_number, wilson_loop,
  hall_conductivity, ...).

>>> drive = hf.floquet.circular(A0=0.3, omega=8.0)
>>> F = hf.floquet.EffectiveModel(hf.models.honeycomb(), drive)
>>> hf.chern_number(F, occ=1)        # light-induced Haldane gap: C = +-1
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Model, _Term


@dataclass
class Drive:
    A: callable          # A(t) -> Cartesian vector (len D)
    omega: float
    name: str = "drive"


def circular(A0, omega, handedness=+1, D=2):
    """A(t) = A0 (cos wt, h sin wt)."""
    def A(t):
        v = np.zeros(max(D, 2))
        v[0], v[1] = A0 * np.cos(omega * t), handedness * A0 * np.sin(omega * t)
        return v
    return Drive(A, omega, f"circular(A0={A0}, w={omega}, h={handedness})")


def linear(A0, omega, direction=(1, 0), D=2):
    """A(t) = A0 cos(wt) e."""
    e = np.zeros(max(D, len(direction)))
    e[:len(direction)] = direction
    e = e / np.linalg.norm(e)
    return Drive(lambda t: A0 * np.cos(omega * t) * e, omega, f"linear(A0={A0}, w={omega})")


def harmonics(model, drive, n_max=4, n_time=128):
    """Fourier components H_m (m = -n_max..n_max) as hopforge Models."""
    T = 2 * np.pi / drive.omega
    ts = np.arange(n_time) * T / n_time
    c = model._compile()
    D = model.lat.D
    Avals = np.array([np.resize(drive.A(t), D) for t in ts])          # (nt, D)
    phases = np.exp(1j * Avals @ c["dc"].T)                            # (nt, nterms)
    # H_m = (1/T) int H(t) e^{i m w t} dt  -> numpy ifft convention
    coef = np.fft.fft(phases, axis=0) / n_time                         # index -m
    out = {}
    for m in range(-n_max, n_max + 1):
        Hm = Model(model.lat, spinful=model.spinful, name=f"H_{m}")
        if m == 0:
            Hm._onsite = model._onsite.copy()
        cm = coef[(-m) % n_time]
        for t_i, t in enumerate(model._terms):
            if abs(cm[t_i]) > 1e-14:
                Hm._terms.append(_Term(t.i, t.j, t.R.copy(), t.T * cm[t_i]))
        Hm._changed()
        out[m] = Hm
    return out


class _HarmonicSet:
    """Exact non-Hermitian harmonics H_m(k) from the bare terms."""

    def __init__(self, model, drive, n_max, n_time):
        self.model, self.drive, self.n_max = model, drive, n_max
        T = 2 * np.pi / drive.omega
        ts = np.arange(n_time) * T / n_time
        c = model._compile()
        D = model.lat.D
        Avals = np.array([np.resize(drive.A(t), D) for t in ts])
        ph = np.exp(1j * Avals @ c["dc"].T)          # forward hopping i->j phase
        self.c = c
        self.fwd = np.fft.fft(ph, axis=0) / n_time            # fwd[q] = (1/N) sum_t ph e^{-2 pi i q t/N}
        self.bwd = np.fft.fft(ph.conj(), axis=0) / n_time     # for the h.c. terms
        self.n_time = n_time

    def Hm(self, m, k, gauge="atomic"):
        """H_m(k) with H(t) = sum_m H_m e^{-i m w t}."""
        model, c = self.model, self.c
        kc, single = model._kcart(k)
        nk, no, ns = len(kc), model.norb, model.ns
        H = np.zeros((nk, no, ns, no, ns), complex)
        if len(c["I"]):
            disp = c["dc"] if gauge == "atomic" else c["Rc"]
            ph = np.exp(1j * kc @ disp.T)
            # H_m = (1/T) int H(t) e^{+i m w t} dt  <->  FFT index -m
            cf = self.fwd[(-m) % self.n_time]       # term i->j
            cb = self.bwd[(-m) % self.n_time]       # term j->i (h.c. with conjugated phase)
            vf = (ph * cf)[:, :, None, None] * c["T"][None]
            vb = (ph.conj() * cb)[:, :, None, None] * c["T"].conj().transpose(0, 2, 1)[None]
            np.add.at(H, (slice(None), c["I"], slice(None), c["J"], slice(None)), vf.transpose(1, 0, 2, 3))
            np.add.at(H, (slice(None), c["J"], slice(None), c["I"], slice(None)), vb.transpose(1, 0, 2, 3))
        H = H.reshape(nk, no * ns, no * ns)
        if m == 0:
            for i in range(no):
                H[:, i * ns:(i + 1) * ns, i * ns:(i + 1) * ns] += model._onsite[i]
        return H[0] if single else H


class _ModelLike:
    """Minimal Model interface for derived Hamiltonians (bands, topology)."""

    def __init__(self, base):
        self.lat = base.lat
        self.spinful = base.spinful
        self.special_points = dict(base.special_points)
        self.info = dict(base.info)
        self._base = base

    @property
    def dim(self):
        return self.lat.dim

    @property
    def ns(self):
        return self._base.ns

    @property
    def norb(self):
        return self._base.norb

    def _kcart(self, k, cartesian=False):
        return self._base._kcart(k, cartesian)

    def eig(self, k=None, vectors=True, gauge="atomic"):
        H = self.hamiltonian(k, gauge=gauge)
        return np.linalg.eigh(H) if vectors else np.linalg.eigvalsh(H)

    def energies(self, k=None):
        return self.eig(k, vectors=False)

    def velocity(self, k=None, gauge="atomic", h=1e-5):
        """Numerical dH/dk (Cartesian) by central differences."""
        k = np.zeros(self.dim) if k is None else np.asarray(k, float)
        single = k.ndim == 1 and self.dim > 1 or k.ndim == 0
        kk = np.atleast_2d(k) if self.dim > 1 else k.reshape(-1, 1)
        B = self.lat.reciprocal
        out = []
        for a in range(self.lat.D):
            dk = np.linalg.lstsq(B.T, np.eye(self.lat.D)[a] * h, rcond=None)[0]
            out.append((self.hamiltonian(kk + dk, gauge) - self.hamiltonian(kk - dk, gauge)) / (2 * h))
        V = np.array(out)
        return V[:, 0] if single else V

    def plot_bands(self, path=None, n=200, **kw):
        from .plotting import plot_bands
        return plot_bands(self, path, n, **kw)

    @property
    def nbands(self):
        return self._base.nbands


class EffectiveModel(_ModelLike):
    """High-frequency Floquet Hamiltonian (first order in 1/omega):
    H_eff = H_0 + sum_{m>0} [H_m, H_-m] / (m omega)."""

    def __init__(self, model, drive, n_max=3, n_time=128):
        super().__init__(model)
        self.drive = drive
        self.n_max = n_max
        self.H = _HarmonicSet(model, drive, n_max, n_time)
        self.name = f"Floquet-eff[{model.name}]"

    def hamiltonian(self, k=None, gauge="atomic", cartesian=False):
        if k is None:
            k = np.zeros(self.dim)
        H = self.H.Hm(0, k, gauge)
        w = self.drive.omega
        for m in range(1, self.n_max + 1):
            Hp, Hm = self.H.Hm(m, k, gauge), self.H.Hm(-m, k, gauge)
            H = H + (Hp @ Hm - Hm @ Hp) / (m * w)
        return 0.5 * (H + np.swapaxes(H.conj(), -1, -2))


class FloquetModel(_ModelLike):
    """Exact Floquet quasienergies in a truncated Sambe space with harmonics
    |m| <= n_max. ``hamiltonian(k)`` returns the Sambe matrix; ``quasienergies(k)``
    returns the nbands quasienergies of the central replica (largest weight in m = 0)."""

    def __init__(self, model, drive, n_max=4, n_time=128):
        super().__init__(model)
        self.drive, self.n_max = drive, n_max
        self.H = _HarmonicSet(model, drive, 2 * n_max, n_time)
        self.name = f"Floquet[{model.name}]"

    def sambe(self, k, gauge="atomic"):
        nb, M, w = self._base.nbands, self.n_max, self.drive.omega
        k = np.atleast_2d(k) if self.dim > 1 else np.asarray(k, float).reshape(-1, 1)
        blocks = {d: self.H.Hm(d, k, gauge) for d in range(-2 * M, 2 * M + 1)}
        size = (2 * M + 1) * nb
        S = np.zeros((len(k), size, size), complex)
        for a, m in enumerate(range(-M, M + 1)):
            for b, n in enumerate(range(-M, M + 1)):
                S[:, a * nb:(a + 1) * nb, b * nb:(b + 1) * nb] = blocks[m - n]
            S[:, a * nb:(a + 1) * nb, a * nb:(a + 1) * nb] -= m * w * np.eye(nb)
        return S

    def quasienergies(self, k, gauge="atomic"):
        nb, M = self._base.nbands, self.n_max
        S = self.sambe(k, gauge)
        w, V = np.linalg.eigh(S)
        weight0 = (np.abs(V[:, M * nb:(M + 1) * nb, :]) ** 2).sum(1)
        idx = np.argsort(-weight0, axis=1)[:, :nb]
        q = np.take_along_axis(w, idx, axis=1)
        return np.sort(q, axis=1)

    def energies(self, k=None):
        return self.quasienergies(np.zeros(self.dim) if k is None else k)

    def hamiltonian(self, k=None, gauge="atomic", cartesian=False):
        return self.sambe(np.zeros(self.dim) if k is None else k, gauge)
