"""Transport / KPM / continuum regression tests against exact results."""
import numpy as np
import hopforge.continuum
import pytest

import hopforge as hf
from hopforge import kpm
from hopforge import models as M
from hopforge import transport as tr


def wire(model, L, W, **kw):
    s = tr.System(model, lambda r: 0 <= r[0] < L and 0 <= r[1] < W, **kw)
    s.attach_lead((-1, 0))
    s.attach_lead((1, 0))
    return s


def test_chain_surface_green_function():
    s = tr.System(M.chain(), lambda r: 0 <= r[0] < 4)
    s.attach_lead((-1,))
    for E in (-1.5, 0.3, 1.9):
        g = s.leads[0].surface_green(E)[0, 0]
        assert np.isclose(g, (E - 1j * np.sqrt(4 - E ** 2)) / 2, atol=1e-8)


def test_quantised_conductance_steps():
    s = wire(M.square(), 12, 10)
    for E in (-3.9, -3.5, -3.0, -2.0, -1.0, 0.1):
        S = s.smatrix(E)
        N = S.num_modes(0)
        assert np.isclose(S.transmission(1, 0), N, atol=1e-6)
        assert abs(S.transmission(0, 0)) < 1e-6


def test_unitarity_and_reciprocity_with_disorder():
    s = wire(M.square(), 12, 8, onsite=tr.anderson(2.0, seed=4))
    S = s.smatrix(-1.0)
    T10, T01 = S.transmission(1, 0), S.transmission(0, 1)
    assert np.isclose(T10, T01, atol=1e-8)
    assert 0 < T10 < S.num_modes(0)


def test_current_conservation():
    s = wire(M.square(), 12, 6, onsite=tr.anderson(1.5, seed=2))
    S = s.smatrix(-1.0)
    T = S.transmission(1, 0)
    pairs, J = S.current(0)
    P = s.positions()
    for xc in (2.5, 7.5):
        cut = (P[pairs[:, 0], 0] - xc) * (P[pairs[:, 1], 0] - xc) < 0
        sign = np.sign(P[pairs[cut, 1], 0] - P[pairs[cut, 0], 0])
        assert np.isclose((J[cut] * sign).sum(), T, atol=1e-6)


def test_quantum_hall_bar():
    B = 2 * np.pi * 0.05
    s = tr.System(M.square(), lambda r: 0 <= r[0] < 50 and 0 <= r[1] < 24,
                  peierls=tr.landau_gauge(B, "x"))
    s.attach_lead((-1, 0)); s.attach_lead((1, 0))
    for x0 in (10, 32):
        s.attach_lead((0, 1), shape=lambda r, x0=x0: x0 <= r[0] < x0 + 6, peierls=None)
    for x0 in (10, 32):
        s.attach_lead((0, -1), shape=lambda r, x0=x0: x0 <= r[0] < x0 + 6, peierls=None)
    for E, nu in ((-3.4, 1), (-2.8, 2)):
        S = s.smatrix(E)
        assert np.isclose(S.four_terminal_resistance(0, 1, 2, 4), 1 / nu, atol=1e-3)
        assert abs(S.four_terminal_resistance(0, 1, 2, 3)) < 1e-3


def test_quantum_spin_hall_edge_conductance():
    # width 15: narrower ribbons show a tiny finite-size gap at the edge Dirac point
    s = wire(M.kane_mele(lso=0.1), 14, 15, onsite=tr.anderson(0.8, seed=1))
    for E in (0.0, 0.1):
        S = s.smatrix(E)
        assert np.isclose(S.transmission(1, 0), 2.0, atol=1e-4)
        up = dict(spin="up")
        assert np.isclose(S.transmission(1, 0, up, up), 1.0, atol=1e-4)
        assert S.num_modes(0) == 2


def test_andreev_doubling():
    N = M.bdg(M.square(), mu=-3.0, delta=0.0)
    Sc = M.bdg(M.square(), mu=-3.0, delta=0.1)
    s = tr.System(N, lambda r: 0 <= r[0] < 8 and 0 <= r[1] < 8)
    s.attach_lead((-1, 0)); s.attach_lead((1, 0), model=Sc)
    e, h = dict(orbitals=["A_e"]), dict(orbitals=["A_h"])
    Ne = s.smatrix(0.0).num_modes(0, e)
    G = tr.andreev_conductance(s, [0.0, 0.05], 0, e, h)
    assert np.allclose(G, 2 * Ne, atol=0.1)


def test_ldos_sum_equals_lead_densities():
    s = wire(M.square(), 8, 5, onsite=tr.anderson(1.0, seed=7))
    S = s.smatrix(-0.7)
    assert np.isclose(S.ldos().sum(), sum(S.density(l).sum() for l in range(2)), rtol=1e-8)


def test_kpm_dos_moments_exact():
    sc = M.honeycomb().supercell([6, 6])
    H = sc.hamiltonian_sparse()
    S, mu = kpm.moments(H, 6, num_vectors=400, seed=1)
    w = np.linalg.eigvalsh(H.toarray())
    e = (w - S.b) / S.a
    exact = [np.cos(n * np.arccos(e)).sum() for n in range(6)]
    assert np.allclose(mu.mean(1).real, exact, atol=0.05 * len(w))


def test_kpm_kubo_bastin_matches_kubo():
    h = M.haldane(t2=0.15)
    sc = h.supercell([20, 20])
    sig = kpm.conductivity(sc.hamiltonian_sparse(), sc.velocity_sparse("x"), sc.velocity_sparse("y"),
                           sc.lat.cell_volume, [-2.0, 0.0], num_moments=300, num_vectors=6)
    assert np.isclose(sig[1], -1.0, atol=0.05)
    assert abs(sig[0]) < 0.05


def test_sparse_matches_dense():
    m = M.kane_mele(lr=0.05).supercell([2, 3])
    k = np.array([0.11, 0.29])
    assert np.allclose(m.hamiltonian_sparse(k).toarray(), m.hamiltonian(k))
    assert np.allclose(m.velocity_sparse("y", k).toarray(), m.velocity(k)[1])


def test_continuum_discretizer():
    ex = "A*(k_x*sigma_x + k_y*sigma_y) + (M - B*(k_x**2 + k_y**2))*sigma_z"
    p = dict(A=1.0, B=1.0, M=0.5)
    m = hf.discretize(ex, "xy", 0.05, p)
    f = hf.continuum.lambdify_continuum(ex, "xy", p)
    kc = np.array([0.3, -0.2])
    assert np.allclose(m.energies(kc @ m.lat.vectors.T / (2 * np.pi)),
                       np.linalg.eigvalsh(f(*kc)), atol=2e-3)
    assert round(hf.chern_number(hf.discretize(ex, "xy", 1.0, p), 1, 30)) == 1
    ti = ("M0*tau_z - B*(k_x**2+k_y**2+k_z**2)*tau_z + "
          "A*(k_x*tau_x*sigma_x + k_y*tau_x*sigma_y + k_z*tau_x*sigma_z)")
    mt = hf.discretize(ti, "xyz", 1.0, dict(M0=1.0, B=0.5, A=1.0))
    assert hf.z2_3d(mt, 2, nk=20, nscan=21) == (1, 0, 0, 0)
