"""Tests for Wannier90 I/O, surface Green's functions, response functions,
mean-field Hubbard and Floquet (all against exact or textbook results)."""
import numpy as np
import pytest

import hopforge as hf
from hopforge import floquet as F
from hopforge import io, meanfield, response, surface
from hopforge import models as M


@pytest.mark.parametrize("model", [M.diamond_fkm(), M.p_orbital_square(lso=0.2), M.pyrochlore()])
def test_wannier90_round_trip(model, tmp_path):
    f = tmp_path / "x_hr.dat"
    model.to_wannier90_hr(str(f))
    A = np.zeros((3, 3)); A[:model.dim, :model.lat.D] = model.lat.vectors
    if model.dim < 3:
        A[2, 2] = 20.0
    pos = np.zeros((model.nbands, 3))
    P = np.repeat(model.lat.positions, model.ns, axis=0)
    pos[:, :P.shape[1]] = P
    r = io.read_wannier90(str(f), lattice=A, positions=pos, spinful=model.spinful, dim=model.dim)
    k = np.random.default_rng(0).random((4, model.dim))
    assert np.allclose(r.hamiltonian(k), model.hamiltonian(k))


def test_poscar_round_trip(tmp_path):
    A = np.array([[3.1, 0, 0], [-1.55, 2.68, 0], [0, 0, 20]])
    P = np.array([[0, 0, 10], [1.55, 0.9, 11.5]])
    f = io.write_poscar(str(tmp_path / "POSCAR"), A, ["Mo", "S"], P)
    A2, sp_, P2 = io.read_poscar(f)
    assert np.allclose(A, A2) and np.allclose(P, P2) and sp_ == ["Mo", "S"]


def test_surface_dirac_cone_and_fermi_arc():
    wd = M.wilson_dirac_ti()
    E = np.linspace(-0.6, 0.6, 61)
    A = surface.spectral_at(wd, np.zeros(2), E, 2, "top", eta=1e-3)
    assert abs(E[np.argmax(A)]) < 1e-9
    A2 = surface.spectral_at(wd, np.array([0.05, 0.0]), E, 2, "top", eta=1e-3)
    slab = np.sort(np.abs(wd.slab(2, 20).energies(np.array([0.05, 0.0]))))[0]
    assert abs(abs(E[np.argmax(A2)]) - slab) < 0.015
    w = M.weyl_minimal()
    inside = surface.spectral_at(w, np.array([0.0, 0.1]), [0.0], 0, eta=2e-3)[0]
    outside = surface.spectral_at(w, np.array([0.0, 0.4]), [0.0], 0, eta=2e-3)[0]
    assert inside > 100 * max(outside, 1e-3)


def test_wiedemann_franz_and_seebeck_sign():
    r = response.boltzmann(M.square(), [-1.0], kT=0.01, nk=300)
    assert abs(r.lorenz[0] / (np.pi ** 2 / 3) - 1) < 0.02
    r2 = response.boltzmann(M.square(), [-3.5, 3.5], kT=0.05, nk=150)
    assert r2.seebeck[0, 0, 0] < 0 < r2.seebeck[1, 0, 0]


def test_graphene_universal_optical_conductivity():
    s = response.optical_conductivity(M.honeycomb(), [0.3], mu=0.0, nk=400, eta=0.02)
    assert abs(s.real[0] / 0.125 - 1) < 0.03


def test_orbital_moment_and_streda():
    E, Om, m = response.orbital_moment(M.haldane(M=0.2), np.random.default_rng(1).random((5, 2)))
    assert np.allclose(m[:, 0], (E[:, 0] - E[:, 1]) * Om[:, 0] / 2)
    h = M.haldane(t2=0.15)
    Mu = response.orbital_magnetization(h, [-0.2, 0.2], nk=150)
    assert np.isclose((Mu[1] - Mu[0]) / 0.4, h.lat.cell_volume / (2 * np.pi), rtol=1e-6)


def test_edelstein_symmetry():
    r = M.rashba_square(alpha=0.3)
    sx = response.edelstein(r, r.spin_operator("x"), "x", mu=-2.0, kT=0.02, nk=120)
    sy = response.edelstein(r, r.spin_operator("y"), "x", mu=-2.0, kT=0.02, nk=120)
    assert abs(sx["total"][0]) < 1e-8 and abs(sy["extrinsic"][0]) > 1e-3
    assert abs(sy["intrinsic"][0]) < 1e-8          # TRS kills the intrinsic term
    rm = M.rashba_square(alpha=0.3, J=0.5)
    assert abs(response.edelstein(rm, rm.spin_operator("x"), "x", mu=-2.0, nk=120)["intrinsic"][0]) > 1e-3


def test_berry_curvature_dipole_fermi_surface_vs_sea():
    q = M.qwz(1.5)
    q.add_hopping_matrix(-0.5j * 0.4 * np.eye(2), ["a", "b"], ["a", "b"], [1, 0])
    q.add_hopping_matrix(0.15 * np.eye(2), ["a", "b"], ["a", "b"], [0, 1])
    mu, kT = -0.3, 0.05
    D = response.berry_curvature_dipole(q, mu, kT, nk=200)[0][0]
    k = hf.kmesh(q, 200)
    h = 1e-4
    E, _, _ = response.orbital_moment(q, k)
    _, Op, _ = response.orbital_moment(q, k + [h, 0])
    _, Om_, _ = response.orbital_moment(q, k - [h, 0])
    f = 0.5 * (1 - np.tanh(0.5 * (E - mu) / kT))
    Dfd = np.sum(f * (Op - Om_) / (2 * 2 * np.pi * h)) / len(k) / q.lat.cell_volume
    assert np.isclose(D, Dfd, rtol=1e-4, atol=1e-10)


def test_hubbard_mean_field():
    g = M.honeycomb().make_spinful()
    weak = meanfield.hubbard(g, 1.5, filling=0.5, init={"A": (0, 0, 1), "B": (0, 0, -1)}, nk=40, kT=0.002)
    strong = meanfield.hubbard(g, 3.0, filling=0.5, init={"A": (0, 0, 1), "B": (0, 0, -1)}, nk=40, kT=0.002)
    assert np.abs(weak.moments).max() < 1e-3          # below U_c ~ 2.23 t
    assert strong.moments[0, 2] > 0.4 and np.isclose(strong.moments[0, 2], -strong.moments[1, 2])
    sq = M.square().make_spinful().supercell([[1, 1], [1, -1]])
    r = meanfield.hubbard(sq, 2.0, filling=0.5, init={0: (0, 0, 1), 1: (0, 0, -1)}, nk=40, kT=0.002)
    assert np.isclose(hf.band_gap(r.model, 2, 40), 2.0 * abs(r.moments[0, 2]), rtol=1e-3)


def test_floquet():
    g = M.honeycomb()
    k = np.random.default_rng(2).random((3, 2))
    d = F.circular(0.4, 8.0)
    HS = F._HarmonicSet(g, d, 6, 128)
    B = g.lat.reciprocal
    for t in (0.0, 0.37):
        Ht = sum(HS.Hm(m, k) * np.exp(-1j * m * d.omega * t) for m in range(-6, 7))
        kred = np.linalg.solve(B.T, (k @ B + d.A(t)).T).T
        assert np.allclose(Ht, g.hamiltonian(kred), atol=1e-8)
    K = g.special_points["K"]
    d = F.circular(0.4, 10.0)
    ge = np.diff(F.EffectiveModel(g, d).energies(K))[0]
    gx = np.diff(F.FloquetModel(g, d).quasienergies(K)[0])[0]
    assert np.isclose(ge, gx, rtol=0.01)
    c_plus = round(hf.chern_number(F.EffectiveModel(g, d), 1, 30))
    c_minus = round(hf.chern_number(F.EffectiveModel(g, F.circular(0.4, 10.0, -1)), 1, 30))
    assert abs(c_plus) == 1 and c_minus == -c_plus
    assert round(hf.chern_number(F.EffectiveModel(g, F.linear(0.4, 10.0)), 1, 30)) == 0


def test_register_model():
    @M.register_model("_test_chain2", 1, "test")
    def _c(t=1.0):
        return M.chain(t)
    assert "_test_chain2" in M.list_models()
    assert np.isclose(M.get("_test_chain2", t=2.0).energies(np.array([0.0]))[0, 0], -4.0)
    M.REGISTRY.pop("_test_chain2")
