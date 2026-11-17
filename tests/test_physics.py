"""Physics regression tests: every number here is a known textbook result."""
import itertools

import numpy as np
import pytest

import hopforge as hf
from hopforge import models as M


# ---------------------------------------------------------------- engine
def test_all_models_hermitian():
    rng = np.random.default_rng(0)
    for name, (f, dim, _) in M.REGISTRY.items():
        m = f()
        k = rng.random((4, m.dim))
        H = m.hamiltonian(k)
        assert np.allclose(H, H.conj().transpose(0, 2, 1)), name


def test_analytic_dispersions():
    k = np.linspace(0, 1, 9)
    assert np.allclose(M.chain().energies(k)[:, 0], -2 * np.cos(2 * np.pi * k))
    kk = np.random.default_rng(1).random((8, 3))
    assert np.allclose(M.cubic().energies(kk)[:, 0], -2 * np.cos(2 * np.pi * kk).sum(1))
    assert np.allclose(M.fcc().energies(np.zeros(3)), -12)
    assert np.allclose(M.bcc().energies(np.zeros(3)), -8)


def test_graphene_dirac_points():
    g = M.honeycomb()
    assert np.allclose(g.energies(g.special_points["K"]), 0, atol=1e-12)
    assert np.allclose(g.energies(np.zeros(2)), [-3, 3])


@pytest.mark.parametrize("model,band,energy", [
    (M.kagome, 2, 2.0), (M.lieb, 1, 0.0), (M.dice, 1, 0.0), (M.checkerboard, 1, 2.0)])
def test_flat_bands(model, band, energy):
    E = model().energies(np.random.default_rng(2).random((40, 2)))
    assert np.allclose(E[:, band], energy)


def test_velocity_matches_finite_difference():
    m = M.kane_mele(lr=0.05)
    k = np.array([0.13, 0.37])
    V = m.velocity(k)
    B = m.lat.reciprocal
    h = 1e-6
    for a in range(2):
        dk = np.linalg.solve(B.T, np.eye(2)[a] * h)  # reduced step for a Cartesian step h
        num = (m.hamiltonian(k + dk) - m.hamiltonian(k - dk)) / (2 * h)
        assert np.allclose(V[a], num, atol=1e-6)


def test_slater_koster_table():
    p = dict(sps=1.3, pps=2.1, ppp=-0.7, pds=1.7, pdp=-0.4, dds=-1.5, ddp=0.8, ddd=-0.2)
    d = np.array([0.3, -0.8, 0.5])
    l, m_, n = d / np.linalg.norm(d)
    assert np.isclose(hf.slater_koster("s", "px", d, p), l * p["sps"])
    assert np.isclose(hf.slater_koster("px", "py", d, p), l * m_ * (p["pps"] - p["ppp"]))
    ref = 3 * l**2 * m_**2 * p["dds"] + (l**2 + m_**2 - 4 * l**2 * m_**2) * p["ddp"] \
        + (n**2 + l**2 * m_**2) * p["ddd"]
    assert np.isclose(hf.slater_koster("dxy", "dxy", d, p), ref)


# ---------------------------------------------------------------- geometry
def test_supercell_folding():
    g = M.honeycomb()
    P = np.array([[2, 1], [0, 1]])
    sc = g.supercell(P)
    ksc = np.array([0.21, 0.37])
    Es, seen = [], set()
    for n in itertools.product(range(-2, 3), repeat=2):
        k = np.linalg.solve(P, np.array(n) + ksc)
        key = tuple(np.round((k % 1.0) * 1e6).astype(int) % 1000000)
        if key in seen:
            continue
        seen.add(key)
        Es += list(g.energies(k))
    assert np.allclose(np.sort(Es), np.sort(sc.energies(ksc)))


def test_ribbon_and_flake_shapes():
    r = M.haldane(t2=0.15).ribbon(1, 20)
    assert r.dim == 1 and r.nbands == 40
    f = M.honeycomb().finite(3)
    assert f.dim == 0 and f.norb == 18


# ---------------------------------------------------------------- topology
def test_chern_numbers():
    assert round(hf.chern_number(M.haldane(), 1, 30)) == 1
    assert round(hf.chern_number(M.haldane(M=1.0), 1, 30)) == 0
    assert round(hf.chern_number(M.qwz(1.0), 1, 30)) == -1
    assert round(hf.chern_number(M.qwz(-1.0), 1, 30)) == 1
    assert round(hf.chern_number(M.qwz(3.0), 1, 30)) == 0
    hz = M.hofstadter(1, 3)
    assert [round(hf.chern_number(hz, [n], 40)) for n in range(3)] == [1, -2, 1]


def test_kubo_curvature_matches_fhs():
    h = M.haldane()
    Om = hf.berry_curvature(h, hf.kmesh(h, 80), occ=1)
    assert np.isclose(2 * np.pi * Om.mean() / h.lat.cell_volume, 1.0, atol=1e-3)


def test_one_d_invariants():
    s = M.ssh()
    G = s.info["chiral"]
    assert round(abs(hf.winding_number(s, G))) == 1
    assert round(hf.winding_number(M.ssh(1, 0.5), G)) == 0
    assert np.isclose(abs(hf.zak_phase(s, gauge="periodic")), np.pi)
    assert np.isclose(hf.polarization(s), 0.25)
    kc = M.kitaev_chain()
    assert round(abs(hf.winding_number(kc, kc.info["chiral"]))) == 1
    assert round(hf.winding_number(M.kitaev_chain(mu=3), kc.info["chiral"])) == 0


def test_z2_two_d():
    assert hf.z2_invariant(M.kane_mele(), 2) == 1
    assert hf.z2_invariant(M.kane_mele(lr=0.05), 2) == 1
    assert hf.z2_invariant(M.kane_mele(lv=0.5), 2) == 0
    assert hf.z2_invariant(M.bhz(), 2) == 1
    assert hf.z2_invariant(M.bhz(M=-1), 2) == 0
    par = np.kron(np.diag([1, -1]), np.eye(2))
    assert hf.fu_kane_parity(M.bhz(), 2, par)[0] == 1


def test_z2_three_d():
    assert hf.z2_3d(M.wilson_dirac_ti(), 2, nk=30, nscan=31) == (1, 0, 0, 0)
    assert hf.z2_3d(M.diamond_fkm(), 2, nk=30, nscan=31) == (1, 1, 1, 1)
    assert hf.z2_3d(M.diamond_fkm(dt=-0.4), 2, nk=30, nscan=31) == (0, 1, 1, 1)


def test_weyl_planes():
    w = M.weyl_minimal()
    assert round(abs(hf.chern_number(w, 1, 30, fixed={2: 0.0}))) == 1
    assert round(hf.chern_number(w, 1, 30, fixed={2: 0.5})) == 0


# ---------------------------------------------------------------- Kubo
def test_quantised_hall():
    assert np.isclose(float(hf.anomalous_hall(M.haldane(), 0.0, 80)), -1.0, atol=1e-3)
    assert np.isclose(abs(float(hf.spin_hall(M.kane_mele(), 0.0, 80))), 1.0, atol=1e-3)


def test_weyl_ahc_value():
    # sigma = (e^2/h) * (Delta k_z / 2 pi) / a  with Delta k_z / 2pi = 1/2, a = 1 A
    s = float(hf.anomalous_hall(M.weyl_minimal(), 0.0, 40))
    assert abs(abs(s) - 0.5 * 3.874045e-5 * 1e8) < 40


def test_orbital_hall_finite():
    s = float(hf.orbital_hall(M.p_orbital_square(), 0.0, 40))
    assert abs(s) > 0.5


# ---------------------------------------------------------------- quiz
def test_quizzes():
    for k in range(1, 8):
        q = hf.Quiz(k)
        assert len(q) == 20
        assert q.check(q.answer_key(), show_explanations=False) == 20
