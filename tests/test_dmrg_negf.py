import numpy as np

import hopforge as hf
from hopforge import dmrg as D
from hopforge import transport as tr


def test_mpo_matches_dense_and_dmrg_matches_ed():
    H = D.heisenberg(8, J2=0.3, h=0.1)
    e = np.linalg.eigvalsh(H.to_dense())[0]
    assert abs(D.dmrg(H, chi=[16, 32], sweeps=4).energy - e) < 1e-8


def test_fermion_signs_against_free_fermions():
    rc = hf.Model(hf.Lattice([[1.0]], [("a", [0])]), spinful=True)
    rc.add_hopping(-np.eye(2) + 0.3j * hf.pauli("y"), "a", "a", [1])
    rc.add_onsite_matrix(0.2 * hf.pauli("z"), ["a"])
    L = 5
    ev = np.linalg.eigvalsh(rc.finite(L).hamiltonian())
    r = D.dmrg(D.from_model(rc, L), chi=[32, 64], sweeps=5)
    assert abs(r.energy - ev[ev < 0].sum()) < 1e-7


def test_tfim_exact_and_excited_gap():
    L = 12
    assert abs(D.dmrg(D.tfim(L, 1, 0.8), chi=[16, 32], sweeps=4).energy - D.tfim_exact_obc(L, 1, 0.8)) < 1e-8
    ex = D.excited_states(D.tfim(L, 1, 1.5), n=2, chi=[16, 32], sweeps=4)
    assert ex[1].energy > ex[0].energy


def test_negf_equilibrium_and_ballistic():
    s = tr.System(hf.models.chain(), lambda r: 0 <= r[0] < 8)
    s.attach_lead((-1,)); s.attach_lead((1,))
    r0 = tr.negf.solve(s, 0.0)
    assert np.allclose(r0.site_density(), 0.5, atol=1e-6)
    r = tr.negf.solve(s, 0.2, hartree=0.5)
    assert r.converged and abs(r.current - 0.2) < 1e-3
    assert np.allclose(r.bond_currents, r.current, atol=1e-4)
