"""GPU back-end: exercise every device code path with a NumPy-backed stand-in
for CuPy (no GPU needed). On a machine with CuPy the same tests run for real
with HOPFORGE_TEST_GPU=1."""
import os
import sys
import types

import numpy as np
import pytest
import scipy.sparse

import hopforge as hf


@pytest.fixture
def fake_cupy(monkeypatch):
    if os.environ.get("HOPFORGE_TEST_GPU"):
        yield
        return
    cp = types.ModuleType("cupy")
    for name in ("asarray", "einsum", "zeros", "eye", "where", "abs", "broadcast_to", "linalg"):
        setattr(cp, name, getattr(np, name))
    cp.asnumpy = np.asarray
    cp.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: 1))
    cpx = types.ModuleType("cupyx"); cpxs = types.ModuleType("cupyx.scipy"); cpxss = types.ModuleType("cupyx.scipy.sparse")
    cpxss.csr_matrix = scipy.sparse.csr_matrix
    cpx.scipy = cpxs; cpxs.sparse = cpxss
    for k, v in {"cupy": cp, "cupyx": cpx, "cupyx.scipy": cpxs, "cupyx.scipy.sparse": cpxss}.items():
        monkeypatch.setitem(sys.modules, k, v)
    yield
    hf.backend.use("numpy")


def test_backend_switch_and_results_identical(fake_cupy):
    m = hf.models.haldane(t2=0.1, phi=np.pi / 2)
    c_cpu = hf.chern_number(m, occ=1, nk=30)
    s_cpu = hf.anomalous_hall(m, mu=0.0, nk=40).sigma
    H = hf.models.honeycomb().supercell([12, 12]).hamiltonian_sparse()
    e_cpu, d_cpu = hf.kpm.dos(H, np.linspace(-2, 2, 21), num_moments=60, num_vectors=3)
    assert "cupy" in hf.backend.available()
    with hf.backend.on("cupy"):
        assert hf.backend.is_gpu() and hf.parallel.resolve_jobs(-1) == 1
        assert abs(hf.chern_number(m, occ=1, nk=30) - c_cpu) < 1e-10
        assert np.allclose(hf.anomalous_hall(m, mu=0.0, nk=40, n_jobs=-1).sigma, s_cpu)
        e, d = hf.kpm.dos(H, np.linspace(-2, 2, 21), num_moments=60, num_vectors=3)
        assert np.allclose(d, d_cpu)
    assert hf.backend.name() == "numpy"


def test_backend_errors():
    with pytest.raises(ValueError):
        hf.backend.use("opencl")
