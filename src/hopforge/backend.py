"""Compute back-end: CPU (NumPy/SciPy) or GPU (CuPy).

    >>> import hopforge as hf
    >>> hf.backend.use("cupy")          # all k-mesh diagonalisations, Kubo kernels
    ...                                 # and KPM recursions now run on the GPU
    >>> hf.anomalous_hall(model, nk=400)
    >>> hf.backend.use("numpy")         # back to the CPU
    >>> with hf.backend.on("cupy"): ...  # temporarily

What runs on the device
-----------------------
* ``Model.eig`` / ``Model.energies``: batched Hermitian eigensolver for the
  whole chunk of k-points (``cupy.linalg.eigh`` on stacked matrices).
* Kubo kernels (AHC, SHC, OHC, Berry curvature, Chern via curvature): matrix
  elements, energy denominators and reductions stay on the device.
* KPM (DOS, LDOS, Kubo-Bastin): the Chebyshev recursion with a
  ``cupyx.scipy.sparse`` matrix and many random vectors at once.

Hamiltonians are still assembled on the host (cheap) and copied once per
chunk; results come back as NumPy arrays, so every other hopforge function
works unchanged. Multiprocessing (``n_jobs``) is disabled automatically on the
GPU back-end: a CUDA context cannot be shared with forked workers, and one
GPU replaces the node-level parallelism. Use ``chunk`` sizes of a few thousand
k-points to keep the device busy.

``use("auto")`` picks CuPy when a CUDA device is visible, otherwise NumPy.
"""
from __future__ import annotations

import contextlib

import numpy as np
import scipy.sparse as _sp

_STATE = {"name": "numpy", "xp": np, "sparse": _sp}


def _load_cupy():
    import cupy
    import cupyx.scipy.sparse as csp
    return cupy, csp


def available():
    """Back-ends usable in this environment."""
    out = ["numpy"]
    try:
        cp, _ = _load_cupy()
        if cp.cuda.runtime.getDeviceCount() > 0:
            out.append("cupy")
    except Exception:
        pass
    return out


def use(name="numpy"):
    """Select the back-end: 'numpy', 'cupy' (needs CuPy + a CUDA GPU) or 'auto'."""
    if name == "auto":
        name = "cupy" if "cupy" in available() else "numpy"
    if name == "numpy":
        _STATE.update(name="numpy", xp=np, sparse=_sp)
    elif name == "cupy":
        try:
            cp, csp = _load_cupy()
        except ImportError as e:
            raise ImportError("the GPU back-end needs CuPy: pip install cupy-cuda12x "
                              "(match your CUDA version)") from e
        _STATE.update(name="cupy", xp=cp, sparse=csp)
    else:
        raise ValueError("backend must be 'numpy', 'cupy' or 'auto'")
    return _STATE["name"]


def name():
    return _STATE["name"]


def is_gpu():
    return _STATE["name"] != "numpy"


def xp():
    """The array module of the active back-end (numpy or cupy)."""
    return _STATE["xp"]


@contextlib.contextmanager
def on(backend):
    old = _STATE["name"]
    use(backend)
    try:
        yield
    finally:
        use(old)


def to_device(a):
    return _STATE["xp"].asarray(a) if is_gpu() else np.asarray(a)


def to_host(a):
    if is_gpu() and not isinstance(a, np.ndarray):
        return _STATE["xp"].asnumpy(a)
    return a


def eigh(H, vectors=True):
    """Batched Hermitian eigensolver on the active device; returns host arrays."""
    if not is_gpu():
        return np.linalg.eigh(H) if vectors else np.linalg.eigvalsh(H)
    x = _STATE["xp"]
    Hd = x.asarray(H)
    if vectors:
        E, V = x.linalg.eigh(Hd)
        return to_host(E), to_host(V)
    return to_host(x.linalg.eigvalsh(Hd))


def eigh_device(H):
    """Like eigh but keeps the results on the device (for fused kernels)."""
    x = _STATE["xp"]
    return x.linalg.eigh(x.asarray(H))


def sparse_to_device(H):
    """scipy.sparse -> device CSR matrix (identity on the CPU back-end)."""
    if not is_gpu():
        return _sp.csr_matrix(H)
    return _STATE["sparse"].csr_matrix(_sp.csr_matrix(H))
