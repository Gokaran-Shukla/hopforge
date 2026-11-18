"""Community model registry: every model carries a reference and a benchmark.

Register a model (in a notebook, a script or your own package)::

    import hopforge as hf

    def check():                          # reproduce one published number
        m = breathing_kagome(ta=1.0, tb=0.5)
        E = m.energies(hf.kmesh(m, 24))
        return 1.5, E[:, 2].mean(), abs(E[:, 2].ptp()) < 1e-10 and abs(E[:, 2].mean() - 1.5) < 1e-10

    @hf.models.register_model("breathing_kagome", dim=2,
                              description="breathing kagome lattice",
                              reference="Ezawa, PRL 120, 026801 (2018)",
                              benchmark=check, author="Your Name")
    def breathing_kagome(ta=1.0, tb=0.5):
        ...

    hf.models.validate("breathing_kagome")    # Hermiticity, dimension, benchmark
    hf.models.validate_all(n_jobs=40)         # the whole library (run in CI)
    hf.models.info("haldane")                 # reference, benchmark, status

A benchmark is either a callable returning ``(published, value, passed)``
or the name of an entry in :mod:`hopforge.benchmarks`. Registered community
benchmarks are also run by ``python -m hopforge.benchmarks``.

Packages can ship models as plugins through an entry point::

    # pyproject.toml of your package
    [project.entry-points."hopforge.models"]
    mymodels = "mypackage.hopforge_models"     # module that calls register_model

``hf.models.load_plugins()`` (called automatically by ``list_models``/``get``)
imports them.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .library import REGISTRY

META: dict = {}
_PLUGINS_LOADED = [False]


# =============================================================================
# registration
# =============================================================================
def register_model(name, dim, description="", reference=None, benchmark=None, author=None, tags=(),
                   replace=False):
    """Decorator adding a model constructor to the registry.

    reference : citation of the paper defining the model (required for the
                community library; shown by ``info`` and ``list_models``)
    benchmark : callable -> (published, value, passed), or the name of a
                ``hopforge.benchmarks`` entry
    """
    def deco(func):
        if name in REGISTRY and not replace:
            raise KeyError(f"model {name!r} already registered (use replace=True)")
        REGISTRY[name] = (func, dim, description)
        META[name] = dict(reference=reference, benchmark=benchmark, author=author, tags=tuple(tags),
                          source="community")
        return func
    return deco


def load_plugins(verbose=False):
    """Import models published by other packages under the ``hopforge.models``
    entry-point group."""
    if _PLUGINS_LOADED[0]:
        return []
    _PLUGINS_LOADED[0] = True
    loaded = []
    try:
        from importlib.metadata import entry_points
        eps = entry_points()
        group = eps.select(group="hopforge.models") if hasattr(eps, "select") else eps.get("hopforge.models", [])
        for ep in group:
            try:
                obj = ep.load()
                if callable(obj) and not isinstance(obj, type(warnings)):
                    obj()
                loaded.append(ep.name)
            except Exception as e:  # pragma: no cover
                warnings.warn(f"hopforge plugin {ep.name!r} failed to load: {e}")
    except Exception:  # pragma: no cover
        pass
    if verbose:
        print("loaded plugins:", loaded)
    return loaded


# =============================================================================
# built-in checks for models without a dedicated literature benchmark
# =============================================================================
def _flat(E, band, value, tol=1e-9):
    b = E[:, band]
    return bool(np.ptp(b) < tol and abs(b.mean() - value) < tol)


def _k(m, n):
    from ..solvers import kmesh
    return kmesh(m, n)


def _chk_triangular():
    from .library import triangular
    m = triangular()
    E = m.energies(np.array([[0, 0], [2 / 3, 1 / 3]]))[:, 0]
    return (-6.0, 3.0), tuple(np.round(E, 6)), bool(np.allclose(E, [-6, 3]))


def _chk_cubic():
    from .library import cubic
    m = cubic(); E = m.energies(_k(m, 12))
    return (-6.0, 6.0), (round(E.min(), 6), round(E.max(), 6)), bool(np.isclose(E.min(), -6) and np.isclose(E.max(), 6))


def _chk_bcc():
    from .library import bcc
    m = bcc(); E = m.energies(_k(m, 12))
    return (-8.0, 8.0), (round(E.min(), 6), round(E.max(), 6)), bool(np.isclose(E.min(), -8) and np.isclose(E.max(), 8))


def _chk_fcc():
    from .library import fcc
    m = fcc(); E = m.energies(_k(m, 12))
    return (-12.0, 4.0), (round(E.min(), 6), round(E.max(), 6)), bool(np.isclose(E.min(), -12) and np.isclose(E.max(), 4))


def _chk_checkerboard():
    from .library import checkerboard
    m = checkerboard(); E = m.energies(_k(m, 20))
    return "flat band at E = 2t", round(float(E[:, 1].mean()), 8), _flat(E, 1, 2.0)


def _chk_creutz():
    from .library import creutz_ladder
    m = creutz_ladder(); E = m.energies(_k(m, 40))
    return (-2.0, 2.0), (round(float(E[:, 0].mean()), 8), round(float(E[:, 1].mean()), 8)), \
        _flat(E, 0, -2.0) and _flat(E, 1, 2.0)


def _chk_dice():
    from .library import dice
    m = dice(); E = m.energies(_k(m, 20))
    return "flat band at E = 0", round(float(np.abs(E[:, 1]).max()), 10), _flat(E, 1, 0.0)


def _chk_pyrochlore():
    from .library import pyrochlore
    m = pyrochlore(); E = m.energies(_k(m, 8))
    return "two flat bands at E = 2t", (round(float(E[:, 2].mean()), 8), round(float(E[:, 3].mean()), 8)), \
        _flat(E, 2, 2.0) and _flat(E, 3, 2.0)


def _chk_altermagnet():
    from .library import altermagnet_dwave
    m = altermagnet_dwave()
    t = np.linspace(0.05, 0.45, 9)
    Sz = m.spin_operator("z")

    def split(ks):
        E, V = m.eig(ks)
        s = np.einsum("kin,ij,kjn->kn", V.conj(), Sz, V).real
        return np.abs(s[:, 0] - s[:, 1]).max(), E

    diag = m.energies(np.stack([t, t], 1))
    axis_up = m.energies(np.stack([t, 0 * t], 1))
    # nodal along G-M: Kramers-like degeneracy; split along G-X
    deg_diag = float(np.abs(diag[:, 0] - diag[:, 1]).max())
    spl_axis = float(np.abs(axis_up[:, 0] - axis_up[:, 1]).max())
    return "split along G-X, degenerate along G-M", (round(spl_axis, 4), round(deg_diag, 10)), \
        spl_axis > 0.1 and deg_diag < 1e-10


def _chk_porbital():
    from .library import p_orbital_square
    from ..kubo import orbital_hall
    m = p_orbital_square()                 # no SOC (spinless)
    o = orbital_hall(m, mu=0.0, nk=60).sigma[0]
    return "orbital Hall conductivity != 0 without SOC", round(float(o), 4), abs(o) > 0.05


def _chk_bdg():
    REG = REGISTRY["bdg_square"][0]
    m = REG(mu=-3.0, delta=0.2)
    E = m.energies(_k(m, 200))
    gap = float(np.abs(E).min())
    return "BCS gap Delta = 0.2 on the Fermi surface", round(gap, 4), abs(gap - 0.2) < 2e-3


def _chk_wdti():
    from .library import wilson_dirac_ti
    from ..topology import z2_3d
    z = tuple(z2_3d(wilson_dirac_ti(), occ=2, nk=10))
    return (1, 0, 0, 0), z, z == (1, 0, 0, 0)


_BUILTIN_CHECKS = {
    "triangular": (_chk_triangular, "Ashcroft & Mermin; E(Gamma) = -6t, E(K) = 3t"),
    "cubic": (_chk_cubic, "Ashcroft & Mermin, Solid State Physics (1976)"),
    "bcc": (_chk_bcc, "Ashcroft & Mermin, Solid State Physics (1976)"),
    "fcc": (_chk_fcc, "Ashcroft & Mermin, Solid State Physics (1976)"),
    "checkerboard": (_chk_checkerboard, "Sun, Gu, Katsura & Das Sarma, PRL 106, 236803 (2011)"),
    "creutz_ladder": (_chk_creutz, "Creutz, PRL 83, 2636 (1999)"),
    "dice": (_chk_dice, "Sutherland, PRB 34, 5208 (1986); Raoux et al., PRL 112, 026402 (2014)"),
    "pyrochlore": (_chk_pyrochlore, "Bergman, Wu & Balents, PRB 78, 125104 (2008)"),
    "altermagnet_dwave": (_chk_altermagnet, "Smejkal, Sinova & Jungwirth, PRX 12, 031042 (2022)"),
    "p_orbital_square": (_chk_porbital, "Go, Jo, Kim & Lee, PRL 121, 086602 (2018)"),
    "bdg_square": (_chk_bdg, "Bardeen, Cooper & Schrieffer (1957); de Gennes (1966)"),
    "wilson_dirac_ti": (_chk_wdti, "Zhang et al., Nat. Phys. 5, 438 (2009)"),
}

# built-in models validated by entries of hopforge.benchmarks
_BENCH_LINKS = {
    "chain": "Hubbard dimer", "square": "Quantised conductance of a ballistic wire",
    "honeycomb": "Graphene Dirac velocity", "haldane": "Haldane phase boundary",
    "kane_mele": "Kane-Mele QSH phase boundary", "qwz": "Qi-Wu-Zhang Chern numbers",
    "bhz": "BHZ quantum spin Hall phase", "kagome": "Kagome flat band", "lieb": "Lieb lattice flat band",
    "bilayer_graphene": "AB bilayer graphene gap under bias", "rashba_square": "Rashba spin-orbit torque symmetry",
    "hofstadter": "Hofstadter TKNN Chern numbers", "diamond_fkm": "Fu-Kane-Mele 3D topological insulator",
    "weyl_minimal": "Weyl semimetal anomalous Hall", "ssh": "SSH polarisation and winding",
    "rice_mele": "Thouless pump", "kitaev_chain": "Kitaev chain phase boundary",
    "bbh_quadrupole": "BBH quadrupole insulator", "p_ip_superconductor": "Chiral p+ip superconductor",
    "weyl_type2": "Type-II Weyl semimetal", "checkerboard_chern": "Flat Chern band (checkerboard)",
    "rhombohedral_graphene": "Chiral decomposition of ABC graphene",
    "graphene_qah": "Graphene QAH from Rashba + exchange",
    "rashba_2deg": "Universal intrinsic spin Hall conductivity",
}


_REFERENCES = {
    "chain": "Ashcroft & Mermin, Solid State Physics (1976)",
    "square": "Ashcroft & Mermin, Solid State Physics (1976)",
    "ssh": "Su, Schrieffer & Heeger, PRL 42, 1698 (1979)",
    "rice_mele": "Rice & Mele, PRL 49, 1455 (1982)",
    "kitaev_chain": "Kitaev, Phys. Usp. 44, 131 (2001)",
    "honeycomb": "Wallace, Phys. Rev. 71, 622 (1947)",
    "haldane": "Haldane, PRL 61, 2015 (1988)",
    "kane_mele": "Kane & Mele, PRL 95, 226801 and 146802 (2005)",
    "qwz": "Qi, Wu & Zhang, PRB 74, 085308 (2006)",
    "bhz": "Bernevig, Hughes & Zhang, Science 314, 1757 (2006)",
    "kagome": "Guo & Franz, PRB 80, 113102 (2009)",
    "lieb": "Lieb, PRL 62, 1201 (1989)",
    "bilayer_graphene": "McCann & Fal'ko, PRL 96, 086805 (2006)",
    "rashba_square": "Bychkov & Rashba, JETP Lett. 39, 78 (1984)",
    "hofstadter": "Hofstadter, PRB 14, 2239 (1976)",
    "diamond_fkm": "Fu, Kane & Mele, PRL 98, 106803 (2007)",
    "weyl_minimal": "Armitage, Mele & Vishwanath, RMP 90, 015001 (2018)",
    "bbh_quadrupole": "Benalcazar, Bernevig & Hughes, Science 357, 61 (2017)",
    "p_ip_superconductor": "Read & Green, PRB 61, 10267 (2000)",
    "weyl_type2": "Soluyanov et al., Nature 527, 495 (2015)",
    "checkerboard_chern": "Sheng, Gu, Sun & Sheng, Nat. Commun. 2, 389 (2011)",
    "rhombohedral_graphene": "Min & MacDonald, PRB 77, 155416 (2008)",
    "graphene_qah": "Tse, Qiao, Yao, MacDonald & Niu, PRB 83, 155447 (2011)",
    "rashba_2deg": "Sinova et al., PRL 92, 126603 (2004)",
}


def _builtin_meta():
    for name in list(REGISTRY):
        if name in META:
            continue
        desc = REGISTRY[name][2]
        if name in _BUILTIN_CHECKS:
            f, ref = _BUILTIN_CHECKS[name]
            META[name] = dict(reference=ref, benchmark=f, author="hopforge", tags=(), source="built-in")
        else:
            META[name] = dict(reference=_REFERENCES.get(name, desc), benchmark=_BENCH_LINKS.get(name),
                              author="hopforge", tags=(), source="built-in")


# =============================================================================
# information and validation
# =============================================================================
@dataclass
class Validation:
    name: str
    passed: bool
    checks: dict = field(default_factory=dict)
    benchmark: object = None
    message: str = ""

    def __repr__(self):
        s = "PASS" if self.passed else "FAIL"
        lines = [f"{self.name}: {s}"] + [f"  {k:12s} {'ok' if v else 'FAILED'}" for k, v in self.checks.items()]
        if self.benchmark is not None:
            lines.append(f"  benchmark    published {self.benchmark[0]}  |  model {self.benchmark[1]}")
        if self.message:
            lines.append("  " + self.message)
        return "\n".join(lines)


def info(name):
    """Registry entry: description, reference, benchmark, author, source."""
    load_plugins()
    _builtin_meta()
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}")
    f, dim, desc = REGISTRY[name]
    meta = dict(META.get(name, {}))
    b = meta.get("benchmark")
    meta["benchmark"] = b if (b is None or isinstance(b, str)) else getattr(b, "__name__", "callable")
    return dict(name=name, dim=dim, description=desc, **meta)


def _run_benchmark(b):
    if b is None:
        return None
    if isinstance(b, str):
        from .. import benchmarks as B
        for n, q, ref, f in B.REGISTRY:
            if n == b:
                return f()
        raise KeyError(f"benchmark {b!r} not found in hopforge.benchmarks")
    return b()


def validate(name, run_benchmark=True, nk_test=5, seed=0):
    """Structural checks (builds, Hermitian H(k), dimension, reproducible) and
    the model's benchmark. Returns a :class:`Validation`."""
    load_plugins()
    _builtin_meta()
    f, dim, desc = REGISTRY[name]
    checks, bench, msg = {}, None, ""
    try:
        m = f()
        checks["builds"] = True
    except Exception as e:
        return Validation(name, False, {"builds": False}, None, f"constructor failed: {e}")
    rng = np.random.default_rng(seed)
    checks["dimension"] = m.dim == dim
    k = rng.random((nk_test, max(m.dim, 1)))
    try:
        H = m.hamiltonian(k)
        checks["hermitian"] = bool(np.allclose(H, H.conj().transpose(0, 2, 1), atol=1e-10))
        checks["finite"] = bool(np.isfinite(H).all())
        checks["reproducible"] = bool(np.allclose(f().hamiltonian(k), H))
    except Exception as e:
        checks["hermitian"] = False
        msg = f"H(k) failed: {e}"
    meta = META.get(name, {})
    checks["reference"] = bool(meta.get("reference"))
    if run_benchmark:
        b = meta.get("benchmark")
        if b is None:
            checks["benchmark"] = False
            msg = msg or "no benchmark registered"
        else:
            try:
                pub, val, ok = _run_benchmark(b)
                bench = (pub, val)
                checks["benchmark"] = bool(ok)
            except Exception as e:
                checks["benchmark"] = False
                msg = f"benchmark error: {e}"
    return Validation(name, all(checks.values()), checks, bench, msg)


def validate_all(run_benchmark=True, n_jobs=None, verbose=True):
    """Validate every registered model (parallel over models)."""
    from ..parallel import pmap
    load_plugins()
    _builtin_meta()
    names = list(REGISTRY)
    res = pmap(lambda n: validate(n, run_benchmark), names, n_jobs)
    if verbose:
        w = max(map(len, names))
        for r in res:
            print(f"{r.name:{w}s}  {'PASS' if r.passed else 'FAIL'}  "
                  f"{'' if r.benchmark is None else str(r.benchmark[1])[:60]}  {r.message}")
        print(f"\n{sum(r.passed for r in res)}/{len(res)} models validated")
    return res


def community_benchmarks():
    """Benchmark entries of community-registered models (callables only), in the
    format of hopforge.benchmarks.REGISTRY."""
    load_plugins()
    out = []
    for name, meta in META.items():
        b = meta.get("benchmark")
        if meta.get("source") == "community" and callable(b):
            out.append((f"[community] {name}", REGISTRY[name][2] or name, meta.get("reference") or "", b))
    return out
