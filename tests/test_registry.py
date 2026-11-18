import sys
import types
from pathlib import Path

import numpy as np
import pytest

import hopforge as hf

ROOT = Path(__file__).resolve().parents[1]


def test_every_model_has_reference_and_benchmark():
    hf.models.info("chain")            # populate built-in metadata
    for name in hf.models.REGISTRY:
        meta = hf.models.META[name]
        assert meta.get("reference"), name
        assert meta.get("benchmark") is not None, f"{name} has no benchmark"


def test_structural_validation_of_library():
    for name in hf.models.REGISTRY:
        v = hf.models.validate(name, run_benchmark=False)
        assert v.passed, v


def test_community_example_and_benchmark_integration():
    sys.path.insert(0, str(ROOT))
    import community.breathing_kagome  # noqa: F401
    v = hf.models.validate("breathing_kagome")
    assert v.passed, v
    names = [e[0] for e in hf.models.community_benchmarks()]
    assert "[community] breathing_kagome" in names
    res = hf.benchmarks.run(names=["[community] breathing_kagome"], verbose=False)
    assert res[0].passed


def test_validation_catches_broken_model():
    @hf.models.register_model("broken_test_model", dim=1, reference="none", benchmark=lambda: (1, 2, False),
                              replace=True)
    def broken():
        m = hf.Model(hf.Lattice([[1.0]], [("a", [0]), ("b", [0])]))
        m.add_hopping(1.0, "a", "b", [1])
        m._onsite[0] = np.array([[1j]])          # non-Hermitian on-site term
        return m
    v = hf.models.validate("broken_test_model")
    assert not v.passed
    del hf.models.REGISTRY["broken_test_model"], hf.models.META["broken_test_model"]


def test_plugin_entry_points(monkeypatch):
    mod = types.ModuleType("fake_hopforge_plugin")

    def load():
        @hf.models.register_model("plugin_chain", dim=1, reference="test", benchmark=lambda: (0, 0, True),
                                  replace=True)
        def plugin_chain():
            return hf.models.chain()

    class EP:
        name = "fake"
        def load(self):
            return load

    class EPS:
        def select(self, group):
            return [EP()] if group == "hopforge.models" else []

    import importlib.metadata as im
    monkeypatch.setattr(im, "entry_points", lambda: EPS())
    hf.models.registry._PLUGINS_LOADED[0] = False
    hf.models.load_plugins()
    assert "plugin_chain" in hf.models.REGISTRY
    assert hf.models.validate("plugin_chain").passed
    del hf.models.REGISTRY["plugin_chain"], hf.models.META["plugin_chain"]
