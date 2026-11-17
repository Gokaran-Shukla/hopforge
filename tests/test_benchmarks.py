"""Every published-result benchmark must pass (see hopforge/benchmarks.py)."""
import pytest

from hopforge import benchmarks as B


@pytest.mark.parametrize("entry", B.REGISTRY, ids=[e[0] for e in B.REGISTRY])
def test_published_result(entry):
    r = B._run_one(entry)
    assert r.passed, f"{r.name}: published {r.published}, hopforge {r.value} {r.note}"
