# Community models

Contributed tight-binding models live here (or in your own package, as a plugin).
**Every model must come with a reference and a benchmark** that reproduces a number
from that reference: a gap, a flat-band energy, an invariant, a critical point.

## Contribute a model in three steps

1. Copy `breathing_kagome.py`, rename it, and write your constructor plus a
   `_benchmark()` that returns `(published, value, passed)`.
2. Register it:
   ```python
   @hf.models.register_model("my_model", dim=2, description="...",
                             reference="Author et al., Journal Vol, page (year)",
                             benchmark=_benchmark, author="Your Name")
   def my_model(...):
       ...
   ```
3. Check it, then open a pull request:
   ```bash
   python -c "import community.my_model, hopforge as hf; print(hf.models.validate('my_model'))"
   pytest -q tests/test_registry.py
   ```

`hf.models.validate` checks that the model builds, has the declared dimension and a
Hermitian, finite, reproducible H(k), cites a reference, and passes its benchmark.
CI runs `hf.models.validate_all()` over the whole library and every file in this folder.

## Ship models from your own package

```toml
# pyproject.toml of your package
[project.entry-points."hopforge.models"]
mymodels = "mypackage.hopforge_models"   # importing it calls register_model
```
After `pip install mypackage`, `hf.models.list_models()` shows your models, and
`python -m hopforge.benchmarks` runs their benchmarks with the built-in ones.
