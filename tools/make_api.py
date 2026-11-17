"""Generate docs/API.md from docstrings:  python tools/make_api.py"""
import inspect

import hopforge as hf

MODS = [("hopforge", hf), ("hopforge.models", hf.models), ("hopforge.transport", hf.transport),
        ("hopforge.kpm", hf.kpm), ("hopforge.surface", hf.surface), ("hopforge.response", hf.response),
        ("hopforge.meanfield", hf.meanfield), ("hopforge.floquet", hf.floquet), ("hopforge.io", hf.io),
        ("hopforge.parallel", hf.parallel), ("hopforge.benchmarks", hf.benchmarks)]


def main():
    out = ["# hopforge API reference\n", "_Generated from docstrings by `tools/make_api.py`._\n"]
    seen = set()
    for name, mod in MODS:
        out.append(f"\n## `{name}`\n")
        out.append((inspect.getdoc(mod) or "").split("\n\n")[0] + "\n")
        for n, obj in sorted(vars(mod).items()):
            if n.startswith("_") or not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if getattr(obj, "__module__", "").split(".")[0] != "hopforge":
                continue
            key = (obj.__module__, n)
            if key in seen:
                continue
            seen.add(key)
            try:
                sig = str(inspect.signature(obj))
            except (TypeError, ValueError):
                sig = "(...)"
            first = (inspect.getdoc(obj) or "").split("\n\n")[0].replace("\n", " ")
            out.append(f"* **`{n}{sig}`**: {first}")
    open("docs/API.md", "w").write("\n".join(out) + "\n")
    print(len(seen), "entries written to docs/API.md")


if __name__ == "__main__":
    main()
