"""Library of published tight-binding models. See :func:`list_models`.

Every model carries a reference and a benchmark; ``validate(name)`` checks it
and ``register_model`` adds community models (see :mod:`.registry`)."""
from .library import *  # noqa: F401,F403
from .library import REGISTRY, get as _get, list_models as _list_models, bdg
from .registry import (register_model, load_plugins, info, validate, validate_all, META,
                       community_benchmarks)


def list_models(*a, **kw):
    load_plugins()
    return _list_models(*a, **kw)


def get(name, **params):
    if name not in REGISTRY:
        load_plugins()
    return _get(name, **params)
