"""Optional framework integrations for Kube.

This package exists so adapters for heavier third-party ecosystems can live
outside the core storage modules and remain optional install-time dependencies.

Availability helpers::

    from kube.integrations import available_backends
    print(available_backends())  # e.g. ['arrow', 'numpy', 'pandas']
"""

from __future__ import annotations


def _check_arrow() -> bool:
    """Return ``True`` if ``pyarrow`` is importable."""
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def _check_numpy() -> bool:
    """Return ``True`` if ``numpy`` is importable."""
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def _check_pandas() -> bool:
    """Return ``True`` if ``pandas`` is importable."""
    try:
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


def _check_torch() -> bool:
    """Return ``True`` if ``torch`` is importable."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _check_datasets() -> bool:
    """Return ``True`` if ``datasets`` (HuggingFace) is importable."""
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        return False


def _check_sklearn() -> bool:
    """Return ``True`` if ``scikit-learn`` is importable."""
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def available_backends() -> list[str]:
    """Return a list of optional backends that are currently available.

    The function exists so users and tooling can introspect the runtime
    environment without importing each backend module individually.
    """
    backends = []
    if _check_arrow():
        backends.append("arrow")
    if _check_numpy():
        backends.append("numpy")
    if _check_pandas():
        backends.append("pandas")
    if _check_torch():
        backends.append("torch")
    if _check_datasets():
        backends.append("datasets")
    if _check_sklearn():
        backends.append("sklearn")
    return backends


__all__ = [
    "_check_arrow",
    "_check_numpy",
    "_check_pandas",
    "_check_torch",
    "_check_datasets",
    "_check_sklearn",
    "available_backends",
]
