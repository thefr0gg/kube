"""NumPy compatibility helpers for Kube storage objects.

This module exists to provide module-level functional equivalents of the
numpy-oriented instance methods on ``Kube``, for callers who prefer a
functional style or who need standalone utilities.

All functions require ``numpy`` to be installed::

    pip install numpy
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from kube.model import Kube

try:
    import numpy as _np
    _NUMPY_AVAILABLE = True
except ImportError:
    _np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False


def _require_numpy():
    if not _NUMPY_AVAILABLE:
        raise ImportError("numpy is required: pip install numpy")


def faces_to_numpy(kube: "Kube", order: str = "xyzf") -> "np.ndarray":
    """Return a ``(W, H, D, 6)`` array of face values for *kube*.

    Delegates to ``kube.faces_to_numpy(order)``.
    """
    _require_numpy()
    return kube.faces_to_numpy(order=order)


def kube_from_numpy(arr: "np.ndarray", *, states: int = 4, path=None) -> "Kube":
    """Build a ``Kube`` from a ``(W, H, D, 6)`` NumPy face array.

    Args:
        arr: 4-D array with shape ``(W, H, D, 6)``.
        states: Number of distinct face values.
        path: Optional file path for a mapped kube; ``None`` for in-memory.

    Returns:
        A populated ``Kube`` instance.
    """
    _require_numpy()
    import polars as pl
    from kube.model import Kube

    if arr.ndim != 4 or arr.shape[3] != 6:
        raise ValueError(f"arr must have shape (W, H, D, 6), got {arr.shape}")
    W, H, D, _ = arr.shape
    xs, ys, zs = _np.meshgrid(
        _np.arange(W), _np.arange(H), _np.arange(D), indexing="ij"
    )
    flat = arr.reshape(-1, 6)
    frame = pl.DataFrame(
        {
            "x": xs.ravel().astype(_np.int32),
            "y": ys.ravel().astype(_np.int32),
            "z": zs.ravel().astype(_np.int32),
            **{f"face_{i}": flat[:, i] for i in range(6)},
        }
    )
    return Kube.from_polars(frame, shape=(W, H, D), states=states, path=path)


def region_to_numpy(kube: "Kube", bounds) -> "np.ndarray":
    """Return a ``(N, 6)`` face array for the cells within *bounds*.

    Delegates to ``kube.region_to_numpy(bounds)``.
    """
    _require_numpy()
    return kube.region_to_numpy(bounds)


def packed_to_numpy(kube: "Kube") -> "np.ndarray":
    """Return a 1-D array of packed cell integers for the full kube.

    Each element is the raw packed integer stored for one cell.
    """
    _require_numpy()
    return kube.to_numpy(packed=True)
