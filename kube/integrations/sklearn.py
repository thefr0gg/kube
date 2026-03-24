"""scikit-learn helpers for Kube storage objects.

This module exists to provide spatial-aware train/test split utilities and
``(X, y)`` array construction, avoiding the need for users to manually
coordinate Kube table exports and numpy conversions.

No direct scikit-learn import is required; the helpers only need ``numpy``.
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


def to_sklearn_xy(
    kube: "Kube",
    target_face: int,
    bounds=None,
) -> "tuple[np.ndarray, np.ndarray]":
    """Return ``(X, y)`` arrays ready for sklearn estimators.

    Delegates to ``kube.to_sklearn_xy(target_face, bounds)``.

    Args:
        kube: Source ``Kube`` instance.
        target_face: Face index (0–5) to use as the label column.
        bounds: Optional region bounds; ``None`` for the full cube.

    Returns:
        Tuple ``(X, y)`` where ``X`` has shape ``(N, 5)`` and ``y`` shape
        ``(N,)``.
    """
    _require_numpy()
    return kube.to_sklearn_xy(target_face, bounds)


def spatial_train_test_split(
    kube: "Kube",
    target_face: int,
    *,
    test_axis: int = 2,
    test_fraction: float = 0.2,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Split the kube on a spatial axis to avoid leakage.

    The kube is divided into a training half and a test half along
    *test_axis*.  Spatial splitting prevents adjacent-cell leakage that
    would inflate accuracy when using random splits on gridded data.

    Args:
        kube: Source ``Kube`` instance.
        target_face: Face index (0–5) used as the label column.
        test_axis: Spatial axis (0=x, 1=y, 2=z) along which to split.
        test_fraction: Fraction of the axis assigned to the test set.

    Returns:
        Tuple ``(X_train, X_test, y_train, y_test)``.
    """
    _require_numpy()

    if test_axis not in (0, 1, 2):
        raise ValueError("test_axis must be 0, 1, or 2")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")

    shape = kube.shape
    axis_len = shape[test_axis]
    split_idx = max(1, int(axis_len * (1.0 - test_fraction)))

    train_bounds = list(kube.full_bounds())
    test_bounds = list(kube.full_bounds())
    train_bounds[test_axis] = (0, split_idx)
    test_bounds[test_axis] = (split_idx, axis_len)

    X_train, y_train = kube.to_sklearn_xy(target_face, tuple(train_bounds))
    X_test, y_test = kube.to_sklearn_xy(target_face, tuple(test_bounds))
    return X_train, X_test, y_train, y_test
