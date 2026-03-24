"""Structural Protocol definition for the Kube backend interface.

This module exists to give type checkers and documentation tools a concrete
description of the operations that any Kube-compatible backend must expose.
The ``Protocol`` is structural (PEP 544) — it is never enforced at runtime.

Example usage::

    from kube.protocol import KubeBackend
    from kube import Kube

    def process(backend: KubeBackend) -> None:
        info = backend.inspect()
        ...

    kube = Kube(8, 8, 8, states=4)
    process(kube)          # passes structural type check
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # Python < 3.8 — should never happen for us
    from typing_extensions import Protocol, runtime_checkable  # type: ignore[assignment]

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa


@runtime_checkable
class KubeBackend(Protocol):
    """Structural interface satisfied by ``Kube`` and compatible backends.

    Any object that exposes these methods is a valid ``KubeBackend``.  No
    subclassing or registration is required.
    """

    def inspect(self) -> dict:
        """Return a metadata dict describing shape, dtype, capacity, etc."""
        ...

    def schema(self, mode: str = "wide_faces") -> dict:
        """Return ``{column_name: dtype_str}`` for the given schema mode."""
        ...

    def read_region(self, bounds) -> Any:
        """Read all cells within *bounds* and return a representation."""
        ...

    def iter_batches(self, batch_size: int) -> Iterator:
        """Yield successive batches of cells as polars DataFrames."""
        ...

    def to_arrow(self, bounds=None) -> "pa.Table":
        """Convert the kube (or a region) to a PyArrow Table."""
        ...

    def to_numpy(self, bounds=None) -> "np.ndarray":
        """Convert the kube (or a region) to a NumPy array."""
        ...

    def close(self) -> None:
        """Release any open file handles or mapped memory."""
        ...
