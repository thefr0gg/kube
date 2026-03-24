"""Public package surface for Kube.

This module exists to make the project pleasant to import from application
code. It re-exports the primary storage classes plus small wrapper functions so
users can work with `kube.create(...)`, `kube.open(...)`, and related helpers
without needing to know the internal module layout.
"""

from __future__ import annotations

from pathlib import Path

from kube.metadata import (
    FEATURE_SCHEMA,
    PIPELINE_VERSION,
    SEQUENCE_LENGTH,
    SOURCE_DATASET,
    SPLIT,
    TOKENIZER_NAME,
    TOKENIZER_PATH,
)
from kube.schemas import (
    SCHEMA_CELL_PACKED,
    SCHEMA_LONG_FACES,
    SCHEMA_SUPERVISED,
    SCHEMA_TOKEN_SEQUENCES,
    SCHEMA_WIDE_FACES,
)
from kube.dataset import (
    ShardedKubeDataset,
    create_sharded,
    import_arrow_text,
    import_parquet_text,
    import_token_ids,
    open_dataset,
)
from kube.logs import configure_logging, get_logger, log_event
from kube.model import Cell, Kube
from kube.renderer import Renderer


def create(path, shape, states: int = 4, *, overwrite: bool = False) -> Kube:
    """Create a mapped `.kube` file through the package root.

    This wrapper exists so callers can stay on the public `kube` namespace
    instead of reaching into `kube.model.Kube.create(...)` directly.
    """
    return Kube.create(path, shape, states=states, overwrite=overwrite)


def open(path, mode: str = "r+") -> Kube:
    """Open an existing `.kube` file through the package root.

    The function exists as the canonical convenience entry point for mapped
    kube storage.
    """
    return Kube.open(path, mode=mode)


def import_file(path, dst_path, states: int = 4, shape=None) -> Kube:
    """Stream a normal file into a new `.kube` container.

    This helper exists to expose the most common disk-to-kube import path as a
    top-level package function.
    """
    return Kube.import_file(path, dst_path, states=states, shape=shape)


def from_bytes(data: bytes, *, states: int = 4, shape=None) -> Kube:
    """Build an in-memory kube from raw bytes.

    This function exists for quick experiments, tests, and workflows that
    already hold payload bytes in memory.
    """
    return Kube.from_bytes(data, states=states, shape=shape)


def from_file(path, *, states: int = 4, shape=None) -> Kube:
    """Build an in-memory kube by eagerly reading a file.

    This wrapper exists for small files and prototyping workflows where memory
    mapping is not required.
    """
    return Kube.from_file(path, states=states, shape=shape)


def from_polars(frame, *, shape, states: int = 4, path=None, metadata=None) -> Kube:
    """Reconstruct a kube from a Polars frame.

    The function exists to support table-first transformation pipelines where a
    kube is materialized only after a frame has been produced or modified.
    """
    return Kube.from_polars(
        frame,
        shape=shape,
        states=states,
        path=path,
        metadata=metadata,
    )


def load(path, states: int = 4, shape=None) -> Kube:
    """Load either a `.kube`, a `.kubed`, or a raw file based on suffix.

    This helper exists as a convenience dispatcher for tools or notebooks that
    want one "do the obvious thing" entry point.
    """
    path = Path(path)
    if path.suffix.lower() == ".kubed":
        return open_dataset(path)
    if path.suffix.lower() == ".kube":
        return Kube.open(path)
    return Kube.from_file(path, states=states, shape=shape)


__all__ = [
    # Core classes
    "Cell",
    "Kube",
    "ShardedKubeDataset",
    "Renderer",
    # Convenience constructors
    "create",
    "create_sharded",
    "open",
    "open_dataset",
    "import_file",
    "import_parquet_text",
    "import_arrow_text",
    "import_token_ids",
    "from_bytes",
    "from_file",
    "from_polars",
    "load",
    # Logging
    "configure_logging",
    "get_logger",
    "log_event",
    # Schema constants
    "SCHEMA_WIDE_FACES",
    "SCHEMA_LONG_FACES",
    "SCHEMA_CELL_PACKED",
    "SCHEMA_SUPERVISED",
    "SCHEMA_TOKEN_SEQUENCES",
    # Metadata key constants
    "SPLIT",
    "TOKENIZER_NAME",
    "TOKENIZER_PATH",
    "SEQUENCE_LENGTH",
    "FEATURE_SCHEMA",
    "SOURCE_DATASET",
    "PIPELINE_VERSION",
]
