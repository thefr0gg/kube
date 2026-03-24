"""HuggingFace Dataset / IterableDataset adapters for Kube.

This module exists to make Kube a drop-in data source for the HuggingFace
``datasets`` and ``transformers`` ecosystems without requiring ``datasets``
as a hard dependency.

Quick start::

    from kube.integrations.hf import to_hf_dataset, to_hf_iterable_dataset

    # Grid kube → in-memory Dataset
    hf_ds = to_hf_dataset(kube)
    assert "face_0" in hf_ds.column_names

    # Sharded token corpus → streaming IterableDataset
    it_ds = to_hf_iterable_dataset(sharded_dataset, sequence_length=256)
    batch = next(iter(it_ds))
    assert "input_ids" in batch
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kube.dataset import ShardedKubeDataset
    from kube.model import Kube

try:
    import datasets as _datasets
    _DATASETS_AVAILABLE = True
except ImportError:
    _datasets = None  # type: ignore[assignment]
    _DATASETS_AVAILABLE = False


def _require_datasets():
    if not _DATASETS_AVAILABLE:
        raise ImportError(
            "The 'datasets' package is required: pip install datasets"
        )


def to_hf_dataset(
    kube: "Kube",
    *,
    schema_mode: str = "wide_faces",
    bounds=None,
    **kwargs,
):
    """Convert a Kube grid to a HuggingFace in-memory ``Dataset``.

    Routing: ``kube.to_arrow(bounds, schema_mode)``
    → ``datasets.Dataset(arrow_table)``.

    Args:
        kube: Source ``Kube`` instance.
        schema_mode: Column layout; ``"wide_faces"`` (default), ``"long_faces"``,
            or ``"cell_packed"``.
        bounds: Optional region bounds.
        **kwargs: Forwarded to ``datasets.Dataset.__init__``.

    Returns:
        A ``datasets.Dataset`` backed by an Arrow table.
    """
    _require_datasets()
    table = kube.to_arrow(bounds, schema_mode=schema_mode)
    return _datasets.Dataset(table, **kwargs)


def to_hf_iterable_dataset(
    kube_dataset: "ShardedKubeDataset",
    sequence_length: int,
    *,
    stride: int | None = None,
    format_keys: tuple[str, ...] = ("input_ids", "attention_mask", "labels"),
):
    """Wrap a ``ShardedKubeDataset`` as a streaming HuggingFace ``IterableDataset``.

    Each yielded example contains the keys in *format_keys*
    (``input_ids``, ``attention_mask``, ``labels`` by default).

    Args:
        kube_dataset: The underlying sharded Kube corpus.
        sequence_length: Token window length.
        stride: Step between adjacent windows; ``None`` means non-overlapping.
        format_keys: Keys to include in each yielded example.

    Returns:
        A ``datasets.IterableDataset``.
    """
    _require_datasets()

    def _gen():
        for sample in kube_dataset.iter_sequences(sequence_length, stride=stride):
            ids = sample["input_ids"]
            example: dict = {}
            if "input_ids" in format_keys:
                example["input_ids"] = ids
            if "attention_mask" in format_keys:
                example["attention_mask"] = [1] * len(ids)
            if "labels" in format_keys:
                example["labels"] = sample["target_ids"]
            yield example

    return _datasets.IterableDataset.from_generator(_gen)


def from_hf_dataset(
    hf_dataset,
    dst_path: str | Path,
    *,
    text_column: str = "text",
    tokenizer_path: str | Path | None = None,
    sequence_length: int = 1024,
) -> "ShardedKubeDataset":
    """Import a HuggingFace ``Dataset`` into a ``.kubed`` root.

    Reads text from *text_column*, tokenizes it (if a tokenizer is provided),
    and imports the tokens into an existing ``.kubed`` dataset root at
    *dst_path*.

    Args:
        hf_dataset: A ``datasets.Dataset`` or any iterable of dicts.
        dst_path: Path to an existing ``.kubed`` dataset root.
        text_column: Column name containing raw text strings.
        tokenizer_path: Path to a ``tokenizers`` JSON file.  Required unless
            the dataset already contains token ID arrays.
        sequence_length: Sequence length recorded in the manifest after import.

    Returns:
        The updated ``ShardedKubeDataset``.
    """
    import polars as pl
    from kube.dataset import open_dataset

    rows = [{text_column: row[text_column]} for row in hf_dataset]
    frame = pl.DataFrame(rows)

    dataset = open_dataset(dst_path, mode="r+")
    dataset.import_text_frame(
        frame,
        text_column=text_column,
        tokenizer_path=tokenizer_path,
    )
    dataset.update_sequence_defaults(sequence_length=sequence_length)
    return dataset
