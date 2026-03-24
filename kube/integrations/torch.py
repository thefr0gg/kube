"""PyTorch data-loading adapters for Kube datasets.

This module exists to turn Kube's sequence iterators into objects that fit
naturally into the PyTorch data pipeline model, especially for iterable causal
language-model training loops.
"""

from __future__ import annotations

from typing import Iterator

from kube.dataset import ShardedKubeDataset

try:
    from torch.utils.data import IterableDataset, get_worker_info
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The optional torch integration requires the 'torch' dependency."
    ) from exc


class KubeCausalLMIterableDataset(IterableDataset):
    """Expose `ShardedKubeDataset.iter_sequences(...)` as a Torch dataset.

    The class exists because PyTorch expects an `IterableDataset` object when
    data should be streamed rather than randomly indexed. It also partitions
    shards across worker processes so multiple dataloader workers do not all
    read the same shard subset.

    Attributes:
        dataset: The underlying Kube sharded dataset.
        sequence_length: Number of input tokens per yielded sample.
        stride: Step size between adjacent windows. `None` means "same as
            sequence_length".
        drop_last: Whether to discard a short tail window at the end.
        prefetch_shards: Number of shard loads to overlap ahead of use.
    """

    def __init__(
        self,
        dataset: ShardedKubeDataset,
        *,
        sequence_length: int,
        stride: int | None = None,
        drop_last: bool = True,
        prefetch_shards: int = 0,
    ) -> None:
        """Capture the sequence-streaming settings used by Torch workers."""
        self.dataset = dataset
        self.sequence_length = int(sequence_length)
        self.stride = stride
        self.drop_last = drop_last
        self.prefetch_shards = int(prefetch_shards)

    def __iter__(self) -> Iterator[dict]:
        """Yield causal LM samples in the format expected by Torch code.

        The method exists to bridge Torch worker execution with Kube's shard
        iterator. If multiple workers are active, each worker takes a disjoint
        slice of shard indices.
        """
        worker = get_worker_info()
        shard_indices = list(range(self.dataset.shard_count))
        if worker is not None and worker.num_workers > 1:
            shard_indices = shard_indices[worker.id::worker.num_workers]
        return self.dataset.iter_sequences(
            self.sequence_length,
            stride=self.stride,
            drop_last=self.drop_last,
            shard_indices=shard_indices,
            prefetch_shards=self.prefetch_shards,
        )
