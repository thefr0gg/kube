"""PyTorch data-loading adapters for Kube datasets and grids.

This module exists to turn Kube's sequence iterators and grid data into
objects that fit naturally into the PyTorch data pipeline model, especially
for iterable causal language-model training loops and spatial ML workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from kube.dataset import ShardedKubeDataset

if TYPE_CHECKING:
    from kube.model import Kube

try:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset, IterableDataset, get_worker_info
    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    Tensor = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]
    IterableDataset = object  # type: ignore[assignment,misc]
    get_worker_info = None  # type: ignore[assignment]


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ModuleNotFoundError(
            "The optional torch integration requires the 'torch' dependency."
        )


class KubeCausalLMIterableDataset(IterableDataset):  # type: ignore[misc]
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


class KubeDataset(Dataset):  # type: ignore[misc]
    """Map-style PyTorch Dataset over a Kube grid's feature frame.

    The full feature frame is materialised once at construction time and held
    in memory.  Suitable for small-to-medium grids.

    Attributes:
        kube: The source ``Kube`` instance.
        bounds: Optional region bounds; ``None`` means the full cube.
        target_face: If set, the named face is returned as a separate
            ``"target"`` tensor alongside ``"features"``.
        dtype: Torch dtype for the feature and target tensors.
    """

    def __init__(
        self,
        kube: "Kube",
        *,
        bounds=None,
        target_face: int | None = None,
        dtype=None,
    ) -> None:
        _require_torch()
        self.kube = kube
        self.bounds = bounds
        self.target_face = target_face
        self.dtype = dtype if dtype is not None else torch.float32

        import polars as pl

        if target_face is not None:
            frame = kube.supervised_frame(
                target_face,
                bounds,
                include_coords=False,
                include_packed=False,
                target_name="target",
                drop_target_face=True,
            )
            feature_cols = [c for c in frame.columns if c != "target"]
            import numpy as np
            self._features = torch.tensor(
                np.stack([frame[c].to_numpy() for c in feature_cols], axis=1),
                dtype=self.dtype,
            )
            self._targets = torch.tensor(frame["target"].to_numpy(), dtype=self.dtype)
        else:
            frame = kube.feature_frame(bounds, include_coords=False, include_packed=False)
            import numpy as np
            self._features = torch.tensor(
                np.stack([frame[c].to_numpy() for c in kube.face_columns], axis=1),
                dtype=self.dtype,
            )
            self._targets = None

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> dict:
        sample: dict = {"features": self._features[idx]}
        if self._targets is not None:
            sample["target"] = self._targets[idx]
        return sample


class KubeRegionDataset(Dataset):  # type: ignore[misc]
    """Map-style PyTorch Dataset that iterates tiles over a bounded region.

    Each item is a ``(tile_W, tile_H, tile_D, 6)`` float tensor covering one
    spatial tile.

    Attributes:
        kube: The source ``Kube`` instance.
        bounds: Region bounds ``((x0,x1),(y0,y1),(z0,z1))``.
        tile_shape: ``(tw, th, td)`` shape of each tile.
        dtype: Torch dtype for the output tensors.
    """

    def __init__(
        self,
        kube: "Kube",
        bounds,
        tile_shape: tuple[int, int, int] = (32, 32, 32),
        *,
        dtype=None,
    ) -> None:
        _require_torch()
        self.kube = kube
        self.bounds = bounds
        self.tile_shape = tile_shape
        self.dtype = dtype if dtype is not None else torch.float32

        (x0, x1), (y0, y1), (z0, z1) = bounds
        tw, th, td = tile_shape
        import math
        self._nx = math.ceil((x1 - x0) / tw)
        self._ny = math.ceil((y1 - y0) / th)
        self._nz = math.ceil((z0 - z0 + (z1 - z0)) / td)
        self._nz = math.ceil((z1 - z0) / td)
        self._x0, self._y0, self._z0 = x0, y0, z0

    def __len__(self) -> int:
        return self._nx * self._ny * self._nz

    def __getitem__(self, idx: int):
        _require_torch()
        tw, th, td = self.tile_shape
        iz = idx % self._nz
        iy = (idx // self._nz) % self._ny
        ix = idx // (self._nz * self._ny)

        x0 = self._x0 + ix * tw
        y0 = self._y0 + iy * th
        z0 = self._z0 + iz * td
        x1 = min(x0 + tw, self._x0 + self._nx * tw)
        y1 = min(y0 + th, self._y0 + self._ny * th)
        z1 = min(z0 + td, self._z0 + self._nz * td)
        tile_bounds = ((x0, x1), (y0, y1), (z0, z1))

        arr = self.kube.faces_to_numpy()
        tile = arr[x0:x1, y0:y1, z0:z1, :]
        return torch.tensor(tile, dtype=self.dtype)


class KubeSupervisedDataset(Dataset):  # type: ignore[misc]
    """Map-style PyTorch Dataset that wraps ``supervised_frame()``.

    Each item is a ``(features_tensor, target_tensor)`` pair.

    Attributes:
        kube: The source ``Kube`` instance.
        target_face: Face index (0–5) used as the label.
        bounds: Optional region bounds; ``None`` means the full cube.
        dtype: Torch dtype for both tensors.
    """

    def __init__(
        self,
        kube: "Kube",
        target_face: int,
        bounds=None,
        *,
        dtype=None,
    ) -> None:
        _require_torch()
        import numpy as np
        self.dtype = dtype if dtype is not None else torch.float32
        X, y = kube.to_sklearn_xy(target_face, bounds)
        self._X = torch.tensor(X, dtype=self.dtype)
        self._y = torch.tensor(y, dtype=self.dtype)

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int) -> tuple:
        return self._X[idx], self._y[idx]


def collate_kube_batch(samples: list[dict]) -> dict:
    """Stack a list of sample dicts into batched tensors.

    Each value in the sample dicts is stacked along a new batch dimension.
    """
    _require_torch()
    keys = samples[0].keys()
    return {k: torch.stack([s[k] for s in samples]) for k in keys}


def to_torch(kube: "Kube", *, dtype=None, device=None) -> dict:
    """Convert a Kube grid to a dict of tensors keyed by face name.

    Args:
        kube: Source ``Kube`` instance.
        dtype: Torch dtype; defaults to ``torch.float32``.
        device: Target device; ``None`` means CPU.

    Returns:
        Dict mapping ``"face_0"`` … ``"face_5"`` to 1-D tensors of shape
        ``(N,)`` where ``N`` is the total cell count.
    """
    _require_torch()
    _dtype = dtype if dtype is not None else torch.float32
    frame = kube.feature_frame(include_coords=False, include_packed=False)
    result = {}
    for col in kube.face_columns:
        import numpy as np
        t = torch.tensor(frame[col].to_numpy(), dtype=_dtype)
        if device is not None:
            t = t.to(device)
        result[col] = t
    return result
