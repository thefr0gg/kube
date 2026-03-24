"""Sharded dataset support for token-oriented Kube workflows.

This module exists to build and iterate `.kubed` dataset roots that are aimed
at language-model and large-corpus pipelines. It handles manifest management,
shard creation, text/token import, resumable sequence iteration, and export of
dataset statistics.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import polars as pl

from kube.model import Kube, _shape_for_bytes


DATASET_FORMAT_VERSION = 1
DEFAULT_SEQUENCE_LENGTH = 1024
DEFAULT_STATES = 256
DEFAULT_TOKEN_BYTES = 4
DEFAULT_TOKENS_PER_SHARD = 1 << 20


@dataclass(frozen=True)
class _LoadedShard:
    """Small transport object for one loaded shard.

    The dataclass exists so prefetch helpers can pass around both shard index
    and token payload together without using anonymous tuples.
    """

    index: int
    token_ids: list[int]


def _ensure_dataset_root(root: str | Path) -> Path:
    """Validate that a path uses the `.kubed` dataset-root convention."""
    root = Path(root)
    if root.suffix.lower() != ".kubed":
        raise ValueError("dataset root must end with .kubed")
    return root


def _manifest_path(root: Path) -> Path:
    """Return the manifest location for a dataset root."""
    return root / "manifest.json"


def _shards_dir(root: Path) -> Path:
    """Return the directory that stores shard `.kube` files."""
    return root / "shards"


def _stats_dir(root: Path) -> Path:
    """Return the directory that stores derived dataset statistics."""
    return root / "stats"


def _documents_path(root: Path) -> Path:
    """Return the standard Parquet path for per-document stats."""
    return _stats_dir(root) / "documents.parquet"


def _tokenizer_copy_path(root: Path) -> Path:
    """Return the dataset-local tokenizer copy path."""
    return root / "tokenizer.json"


def _read_table(path: str | Path) -> pl.DataFrame:
    """Load a supported tabular input file into Polars.

    The helper exists because dataset import commands accept both Parquet and
    Arrow/IPC sources.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in {".arrow", ".ipc", ".feather"}:
        return pl.read_ipc(path)
    raise ValueError(f"unsupported table format for {path}")


def _require_tokenizers():
    """Import the optional `tokenizers` dependency with a clear error."""
    try:
        from tokenizers import Tokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Built-in text tokenization requires the optional 'tokenizers' dependency. "
            "Install it with Poetry extras or pip before using text-import helpers."
        ) from exc
    return Tokenizer


def _load_tokenizer(tokenizer=None, tokenizer_path: str | Path | None = None):
    """Return a tokenizer instance from an object or on-disk path.

    This helper exists so text-import APIs can accept either a preloaded
    tokenizer or a path to one.
    """
    if tokenizer is not None:
        return tokenizer
    if tokenizer_path is None:
        raise ValueError("either tokenizer or tokenizer_path must be provided")
    Tokenizer = _require_tokenizers()
    return Tokenizer.from_file(str(tokenizer_path))


def _infer_eos_token_id(tokenizer) -> int | None:
    """Best-effort inference of a tokenizer EOS token ID.

    The function exists to make text-import helpers usable with fewer required
    arguments when a common EOS token is present.
    """
    token_to_id = getattr(tokenizer, "token_to_id", None)
    if callable(token_to_id):
        for token in ("</s>", "<|endoftext|>", "<eos>", "[EOS]"):
            token_id = token_to_id(token)
            if token_id is not None:
                return int(token_id)
    return None


class ShardedKubeDataset:
    """High-level representation of a `.kubed` dataset root.

    The class exists to group together:

    - the dataset manifest
    - the shard layout
    - text/token ingestion logic
    - sequence iteration with resume-state support
    - dataset statistics export

    Attributes:
        root: Filesystem path to the dataset root directory.
        manifest: In-memory manifest dictionary describing shards and defaults.
        mode: Access mode, usually `"r"` or `"r+"`.
    """

    def __init__(self, root: str | Path, manifest: dict[str, Any], *, mode: str = "r"):
        """Create a dataset wrapper around an existing or newly built manifest."""
        self.root = _ensure_dataset_root(root)
        self.manifest = manifest
        self.mode = mode

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        shards: int,
        states: int = DEFAULT_STATES,
        token_bytes: int = DEFAULT_TOKEN_BYTES,
        tokens_per_shard: int = DEFAULT_TOKENS_PER_SHARD,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        dataset_name: str | None = None,
        tokenizer_metadata: dict[str, Any] | None = None,
        creation_parameters: dict[str, Any] | None = None,
    ) -> "ShardedKubeDataset":
        """Create a new empty sharded dataset root on disk.

        This classmethod exists for the "prepare storage first, ingest later"
        workflow common in LLM preprocessing pipelines.
        """
        root = _ensure_dataset_root(root)
        if shards <= 0:
            raise ValueError("shards must be positive")
        if token_bytes not in {1, 2, 4, 8}:
            raise ValueError("token_bytes must be one of 1, 2, 4, or 8")
        if tokens_per_shard <= 0:
            raise ValueError("tokens_per_shard must be positive")

        root.mkdir(parents=True, exist_ok=True)
        _shards_dir(root).mkdir(parents=True, exist_ok=True)
        _stats_dir(root).mkdir(parents=True, exist_ok=True)

        shard_specs = []
        shard_capacity_bytes = tokens_per_shard * token_bytes
        shard_shape = _shape_for_bytes(shard_capacity_bytes, states)
        for index in range(shards):
            name = f"shard-{index:06d}.kube"
            path = _shards_dir(root) / name
            kube = Kube.create(
                path,
                shard_shape,
                states=states,
                source_byte_count=0,
                metadata={
                    "dataset_role": "llm-shard",
                    "token_bytes": token_bytes,
                    "token_count": 0,
                    "document_count": 0,
                    "shard_index": index,
                },
            )
            kube.close()
            shard_specs.append(
                {
                    "index": index,
                    "name": name,
                    "path": f"shards/{name}",
                    "shape": list(shard_shape),
                    "token_count": 0,
                    "byte_count": 0,
                    "document_count": 0,
                }
            )

        manifest = {
            "format": "kubed",
            "format_version": DATASET_FORMAT_VERSION,
            "dataset_name": dataset_name or root.stem,
            "states": states,
            "itemsize": Kube(*shard_shape, states=states).itemsize,
            "token_bytes": token_bytes,
            "tokens_per_shard": tokens_per_shard,
            "total_token_count": 0,
            "total_document_count": 0,
            "tokenizer_metadata": dict(tokenizer_metadata or {}),
            "sequence_defaults": {
                "sequence_length": int(sequence_length),
                "stride": int(sequence_length),
            },
            "creation_parameters": dict(creation_parameters or {}),
            "documents_path": str(_documents_path(root).relative_to(root)),
            "shards": shard_specs,
        }
        dataset = cls(root, manifest, mode="r+")
        dataset._write_manifest()
        return dataset

    @classmethod
    def open(cls, root: str | Path, mode: str = "r") -> "ShardedKubeDataset":
        """Open an existing dataset by reading its manifest."""
        root = _ensure_dataset_root(root)
        manifest = json.loads(_manifest_path(root).read_text(encoding="utf-8"))
        return cls(root, manifest, mode=mode)

    def _write_manifest(self) -> None:
        """Persist the in-memory manifest back to disk."""
        if self.mode == "r":
            raise PermissionError("dataset is opened in read-only mode")
        _manifest_path(self.root).write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    def close(self) -> None:
        """Provide a no-op close method for API symmetry with `Kube`.

        The dataset itself does not keep shards permanently open, but a `close`
        method exists so callers can use datasets in context-manager style.
        """
        return None

    def __enter__(self) -> "ShardedKubeDataset":
        """Support `with` statements for dataset wrappers."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Match the context-manager protocol with a no-op cleanup hook."""
        self.close()

    def _ensure_writable(self) -> None:
        """Reject mutating dataset operations in read-only mode."""
        if self.mode == "r":
            raise PermissionError("dataset is opened in read-only mode")

    @property
    def shard_count(self) -> int:
        """Return the number of shard entries in the manifest."""
        return len(self.manifest["shards"])

    @property
    def token_bytes(self) -> int:
        """Return the configured fixed width of stored token IDs."""
        return int(self.manifest["token_bytes"])

    @property
    def total_token_count(self) -> int:
        """Return the total token count across all shards."""
        return int(self.manifest["total_token_count"])

    @property
    def total_document_count(self) -> int:
        """Return the total logical document count across the dataset."""
        return int(self.manifest["total_document_count"])

    @property
    def default_sequence_length(self) -> int:
        """Return the manifest's default sequence length for iteration."""
        return int(self.manifest["sequence_defaults"]["sequence_length"])

    def inspect(self) -> dict[str, Any]:
        """Return a compact summary suitable for CLI or notebook inspection."""
        tokenizer_file = _tokenizer_copy_path(self.root)
        return {
            "root": str(self.root),
            "dataset_name": self.manifest["dataset_name"],
            "format_version": self.manifest["format_version"],
            "states": self.manifest["states"],
            "token_bytes": self.token_bytes,
            "shard_count": self.shard_count,
            "total_token_count": self.total_token_count,
            "total_document_count": self.total_document_count,
            "sequence_defaults": dict(self.manifest["sequence_defaults"]),
            "tokenizer_metadata": dict(self.manifest["tokenizer_metadata"]),
            "tokenizer_file": str(tokenizer_file) if tokenizer_file.exists() else None,
        }

    def update_sequence_defaults(self, *, sequence_length: int, stride: int | None = None) -> None:
        """Update the manifest's default sequence iteration settings.

        This exists so ingest commands can set dataset-local defaults that later
        tooling can reuse.
        """
        self._ensure_writable()
        if stride is None:
            stride = sequence_length
        self.manifest["sequence_defaults"] = {
            "sequence_length": int(sequence_length),
            "stride": int(stride),
        }
        self._write_manifest()

    def shard_frame(self) -> pl.DataFrame:
        """Return the shard manifest rows as a Polars frame."""
        return pl.DataFrame(self.manifest["shards"])

    def document_frame(self) -> pl.DataFrame:
        """Return per-document stats if they have been materialized.

        The method exists so users can inspect logical document boundaries after
        ingestion.
        """
        path = self.root / self.manifest.get("documents_path", "")
        if not path.exists():
            return pl.DataFrame(
                {
                    "document_index": [],
                    "start_token": [],
                    "token_count": [],
                    "start_shard_index": [],
                }
            )
        return pl.read_parquet(path)

    def export_stats_parquet(self, path: str | Path) -> Path:
        """Write shard-level statistics to a Parquet file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = self.shard_frame().with_columns(
            pl.lit(self.manifest["dataset_name"]).alias("dataset_name"),
            pl.lit(self.manifest["states"]).alias("states"),
            pl.lit(self.token_bytes).alias("token_bytes"),
            pl.lit(self.total_token_count).alias("total_token_count"),
            pl.lit(self.total_document_count).alias("total_document_count"),
            pl.lit(json.dumps(self.manifest["tokenizer_metadata"], sort_keys=True)).alias("tokenizer_metadata"),
        )
        frame.write_parquet(path)
        return path

    def export_document_parquet(self, path: str | Path) -> Path:
        """Write the per-document stats table to Parquet."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.document_frame().write_parquet(path)
        return path

    def export_sequence_sample_parquet(
        self,
        path: str | Path,
        *,
        sequence_length: int,
        count: int,
        stride: int | None = None,
    ) -> Path:
        """Export a sampled set of sequence windows to Parquet.

        This helper exists for lightweight inspection outside the runtime API.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_sequences_frame(sequence_length=sequence_length, count=count, stride=stride).write_parquet(path)
        return path

    def save_resume_state(self, path: str | Path, state: dict[str, Any]) -> Path:
        """Persist a sequence-iterator resume state to JSON.

        Resume states exist so long preprocessing or training scans can restart
        from a known token offset instead of starting over.
        """
        if Path(state["manifest_path"]) != _manifest_path(self.root):
            raise ValueError("resume state does not belong to this dataset")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return path

    def load_resume_state(self, path: str | Path) -> dict[str, Any]:
        """Load and validate a saved resume state."""
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if Path(state["manifest_path"]) != _manifest_path(self.root):
            raise ValueError("resume state does not belong to this dataset")
        return state

    def _shard_path(self, shard_index: int) -> Path:
        """Resolve one shard index into its on-disk `.kube` path."""
        return self.root / self.manifest["shards"][shard_index]["path"]

    def _clear_dataset_storage(self) -> None:
        """Reset every shard and derived stats artifact in the dataset.

        This private helper exists so new imports start from a clean, consistent
        dataset state.
        """
        self._ensure_writable()
        for shard in self.manifest["shards"]:
            kube = Kube.open(self.root / shard["path"], mode="r+")
            kube.encode(b"")
            kube.update_metadata(
                dataset_role="llm-shard",
                token_bytes=self.token_bytes,
                token_count=0,
                document_count=0,
                shard_index=int(shard["index"]),
            )
            kube.close()
            shard["token_count"] = 0
            shard["byte_count"] = 0
            shard["document_count"] = 0
        documents = _documents_path(self.root)
        if documents.exists():
            documents.unlink()
        self.manifest["total_token_count"] = 0
        self.manifest["total_document_count"] = 0
        self._write_manifest()

    def _flush_shard_tokens(self, shard_index: int, token_ids: Sequence[int], *, document_count: int) -> None:
        """Write one shard's token payload and refresh its manifest row."""
        shard = self.manifest["shards"][shard_index]
        kube = Kube.open(self.root / shard["path"], mode="r+")
        kube.encode_token_ids(token_ids, token_bytes=self.token_bytes)
        kube.update_metadata(
            dataset_role="llm-shard",
            token_bytes=self.token_bytes,
            token_count=len(token_ids),
            document_count=document_count,
            shard_index=shard_index,
        )
        kube.close()
        shard["token_count"] = len(token_ids)
        shard["byte_count"] = len(token_ids) * self.token_bytes
        shard["document_count"] = document_count

    def _empty_remaining_shards(self, start_index: int) -> None:
        """Zero all shards that were not filled by the current import."""
        for shard_index in range(start_index, self.shard_count):
            self._flush_shard_tokens(shard_index, [], document_count=0)

    def _ingest_documents(
        self,
        documents: Iterator[dict[str, Any]],
        *,
        tokenizer_metadata: dict[str, Any] | None = None,
    ) -> "ShardedKubeDataset":
        """Pack a stream of tokenized documents into dataset shards.

        This is the core ingest routine. It exists so both text-import and
        token-import paths can share the same shard-filling logic and document
        bookkeeping.
        """
        self._ensure_writable()
        self._clear_dataset_storage()

        shard_capacity = int(self.manifest["tokens_per_shard"])
        current_shard = 0
        shard_tokens: list[int] = []
        shard_document_count = 0
        total_tokens = 0
        total_documents = 0
        document_rows: list[dict[str, Any]] = []

        for document in documents:
            token_ids = [int(token_id) for token_id in document["token_ids"]]
            if current_shard >= self.shard_count and token_ids:
                raise ValueError("dataset capacity exceeded; create more shards or increase tokens_per_shard")
            doc_start_token = total_tokens
            doc_start_shard = current_shard

            while token_ids:
                if current_shard >= self.shard_count:
                    raise ValueError("dataset capacity exceeded; create more shards or increase tokens_per_shard")
                available = shard_capacity - len(shard_tokens)
                if available <= 0:
                    self._flush_shard_tokens(current_shard, shard_tokens, document_count=shard_document_count)
                    current_shard += 1
                    shard_tokens = []
                    shard_document_count = 0
                    continue
                take = min(available, len(token_ids))
                shard_tokens.extend(token_ids[:take])
                token_ids = token_ids[take:]
                total_tokens += take
            shard_document_count += 1
            document_rows.append(
                {
                    "document_index": total_documents,
                    "start_token": doc_start_token,
                    "token_count": int(document["token_count"]),
                    "start_shard_index": doc_start_shard,
                }
            )
            total_documents += 1

        if current_shard < self.shard_count:
            self._flush_shard_tokens(current_shard, shard_tokens, document_count=shard_document_count)
            current_shard += 1
        self._empty_remaining_shards(current_shard)

        self.manifest["total_token_count"] = total_tokens
        self.manifest["total_document_count"] = total_documents
        if tokenizer_metadata is not None:
            self.manifest["tokenizer_metadata"] = dict(tokenizer_metadata)
        self._write_manifest()

        if document_rows:
            documents_frame = pl.DataFrame(document_rows)
            _stats_dir(self.root).mkdir(parents=True, exist_ok=True)
            documents_frame.write_parquet(_documents_path(self.root))
        return self

    def import_token_frame(
        self,
        frame: pl.DataFrame,
        *,
        tokens_column: str = "input_ids",
        eos_token_id: int | None = None,
    ) -> "ShardedKubeDataset":
        """Import a frame containing pre-tokenized rows.

        The method exists for workflows where tokenization happened upstream and
        Kube only needs to store and shard the token IDs.
        """
        if tokens_column not in frame.columns:
            raise ValueError(f"frame must contain '{tokens_column}'")

        def documents() -> Iterator[dict[str, Any]]:
            for row in frame.select(tokens_column).iter_rows(named=True):
                token_ids = [int(token_id) for token_id in row[tokens_column]]
                if eos_token_id is not None:
                    token_ids.append(int(eos_token_id))
                yield {"token_ids": token_ids, "token_count": len(token_ids)}

        tokenizer_metadata = dict(self.manifest.get("tokenizer_metadata", {}))
        if eos_token_id is not None:
            tokenizer_metadata["eos_token_id"] = int(eos_token_id)
        return self._ingest_documents(documents(), tokenizer_metadata=tokenizer_metadata)

    def import_text_frame(
        self,
        frame: pl.DataFrame,
        *,
        text_column: str = "text",
        tokenizer=None,
        tokenizer_path: str | Path | None = None,
        eos_token_id: int | None = None,
    ) -> "ShardedKubeDataset":
        """Tokenize a text frame and import it into dataset shards.

        This method exists for the common convenience path where Kube handles
        tokenization during ingestion.
        """
        if text_column not in frame.columns:
            raise ValueError(f"frame must contain '{text_column}'")
        tokenizer = _load_tokenizer(tokenizer=tokenizer, tokenizer_path=tokenizer_path)
        if eos_token_id is None:
            eos_token_id = _infer_eos_token_id(tokenizer)

        def documents() -> Iterator[dict[str, Any]]:
            for row in frame.select(text_column).iter_rows(named=True):
                text = "" if row[text_column] is None else str(row[text_column])
                encoding = tokenizer.encode(text)
                token_ids = [int(token_id) for token_id in encoding.ids]
                if eos_token_id is not None:
                    token_ids.append(int(eos_token_id))
                yield {"token_ids": token_ids, "token_count": len(token_ids)}

        tokenizer_metadata = {
            "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
            "eos_token_id": eos_token_id,
        }
        if tokenizer_path is not None:
            shutil.copy2(tokenizer_path, _tokenizer_copy_path(self.root))
        return self._ingest_documents(documents(), tokenizer_metadata=tokenizer_metadata)

    def _load_shard_tokens(self, shard_index: int) -> _LoadedShard:
        """Load one shard's token payload into memory."""
        kube = Kube.open(self._shard_path(shard_index), mode="r")
        try:
            token_ids = kube.read_token_ids(token_bytes=self.token_bytes)
        finally:
            kube.close()
        return _LoadedShard(shard_index, token_ids)

    def _iter_loaded_shards(
        self,
        shard_indices: Sequence[int],
        *,
        prefetch_shards: int = 0,
    ) -> Iterator[_LoadedShard]:
        """Yield loaded shards, optionally with one-shard-ahead prefetch.

        The iterator exists to hide the prefetching mechanics from higher-level
        sequence iteration code.
        """
        shard_indices = list(shard_indices)
        if prefetch_shards <= 0 or len(shard_indices) <= 1:
            for shard_index in shard_indices:
                yield self._load_shard_tokens(shard_index)
            return

        with ThreadPoolExecutor(max_workers=1) as executor:
            future: Future[_LoadedShard] | None = executor.submit(self._load_shard_tokens, shard_indices[0])
            for next_index in shard_indices[1:]:
                assert future is not None
                current = future.result()
                future = executor.submit(self._load_shard_tokens, next_index)
                yield current
            assert future is not None
            yield future.result()

    def _prefix_token_counts(self) -> list[int]:
        """Build cumulative token counts per shard.

        This helper exists so global token offsets can be translated into
        shard-local coordinates.
        """
        prefix = [0]
        running = 0
        for shard in self.manifest["shards"]:
            running += int(shard["token_count"])
            prefix.append(running)
        return prefix

    def _state_from_global(
        self,
        global_token_offset: int,
        *,
        sequence_length: int,
        stride: int,
        epoch: int,
        shuffle_seed: int | None,
    ) -> dict[str, Any]:
        """Convert a global token offset into a validated resume-state object.

        Resume states exist because long scans over sharded datasets need a
        stable, serializable checkpoint format.
        """
        prefix = self._prefix_token_counts()
        bounded = min(max(0, int(global_token_offset)), self.total_token_count)
        shard_index = max(0, self.shard_count - 1)
        local_offset = 0
        for index in range(self.shard_count):
            if bounded < prefix[index + 1] or index == self.shard_count - 1:
                shard_index = index
                local_offset = bounded - prefix[index]
                break
        return {
            "manifest_path": str(_manifest_path(self.root)),
            "root": str(self.root),
            "shard_index": shard_index,
            "token_offset": int(local_offset),
            "sequence_length": int(sequence_length),
            "stride": int(stride),
            "epoch": int(epoch),
            "shuffle_seed": shuffle_seed,
        }

    def iter_sequences(
        self,
        sequence_length: int,
        *,
        stride: int | None = None,
        drop_last: bool = True,
        start_state: dict[str, Any] | None = None,
        shard_indices: Sequence[int] | None = None,
        prefetch_shards: int = 0,
        epoch: int = 0,
        shuffle_seed: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield causal LM samples across all selected shards.

        Each sample contains:

        - `input_ids`
        - `target_ids`
        - `shard_index`
        - `token_offset`
        - `resume_state`

        The method exists as the core streaming interface for language-model
        training and inspection workflows.
        """
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if stride is None:
            stride = sequence_length
        stride = int(stride)
        if stride <= 0:
            raise ValueError("stride must be positive")

        all_indices = list(range(self.shard_count)) if shard_indices is None else [int(index) for index in shard_indices]
        prefix = self._prefix_token_counts()
        start_global = 0
        if start_state is not None:
            if Path(start_state["manifest_path"]) != _manifest_path(self.root):
                raise ValueError("resume state does not belong to this dataset")
            shard_index = int(start_state["shard_index"])
            local_offset = int(start_state["token_offset"])
            start_global = prefix[shard_index] + local_offset

        buffer: list[int] = []
        buffer_start_global = prefix[all_indices[0]] if all_indices else 0
        next_sample_global = start_global
        initialized = False

        for loaded in self._iter_loaded_shards(all_indices, prefetch_shards=prefetch_shards):
            shard_start = prefix[loaded.index]
            shard_end = prefix[loaded.index + 1]
            tokens = loaded.token_ids
            if not initialized:
                buffer_start_global = shard_start
                initialized = True
            if start_global >= shard_end:
                continue
            if start_global > shard_start:
                tokens = tokens[start_global - shard_start:]
                shard_start = start_global
                if not buffer:
                    buffer_start_global = shard_start
                    next_sample_global = start_global
            if not buffer:
                buffer_start_global = shard_start
            buffer.extend(tokens)

            while next_sample_global + sequence_length < buffer_start_global + len(buffer):
                local_start = next_sample_global - buffer_start_global
                window = buffer[local_start:local_start + sequence_length + 1]
                sample_origin = self._state_from_global(
                    next_sample_global,
                    sequence_length=sequence_length,
                    stride=stride,
                    epoch=epoch,
                    shuffle_seed=shuffle_seed,
                )
                sample = {
                    "input_ids": window[:-1],
                    "target_ids": window[1:],
                    "shard_index": sample_origin["shard_index"],
                    "token_offset": sample_origin["token_offset"],
                }
                next_state = self._state_from_global(
                    next_sample_global + stride,
                    sequence_length=sequence_length,
                    stride=stride,
                    epoch=epoch,
                    shuffle_seed=shuffle_seed,
                )
                sample["resume_state"] = next_state
                yield sample
                next_sample_global += stride
                drop = next_sample_global - buffer_start_global
                if drop > 0:
                    buffer = buffer[drop:]
                    buffer_start_global = next_sample_global

        if not drop_last and len(buffer) > 1:
            sample = {
                "input_ids": buffer[:-1],
                "target_ids": buffer[1:],
                "shard_index": int(self.manifest["shards"][-1]["index"]) if self.manifest["shards"] else 0,
                "token_offset": 0,
                "resume_state": self._state_from_global(
                    self.total_token_count,
                    sequence_length=sequence_length,
                    stride=stride,
                    epoch=epoch,
                    shuffle_seed=shuffle_seed,
                ),
            }
            yield sample

    def iter_sequence_batches(
        self,
        batch_size: int,
        sequence_length: int,
        *,
        stride: int | None = None,
        drop_last: bool = True,
        start_state: dict[str, Any] | None = None,
        shard_indices: Sequence[int] | None = None,
        prefetch_shards: int = 0,
        epoch: int = 0,
        shuffle_seed: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Group sequence samples into batch dictionaries.

        This exists because many training loops want simple Python lists of
        samples before converting them into framework tensors.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batch: list[dict[str, Any]] = []
        for sample in self.iter_sequences(
            sequence_length,
            stride=stride,
            drop_last=drop_last,
            start_state=start_state,
            shard_indices=shard_indices,
            prefetch_shards=prefetch_shards,
            epoch=epoch,
            shuffle_seed=shuffle_seed,
        ):
            batch.append(sample)
            if len(batch) >= batch_size:
                yield {
                    "input_ids": [item["input_ids"] for item in batch],
                    "target_ids": [item["target_ids"] for item in batch],
                    "shard_indices": [item["shard_index"] for item in batch],
                    "token_offsets": [item["token_offset"] for item in batch],
                    "resume_state": batch[-1]["resume_state"],
                }
                batch = []
        if batch:
            yield {
                "input_ids": [item["input_ids"] for item in batch],
                "target_ids": [item["target_ids"] for item in batch],
                "shard_indices": [item["shard_index"] for item in batch],
                "token_offsets": [item["token_offset"] for item in batch],
                "resume_state": batch[-1]["resume_state"],
            }

    def resume_iterator(self, state: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Resume sequence iteration from a previously saved resume state."""
        return self.iter_sequences(
            state["sequence_length"],
            stride=state["stride"],
            start_state=state,
            epoch=state.get("epoch", 0),
            shuffle_seed=state.get("shuffle_seed"),
        )

    def sample_sequences_frame(
        self,
        *,
        sequence_length: int,
        count: int,
        stride: int | None = None,
    ) -> pl.DataFrame:
        """Materialize a small sample of sequence windows as a Polars frame."""
        rows = []
        for sequence_index, sample in enumerate(self.iter_sequences(sequence_length, stride=stride)):
            if sequence_index >= count:
                break
            rows.append(
                {
                    "sequence_index": sequence_index,
                    "shard_index": sample["shard_index"],
                    "token_offset": sample["token_offset"],
                    "input_ids": sample["input_ids"],
                    "target_ids": sample["target_ids"],
                }
            )
        return pl.DataFrame(rows)

    def iter_sequence_frame_batches(
        self,
        batch_size: int,
        sequence_length: int,
        *,
        stride: int | None = None,
        drop_last: bool = True,
        start_state: dict[str, Any] | None = None,
        shard_indices: Sequence[int] | None = None,
        prefetch_shards: int = 0,
        epoch: int = 0,
        shuffle_seed: int | None = None,
    ) -> Iterator[pl.DataFrame]:
        """Yield Polars DataFrame batches of sequence samples.

        The method exists for downstream tooling that prefers tabular batches
        over plain Python batch dictionaries.
        """
        for batch_index, batch in enumerate(
            self.iter_sequence_batches(
                batch_size,
                sequence_length,
                stride=stride,
                drop_last=drop_last,
                start_state=start_state,
                shard_indices=shard_indices,
                prefetch_shards=prefetch_shards,
                epoch=epoch,
                shuffle_seed=shuffle_seed,
            )
        ):
            yield pl.DataFrame(
                {
                    "batch_index": [batch_index] * len(batch["input_ids"]),
                    "sample_index": list(range(len(batch["input_ids"]))),
                    "shard_index": batch["shard_indices"],
                    "token_offset": batch["token_offsets"],
                    "input_ids": batch["input_ids"],
                    "target_ids": batch["target_ids"],
                }
            )


def create_sharded(
    root: str | Path,
    *,
    shards: int,
    states: int = DEFAULT_STATES,
    token_bytes: int = DEFAULT_TOKEN_BYTES,
    tokens_per_shard: int = DEFAULT_TOKENS_PER_SHARD,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    dataset_name: str | None = None,
) -> ShardedKubeDataset:
    """Public convenience wrapper for `ShardedKubeDataset.create(...)`."""
    return ShardedKubeDataset.create(
        root,
        shards=shards,
        states=states,
        token_bytes=token_bytes,
        tokens_per_shard=tokens_per_shard,
        sequence_length=sequence_length,
        dataset_name=dataset_name,
    )


def open_dataset(root: str | Path, mode: str = "r") -> ShardedKubeDataset:
    """Public convenience wrapper for `ShardedKubeDataset.open(...)`."""
    return ShardedKubeDataset.open(root, mode=mode)


def import_parquet_text(
    src_path: str | Path,
    root: str | Path,
    *,
    text_column: str = "text",
    tokenizer=None,
    tokenizer_path: str | Path | None = None,
    eos_token_id: int | None = None,
) -> ShardedKubeDataset:
    """Read a Parquet text table and import it into a dataset root.

    This helper exists to provide one-step "load table then ingest" behavior at
    the package level.
    """
    dataset = open_dataset(root, mode="r+")
    return dataset.import_text_frame(
        _read_table(src_path),
        text_column=text_column,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        eos_token_id=eos_token_id,
    )


def import_arrow_text(
    src_path: str | Path,
    root: str | Path,
    *,
    text_column: str = "text",
    tokenizer=None,
    tokenizer_path: str | Path | None = None,
    eos_token_id: int | None = None,
) -> ShardedKubeDataset:
    """Read an Arrow/IPC text table and import it into a dataset root."""
    dataset = open_dataset(root, mode="r+")
    return dataset.import_text_frame(
        _read_table(src_path),
        text_column=text_column,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        eos_token_id=eos_token_id,
    )


def import_token_ids(
    src_path: str | Path,
    root: str | Path,
    *,
    tokens_column: str = "input_ids",
    eos_token_id: int | None = None,
) -> ShardedKubeDataset:
    """Read a token-ID table and import it into a dataset root."""
    dataset = open_dataset(root, mode="r+")
    return dataset.import_token_frame(
        _read_table(src_path),
        tokens_column=tokens_column,
        eos_token_id=eos_token_id,
    )
