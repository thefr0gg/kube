# Changelog

All notable changes to Kube are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [PEP 440](https://peps.python.org/pep-0440/) (e.g. `0.0.1b1` for the first beta).

---

## [0.1.0] – 2026-11-07

First release outside of beta.

### Web viewer

- Replaced the native raylib + Clay renderer with a FastAPI backend and a
  WebGL (three.js) frontend (`kube view`) — no native renderer dependency,
  no build step for the frontend
- `kube/web/server.py`: `/api/state`, `/api/view`, `/api/mesh`, `/api/cell`,
  `/ws/edit` WebSocket
- `kube/web/mesh.py`: vectorized numpy shell/slice mesh builder
- `kube/web/serve.py`: port selection, uvicorn startup, browser launch

### Performance

- `to_polars` (and everything that routes through it — `feature_frame`,
  `faces_to_numpy`, `to_pandas`, `to_arrow`, `to_parquet`, `to_sklearn_xy`)
  now unpacks faces via vectorized numpy instead of a per-cell Python loop
- `flush()` skips the on-disk header rewrite when metadata hasn't changed,
  cutting the viewer's per-edit round trip from ~130 to ~15k edits/s
- `read_token_ids` decodes via `np.frombuffer` instead of a per-token
  `int.from_bytes` loop
- native batch loops (`set_packed_batch`, `get_faces_batch`,
  `set_faces_batch`) branch on itemsize once per call instead of once per
  element

### Benchmark suite

- Rewritten around four realistic usage scenarios — `stream_import`,
  `table_conversion`, `lm_dataset`, `viewer_ops` — in place of raw
  micro-ops and the old binary-occupancy comparison

### CI

- Cross-platform test matrix (Linux, Windows, macOS) across Python 3.11/3.12
- Dedicated full-extras job (arrow, hf, torch, examples)
- Removed stale raylib system dependencies

### Other

- `examples/benchmark_kube_vs_binary.py` removed
- Full rewrite of `docs/guide/` and `docs/reference.md`

## [0.0.1b1] – 2026-03-24

Initial public beta. Everything is new.

### Core storage (`.kube`)

- `Kube` class: face-addressable, memory-mapped 3-D grid with configurable states per face
- `PackedBuffer`: native C storage engine (`kube_storage`) for packed-cell IO; replaces the previous pure-Python path for all read/write operations
- `CompiledFaceBatch`: pre-compiled random-access operation batch for amortised `ctypes` overhead
- `Cell` value type with `__repr__` and `__str__` representations
- `Kube.__str__` human-readable summary (shape, states, bits/face, bytes, dtype)
- Memory-mapped create/open/close lifecycle with `mode="r"` / `"r+"` guard
- `iter_region(bounds, tile_shape)` for deterministic tile-by-tile region walks
- `to_polars`, `from_polars`, `feature_frame`, `supervised_frame`, `iter_batches`

### Universal data backend (interop layer)

- `kube/schemas.py`: schema mode constants (`SCHEMA_WIDE_FACES`, `SCHEMA_LONG_FACES`, `SCHEMA_CELL_PACKED`, `SCHEMA_SUPERVISED`, `SCHEMA_TOKEN_SEQUENCES`)
- `kube/metadata.py`: standard manifest key constants (`SPLIT`, `TOKENIZER_NAME`, `TOKENIZER_PATH`, `SEQUENCE_LENGTH`, `FEATURE_SCHEMA`, `SOURCE_DATASET`, `PIPELINE_VERSION`)
- `kube/protocol.py`: structural `KubeBackend` Protocol for type-checker integration
- `Kube.to_arrow(bounds, schema_mode)`: zero-copy Arrow IPC table via Polars
- `Kube.from_arrow(table, shape, states)`: classmethod round-trip
- `Kube.to_numpy(bounds, packed)`, `Kube.faces_to_numpy(order)`, `Kube.__array__`
- `Kube.from_numpy(arr, states, path)`: classmethod
- `Kube.to_pandas(bounds)`, `Kube.from_pandas(df, shape, states)`
- `Kube.to_parquet(path, bounds, schema_mode)`, `Kube.region_to_parquet`
- `Kube.to_sklearn_xy(target_face, bounds)`, `Kube.schema(mode)`
- `kube/integrations/numpy_compat.py`: module-level `faces_to_numpy`, `kube_from_numpy`, `region_to_numpy`, `packed_to_numpy`
- `kube/integrations/sklearn.py`: `to_sklearn_xy`, `spatial_train_test_split`
- `kube/integrations/hf.py`: `to_hf_dataset`, `to_hf_iterable_dataset`, `from_hf_dataset`
- `kube/integrations/torch.py`: `KubeDataset`, `KubeRegionDataset`, `KubeSupervisedDataset`, `collate_kube_batch`, `to_torch`; existing `KubeCausalLMIterableDataset` retained
- `kube/integrations/__init__.py`: `available_backends()` returning list of importable optional backends

### Sharded dataset (`.kubed`)

- `ShardedKubeDataset`: multi-shard token corpus with manifest, document index, and resume state
- `create_sharded`, `open_dataset`, `import_parquet_text`, `import_arrow_text`, `import_token_ids`
- `ShardedKubeDataset.to_arrow(sequence_length, count)`, `.to_parquet(path, sequence_length)`
- `export_stats_parquet`, `export_document_parquet`, `export_sequence_sample_parquet`
- `shard_frame`, `document_frame` Polars previews
- `iter_sequences`, `iter_sequence_batches` with resume-state support and shard prefetch

### CLI (`kube` entry point)

- `kube create`, `kube open`, `kube inspect`, `kube export`, `kube view`, `kube demo`
- `kube export-arrow <src.kube> <dst.arrow>` with `--schema-mode` and `--bounds`
- `kube export-parquet <src.kube> <dst.parquet>` with `--schema-mode`, `--bounds`, `--compression`
- `kube dataset create`, `kube dataset inspect`, `kube dataset sample`, `kube dataset stats`
- `kube benchmark`: multi-workload benchmark suite covering 10 workloads (random read/write, neighborhood, region extract, full scan, histogram, polars pipeline, batch iter, lm sequence iter, mutation) with `--size`, `--iterations`, `--workloads`, `--output json` flags

### Native renderer (`kube view`)

- raylib + Clay UI layer for interactive 3-D kube inspection and editing
- Shell/slice navigation, face-state painting, zoom, LOD controls
- Inspector panel: coordinate display, STATE CHIPS chip rail (dynamic width to prevent overflow), packed-value readout
- Vendored Clay UI source (`third_party/clay/clay.h`) — no separate install required

### Build system

- `build.py` cross-platform CMake build script with `--clean`, `--cmake-only`, `--build-dir`, `--raylib-root`, `--use-system-raylib` flags
- `build.bat` (Windows MinGW64) and `build.sh` (Linux/macOS) convenience wrappers
- Poetry build hook compiles both `kube_storage` and `kube_renderer` on `poetry install`

### Examples

- `examples/roundtrip_file_container.py`: binary round-trip with Arrow/Parquet export and schema display
- `examples/tile_region_processing.py`: tiled region walk with optional Arrow region export
- `examples/export_dataset_reports.py`: shard/document/sequence QA report export
- `examples/import_text_corpus.py`: Parquet/Arrow text corpus ingest with metadata constants
- `examples/train_kube_lm.py`: minimal causal LM training loop with HuggingFace bootstrap
- `examples/benchmark_kube_vs_binary.py`: storage and throughput comparison vs. binary occupancy grid

---

[0.1.0]: https://github.com/thefr0gg/kube/releases/tag/v0.1.0
[0.0.1b1]: https://github.com/thefr0gg/kube/releases/tag/v0.0.1b1
