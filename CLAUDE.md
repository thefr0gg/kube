# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Kube Is

Kube is a spatial-native data system. It models data as a grid of cells where each
cell has six oriented faces (top/bottom/left/right/front/back), and each face
stores a discrete state in `0..states-1`. This face-addressable model is the
core abstraction everything else builds on: the `.kube` memory-mapped file
format, the `.kubed` sharded dataset format for LLM-scale corpora, the Python
API, the CLI, and the localhost web viewer/editor all operate on this same
cell/face/state structure.

The project is intentionally usable without the viewer — it doubles as a
Polars-native storage/dataset library for ML pipelines (see
`kube/dataset.py`, `kube/integrations/`).

## Build & Dev Commands

Install (also attempts to build native libs via the Poetry build hook):

```bash
poetry install                 # core only
poetry install -E arrow        # + pyarrow (Arrow/Parquet export)
poetry install -E hf           # + datasets/transformers/tokenizers
poetry install -E torch        # + torch integration
poetry install -E examples     # everything the example scripts need
```

Build the native storage library directly (needed after editing `src/*.c` —
the Poetry hook only runs on install):

```bash
python scripts/build.py                                  # Linux/default
python scripts/build.py --generator "MinGW Makefiles"     # Windows MinGW64
python scripts/build.py --clean                           # wipe build/ first
python scripts/build.py --cmake-only                      # configure only, no compile
```

Native build needs `cmake >= 3.20` and a C toolchain — no other native
dependency. The web viewer (`kube view`) needs no native build at all; it's
pure Python (FastAPI/uvicorn) talking to the same `kube_storage`-backed `Kube`
object.

Run tests:

```bash
poetry run python -m unittest tests.test_kube -q
poetry run python -m unittest tests.test_kube.SomeTestClass.test_some_method -q   # single test
poetry run python -m unittest tests.test_dataset -q
poetry run python -m unittest tests.test_interop -q
poetry run python -m unittest tests.test_web -q
```

Quick syntax check without full test run:

```bash
poetry run python -m py_compile kube\model.py kube\main.py kube\web\server.py
```

Benchmark suite (scenario-based: streaming import, table conversion, LLM
dataset iteration, viewer ops — see `kube/benchmark/`):

```bash
kube benchmark --size 64 --iterations 5
kube benchmark --size 64 --iterations 5 --output json > results.json
```

## Architecture

**One native shared library, one Python package.** `CMakeLists.txt` builds
only `kube_storage` (packed-cell IO hot paths: get/set face, batched get/set,
zeroing — `src/kube_storage.c`). It backs every `Kube`/dataset operation and
lands in `kube/` (not `build/`) as `kube_storage.{dll,so,dylib}` so it ships
inside the wheel. There is no native renderer anymore — `kube view` is a pure
Python FastAPI server (see below).

**`kube/_dll.py` is the only ctypes boundary.** It resolves the
platform-specific library filename, loads+caches the storage DLL, and
declares every `argtypes`/`restype` signature for its native functions.
Nothing else in the codebase should touch `ctypes` directly — go through this
module.

**`kube/web/` is the viewer**, a FastAPI backend + WebGL (three.js) frontend
replacing the old native raylib/Clay editor. `kube/web/server.py` exposes
`/api/state`, `/api/view`, `/api/mesh`, `/api/cell`, and a `/ws/edit`
WebSocket; `kube/web/mesh.py` builds shell/slice geometry via vectorized numpy
over `Kube.faces_to_numpy()` (no C involved); `kube/web/static/` is the
frontend (plain ES modules, three.js from a CDN, no build step);
`kube/web/serve.py` picks a free port, starts uvicorn, and opens the browser.
The server is localhost-only by design — never add auth scaffolding or an
externally-facing default.

**`kube/model.py` (the `Kube` class) is the core storage/editing object** and
the largest module. It owns the packed-cell representation, mmap-backed file
IO, metadata, and all the interop surface (`to_polars`, `to_arrow`,
`to_numpy`/`faces_to_numpy`, `to_pandas`, `to_sklearn_xy`, `from_*` round-trip
constructors, batching/region iteration). Functional equivalents for some of
this live in `kube/integrations/` (`numpy_compat.py`, `sklearn.py`, `hf.py`,
`torch.py`) — these are the thin adapters; the heavy logic stays in `model.py`.

**`kube/dataset.py` (`ShardedKubeDataset`) is the separate `.kubed` path** for
sharded, resumable LLM-scale token corpora — distinct from the single-file
`.kube` format in `model.py` but built on the same storage primitives. Handles
causal-LM sequence iteration (`iter_sequences`, `iter_sequence_batches`),
resume-state checkpointing, and token/text import.

**`kube/main.py` is the CLI**, a thin layer dispatching to `model.py` /
`dataset.py` operations (create/import/inspect/export/stats/view plus the
`dataset-*` subcommands).

**`kube/benchmark/`** is a self-contained package (`_backends.py` defines
comparison backends like numpy/binary-occupancy, `_workloads.py` defines the
operations benchmarked, `_runner.py` drives iterations, `_reporter.py` formats
output, `_cli.py` wires it to `kube benchmark`).

## Native Build Gotchas

- CMake caches its generator in `CMakeCache.txt`; `scripts/build.py` detects
  a generator mismatch and auto-deletes the stale top-level *and* nested
  (FetchContent) caches rather than failing — see `_reset_incompatible_cache`
  and `_reset_incompatible_nested_caches` in `scripts/build.py` if a rebuild
  misbehaves.
- On Windows, `scripts/build.py` prefers Ninja, then MinGW Makefiles (if `gcc`+
  `mingw32-make` are found), then NMake (if MSVC `cl`/`nmake` are found) —
  plain `cmake` on Windows otherwise defaults to NMake and fails outside a VS
  developer shell.
- Preserve `.kube` file format compatibility unless a change is an intentional
  format revision — this is a storage format other tooling may depend on.
- Keep mesh-building logic in `kube/web/mesh.py` vectorized (numpy), not
  per-cell Python loops.
