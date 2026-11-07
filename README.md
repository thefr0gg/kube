# Kube

Kube is a spatial-native data system for datasets that need structure, locality,
and direct editability.

Contributor setup and workflow notes live in [CONTRIBUTING.md](https://github.com/ramsy0dev/kube/blob/main/CONTRIBUTING.md).

Instead of treating data as only rows, tokens, or voxels, Kube models it as a
grid of cells whose six faces carry discrete state. That gives you a compact
representation that can be memory-mapped, queried like a table, streamed like a
dataset, and inspected interactively when human debugging matters.

The project includes:

- a `.kube` file format for storage larger than RAM
- a `.kubed` sharded dataset format for LLM-scale corpora
- a Python library for programmatic access
- a CLI for import/export/inspect/view workflows
- a native storage engine for packed-cell IO, batching, and ingest hot paths
- a localhost web viewer/editor (FastAPI backend + WebGL frontend) for 3D kube rendering and editing

## New here? Start with the guide

**[docs/guide/](docs/guide/README.md)** is a step-by-step walkthrough: what a
cell/face/state actually is, the storage model, the Python API, the CLI, the
viewer, sharded datasets, and concrete recipes for dropping Kube into a
Polars, Pandas, PyTorch, scikit-learn, or HuggingFace project you already
have. Read it top to bottom if you're new; the rest of this README is a
quick pitch and install reference.

Other docs:

- **[docs/guide/](docs/guide/README.md)** — the step-by-step guide (start here).
- **[docs/reference.md](docs/reference.md)** — the deep reference: exact
  `.kube` file format, native storage engine, web viewer architecture, native
  build internals.
- **[docs/library-integration.md](docs/library-integration.md)** — a
  condensed ML/data integration cookbook with an API summary table.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — build setup and contribution workflow.

## Identity

Kube is:

- a face-addressable storage model
- a memory-mapped runtime for large structured data
- a bridge between spatial data, table workflows, and sequence datasets
- a debugging surface for representations that are hard to understand as raw bytes alone

Kube is not:

- only a cube visualizer
- only a voxel editor
- only an LLM dataset wrapper
- only a file format

The viewer matters, but it is one interface onto the system. The deeper idea is
that Kube gives you a single representation that can move between:

- raw bytes
- packed disk-backed storage
- DataFrame-style analytics
- minibatch iteration
- native interactive inspection

## Data Measure System

Kube measures data in three layers:

- `cell`: the base spatial unit
- `face`: one of the six oriented surfaces of that cell
- `state`: the discrete value stored on a face

That means Kube is not limited to plain binary `0/1` occupancy. Each cell
carries six measurements — top, bottom, left, right, front, back — and each
face stores a state in the range `0..states-1`. If `states=2`, you get a
binary system; if `states=4`, `8`, or `256`, each face can represent richer
categories or intensity bands instead of a single bit.

See **[docs/guide/01-concept.md](docs/guide/01-concept.md)** for the full
walkthrough of this idea, with runnable examples.

## Why Kube

Kube is designed for workflows where you want:

- a compact face-based representation instead of plain `0/1` data
- random access edits at cell or face level
- memory-mapped storage for large inputs
- streaming import/export without loading the full source into memory
- a table-oriented API that works well with Polars and ML/data tooling
- sharded, resumable token-sequence iteration for LLM pretraining pipelines

In practice, that makes Kube a good fit for intermediate representations in
data or ML pipelines, inspectable storage for generated or transformed
artifacts, spatial datasets where adjacency and orientation matter, and
sequence corpora that still benefit from a unified storage/runtime layer.

## Install

Clone the repository first:

```bash
git clone https://github.com/ramsy0dev/kube.git
cd kube
```

### Install the core library

If you want the Python library and CLI without the optional LLM/demo extras:

```bash
poetry install
```

Or with pip from a local checkout:

```bash
pip install .
```

This installs the core dependency set, including:

- `polars`, `numpy`, `fastapi`, `uvicorn`
- the `kube` Python package
- the `kube` CLI entry point
- the native `kube_storage` library used by Kube's packed-cell runtime

### Install optional extras

Kube keeps the heavier ML and Hugging Face dependencies optional.

```bash
poetry install -E arrow      # Arrow IPC and Parquet export (pyarrow)
poetry install -E hf         # Hugging Face dataset/tokenizer helpers
poetry install -E torch      # torch integration
poetry install -E examples   # everything the example LM training script needs
```

### Native library build

Kube's storage runtime always uses the native `kube_storage` library. When
installing from source with Poetry, the project build hook attempts to
compile it automatically — you'll typically need `cmake` and a working C
toolchain.

`kube view` / `Kube.view(...)` need no native library beyond `kube_storage` —
the viewer is a FastAPI server (serving a WebGL frontend) that talks to the
same storage engine through the normal Python API.

If you need to build it manually:

```bash
.\build.bat           # Windows with MinGW64
./build.sh             # Linux
python build.py --generator "MinGW Makefiles"   # or call the script directly
```

Useful flags: `--clean`, `--cmake-only`, `--build-dir <path>`. See
[docs/reference.md § Build and Native Library Loading](docs/reference.md#build-and-native-library-loading)
for the full CMake/loading details.

## A quick taste

The CLI:

```bash
kube create data/sample.kube --shape 64x64x64 --states 4
kube import input.bin data/large.kube --states 4
kube inspect data/large.kube
kube stats data/large.kube --slice z=20 --normalize
kube view data/large.kube
```

The Python API:

```python
from kube import Kube

kube = Kube.from_bytes(b"hello kube", states=4)
cell = kube.get_cell(0, 0, 0)
print(cell.faces)

kube.set_face(0, 0, 0, 4, 3)
frame = kube.to_polars()
print(frame.head())
```

That's one command and one snippet — not the whole surface. For the full
tour (CLI subcommands, sharded datasets, table/array interop, integration
recipes), go to **[docs/guide/](docs/guide/README.md)**. For a condensed
quick-lookup API table, see
**[docs/library-integration.md § API Summary](docs/library-integration.md)**.

## Benchmark Results

The suite times realistic Kube usage scenarios, not raw micro-ops: streaming
a large file into a `.kube` container, converting a stored kube to a table
for a pipeline, importing and iterating an LLM token dataset, and the web
viewer's edit and mesh-build hot paths.

Run it yourself:

```bash
kube benchmark --size 64 --iterations 5
kube benchmark --size 64 --iterations 5 --output json > results.json
kube benchmark --scenarios stream_import,lm_dataset
```

Results at `--size 64`, 5 iterations (Windows, MinGW64, w64devkit gcc, pyarrow/pandas installed):

```
kube benchmark  v0.0.1b1
size=64  states=4  5 iters
------------------------------------------------------------------------------
Scenario           Backend                                Throughput    Best ms    Mean ms
------------------------------------------------------------------------------
stream_import      import_file (streaming)                890k ops/s      294.5      316.4
stream_import      from_file (load into RAM first)        1.1M ops/s      247.4      265.3

table_conversion   to_polars                              170k ops/s     1545.1     1648.6
table_conversion   faces_to_numpy                         179k ops/s     1466.8     1781.3
table_conversion   to_pandas                              152k ops/s     1721.5     2113.5
table_conversion   to_arrow                               176k ops/s     1491.1     1562.1
table_conversion   to_parquet                             151k ops/s     1738.8     1815.8

lm_dataset         import_token_ids                        12k ops/s      353.5      521.5
lm_dataset         iter_sequence_batches                   90k tok/s       45.6       47.1

viewer_ops         set_face+flush (edit round trip)      520k edits/s        0.5        0.7
viewer_ops         build_shell_mesh                      171k cells/s     1531.5     1668.3
viewer_ops         build_slice_mesh                        2k cells/s     1787.4     1996.3
------------------------------------------------------------------------------
```

Key takeaways:

- **`stream_import`** — at this size the file fits comfortably in memory, so eager `from_file` edges out streaming `import_file`; streaming's advantage is staying flat as file size grows past what fits in RAM, not winning at small sizes.
- **`table_conversion`** — all five export formats land within a similar band, since every one of them routes through `to_polars()` internally (see [Chapter 4 of the guide](docs/guide/04-tables-and-interop.md)); the cost is largely the one-time Polars frame construction, not the format-specific serialization after that.
- **`lm_dataset`** — `import_token_ids` is a one-time cost per dataset; `iter_sequence_batches` is what a training loop actually pays per step, and it's measured separately here for that reason.
- **`viewer_ops`** — `set_face+flush` edits are sub-millisecond even per-call, confirming the web viewer's "flush immediately on every edit" design doesn't bottleneck interactive use; `build_slice_mesh` is slower than `build_shell_mesh` at the same grid size because a slice renders all 6 faces of every cell in the plane instead of only boundary faces.

For a model-size comparison instead (packed cells vs. plain binary
occupancy), see [`examples/benchmark_kube_vs_binary.py`](examples/benchmark_kube_vs_binary.py).

## Project Layout

- [kube/model.py](kube/model.py): storage model and file format logic
- [kube/main.py](kube/main.py): CLI
- [kube/web/server.py](kube/web/server.py): FastAPI app (REST + WebSocket) for the viewer
- [kube/web/static/](kube/web/static/): WebGL (three.js) frontend
- [src/kube_storage.c](src/kube_storage.c): native packed-cell storage engine
- [tests/](tests/): test suite

See [docs/reference.md § Repository Map](docs/reference.md#repository-map)
for the full breakdown.

## Current Notes

- The Python library depends on `polars`, `numpy`, `fastapi`, and `uvicorn`.
- Built-in text tokenization uses the optional `tokenizers` dependency.
- Arrow and Parquet export require the optional `pyarrow` dependency (`pip install pyarrow` or `poetry install -E arrow`).
- Pandas interop requires `pandas`, which is not declared as a hard dependency because it is typically already present in data/ML environments.
- The optional torch adapter lives in `kube.integrations.torch`.
- The optional HuggingFace adapter lives in `kube.integrations.hf` (requires `datasets`).
- The sklearn spatial split helper lives in `kube.integrations.sklearn` (scikit-learn itself is not a hard dependency).
- `kube.integrations.available_backends()` lists which optional backends are currently importable.
- The viewer is localhost-only by design — there is no auth and it should never be exposed beyond the local machine.
