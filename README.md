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

Run the suite yourself:

```bash
kube benchmark --size 64 --iterations 5
kube benchmark --size 64 --iterations 5 --output json > results.json
```

Results on a 64³ grid, 4 states, 5 iterations (Windows, MinGW64, w64devkit gcc):

```
kube benchmark  v0.0.0+local
size=64³  s=4 states  5 iters
────────────────────────────────────────────────────────────────────────
Workload               Backend                Throughput   Best ms   Mean ms
────────────────────────────────────────────────────────────────────────
random_read            kube                   5.0M ops/s      39.7      46.3
random_read            numpy                 22.5M ops/s       8.9       9.8
random_read            binary 1 bit/cell       1.8M ops/s     111.8     117.8

random_write           kube                 399.1M ops/s       0.5       0.7
random_write           numpy                 54.0M ops/s       3.7       4.2
random_write           binary 1 bit/cell       1.3M ops/s     150.5     164.4

neighborhood           kube                   6.1M ops/s       9.7      10.5
neighborhood           numpy                 33.1M ops/s       1.8       1.9

region_extract         kube lower octant        88k ops/s     373.2     439.3
region_extract         numpy                135.2M ops/s       0.2       0.3

full_scan              kube                    88k ops/s    2991.9    3239.0
full_scan              numpy reshape+sum     200.6M ops/s       1.3       1.5

histogram              kube                   108k ops/s    2437.2    3581.2
histogram              numpy np.bincount     145.7M ops/s       1.8       1.9
histogram              polars                  90k ops/s    2905.8    2977.0

polars_pipeline        kube filter+group_by        87k ops/s    3000.6    3180.9
polars_pipeline        numpy→polars→filter+group_by      46.2M ops/s       5.7       7.6

batch_iter[1k]         kube                    90k ops/s    2918.7    4233.1
batch_iter[64k]        kube                    87k ops/s    3014.2    3256.6
batch_iter[512k]       kube                    88k ops/s    2989.9    3251.0

mutation               kube interleaved r+w      10.3M ops/s      19.4      22.3
mutation               numpy                 37.9M ops/s       5.3       5.6
mutation               binary 1 bit/cell       1.5M ops/s     129.1     140.7

export_arrow           kube wide_faces         85k ops/s    3068.4    3642.5
export_arrow           kube long_faces        440k ops/s    3578.3    3971.7
export_arrow           kube cell_packed       174k ops/s    1509.3    1686.8

export_parquet         kube wide_faces        129k ops/s    2029.6    2343.8

export_numpy           kube faces_to_numpy       135k ops/s    1942.6    1991.8
export_numpy           kube to_numpy flat       149k ops/s    1754.2    1852.7

export_pandas          kube to_pandas         141k ops/s    1864.3    1910.4

sklearn_xy             kube to_sklearn_xy       146k ops/s    1795.4    1840.6
────────────────────────────────────────────────────────────────────────
```

Key takeaways:

- **random_write** is the fastest single-cell path — the native storage engine batches packed-cell writes without per-call Python overhead.
- **random_read** is slower than numpy because numpy uses direct array indexing while Kube unpacks face values from packed storage on each access.
- **binary 1 bit/cell** is consistently the slowest backend despite storing less data — the Python loop over individual cells dominates.
- **full_scan / histogram / polars_pipeline** throughputs reflect the cost of materializing a full Polars frame on each call; the frame itself is fast once built.
- **export_arrow / export_parquet / export_numpy / export_pandas** throughputs cover the full round-trip including frame construction and format serialization.

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
