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
- a native editor that uses raylib for the window, input, and 3D kube rendering, plus Clay for a compact terminal-style UI layer

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

That means Kube is not limited to plain binary `0/1` occupancy.

Each cell carries six measurements, one per face:

- top
- bottom
- left
- right
- front
- back

And each face stores a state in the range `0..states-1`.

So if `states=2`, you get a binary system. But if `states=4`, `states=8`, or
`states=256`, each face can represent richer categories or intensity bands
instead of a single bit.

This makes Kube useful for data that is better described as:

- discrete levels instead of yes/no flags
- directional labels instead of undirected values
- local measurements that depend on orientation
- compact categorical state fields that still need spatial locality

## Why Kube

Kube is designed for workflows where you want:

- a compact face-based representation instead of plain `0/1` data
- random access edits at cell or face level
- memory-mapped storage for large inputs
- streaming import/export without loading the full source into memory
- a table-oriented API that works well with Polars and ML/data tooling
- sharded, resumable token-sequence iteration for LLM pretraining pipelines

In practice, that makes Kube a good fit for:

- intermediate representations in data or ML pipelines
- inspectable storage for generated or transformed artifacts
- spatial datasets where adjacency and orientation matter
- sequence corpora that still benefit from a unified storage/runtime layer

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

- `polars`
- the `kube` Python package
- the `kube` CLI entry point
- the native `kube_storage` library used by Kube's packed-cell runtime

### Install optional extras

Kube keeps the heavier ML and Hugging Face dependencies optional.

Install Arrow/Parquet support:

```bash
poetry install -E arrow
```

Install the Hugging Face/text-import stack:

```bash
poetry install -E hf
```

Install the torch integration:

```bash
poetry install -E torch
```

Install everything needed for the example LM training scripts:

```bash
poetry install -E examples
```

The current optional extras are:

- `arrow`: Arrow IPC and Parquet export via `pyarrow`
- `llm`: built-in tokenization support via `tokenizers`
- `torch`: torch integration plus tokenizers
- `hf`: Hugging Face dataset/tokenizer helpers
- `examples`: everything needed by the example training scripts

### Native libraries build

Kube's storage runtime now always uses the native `kube_storage` library, and
`kube view` / `Kube.view(...)` also need the native `kube_renderer` shared
library.

When installing from source with Poetry, the project build hook now attempts to compile both native libraries automatically during package build/install.

That means you will typically need:

- `cmake`
- a working C toolchain
- either a system `raylib` package or network access so CMake can fetch raylib

If you already have an unpacked Windows raylib bundle with `include/` and `lib/`
folders, point the build at it explicitly before installing:

```bash
KUBE_RAYLIB_ROOT="/path/to/raylib"
poetry install
```

Kube will then use `include/raylib.h` plus the library files inside `lib/`
instead of trying `find_package(raylib)` first.

The native editor UI is now built on top of vendored `Clay` source in
`third_party/clay/clay.h`. Clay is compiled as part of the existing native
renderer target, so you do not need to install a separate UI library.

## Build The Native Libraries

The Python package now depends on the native storage engine, and the viewer also
depends on the native renderer. Source installs should build both.

Windows with MinGW64:

```powershell
.\build.bat
```

Linux:

```bash
./build.sh
```

You can also call the build script directly:

```bash
python build.py --generator "MinGW Makefiles"
python3 build.py
```

Useful build flags:

- `--clean`: remove the current build directory first
- `--cmake-only`: configure without compiling
- `--build-dir <path>`: use a custom build directory
- `--use-system-raylib` or `--no-use-system-raylib`: control how raylib is resolved
- `--raylib-root <path>`: point directly at an unpacked raylib folder with `include/` and `lib/`

Important native sources:

- [src/kube_storage.c](src/kube_storage.c): packed-cell storage engine used by the Python runtime
- [src/kube_renderer.c](src/kube_renderer.c): raylib renderer/editor loop
- [src/kube_ui_clay.c](src/kube_ui_clay.c): Clay-based editor UI

## CLI

Create an empty kube file:

```bash
kube create data/sample.kube --shape 64x64x64 --states 4
```

Import a large file into a mapped `.kube` container:

```bash
kube import input.bin data/large.kube --states 4
```

Inspect metadata and storage stats:

```bash
kube inspect data/large.kube
```

Export bytes back out:

```bash
kube export data/large.kube restored.bin
```

Export the grid as Arrow IPC:

```bash
kube export-arrow data/large.kube out.arrow
kube export-arrow data/large.kube out.arrow --schema-mode cell_packed
```

Export the grid as Parquet:

```bash
kube export-parquet data/large.kube out.parquet
kube export-parquet data/large.kube out.parquet --schema-mode long_faces --compression zstd
```

Compute per-face histograms:

```bash
kube stats data/large.kube
kube stats data/large.kube --slice z=20 --normalize
```

Create and inspect a sharded dataset:

```bash
kube dataset-create corpora/wiki.kubed --shards 8 --token-bytes 4 --tokens-per-shard 1048576
kube dataset-inspect corpora/wiki.kubed
kube dataset-stats corpora/wiki.kubed
```

Import tokenized rows into a sharded dataset:

```bash
kube dataset-import-tokens train.parquet corpora/wiki.kubed --tokens-column input_ids --sequence-length 2048
kube dataset-sample corpora/wiki.kubed --sequence-length 2048 --count 3
kube dataset-sample corpora/wiki.kubed --sequence-length 2048 --count 3 --resume-state-out checkpoints/wiki-sample.json
```

Import raw text with built-in tokenization:

```bash
kube dataset-import-text raw_text.parquet corpora/wiki.kubed --text-column text --tokenizer tokenizer.json --sequence-length 2048
```

Open the editor:

```bash
kube view data/large.kube --window-size 1440x900
kube view data/large.kube --slice z=20 --lod 2
kube view data/large.kube --read-only
```

## Python Library

### Quick start

```python
from kube import Kube

kube = Kube.from_bytes(b"hello kube", states=4)
cell = kube.get_cell(0, 0, 0)
print(cell.faces)

kube.set_face(0, 0, 0, 4, 3)
frame = kube.to_polars()
print(frame.head())
```

### Work with mapped files

```python
from kube import create, open as open_kube

with create("data/train.kube", (128, 128, 128), states=4) as kube:
    kube.update_metadata(dataset="train", owner="pipeline-a")
    kube.set_face(0, 0, 0, 0, 1)
    kube.flush()

with open_kube("data/train.kube", mode="r") as kube:
    print(kube.inspect())
```

### Stream large files

```python
from kube import import_file

kube = import_file("big-input.bin", "data/big.kube", states=4)
print(kube.memory_stats())
kube.close()
```

### Sharded LLM datasets

```python
from kube import create_sharded, import_token_ids, open_dataset

dataset = create_sharded(
    "datasets/wiki.kubed",
    shards=8,
    token_bytes=4,
    tokens_per_shard=1 << 20,
    sequence_length=2048,
)

dataset = import_token_ids("train.parquet", "datasets/wiki.kubed", tokens_column="input_ids")
reopened = open_dataset("datasets/wiki.kubed")
print(reopened.inspect())
```

### Table-oriented access with Polars

```python
from kube import Kube

kube = Kube(8, 8, 8, states=4)
frame = kube.to_polars()
features = kube.feature_frame(include_coords=True, include_packed=False)

for batch in kube.iter_batches(1024, include_packed=False):
    # Send each batch into your preprocessing or training code.
    print(batch.shape)
```

### Interop with NumPy, Arrow, Parquet, Pandas, and sklearn

```python
import numpy as np
from kube import Kube

kube = Kube(8, 8, 8, states=4)

# NumPy — (W, H, D, 6) array of face values
arr = kube.faces_to_numpy()          # shape (8, 8, 8, 6), dtype uint8
arr = np.array(kube)                 # same, via __array__ protocol

# Arrow — zero-copy via Polars' Arrow-native memory model
table = kube.to_arrow()              # pa.Table with face_0..face_5 columns
table = kube.to_arrow(schema_mode="long_faces")   # melted N×6 rows
table = kube.to_arrow(schema_mode="cell_packed")  # packed integer column

# Parquet
kube.to_parquet("out.parquet")
kube.to_parquet("region.parquet", bounds=((0, 4), (0, 4), (0, 4)))

# Pandas
df = kube.to_pandas()

# sklearn — get (X, y) feature/label arrays
X, y = kube.to_sklearn_xy(target_face=0)   # X shape (N, 5), y shape (N,)

# Round-trips
from kube.model import Kube
kube2 = Kube.from_arrow(table, shape=(8, 8, 8), states=4)
kube3 = Kube.from_pandas(df, shape=(8, 8, 8), states=4)
```

Functional-style helpers live in `kube.integrations`:

```python
from kube.integrations.numpy_compat import faces_to_numpy, kube_from_numpy, packed_to_numpy
from kube.integrations.sklearn import to_sklearn_xy, spatial_train_test_split
from kube.integrations.hf import to_hf_dataset, to_hf_iterable_dataset

# Spatial split avoids leakage from adjacent cells
X_tr, X_te, y_tr, y_te = spatial_train_test_split(kube, target_face=0, test_axis=2)

# HuggingFace (requires datasets extra)
hf_ds = to_hf_dataset(kube)                           # in-memory Dataset
it_ds = to_hf_iterable_dataset(sharded_ds, seq_len=256)  # streaming
```

### Analytics and supervised-learning helpers

```python
histogram = kube.face_histogram(normalize=True)
slice_only = kube.to_polars(bounds=kube.slice_bounds("z", 4))

train_frame = kube.supervised_frame(
    target_face=4,
    include_coords=True,
    include_packed=False,
)

for batch in kube.iter_supervised_batches(
    2048,
    target_face=4,
    include_coords=True,
):
    print(batch.columns)
```

### LLM sequence iteration

```python
from kube import open_dataset

dataset = open_dataset("datasets/wiki.kubed")

for sample in dataset.iter_sequences(2048, stride=2048, prefetch_shards=1):
    print(len(sample["input_ids"]), len(sample["target_ids"]))
    resume_state = sample["resume_state"]
    break

for batch in dataset.iter_sequence_batches(8, 2048):
    print(len(batch["input_ids"]))
    break

resumed = dataset.resume_iterator(resume_state)
next_sample = next(resumed)

dataset.save_resume_state("checkpoints/wiki.json", resume_state)
loaded_state = dataset.load_resume_state("checkpoints/wiki.json")

for frame_batch in dataset.iter_sequence_frame_batches(8, 2048):
    print(frame_batch.columns)
    break
```

### Rebuild a kube from a Polars frame

```python
from kube import from_polars

restored = from_polars(frame, shape=(8, 8, 8), states=4)
```

The input frame must contain:

- `x`, `y`, `z`
- either `packed`
- or all six face columns: `face_0` through `face_5`

### More workflow examples

Use Kube as a reversible binary container:

```python
from pathlib import Path
from kube import from_bytes

payload = Path("artifact.bin").read_bytes()
kube = from_bytes(payload, states=4, path="data/artifact.kube")
assert kube.to_bytes()[: len(payload)] == payload
```

Process a large region tile by tile:

```python
from kube import open as open_kube

with open_kube("data/world.kube", mode="r") as kube:
    for origin, tile in kube.iter_region(((0, 128), (0, 128), (0, 64)), tile_shape=(32, 32, 16)):
        print(origin, tile.height)
```

Export dataset QA tables:

```python
dataset = open_dataset("datasets/wiki.kubed")
dataset.export_stats_parquet("reports/wiki-shards.parquet")
dataset.export_document_parquet("reports/wiki-documents.parquet")
```

Import text tables directly into a dataset:

```python
dataset = import_parquet_text(
    "news.parquet",
    "datasets/news.kubed",
    text_column="body",
    tokenizer_path="tokenizer.json",
)
```

Benchmark Kube against a one-bit binary occupancy grid:

```bash
python examples/benchmark_kube_vs_binary.py --shape 64x64x64 --states 4 --ops 200000
```

That benchmark focuses on the core tradeoff:

- binary occupancy is smaller because it stores one bit per cell
- Kube stores six oriented face states per cell, so it carries much richer local information
- the output prints both storage math and random read/write timings

The wiki has a larger worked-example section with more context for each pattern.

## Public API Highlights

Main types:

- `Kube`: the primary storage and editing object
- `Cell`: one six-face cell

Top-level helpers:

- `create(...)`
- `create_sharded(...)`
- `open(...)`
- `open_dataset(...)`
- `import_file(...)`
- `import_parquet_text(...)`
- `import_arrow_text(...)`
- `import_token_ids(...)`
- `from_bytes(...)`
- `from_file(...)`
- `from_polars(...)`
- `load(...)`

Important `Kube` methods:

- `get_cell(...)`, `set_cell(...)`
- `get_face(...)`, `set_face(...)`
- `slice_bounds(...)`
- `to_polars(...)`
- `feature_frame(...)`
- `supervised_frame(...)`
- `face_histogram(...)`
- `iter_batches(...)`
- `iter_supervised_batches(...)`
- `read_region(...)`
- `iter_region(...)`
- `update_metadata(...)`
- `flush()`
- `view(...)`
- `schema(mode)` — column names + dtypes for a schema mode
- `to_arrow(bounds, schema_mode)` — PyArrow Table
- `from_arrow(table, shape, states)` — classmethod
- `to_numpy(bounds, packed)` — NumPy array
- `faces_to_numpy()` — `(W, H, D, 6)` array
- `to_pandas(bounds)` — Pandas DataFrame
- `from_pandas(df, shape, states)` — classmethod
- `to_parquet(path, bounds, schema_mode)` — Parquet file
- `to_sklearn_xy(target_face, bounds)` — `(X, y)` arrays

Important `ShardedKubeDataset` methods:

- `inspect()`
- `shard_frame()`
- `document_frame()`
- `import_text_frame(...)`
- `import_token_frame(...)`
- `iter_sequences(...)`
- `iter_sequence_batches(...)`
- `iter_sequence_frame_batches(...)`
- `resume_iterator(...)`
- `save_resume_state(...)`
- `load_resume_state(...)`
- `sample_sequences_frame(...)`
- `export_stats_parquet(...)`
- `to_arrow(sequence_length, count)` — PyArrow Table of sequences
- `to_parquet(path, sequence_length, count)` — Parquet file
- `to_hf_dataset(sequence_length, count)` — HuggingFace Dataset
- `to_hf_iterable_dataset(sequence_length)` — streaming IterableDataset
- `iter_lm_batches(batch_size, sequence_length)` — padded LM batches

## Machine Learning And Data Pipeline Use

Kube is intentionally usable without the renderer. For model training and feature engineering, the most relevant pieces are:

- mapped `.kube` files via `Kube.open(...)`
- sharded `.kubed` datasets via `open_dataset(...)`
- Polars frames via `to_polars(...)` or `feature_frame(...)`
- supervised datasets via `supervised_frame(...)`
- streaming minibatches via `iter_batches(...)`
- supervised minibatches via `iter_supervised_batches(...)`
- per-face distribution summaries via `face_histogram(...)`
- causal LM sequences via `iter_sequences(...)`
- causal LM batches via `iter_sequence_batches(...)`
- round-tripping model outputs back into storage with `from_polars(...)`

Typical workflow:

1. Import raw text or token IDs into a sharded `.kubed` dataset.
2. Open the dataset in read-only or read-write mode.
3. Iterate causal LM windows with `iter_sequences(...)` or `iter_sequence_batches(...)`.
4. Resume from `resume_state` if preprocessing or training is interrupted.
5. Save checkpoints with `save_resume_state(...)` when you need restartable scans across long jobs.
6. Optionally export shard or document stats to Parquet for dataset inspection.
7. Write transformed or predicted outputs back into a new `Kube` or `.kubed` dataset.

See [docs/library-integration.md](docs/library-integration.md) for a more detailed guide.

## Editor Controls

- Left click: select a visible cell/face
- Right drag: orbit camera
- Mouse wheel: zoom
- `H` or `F1`: toggle help
- `` ` `` or `F2`: open the built-in Python console
- `Esc`: close help or console overlays
- `Ctrl+Q`: close the viewer
- `Tab`: toggle shell/slice mode
- `X`, `Y`, `Z`: choose slice axis
- `PageUp`, `PageDown`: move slice
- `1-6`: select a face in the inspector
- `[` and `]`: decrement or increment selected face value
- numpad digits: set face values directly when `states <= 10`

### Console Notes

The built-in Python console now supports:

- `Tab` autocompletion for Python names and dotted attributes
- a compact floating terminal panel with a close button
- separate completion hints instead of mixing suggestions into command output
- labeled `output`, `result`, and `error` sections for executed code

## Project Layout

- [kube/model.py](kube/model.py): storage model and file format logic
- [kube/main.py](kube/main.py): CLI
- [kube/renderer.py](kube/renderer.py): Python/native bridge
- [src/kube_renderer.c](src/kube_renderer.c): native raylib renderer and editor loop
- [tests/test_kube.py](tests/test_kube.py): tests

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

## Current Notes

- The Python library depends on `polars`.
- Built-in text tokenization uses the optional `tokenizers` dependency.
- Arrow and Parquet export require the optional `pyarrow` dependency (`pip install pyarrow` or `poetry install -E arrow`).
- NumPy interop requires `numpy`. Pandas interop requires `pandas`. These are not declared as extras because they are typically already present in data/ML environments.
- The optional torch adapter lives in `kube.integrations.torch`.
- The optional HuggingFace adapter lives in `kube.integrations.hf` (requires `datasets`).
- The sklearn spatial split helper lives in `kube.integrations.sklearn` (requires `numpy`; scikit-learn itself is not a hard dependency).
- `kube.integrations.available_backends()` lists which optional backends are currently importable.
- The renderer requires building the native `kube_renderer` library.
- The current visualization path focuses on shell and slice interaction for large files, not full volumetric rendering.
