# Kube

Kube is a spatial-native data system for datasets that need structure, locality,
and direct editability.

Contributor setup and workflow notes live in [CONTRIBUTING.md](/c:/Users/0rams/Documents/Coding/Projects/kube/CONTRIBUTING.md).

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
git clone <your-kube-repo-url>
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

```powershell
$env:KUBE_RAYLIB_ROOT="C:\Users\0rams\Documents\Repos\raylib-5.5_win64_mingw-w64"
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

- [src/kube_storage.c](/c:/Users/0rams/Documents/Coding/Projects/kube/src/kube_storage.c): packed-cell storage engine used by the Python runtime
- [src/kube_renderer.c](/c:/Users/0rams/Documents/Coding/Projects/kube/src/kube_renderer.c): raylib renderer/editor loop
- [src/kube_ui_clay.c](/c:/Users/0rams/Documents/Coding/Projects/kube/src/kube_ui_clay.c): Clay-based editor UI

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

See [docs/library-integration.md](/c:/Users/0rams/Documents/Coding/Projects/kube/docs/library-integration.md) for a more detailed guide.

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

- [kube/model.py](/c:/Users/0rams/Documents/Coding/Projects/kube/kube/model.py): storage model and file format logic
- [kube/main.py](/c:/Users/0rams/Documents/Coding/Projects/kube/kube/main.py): CLI
- [kube/renderer.py](/c:/Users/0rams/Documents/Coding/Projects/kube/kube/renderer.py): Python/native bridge
- [src/kube_renderer.c](/c:/Users/0rams/Documents/Coding/Projects/kube/src/kube_renderer.c): native raylib renderer and editor loop
- [tests/test_kube.py](/c:/Users/0rams/Documents/Coding/Projects/kube/tests/test_kube.py): tests

## Current Notes

- The Python library depends on `polars`.
- Built-in text tokenization uses the optional `tokenizers` dependency.
- The optional torch adapter lives in `kube.integrations.torch`.
- The renderer requires building the native `kube_renderer` library.
- The current visualization path focuses on shell and slice interaction for large files, not full volumetric rendering.
