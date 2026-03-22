# Kube Wiki

This file is a code-oriented guide to how Kube works today.

Kube's packed-cell runtime now uses the native `kube_storage` engine and does
not expose a public Python fallback path.

It focuses on the pieces that are implemented in this repository:

- the `Kube` storage model
- the `.kube` on-disk format
- the Python API and CLI
- the native storage engine used for packed random IO and ingest batching
- the native renderer/editor bridge, with raylib handling the 3D scene and Clay handling editor UI layout

For pipeline-oriented usage examples, also see `README.md` and `docs/library-integration.md`.

## What Kube Is

Kube is best thought of as a spatial-native data substrate.

The cube is the storage grammar, not the whole identity.

That means Kube is trying to do three things at once:

- give structured data a compact face-addressable physical layout
- keep that layout usable from Python, analytics, and ML workflows
- make the representation inspectable when humans need to debug it visually

Kube stores data as a 3D grid of cells. Each cell has 6 faces, and each face stores a discrete state value.

The important mental model is:

- one cell = 6 face values
- one face value is in the range `0..states-1`
- `states` must be a power of two and at least `2`
- cells are stored in a packed integer form so the dataset can stay compact and memory-mappable

This is Kube's measurement system:

- the cell is the spatial address
- the face is the oriented measurement slot
- the state is the discrete measurement value

So Kube is not inherently a binary `0/1` system.

If `states=2`, each face is effectively one bit.

If `states=4`, `8`, `16`, or `256`, each face can encode richer local meaning,
such as:

- category or class
- confidence band
- intensity band
- transition type
- directional role

That is why "cell + faces + states" is the core abstraction, not "cube full of
booleans."

Kube supports two main usage modes:

- in-memory kubes for quick manipulation
- memory-mapped `.kube` files for larger-than-RAM or persistent data

There is also a dataset layer for sharded `.kubed` corpora, but this wiki focuses primarily on the core `Kube` object and the viewer/editor path.

Another way to say it:

- `.kube` is the storage unit
- `Kube` is the runtime object
- Polars/minibatch helpers are the analysis surface
- the native editor is the inspection surface
- `.kubed` is the scaled dataset packaging layer

So while the renderer is visible, the project itself is broader than visualization.

## Use Cases

Kube is most useful when your data naturally fits a "small discrete state per face" model and you want compact storage plus random access.

### 1. Interactive inspection and editing

Use Kube when you want to:

- inspect a large cube without loading a second decoded representation
- click through visible cells and faces
- make targeted edits and persist them immediately
- debug how a generated cube is actually encoded

Good fit:

- procedural content or rule-based cube generation
- labeled voxel-like data where each face matters independently
- data validation workflows where human inspection is important

### 2. Disk-backed datasets larger than RAM

Use `.kube` files when you want:

- memory-mapped access
- stable on-disk persistence
- per-cell or per-face random writes
- streaming import/export from ordinary byte-oriented files

Good fit:

- large binary payloads that should be addressable through a compact face-state encoding
- persistent training or preprocessing artifacts
- intermediate representations that need to survive across pipeline stages

### 3. Table-oriented analytics and feature engineering

Use the Polars-facing APIs when you want:

- face values as columns
- deterministic iteration over the whole cube or a slice
- minibatches without materializing everything at once
- easy integration with ETL and analytics code

Good fit:

- feature extraction
- distribution analysis
- supervised learning setups where one face becomes the label
- slice-by-slice or tile-by-tile preprocessing

### 4. Sharded corpora for language-model workflows

Use `.kubed` datasets when you want:

- many shards under one dataset root
- token-ID or raw-text ingestion
- resumable sequence iteration
- sequence batching for long-running preprocessing or training jobs

Good fit:

- LLM pretraining corpora
- restartable dataset scans
- exporting corpus stats for inspection

### 5. Format conversion and round-tripping

Use Kube when you need to move between:

- raw bytes
- packed cell storage
- decoded face tables
- transformed output frames

Good fit:

- converting binary payloads into a structured inspection format
- running transformations in Polars and writing the result back
- preserving storage compactness while still exposing decoded columns to tooling

## Repository Map

- `kube/model.py`: core storage model, file IO, packing/unpacking, Polars export, region and batch iteration
- `kube/renderer.py`: Python wrapper that passes a live Kube buffer into the native renderer
- `kube/_dll.py`: native library discovery and `ctypes` signature setup
- `kube/main.py`: CLI entry points
- `src/kube_storage.c`: native packed-cell storage engine for random IO, face batches, contiguous packed writes, and ingest hot paths
- `src/kube_renderer.c`: native entry point, geometry builder, camera/input loop, and 3D viewport rendering
- `src/kube_ui_clay.c`: Clay-backed editor UI layout, rendering, and hit handling
- `src/kube_ui_clay.h`: native UI bridge types shared with the renderer loop
- `build.py`: configure/build helper for the native libraries
- `CMakeLists.txt`: native build definition
- `tests/test_kube.py`: regression coverage for the core storage API
- `tests/test_dataset.py`: regression coverage for sharded dataset ingest and sequence iteration
- `docs/library-integration.md`: library-facing usage notes for ETL/ML workflows

## Native Sources

Kube now has two native layers with distinct responsibilities.

### `src/kube_storage.c`

This file is the native storage engine used by the Python runtime itself.

It exists so packed-cell access, face batches, clears, and ingest writes do not
have to route through Python loops once the buffer is addressable.

The current exported responsibilities are:

- single packed-value reads and writes
- single decoded face reads and writes
- batched face reads and writes
- contiguous packed-value batch writes
- byte-range zeroing for clears and trailing cleanup

This is the source that backs:

- native-only `Kube.get_packed(...)`
- native-only `Kube.get_face(...)`
- native-only `Kube.set_face(...)`
- `get_faces_batch(...)` and `set_faces_batch(...)`
- batched `encode(...)`, `_stream_import(...)`, and token-shard ingest

### `src/kube_renderer.c`

This file owns the native viewer/editor loop:

- raylib window lifecycle
- camera and viewport rendering
- pointer and keyboard handling
- callback bridge back into Python

### `src/kube_ui_clay.c` and `src/kube_ui_clay.h`

These files own the Clay-based editor chrome layered on top of the renderer:

- help panel
- inspector
- console
- terminal-style overlay layout
- drag, scroll, and hit behavior

## Core Data Model

### Cell

A `Cell` is the logical unit of data:

- it always contains exactly 6 faces
- it validates that every face value is within the configured state range
- it can convert to and from a packed integer representation

The face order is fixed by index:

- `0`: top
- `1`: bottom
- `2`: left
- `3`: right
- `4`: front
- `5`: back

### Packed representation

`states` determines `bits_per_face`:

- `states=2` -> `1` bit per face
- `states=4` -> `2` bits per face
- `states=256` -> `8` bits per face

Each cell uses:

```text
bits_per_cell = 6 * log2(states)
```

Faces are packed into one integer, with face `0` occupying the lowest bits, then face `1`, and so on.

The storage item size is chosen automatically:

- up to 8 total bits -> `UInt8`
- up to 16 total bits -> `UInt16`
- up to 32 total bits -> `UInt32`
- otherwise -> `UInt64`

This logic lives in `_itemsize_for_states(...)` in `kube/model.py`.

### Grid indexing

Cells are stored in a flat contiguous buffer using:

```text
flat_index = x * (H * D) + y * D + z
```

That means:

- `x` is the outermost stride
- `z` is the fastest-changing coordinate

The same layout is used by Python, the native storage engine, and the native
renderer, which is why both storage and rendering can read the mapped buffer
directly without copying it.

## The `.kube` File Format

### Header

Every `.kube` file starts with a fixed 4096-byte header.

`kube/model.py` defines:

- `MAGIC = b"KUBE"`
- `FORMAT_VERSION = 1`
- `HEADER_SIZE = 4096`

The header includes:

- magic and format version
- shape: `W`, `H`, `D`
- `states`
- `itemsize`
- payload offset
- cell count
- source byte count
- metadata length
- JSON metadata blob

The packed header prefix uses `HEADER_STRUCT = struct.Struct("<4sHHQQQIIQQQI")`.

Important consequence:

- metadata is stored in the fixed header area
- metadata must fit inside the 4 KiB header
- `update_metadata(...)` rewrites the header for mapped files

### Payload

After the header, the payload is a contiguous array of packed cell values.

Kube uses a `PackedBuffer` abstraction so the payload can be backed by:

- a Python `bytearray` for in-memory kubes
- an `mmap` for disk-backed kubes

### Read/write modes

`Kube.open(path, mode=...)` supports:

- `mode="r"`: safe read-only access
- `mode="r+"`: read/write mapped access

When opened with `mode="r"`:

- Python mutation APIs raise `PermissionError`
- the file is mapped with copy-on-write semantics for the buffer view
- `flush()` becomes a no-op

When created or opened for writing:

- face edits modify the mapped payload directly
- `flush()` persists mapped updates and refreshes the header when needed

## Import/Export Model

Kube can treat ordinary byte streams as a packed sequence of face states.

The core helpers are:

- `_pack_stream_to_values(...)`
- `_unpack_values_to_bytes(...)`
- `_packed_value_to_stream_chunk(...)`

This is what powers:

- `Kube.from_bytes(...)`
- `Kube.from_file(...)`
- `Kube.import_file(...)`
- `Kube.export_file(...)`
- `Kube.to_bytes()`

There are two supported import styles:

- eager import for in-memory payloads
- streaming import for large files

`Kube.import_file(...)` streams from disk, packs incrementally, and clears unused trailing cells if the target cube is larger than the source payload.

## Worked Examples

This section is intentionally practical. It shows the most common ways to use the project without requiring you to reverse-engineer the API from source.

### Example 1: Create an in-memory kube and inspect one cell

```python
from kube import Kube

kube = Kube(4, 4, 4, states=4)
kube.set_face(0, 0, 0, 0, 3)
kube.set_face(0, 0, 0, 4, 1)

cell = kube.get_cell(0, 0, 0)
print(cell.faces)   # [3, 0, 0, 0, 1, 0]
print(kube.get_packed(0, 0, 0))
```

Use this pattern when:

- you want a lightweight scratch object
- the cube fits comfortably in memory
- you are prototyping packing or face-level logic

### Example 2: Create a mapped `.kube` file and persist edits

```python
from kube import create, open as open_kube

with create("data/sample.kube", (32, 32, 32), states=4) as kube:
    kube.update_metadata(dataset="sample", owner="local-dev")
    kube.set_face(0, 0, 0, 0, 2)
    kube.set_face(0, 0, 0, 4, 3)
    kube.flush()

with open_kube("data/sample.kube", mode="r") as reopened:
    print(reopened.inspect())
    print(reopened.get_cell(0, 0, 0).faces)
```

Use this pattern when:

- you want persistence across sessions
- you need memory-mapped storage
- your workflow may outgrow RAM later

### Example 3: Import a normal file and export it back out

CLI:

```bash
kube import input.bin data/large.kube --states 4
kube inspect data/large.kube
kube export data/large.kube restored.bin
```

Python:

```python
from kube import import_file

kube = import_file("input.bin", "data/large.kube", states=4)
print(kube.memory_stats())
kube.close()
```

Use this pattern when:

- you need a compact, memory-mappable container for ordinary bytes
- you want to inspect or transform the imported content later
- you need streaming import instead of loading the whole source file at once

### Example 4: Convert a kube into a Polars table

```python
from kube import Kube

kube = Kube.from_bytes(b"hello kube", states=4)
frame = kube.to_polars()
features = kube.feature_frame(include_coords=True, include_packed=False)

print(frame.columns)
print(features.head())
```

Use this pattern when:

- you want face values as ordinary columns
- you are feeding the data into Polars-first analytics or ML code
- you want to inspect slices or subsets as tables

### Example 5: Train on one face as the label

```python
from kube import open as open_kube

with open_kube("data/train.kube", mode="r") as kube:
    train_frame = kube.supervised_frame(
        target_face=4,
        include_coords=True,
        include_packed=False,
    )

    for batch in kube.iter_supervised_batches(
        2048,
        target_face=4,
        include_coords=True,
        include_packed=False,
    ):
        print(batch.columns)
        break
```

Use this pattern when:

- one face is the prediction target
- the other faces are inputs
- you want to keep the storage format unchanged while exposing a supervised-learning table

### Example 6: Analyze a single slice

CLI:

```bash
kube stats data/sample.kube --slice z=20 --normalize
```

Python:

```python
from kube import open as open_kube

with open_kube("data/sample.kube", mode="r") as kube:
    bounds = kube.slice_bounds("z", 20)
    frame = kube.to_polars(bounds=bounds, include_packed=False)
    histogram = kube.face_histogram(bounds=bounds, normalize=True)
    print(frame.height)
    print(histogram.head())
```

Use this pattern when:

- you want 2D inspection of one plane
- you are doing drift checks or local debugging
- the full cube is too large to examine at once

### Example 7: Open the native viewer

CLI:

```bash
kube view data/sample.kube --window-size 1440x900
kube view data/sample.kube --slice z=20 --lod 2
kube view data/sample.kube --read-only
```

Python:

```python
from kube import open as open_kube

with open_kube("data/sample.kube", mode="r+") as kube:
    kube.view(window_size=(1440, 900), lod=2)
```

Use this pattern when:

- you need manual visual inspection
- you want click-to-select face editing
- you want shell or slice exploration of a mapped file

### Example 8: Rebuild a kube from a transformed frame

```python
import polars as pl
from kube import from_polars

frame = pl.DataFrame(
    {
        "x": [0, 0],
        "y": [0, 1],
        "z": [0, 0],
        "face_0": [1, 2],
        "face_1": [0, 0],
        "face_2": [3, 3],
        "face_3": [1, 1],
        "face_4": [2, 2],
        "face_5": [0, 0],
    }
)

restored = from_polars(
    frame,
    shape=(8, 8, 8),
    states=4,
    path="data/restored.kube",
    metadata={"producer": "transform-step"},
)
restored.close()
```

Use this pattern when:

- a transform step runs outside the core Kube API
- you want to write model outputs back into packed storage
- you want a new artifact rather than mutating the source in place

### Example 9: Work with sharded `.kubed` corpora

CLI:

```bash
kube dataset-create corpora/wiki.kubed --shards 8 --token-bytes 4 --tokens-per-shard 1048576
kube dataset-import-tokens train.parquet corpora/wiki.kubed --tokens-column input_ids --sequence-length 2048
kube dataset-sample corpora/wiki.kubed --sequence-length 2048 --count 3 --resume-state-out checkpoints/wiki.json
```

Python:

```python
from kube import create_sharded, import_token_ids, open_dataset

dataset = create_sharded(
    "corpora/wiki.kubed",
    shards=8,
    token_bytes=4,
    tokens_per_shard=1 << 20,
    sequence_length=2048,
)

dataset = import_token_ids("train.parquet", "corpora/wiki.kubed", tokens_column="input_ids")

for sample in open_dataset("corpora/wiki.kubed").iter_sequences(2048, stride=2048):
    print(sample["input_ids"][:8])
    print(sample["resume_state"])
    break
```

Use this pattern when:

- you are working with tokenized corpora
- dataset size makes sharding important
- you need resumable iteration over long scans

### Example 10: Train a tiny Torch causal LM on a `.kubed` dataset

An end-to-end example script now lives at:

- `examples/train_kube_lm.py`

Example:

```bash
python examples/train_kube_lm.py corpora/wiki.kubed --vocab-size 50257 --sequence-length 256 --batch-size 8 --steps 200 --checkpoint-out checkpoints/tiny-lm.pt --resume-state-out checkpoints/tiny-lm-resume.json
```

What this script does:

- opens the dataset with `open_dataset(...)`
- streams batches with `iter_sequence_batches(...)`
- trains a small decoder-only Transformer in PyTorch
- optionally saves a model checkpoint
- optionally saves a Kube dataset resume state for later continuation

Notes:

- `torch` must be installed because this is an optional dependency
- `--vocab-size` is required when training an existing dataset; the Hugging Face bootstrap path can infer it from the downloaded tokenizer
- this is a compact example script, not a full production trainer

The same script can also bootstrap a tiny demo corpus from Hugging Face, convert it into `.kubed`, and then train on it:

```bash
python examples/train_kube_lm.py data/wikitext-demo.kubed --bootstrap-hf --hf-dataset wikitext --hf-config wikitext-2-raw-v1 --hf-split train[:1%] --hf-tokenizer gpt2 --sequence-length 128 --batch-size 4 --steps 50 --bootstrap-overwrite
```

That flow does three things in one command:

- downloads a small text split from Hugging Face
- tokenizes the text and stores it in a sharded Kube dataset
- trains the tiny causal LM on the generated `.kubed` data

For the bootstrap path, install the optional Hugging Face dependencies used by the example:

- `datasets`
- `transformers`
- `torch`

### Example 11: Use Kube as a reversible binary container

```python
from pathlib import Path
from kube import from_bytes

payload = Path("artifact.bin").read_bytes()
kube = from_bytes(payload, states=4, path="data/artifact.kube")

restored = kube.to_bytes()
assert restored[: len(payload)] == payload
kube.close()
```

Use this pattern when:

- you want a memory-mappable wrapper around an ordinary binary file
- you need byte-for-byte round-tripping after inspection or transformation
- you want to keep one artifact in Kube storage instead of maintaining two formats

### Example 12: Process a large cube region tile by tile

```python
from kube import open as open_kube

with open_kube("data/world.kube", mode="r") as kube:
    bounds = ((0, 128), (0, 128), (0, 64))

    for origin, tile in kube.iter_region(bounds, tile_shape=(32, 32, 16)):
        features = tile.select(["x", "y", "z", "face_0", "face_4"])
        print("tile origin:", origin, "rows:", features.height)
```

Use this pattern when:

- the cube is too large to materialize as one DataFrame
- you are running deterministic preprocessing over spatial neighborhoods
- you want a tiled ETL or feature-extraction pass

### Example 13: Export dataset QA tables for external inspection

```python
from kube import open_dataset

dataset = open_dataset("corpora/wiki.kubed")

print(dataset.shard_frame().head())
print(dataset.document_frame().head())

dataset.export_stats_parquet("reports/wiki-shards.parquet")
dataset.export_document_parquet("reports/wiki-documents.parquet")
dataset.export_sequence_sample_parquet(
    "reports/wiki-sample.parquet",
    sequence_length=512,
    count=16,
)
```

Use this pattern when:

- you want corpus QA artifacts outside the runtime API
- you need shard-level or document-level reporting for dashboards
- you want a small sequence sample for manual inspection in Polars, DuckDB, or notebooks

### Example 14: Ingest text tables directly into a `.kubed` dataset

```python
from kube import create_sharded, import_parquet_text

dataset = create_sharded(
    "corpora/news.kubed",
    shards=4,
    token_bytes=4,
    tokens_per_shard=1 << 18,
    sequence_length=1024,
)

dataset = import_parquet_text(
    "news.parquet",
    "corpora/news.kubed",
    text_column="body",
    tokenizer_path="tokenizer.json",
)

print(dataset.inspect())
```

Use this pattern when:

- your source corpus already lives in Parquet or Arrow-style tables
- you want one-step text ingestion instead of pre-tokenizing separately
- you are preparing a sharded dataset for sequence iteration or training

### Example 15: Benchmark Kube against a binary occupancy system

Use the runnable example script:

```bash
python examples/benchmark_kube_vs_binary.py --shape 64x64x64 --states 4 --ops 200000
```

What it compares:

- a one-bit binary occupancy grid with no face orientation
- a Kube grid where each cell stores six oriented face states

What it reports:

- payload bytes for both systems
- bits per cell and local state-space size
- random write throughput
- random read throughput

How to interpret it:

- binary occupancy wins on raw compactness because it stores less information
- Kube is intentionally richer: one cell has six measurement slots, not one bit
- the benchmark is meant to show the tradeoff between expressiveness and storage/runtime cost, not to claim the models solve the exact same problem

## Main Python API

### Construction and opening

The common entry points are:

- `Kube(...)`: create an in-memory kube
- `Kube.create(path, shape, ...)`: create a mapped `.kube`
- `Kube.open(path, mode=...)`: open an existing `.kube`
- `Kube.from_bytes(...)`
- `Kube.from_file(...)`
- `from_polars(...)`

The package also re-exports convenience helpers from `kube/__init__.py`:

- `create(...)`
- `open(...)`
- `import_file(...)`
- `from_bytes(...)`
- `from_file(...)`
- `from_polars(...)`
- `load(...)`

### Reading and writing cells

Important low-level methods on `Kube`:

- `get_packed(x, y, z)`
- `get_face(x, y, z, face_idx)`
- `set_face(x, y, z, face_idx, value)`
- `get_cell(x, y, z)`
- `set_cell(x, y, z, cell)`

`set_face(...)` edits only the selected face bits in-place. It does not automatically flush unless you are using the renderer, which explicitly calls back into `flush()`.

### Metadata and inspection

Useful methods:

- `memory_stats()`
- `inspect()`
- `update_metadata(...)`
- `flush()`
- `close()`

Useful labels for UI and logging:

- `file_label`
- `storage_label`

### Table-oriented access

Kube is designed to bridge into Polars cleanly.

Key methods:

- `to_polars(...)`
- `feature_frame(...)`
- `supervised_frame(...)`
- `face_histogram(...)`
- `read_region(...)`
- `iter_region(...)`
- `iter_batches(...)`
- `iter_supervised_batches(...)`

Typical column groups are:

- coordinates: `x`, `y`, `z`
- packed value: `packed`
- decoded faces: `face_0` through `face_5`

## Renderer Architecture

The native viewer is not a separate file parser. It is a live view over the current Kube buffer.

The call chain is:

```text
Kube.view(...)
  -> Renderer(...).run()
  -> load_dll().kube_run(...)
  -> native event/render loop in src/kube_renderer.c
  -> Clay-backed UI module in src/kube_ui_clay.c
```

### What Python passes to native code

`Renderer.run()` sends the renderer:

- the raw buffer address from `kube.buffer_address()`
- `W`, `H`, `D`
- `states`
- `itemsize`
- initial view mode
- initial slice axis and slice index
- initial LOD
- window width and height
- `read_only` flag
- file/storage labels for the HUD
- a `set_face` callback
- a `flush` callback
- a `run_code` callback for the built-in Python console

This is why the native code can read the data directly and why edits in the viewer change the real Kube object immediately.

### Native callback contract

`kube/_dll.py` defines three `ctypes` callbacks:

- `SET_FACE_CB`
- `FLUSH_CB`
- `RUN_CODE_CB`

`kube/renderer.py` wraps Python methods in these callbacks:

- `set_face(...)` forwards to `self.kube.set_face(...)`
- `flush()` forwards to `self.kube.flush()`
- `run_code(...)` evaluates Python inside a persistent debug namespace, handles autocomplete requests, and returns formatted output text

If either callback throws in Python, the wrapper returns `0`, which the C renderer treats as a failed edit/autosave operation.

### Built-in Python console

The editor now includes a built-in Python console for live debugging.

Open it with:

- `` ` ``
- `F2`

Basic behavior:

- `Enter` executes the current command
- `Tab` autocompletes Python names and dotted attributes
- `Esc`, `` ` ``, or `F2` closes the console
- `Ctrl+Q` exits the viewer
- `Ctrl+V` pastes from the clipboard
- `Up` recalls the last command

UI behavior:

- the console opens as a compact floating terminal panel rather than a full-width overlay
- its height is about half the window height and its width is intentionally narrow
- the panel can be dragged from its header bar
- the header includes a close button
- completion matches are shown in a dedicated hint panel above the output area
- executed code renders labeled `output`, `result`, or `error` blocks so printed text and return values stay separate

The console runs against the live `Kube` object already opened by the editor.
Its Python namespace persists between commands, so variables you assign stay
available for later commands.

Useful built-in objects:

- `kube`: the live `Kube` instance
- `selection`: selected coordinates plus active face, or `None`
- `hover`: hovered coordinates plus face, or `None`
- `selected_cell`, `hover_cell`: decoded `Cell` objects
- `selected_face_value`, `hover_face_value`
- `selected_packed`, `hover_packed`
- `_`: last expression result

Useful built-in helpers:

- `cell(x, y, z)`: return one decoded cell
- `packed(x, y, z)`: return one packed integer
- `face(x, y, z, face_idx)`: return one decoded face value
- `selection_bounds(radius=0)`: region bounds around the selected cell
- `inspect_selection()`: dict with packed value, face values, and active face value
- `frame(bounds=None, radius=None, include_packed=True)`: materialize a Polars frame for a region or selected neighborhood
- `histogram(bounds=None, radius=None, normalize=False)`: face-value histogram for a region or selected neighborhood
- `set_selected(value, face_idx=None)`: mutate the selected face and flush immediately
- `console_api()`: return a summary of the console namespace itself

Example console commands:

```python
kube.inspect()
inspect_selection()
frame(radius=1)
histogram(radius=2, normalize=True)
set_selected(3)
```

Autocomplete examples:

```python
kub<Tab>
selection_<Tab>
kube.insp<Tab>
```

## Native Renderer Behavior

### Entry point

The exported function is:

```c
KUBE_API void kube_run(...)
```

It owns the full interactive window lifecycle:

- create window
- initialize font and Clay UI state
- build geometry from the packed grid
- process input
- rebuild geometry when needed
- draw the scene, then render Clay UI commands on top
- close and free GPU resources

### View modes

The renderer has two modes:

- `VIEW_SHELL`
- `VIEW_SLICE`

#### Shell mode

Shell mode shows only the camera-facing exterior faces of the cube.

Important details:

- only outer boundary faces are eligible
- only the 3 currently camera-facing sides are enabled
- when the camera crosses to the opposite side, visible-face selection changes
- that visibility change triggers a geometry rebuild

This keeps rendering cheaper for large grids and makes the outer surface easier to inspect.

#### Slice mode

Slice mode shows one planar slice of cells along a chosen axis:

- `X`
- `Y`
- `Z`

In slice mode:

- only cells in the selected slice index are considered
- all 6 faces of those cells are rendered
- the selected face for hover/edit is inferred from camera direction for the slice plane

### LOD sampling

The renderer supports level-of-detail sampling through `view.lod`.

Behavior:

- `lod=1` means full sampling
- larger values skip interior positions on the visible shell or slice grid
- boundary rows and edges are preserved so the structure still reads visually

Changing LOD triggers a rebuild of the generated mesh.

### Geometry generation

The renderer does not draw cubes one by one every frame. It builds batched face geometry.

The rough flow is:

1. count visible faces for the current view state
2. allocate CPU arrays for positions, colors, and indices
3. decode the packed face values for each visible face
4. assign a color based on the face state
5. upload the geometry to GPU buffers

Key helpers:

- `build_geometry(...)`
- `upload_geo(...)`
- `gpu_free(...)`
- `geo_free(...)`

Coloring rules:

- for `states <= 8`, a fixed neon palette is used
- for larger state counts, colors come from `ColorFromHSV(...)`

### Picking and selection

The viewer supports mouse picking of visible cells.

The interaction flow is:

- move the mouse over the viewport
- `pick_cell(...)` determines the hovered cell and face
- left click stores that cell in `EditorState`
- the inspector panel shows the selected cell's packed value and per-face values

Selection state includes:

- whether a selection exists
- selected `x`, `y`, `z`
- selected face index
- a short status string used for HUD feedback

### Editing model

Edits are live and callback-driven.

The key helpers are:

- `try_edit_face(...)`
- `try_edit_face_delta(...)`

An edit path looks like this:

1. user chooses a selected cell and face
2. user presses a key or clicks a UI control
3. native code calls `set_face_cb(...)`
4. Python mutates the real `Kube`
5. native code calls `flush_cb(...)`
6. the renderer marks geometry dirty and rebuilds it

If the viewer is `read_only`, edits are blocked before the callback runs.

### Autosave semantics

The renderer HUD distinguishes between:

- `READ ONLY`
- `AUTOSAVE`

In writable mode, every successful viewer edit attempts to flush immediately. That means the renderer is intentionally biased toward "edit and persist now" rather than batching unsaved local changes.

## Viewer Controls

Current controls implemented in the renderer:

- left click: select a visible cell
- right drag: orbit camera
- mouse wheel: zoom
- `Tab`: toggle shell/slice mode
- `X`, `Y`, `Z`: choose slice axis
- `PageUp`, `PageDown`: move the active slice
- `-` or numpad subtract: increase LOD value
- `=` or numpad add: decrease LOD value
- `1` through `6`: choose the active face in the inspector
- `[` and `]`: decrement or increment the selected face value
- numpad digits: set face values directly when `states <= 10`
- `` ` `` or `F2`: open the built-in Python console
- `Esc`: close help or console overlays
- `Ctrl+Q`: close the viewer

The on-screen controls in the bottom bar mirror the same mode, axis, slice, and LOD actions.

## Build and Native Library Loading

### Build flow

The native renderer is built through `build.py`, which wraps CMake.

The normal commands are:

```bash
python build.py
```

On Windows with MinGW:

```powershell
python build.py --generator "MinGW Makefiles"
```

Useful flags:

- `--clean`
- `--cmake-only`
- `--reconfigure`
- `--build-dir`
- `--use-system-raylib`
- `--no-use-system-raylib`

### CMake behavior

`CMakeLists.txt`:

- builds `src/kube_storage.c` as the storage shared library
- builds `src/kube_renderer.c` plus the Clay UI module as the renderer shared library
- exposes a combined `kube_native` target that builds both
- links `raylib` only on the renderer side
- prefers a system `raylib` if available
- otherwise fetches raylib 5.0 with `FetchContent`

The built shared libraries are written directly into the Python package directory:

- Windows: `kube/kube_storage.dll` and `kube/kube_renderer.dll`
- Linux: `kube/libkube_storage.so` and `kube/libkube_renderer.so`
- macOS: `kube/libkube_storage.dylib` and `kube/libkube_renderer.dylib`

### Runtime loading

`kube/_dll.py` chooses the correct library names for the platform and loads
them with `ctypes.CDLL(...)`.

If either required library is missing, Kube raises a `FileNotFoundError` with
build instructions instead of failing silently.

## CLI Overview

The CLI entry point lives in `kube/main.py`.

Core commands:

- `create`
- `import`
- `export`
- `inspect`
- `stats`
- `view`
- `demo`

There is also a dataset-oriented command group for `.kubed` roots:

- `dataset-create`
- `dataset-import-text`
- `dataset-import-tokens`
- `dataset-inspect`
- `dataset-stats`
- `dataset-sample`

Parsing helpers worth knowing:

- `_parse_shape(...)`
- `_parse_slice(...)`
- `_parse_window_size(...)`

### CLI recipes

Create an empty mapped kube:

```bash
kube create data/sample.kube --shape 64x64x64 --states 4
```

Inspect storage stats and metadata:

```bash
kube inspect data/sample.kube
```

Compute histograms for the whole cube or one slice:

```bash
kube stats data/sample.kube
kube stats data/sample.kube --slice y=12 --normalize
```

Open the viewer in a narrow debug slice:

```bash
kube view data/sample.kube --slice x=8 --lod 2 --read-only
```

## Testing Coverage

`tests/test_kube.py` currently covers the storage layer more than the renderer.

Covered areas include:

- in-memory round-tripping
- mapped create/open/edit flows
- overwrite behavior
- streaming import/export
- read-only mutation protection
- region iteration
- Polars conversion
- batch iteration
- supervised-frame helpers
- basic view argument validation

The native storage engine is validated indirectly through the Python runtime and
the automated core/dataset tests, while the native renderer itself is still
validated mostly through the Python bridge and manual use rather than a
dedicated automated UI test suite.

## Practical Constraints and Gotchas

- `states` must be a power of two. Many packing assumptions depend on this.
- Metadata must fit in the fixed 4096-byte header.
- The viewer requires the native library to be built first.
- The renderer is a shell/slice inspector, not a full volumetric renderer.
- Viewer edits flush immediately in writable mode.
- `Kube.set_face(...)` mutates the buffer but does not flush by itself.
- `window_size` must be a `(width, height)` pair in Python.
- Slice mode and shell mode rebuild geometry when their visible set changes, so very large cubes still depend heavily on LOD for interactivity.

## Choosing The Right Path

If you are deciding which API or workflow to use, this is the short version:

- use `Kube(...)` for small in-memory experiments
- use `create(...)` and `open(...)` for persistent `.kube` files
- use `import_file(...)` when the source is already a byte stream or binary file
- use `to_polars(...)` or `feature_frame(...)` for analytics and feature extraction
- use `supervised_frame(...)` or `iter_supervised_batches(...)` when one face is the label
- use `from_polars(...)` when a transform step already produced a table
- use `examples/train_kube_lm.py` as the simplest starting point for Torch-based LM experiments on `.kubed`
- use `kube view` or `Kube.view(...)` when human inspection or editing is part of the workflow
- use `.kubed` dataset APIs when the problem is really token-sequence storage and iteration rather than cube inspection
