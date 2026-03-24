# Library Integration Guide

This guide focuses on using `kube` as a Python library in data engineering, machine learning, analytics, and batch-processing pipelines.

## When To Use The Library API

Use the Python API when you want to:

- build `.kube` files inside ETL jobs
- build sharded `.kubed` datasets for LLM corpora
- open mapped kube files from training or inference code
- transform kube contents into Polars frames
- iterate in batches without materializing the whole cube as one large table
- write model outputs back into kube storage

## Core Patterns

### 1. Create a training dataset container

```python
from kube import create

with create("datasets/train.kube", (256, 256, 128), states=4) as kube:
    kube.update_metadata(split="train", source="preprocessed-v2")
```

### 1b. Create a sharded LLM dataset

```python
from kube import create_sharded

dataset = create_sharded(
    "datasets/corpus.kubed",
    shards=8,
    token_bytes=4,
    tokens_per_shard=1 << 20,
    sequence_length=2048,
)
```

### 2. Open a mapped kube for feature extraction

```python
from kube import open as open_kube

with open_kube("datasets/train.kube", mode="r") as kube:
    features = kube.feature_frame(include_coords=True, include_packed=False)
```

`feature_frame(...)` is usually the simplest choice for ML features because it returns face columns directly.

### 3. Stream minibatches

```python
with open_kube("datasets/train.kube", mode="r") as kube:
    for batch in kube.iter_batches(4096, include_packed=False):
        # Convert or hand off to your framework.
        # Example columns: x, y, z, face_0..face_5
        pass
```

This pattern is useful when the kube is larger than RAM or when you want deterministic chunking for preprocessing.

### 4. Build a supervised training frame

```python
with open_kube("datasets/train.kube", mode="r") as kube:
    training_frame = kube.supervised_frame(
        target_face=4,
        include_coords=True,
        include_packed=False,
    )
```

This is useful when one face should be treated as the label and the remaining faces should be used as features.

### 5. Stream supervised batches

```python
with open_kube("datasets/train.kube", mode="r") as kube:
    for batch in kube.iter_supervised_batches(
        4096,
        target_face=4,
        include_coords=True,
    ):
        pass
```

### 6. Measure face distributions

```python
with open_kube("datasets/train.kube", mode="r") as kube:
    histogram = kube.face_histogram(normalize=True)
```

This is handy for dataset balancing checks, drift detection, and basic exploratory analysis.

### 7. Store predictions back into a kube

```python
import polars as pl
from kube import from_polars

predictions = pl.DataFrame(
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

predicted_kube = from_polars(
    predictions,
    shape=(64, 64, 64),
    states=4,
    path="datasets/predictions.kube",
    metadata={"producer": "model-v1"},
)
predicted_kube.close()
```

### 8. Iterate causal LM sequences

```python
from kube import open_dataset

dataset = open_dataset("datasets/corpus.kubed")

for sample in dataset.iter_sequences(2048, stride=2048, prefetch_shards=1):
    input_ids = sample["input_ids"]
    target_ids = sample["target_ids"]
    resume_state = sample["resume_state"]
    break

dataset.save_resume_state("checkpoints/corpus.json", resume_state)
loaded_state = dataset.load_resume_state("checkpoints/corpus.json")

for frame_batch in dataset.iter_sequence_frame_batches(8, 2048, start_state=loaded_state):
    print(frame_batch.columns)
    break
```

### 9. Use the optional torch adapter

```python
from kube import open_dataset
from kube.integrations.torch import KubeCausalLMIterableDataset

dataset = open_dataset("datasets/corpus.kubed")
torch_dataset = KubeCausalLMIterableDataset(dataset, sequence_length=2048)
```

### 10. Export to Arrow or Parquet

```python
from kube import open as open_kube

with open_kube("datasets/train.kube", mode="r") as kube:
    # Arrow IPC — near-zero-copy via Polars' Arrow-native memory
    table = kube.to_arrow()                             # pa.Table, wide_faces layout
    table = kube.to_arrow(schema_mode="long_faces")     # melted N×6 rows
    table = kube.to_arrow(schema_mode="cell_packed")    # single packed-int column

    # Parquet
    kube.to_parquet("out/train.parquet")
    kube.to_parquet("out/region.parquet", bounds=((0, 32), (0, 32), (0, 16)))
```

Schema modes:

| Mode | Columns | Rows |
|------|---------|------|
| `wide_faces` (default) | `x, y, z, face_0..face_5` | N (one per cell) |
| `long_faces` | `x, y, z, face, value` | N×6 (one per face) |
| `cell_packed` | `x, y, z, packed` | N (packed integer) |

### 11. Export to NumPy

```python
import numpy as np
from kube import open as open_kube

with open_kube("datasets/train.kube", mode="r") as kube:
    # (W, H, D, 6) spatial array
    arr = kube.faces_to_numpy()

    # (N, 6) flat array — same data, no reshape
    flat = kube.to_numpy()

    # NumPy array protocol — np.array(kube) works too
    arr2 = np.array(kube)

    # Bounded region
    region = kube.region_to_numpy(((0, 8), (0, 8), (0, 8)))
```

Functional equivalents are available in `kube.integrations.numpy_compat`:

```python
from kube.integrations.numpy_compat import faces_to_numpy, kube_from_numpy, packed_to_numpy

arr = faces_to_numpy(kube)          # (W, H, D, 6)
packed = packed_to_numpy(kube)      # 1-D packed integers

# Round-trip: numpy array → Kube
restored = kube_from_numpy(arr, states=4)
```

### 12. sklearn-compatible (X, y) arrays and spatial splitting

```python
from kube.integrations.sklearn import to_sklearn_xy, spatial_train_test_split

X, y = to_sklearn_xy(kube, target_face=3)   # X shape (N, 5), y shape (N,)

# Split along a spatial axis to avoid adjacent-cell leakage
X_train, X_test, y_train, y_test = spatial_train_test_split(
    kube,
    target_face=3,
    test_axis=2,       # split along z-axis
    test_fraction=0.2,
)
```

### 13. Export a sharded dataset to HuggingFace

```python
from kube import open_dataset
from kube.integrations.hf import to_hf_dataset, to_hf_iterable_dataset

dataset = open_dataset("datasets/corpus.kubed")

# In-memory Dataset (materialises all windows)
hf_ds = dataset.to_hf_dataset(sequence_length=1024, count=10_000)

# Streaming IterableDataset (no materialisation)
it_ds = dataset.to_hf_iterable_dataset(sequence_length=1024)

# Or use the module-level helper
it_ds2 = to_hf_iterable_dataset(dataset, sequence_length=1024)
```

### 14. HuggingFace Trainer-compatible data collator

```python
from kube.dataset import KubeDataCollatorForCausalLM

collator = KubeDataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id)

# Use with HuggingFace Trainer
from transformers import Trainer
trainer = Trainer(
    model=model,
    train_dataset=it_ds,
    data_collator=collator,
)
```

### 15. Schema and metadata constants

```python
from kube.schemas import SCHEMA_WIDE_FACES, SCHEMA_LONG_FACES, schema_columns
from kube.metadata import SPLIT, TOKENIZER_NAME, SEQUENCE_LENGTH

# Inspect expected column layout before exporting
cols = schema_columns(SCHEMA_WIDE_FACES, states=4)
# {'x': 'UInt32', 'y': 'UInt32', 'z': 'UInt32',
#  'face_0': 'UInt8', ..., 'face_5': 'UInt8'}

# Use as standard metadata keys
kube.update_metadata({SPLIT: "train", TOKENIZER_NAME: "gpt2"})
```

### 16. Check which optional backends are available

```python
from kube.integrations import available_backends
print(available_backends())
# e.g. ['arrow', 'numpy', 'pandas', 'torch']
```

## Recommended Table Shapes

`to_polars(...)` returns a table with:

- `x`, `y`, `z`
- optional `packed`
- optional `face_0` through `face_5`

For most analytics and ML tasks:

- use `feature_frame(include_coords=True, include_packed=False)` for model inputs
- use `supervised_frame(target_face=...)` when one face acts as the label
- keep `packed` if you want the compact encoded value as an extra feature
- drop coordinates if spatial position should not be part of the model

For language-model pretraining:

- store corpora in `.kubed` roots
- ingest either token-id rows or raw text plus a tokenizer
- iterate with `iter_sequences(...)` or `iter_sequence_batches(...)`
- use `iter_sequence_frame_batches(...)` when downstream tooling expects Polars frames
- store and reuse `resume_state` for restartable scans

## Integration Ideas

### Polars-first analytics

```python
summary = (
    kube.to_polars(include_coords=False)
    .group_by("face_0")
    .len()
    .sort("face_0")
)
```

### Training pipeline handoff

You can convert a batch from Polars into the structure your training stack expects:

- Arrow tables
- pandas DataFrames
- Python lists or dicts
- tensors through your framework-specific adapter layer

Keeping `kube` itself free of heavyweight ML dependencies makes it easier to integrate into multiple stacks.

### Offline feature engineering

Use `iter_region(...)` when you want spatial tiling behavior, and `iter_batches(...)` when you want simple row-count based batches.

Use `slice_bounds(...)` if you want to operate on a single `x`, `y`, or `z` slice while keeping the same table-oriented APIs.

## Suggested Conventions

- Use one `.kube` file per dataset split or pipeline stage.
- Store pipeline provenance with `update_metadata(...)`.
- Open training data with `mode="r"` when you want safe read-only access.
- Write transformed or predicted outputs into a new `.kube` file rather than mutating your source dataset in place.

## API Summary

Top-level:

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

On `Kube`:

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
- `schema(mode)` — column layout dict for a given schema mode
- `to_arrow(bounds, schema_mode)` — PyArrow Table
- `from_arrow(table, shape, states)` — classmethod
- `to_numpy(bounds, packed)` — NumPy array
- `faces_to_numpy()` — `(W, H, D, 6)` spatial array
- `region_to_numpy(bounds)` — `(N, 6)` bounded array
- `to_pandas(bounds)` — Pandas DataFrame
- `from_pandas(df, shape, states)` — classmethod
- `to_parquet(path, bounds, schema_mode)` — Parquet file
- `to_sklearn_xy(target_face, bounds)` — `(X, y)` arrays

On `ShardedKubeDataset`:

- `inspect()`
- `shard_frame()`
- `document_frame()`
- `iter_sequences(...)`
- `iter_sequence_batches(...)`
- `iter_sequence_frame_batches(...)`
- `resume_iterator(...)`
- `save_resume_state(...)`
- `load_resume_state(...)`
- `sample_sequences_frame(...)`
- `export_stats_parquet(...)`
- `to_arrow(sequence_length, count)` — PyArrow Table of sequence windows
- `to_parquet(path, sequence_length, count)` — Parquet file
- `to_hf_dataset(sequence_length, count)` — HuggingFace in-memory Dataset
- `to_hf_iterable_dataset(sequence_length)` — streaming IterableDataset
- `iter_lm_batches(batch_size, sequence_length)` — padded LM batches

## Limitations

- The current viewer targets shell and slice exploration, not volumetric ML visualization.
- The storage format is optimized for face-state data, not arbitrary tensor metadata schemas.
- `from_polars(...)` currently expects explicit coordinates and either packed values or full face columns.
- The first `.kubed` implementation is single-machine oriented and keeps distributed orchestration out of scope.
