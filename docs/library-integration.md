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

## Limitations

- The current viewer targets shell and slice exploration, not volumetric ML visualization.
- The storage format is optimized for face-state data, not arbitrary tensor metadata schemas.
- `from_polars(...)` currently expects explicit coordinates and either packed values or full face columns.
- The first `.kubed` implementation is single-machine oriented and keeps distributed orchestration out of scope.
