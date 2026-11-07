# 8. Integrating Kube

This is the payoff chapter. You understand the concept, the storage model,
the API, and the tools around it. Now: concrete recipes for dropping Kube
into a project that already exists, organized by what you're already using.

Each recipe shows a short "before" — what the step looks like today — and
an "after" — the same step with Kube in it.

## I have a Polars pipeline

**Before** — some step in your pipeline produces a Polars DataFrame and
hands it to the next step directly:

```python
import polars as pl

frame = pl.DataFrame({
    "x": [0, 0, 1, 1], "y": [0, 1, 0, 1], "z": [0, 0, 0, 0],
    "face_0": [1, 2, 3, 0], "face_1": [0, 0, 0, 0], "face_2": [0, 0, 0, 0],
    "face_3": [0, 0, 0, 0], "face_4": [0, 0, 0, 0], "face_5": [0, 0, 0, 0],
})
next_step(frame)
```

**After** — materialize it as a kube first, so the data is now compact,
memory-mappable, and inspectable (via the CLI or viewer) between steps,
while `next_step` keeps getting the same kind of Polars frame it always did:

```python
from kube import from_polars

kube = from_polars(frame, shape=(2, 2, 1), states=4)
kube.to_parquet("checkpoints/step_output.parquet")  # or keep going:
next_step(kube.to_polars())
```

Nothing about `next_step` has to change — `to_polars()` gives it back the
same column shape it expected. You've gained a durable, inspectable
checkpoint for free.

## I have a Pandas pipeline

**Before:**

```python
import pandas as pd
df = pd.read_parquet("features.parquet")
model.fit(df.drop(columns=["label"]), df["label"])
```

**After** — same idea, with Kube as the storage layer in between:

```python
from kube import Kube

kube = Kube.from_pandas(df, shape=(8, 8, 8), states=4)
df_restored = kube.to_pandas()
model.fit(df_restored.drop(columns=["label"]), df_restored["label"])
```

(`to_pandas()` needs `pyarrow` installed alongside pandas — see
[Chapter 4](04-tables-and-interop.md#pandas).)

## I just want a drop-in binary container

If you don't care about the spatial structure at all and just want Kube as
a compact, memory-mappable, inspectable container for an existing binary
artifact (model weights, a serialized blob, anything byte-oriented):

```python
from kube import from_bytes

payload = open("model_weights.bin", "rb").read()
kube = from_bytes(payload, states=4, path="artifacts/model_weights.kube")

restored = kube.to_bytes()
assert restored[: len(payload)] == payload
```

For files too large to read into memory first, stream it instead with
`import_file(...)` (see [Chapter 3](03-python-api.md#streaming-a-file-in)).
The full version of this pattern, including Arrow/Parquet export, is in
[`examples/roundtrip_file_container.py`](../../examples/roundtrip_file_container.py).

## I have a scikit-learn workflow

**Before:**

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

**After** — if your features are naturally gridded/spatial, store them as a
kube and let it produce `(X, y)` arrays directly, with one face as the
label:

```python
from kube import Kube
from kube.integrations.sklearn import spatial_train_test_split

kube = Kube(8, 8, 8, states=4)
# ... populate kube via set_face / set_faces_batch ...

X_train, X_test, y_train, y_test = spatial_train_test_split(
    kube, target_face=0, test_axis=2,
)

model = LogisticRegression()
model.fit(X_train, y_train)
```

Use `spatial_train_test_split` instead of a random split whenever adjacent
cells could leak information between train and test — a random split on
gridded data tends to overstate accuracy because neighboring cells are
rarely independent. It splits along one spatial axis instead, so train and
test come from genuinely separate regions. If leakage isn't a concern, plain
`kube.to_sklearn_xy(target_face=0)` gives you the full `(X, y)` pair without
splitting.

## I have a PyTorch training loop

Kube's `kube.integrations.torch` module has four dataset adapters, each
matching a different existing shape of PyTorch code (requires `torch`:
`poetry install -E torch`).

**If you already load the whole grid into memory as features:**

```python
# Before: your own feature-loading Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.X = load_features(path)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return {"features": self.X[i]}

# After
from kube import open as open_kube
from kube.integrations.torch import KubeDataset

kube = open_kube("data/features.kube", mode="r")
dataset = KubeDataset(kube, target_face=0)   # yields {"features": ..., "target": ...}
loader = torch.utils.data.DataLoader(dataset, batch_size=64, collate_fn=None)
```

**If you process spatial data tile by tile** (large grids that don't fit
as one in-memory frame):

```python
from kube.integrations.torch import KubeRegionDataset

bounds = ((0, 128), (0, 128), (0, 64))
dataset = KubeRegionDataset(kube, bounds, tile_shape=(32, 32, 16))
# each item: a (32, 32, 16, 6) float tensor
```

**If you want `(X, y)` tensors directly, sklearn-style but for torch:**

```python
from kube.integrations.torch import KubeSupervisedDataset

dataset = KubeSupervisedDataset(kube, target_face=0)
# each item: (features_tensor, target_tensor)
```

**If you're training a causal LM on a `.kubed` dataset** (streaming, not
map-style — see [Chapter 7](07-datasets.md)):

```python
from kube import open_dataset
from kube.integrations.torch import KubeCausalLMIterableDataset

sharded = open_dataset("corpora/wiki.kubed")
torch_dataset = KubeCausalLMIterableDataset(sharded, sequence_length=2048)
loader = torch.utils.data.DataLoader(torch_dataset, batch_size=8)
```

This one partitions shards across DataLoader workers automatically, so
`num_workers > 1` doesn't duplicate reads. A full working training script
using this pattern is in
[`examples/train_kube_lm.py`](../../examples/train_kube_lm.py).

For any of the map-style datasets, `collate_kube_batch` stacks a list of
sample dicts into one batched dict — hand it to `DataLoader(..., collate_fn=collate_kube_batch)`
if the default collation doesn't fit your sample shape. For a one-shot
conversion with no Dataset class at all, `to_torch(kube)` returns a plain
`{"face_0": tensor, ..., "face_5": tensor}` dict.

## I have a HuggingFace Datasets/Trainer pipeline

Requires `datasets` (`poetry install -E hf`).

**A kube grid as an in-memory `datasets.Dataset`:**

```python
from kube.integrations.hf import to_hf_dataset

hf_ds = to_hf_dataset(kube)   # datasets.Dataset, Arrow-backed
```

**A `.kubed` sharded dataset as a streaming `IterableDataset`** — the HF
equivalent of the torch iterable dataset above:

```python
from kube import open_dataset
from kube.integrations.hf import to_hf_iterable_dataset

sharded = open_dataset("corpora/wiki.kubed")
it_ds = to_hf_iterable_dataset(sharded, sequence_length=1024)
# yields {"input_ids": [...], "attention_mask": [...], "labels": [...]}
```

**Training with `Trainer`** — pad variable-length sequences with Kube's
built-in collator instead of writing your own:

```python
from transformers import Trainer
from kube.dataset import KubeDataCollatorForCausalLM

collator = KubeDataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id)
trainer = Trainer(model=model, train_dataset=it_ds, data_collator=collator, ...)
```

The collator pads `input_ids` with `pad_token_id` and pads `labels` with
`-100` (which `CrossEntropyLoss` ignores automatically) — both standard HF
conventions, so `Trainer` needs no other changes to accept it.

**Going the other way** — importing an existing HF `Dataset` into Kube —
use `from_hf_dataset(...)`, the mirror of `to_hf_dataset(...)`.

## What's next

You've seen the shape of the integration pattern: wherever your project
already has a table, array, or dataset abstraction, Kube has an adapter that
produces exactly that shape, backed by the same compact packed-cell storage
underneath. Pick the adapter that matches what you already have — you don't
need to rewrite the rest of your pipeline to adopt Kube at one step.

Next: [Chapter 9 — Next Steps](09-next-steps.md).
