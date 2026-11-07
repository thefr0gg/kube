# 7. Sharded Datasets

Everything so far has been one `.kube` file. For token-sequence corpora at
LLM-training scale, Kube has a second format built on the same storage
primitives: `.kubed` — a **sharded** dataset root, meant for datasets too
large or too long-running to treat as a single file.

## When to reach for this

Use `.kube` (Chapters 1-6) when you have one spatial/structured grid. Use
`.kubed` when you have a long sequence of tokens — text, pretraining data,
anything you'll iterate over in fixed-length windows across potentially many
shard files.

## Creating a dataset

```bash
kube dataset-create corpora/wiki.kubed --shards 8 --token-bytes 4 --tokens-per-shard 1048576 --sequence-length 2048
```

Or from Python:

```python
from kube import create_sharded

dataset = create_sharded(
    "corpora/wiki.kubed",
    shards=8,
    token_bytes=4,
    tokens_per_shard=1 << 20,
    sequence_length=2048,
)
dataset.close()
```

This creates an empty dataset root: a manifest plus `shards` shard files,
each capable of holding `tokens_per_shard` tokens. Nothing is imported yet.

## Importing tokens

If you already have token IDs (e.g. from a tokenizer you're running
yourself), import them directly from a Parquet/Arrow table:

```bash
kube dataset-import-tokens train.parquet corpora/wiki.kubed --tokens-column input_ids --sequence-length 2048
```

```python
from kube import import_token_ids

dataset = import_token_ids("train.parquet", "corpora/wiki.kubed", tokens_column="input_ids")
print(dataset.inspect()["total_token_count"])
```

If you have raw text instead, `dataset-import-text` / `import_parquet_text`
tokenize it first using a `tokenizers`-compatible tokenizer file — see
[docs/library-integration.md](../library-integration.md) for the full
signature.

## Iterating sequences

The whole point of `.kubed` is fixed-length causal-LM windows, with resume
support for long-running jobs:

```python
from kube import open_dataset

dataset = open_dataset("corpora/wiki.kubed")

for sample in dataset.iter_sequences(2048, stride=2048):
    print(sample["input_ids"][:8], sample["target_ids"][:8])
    resume_state = sample["resume_state"]
    break
```

`target_ids` is `input_ids` shifted by one — the standard next-token
prediction target. `stride` controls overlap between consecutive windows
(`stride == sequence_length` means no overlap).

## Resuming a scan

If a long preprocessing or training job gets interrupted, pick up exactly
where it left off instead of rescanning from the start:

```python
resumed = dataset.resume_iterator(resume_state)
next_sample = next(resumed)

dataset.save_resume_state("checkpoints/wiki.json", resume_state)
loaded_state = dataset.load_resume_state("checkpoints/wiki.json")
```

## Inspecting and sampling from the CLI

```bash
kube dataset-inspect corpora/wiki.kubed
kube dataset-stats corpora/wiki.kubed
kube dataset-sample corpora/wiki.kubed --sequence-length 2048 --count 3
```

`dataset-sample` is the quickest way to eyeball what a dataset will actually
hand your training loop — it prints a handful of `(input_ids, target_ids)`
pairs plus a resume state, without writing anything.

## Batches for training

For actual training loops, `iter_sequence_batches`/`iter_lm_batches` give you
padded, batch-shaped output directly — covered with full context in
[Chapter 8](08-integrating.md#i-have-a-pytorch-training-loop).

Next: [Chapter 8 — Integrating Kube](08-integrating.md), the payoff chapter.
