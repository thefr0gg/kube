# 3. The Python API

You've seen `get_face`/`set_face`/`get_cell` already. This chapter rounds
out the methods you'll actually reach for day to day: batch edits, streaming
import, and region iteration. For the exhaustive method list, see
[library-integration.md § API Summary](../library-integration.md).

## Single reads and writes

```python
from kube import Kube

kube = Kube(4, 4, 4, states=4)
kube.set_face(0, 0, 0, 0, 3)
print(kube.get_face(0, 0, 0, 0))   # 3
print(kube.get_packed(0, 0, 0))    # the packed integer for that cell
```

These are the simplest and slowest path — fine for a handful of edits, not
for touching thousands of cells in a loop.

## Batch reads and writes

When you have many scattered edits or reads, batch them into one call
instead of looping `set_face`/`get_face` one at a time:

```python
ops = [(0, 0, 0, 0, 3), (1, 0, 0, 0, 2), (2, 0, 0, 0, 1)]
kube.set_faces_batch(ops)

values = kube.get_faces_batch([(0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (2, 0, 0, 0, 0)])
print(values)
# [3, 2, 1]
```

Each op is `(x, y, z, face_idx, value)` for writes, or the same shape with
`value` ignored for reads. This routes through the native `kube_storage`
accelerator once instead of crossing the Python/native boundary per cell —
it matters once you're touching more than a few dozen faces.

## Streaming a file in

`import_file` treats an ordinary file as a stream of bytes and packs it into
a `.kube` container without loading the whole source into memory first:

```python
from kube import import_file

kube = import_file("input.bin", "data/large.kube", states=4)
print(kube.memory_stats())
kube.close()
```

This is how you'd bring an existing binary artifact — a dataset dump, a
serialized model, anything byte-oriented — into Kube's storage model. Round
trip it back out with `to_bytes()` or `export_file(...)`.

## Iterating a region

For large kubes, materializing the whole grid as one table or array isn't
always practical. `iter_region` walks a bounded area tile by tile:

```python
bounds = ((0, 128), (0, 128), (0, 64))
for origin, tile in kube.iter_region(bounds, tile_shape=(32, 32, 16)):
    print(origin, tile.height)
```

Each `tile` is a Polars DataFrame for that sub-cube (see
[Chapter 4](04-tables-and-interop.md) for what's in it). Use this for
deterministic, memory-bounded passes over data too large to hold as one
frame.

## Batches for training loops

`iter_batches` yields fixed-size row batches across the whole kube (or a
bounded region), which is the shape most training loops expect:

```python
for batch in kube.iter_batches(4096, include_packed=False):
    print(batch.columns)
    # process batch...
```

There's a supervised variant, `iter_supervised_batches(...)`, that splits
one face off as the label column — see
[Chapter 8](08-integrating.md#i-have-a-pytorch-training-loop) for that in
context.

## Flushing and closing

- `flush()` persists pending writes for a mapped `.kube` file. It's a no-op
  for in-memory kubes and for anything opened `mode="r"`.
- `close()` releases the mapped buffer. Prefer the context-manager form
  (`with create(...) as kube:` / `with open_kube(...) as kube:`) so this
  happens automatically even if an exception is raised.

## What's next

This chapter covered reading and writing. The next one covers getting data
*out* of Kube in whatever shape your other tools already expect — Polars,
NumPy, Pandas, Arrow, Parquet — which is usually the more interesting half
of integrating Kube into an existing project.

Next: [Chapter 4 — Tables and Interop](04-tables-and-interop.md).
