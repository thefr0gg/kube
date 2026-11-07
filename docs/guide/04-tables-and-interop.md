# 4. Tables and Interop

Kube's native storage is packed cells, but almost nobody wants to work with
packed integers directly. This chapter covers getting a kube into whatever
table or array shape your existing tools already expect — and back.

Every conversion routes through the same place: `to_polars()`. Polars uses
Arrow-native memory internally, so the other formats (Arrow, Parquet,
NumPy, Pandas) build on top of it with little or no extra copying.

## Polars

```python
from kube import Kube

kube = Kube(4, 4, 4, states=4)
kube.set_face(0, 0, 0, 0, 3)

frame = kube.to_polars()
print(frame.columns)
# ['x', 'y', 'z', 'packed', 'face_0', 'face_1', 'face_2', 'face_3', 'face_4', 'face_5']
print(frame.height)
# 64
```

Every row is one cell: coordinates, the packed integer, and all six decoded
faces as separate columns. If you want a leaner frame for feature
engineering, `feature_frame(...)` lets you drop the packed column:

```python
features = kube.feature_frame(include_coords=True, include_packed=False)
print(features.columns)
# ['x', 'y', 'z', 'face_0', 'face_1', 'face_2', 'face_3', 'face_4', 'face_5']
```

Rebuilding a kube from a (possibly transformed) frame is the mirror image:

```python
from kube import from_polars

restored = from_polars(frame, shape=(4, 4, 4), states=4)
print(restored.get_face(0, 0, 0, 0))
# 3
```

The frame needs `x`, `y`, `z`, plus either `packed` or all six `face_0..5`
columns.

## NumPy

```python
arr = kube.faces_to_numpy()   # (W, H, D, 6), dtype uint8/uint16/uint32
print(arr.shape, arr.dtype)
# (4, 4, 4, 6) uint8

arr2 = np.array(kube)         # same thing, via the __array__ protocol
```

`faces_to_numpy()` is the one to reach for if some other part of your
pipeline already expects a plain spatial array instead of a table.

## Pandas

```python
df = kube.to_pandas()
print(list(df.columns)[:3], len(df))
# ['x', 'y', 'z'] 64
```

> **Note:** `to_pandas()` goes through Polars' pandas bridge, which needs
> `pyarrow` installed even though you're asking for pandas, not Arrow.
> Install it alongside pandas: `pip install pyarrow pandas`.

## Arrow and Parquet

Requires `pyarrow` (`poetry install -E arrow`, or `pip install pyarrow`):

```python
table = kube.to_arrow()                            # wide_faces layout
print(table.column_names)
# ['x', 'y', 'z', 'face_0', 'face_1', 'face_2', 'face_3', 'face_4', 'face_5']

table_long = kube.to_arrow(schema_mode="long_faces")  # one row per face
print(table_long.column_names)
# ['x', 'y', 'z', 'face', 'value']

kube.to_parquet("out.parquet")
```

Three schema modes are available everywhere a `schema_mode` argument
appears (`to_arrow`, `to_parquet`, the CLI's `export-arrow`/`export-parquet`):

| mode | shape |
|---|---|
| `wide_faces` (default) | one row per cell, `face_0..face_5` columns |
| `long_faces` | one row per `(cell, face)` pair — "melted" |
| `cell_packed` | one row per cell, single `packed` column |

Pick `wide_faces` for most analytics, `long_faces` if your downstream tool
wants tidy/long data, `cell_packed` if you only need the compact round-trip
representation.

## Picking a format

If your project already has a table library it leans on, use that one —
there's no "best" format here, just whichever one the rest of your pipeline
already speaks:

- Already on Polars → `to_polars()` / `from_polars()`, nothing else needed.
- Already on Pandas → `to_pandas()` (install `pyarrow` alongside it).
- Feeding NumPy-based code (sklearn, custom array code) → `faces_to_numpy()`.
- Writing files for other tools/languages to read → `to_parquet()` or
  `to_arrow()`.

[Chapter 8](08-integrating.md) picks this up again with full before/after
recipes for specific ecosystems (PyTorch, scikit-learn, HuggingFace).

Next: [Chapter 5 — The CLI](05-cli.md).
