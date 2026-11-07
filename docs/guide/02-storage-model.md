# 2. The Storage Model

Chapter 1 covered the idea: a grid of cells, six faces each, a small integer
per face. This chapter covers how Kube actually stores that grid — in
memory, and on disk.

## In-memory kubes

The simplest way to use Kube is entirely in memory:

```python
from kube import Kube

kube = Kube(4, 4, 4, states=4)
print(kube.shape)      # (4, 4, 4)
print(kube.dtype)       # UInt16
print(kube.capacity_bytes())  # 96
```

A `Kube(W, H, D, states=...)` is a `W × H × D` grid, backed by a plain
Python `bytearray`. It behaves exactly like the `Cell` from Chapter 1, just
addressed by coordinates. This is the right choice for small, throwaway
grids — prototypes, tests, scratch work.

### Why `dtype` is `UInt16`

Each cell packs its six face values into **one integer**, not six separate
bytes. The width of that integer depends on `states`:

```text
bits_per_cell = 6 * log2(states)
```

For `states=4`, that's `6 * 2 = 12` bits per cell — more than fits in a
`UInt8` (8 bits), so Kube rounds up to the next size that fits:

| bits needed | storage width |
|---|---|
| ≤ 8 | `UInt8` |
| ≤ 16 | `UInt16` |
| ≤ 32 | `UInt32` |
| otherwise | `UInt64` |

This is also why `states` must be a power of two: it keeps each face at a
fixed, predictable bit width inside the packed integer, instead of needing
variable-width encoding.

## Memory-mapped `.kube` files

In-memory kubes disappear when the process exits, and they have to fit in
RAM. For anything persistent, or larger than memory, use a mapped `.kube`
file instead — same API, different backing storage:

```python
from kube import create, open as open_kube

with create("data/sample.kube", (32, 32, 32), states=4) as kube:
    kube.update_metadata(dataset="sample", owner="local-dev")
    kube.set_face(0, 0, 0, 0, 2)
    kube.flush()

with open_kube("data/sample.kube", mode="r") as kube:
    print(kube.inspect()["dtype"])   # UInt16
    print(kube.get_face(0, 0, 0, 0))  # 2
```

A `.kube` file is a fixed 4096-byte header (magic bytes, shape, `states`,
your metadata as JSON) followed by the packed cell payload. Opening it with
`mode="r+"` (the default when creating) memory-maps the file for
reading *and* writing — edits go straight to the mapped pages, and
`flush()` persists them.

### Read-only means read-only

Open with `mode="r"` and Kube refuses mutation outright, rather than letting
you silently write to a copy-on-write page that never reaches disk:

```python
with open_kube("data/sample.kube", mode="r") as kube:
    kube.set_face(0, 0, 0, 0, 1)
# PermissionError: kube is opened in read-only mode
```

This matters once you start sharing `.kube` files between a writer process
and one or more readers (the [viewer](06-viewer.md) uses exactly this:
`--read-only` opens in `mode="r"`).

## What memory-mapping buys you

- **Larger than RAM**: the OS pages the file in on demand instead of loading
  it all up front.
- **Persistence**: close the process, reopen the file later, pick up exactly
  where you left off.
- **Streaming import**: `import_file(...)` can pack a file far bigger than
  available memory straight into a `.kube` container without ever holding
  the whole thing in RAM at once (see [Chapter 3](03-python-api.md)).

## Choosing between them

- Prototyping, tests, or a grid that comfortably fits in memory → `Kube(...)`
- Anything you want to persist, reopen later, or that might outgrow RAM →
  `create(...)` / `open(...)`

Both expose the exact same `get_face`/`set_face`/`get_cell` API from
Chapter 1 — switching between them later is a one-line change, not a
rewrite.

Next: [Chapter 3 — The Python API](03-python-api.md), a tour of the methods
you'll actually reach for day to day.
