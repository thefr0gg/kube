# 1. The Concept

Before touching files, CLI commands, or dataframes, it's worth spending five
minutes on the one idea Kube is built around: **a cell with six labeled
faces, each holding a small number.**

## The problem with plain bits

A lot of spatial and grid-like data gets modeled as a simple occupancy grid:
each position in a 3D grid is either `0` or `1`. That's compact, but it
throws away information. A single bit can't tell you *which side* of a cell
something happened on, or represent more than two states (on/off) at that
position.

Kube starts from a richer unit instead.

## Cell, face, state

- A **cell** is one address in a 3D grid — an `(x, y, z)` position.
- Every cell has exactly **six faces**: `top`, `bottom`, `left`, `right`,
  `front`, `back`. These are fixed, in that order, by face index `0..5`.
- Every face holds a **state** — an integer from `0` up to `states - 1`.

`states` is a number you choose once per kube (it must be a power of two,
at least `2`). It controls how much each face can express:

| `states` | what each face can hold |
|---|---|
| `2` | a single bit — back to plain on/off, if that's all you need |
| `4` | 4 categories, confidence bands, or small intensity levels |
| `256` | a full byte of information per face |

So a cell isn't "on or off." It's six independent small measurements, one
per direction, that can represent whatever your data actually means:
category, confidence, direction, intensity — anything you can enumerate.

## A cell, by hand

You can build and inspect one `Cell` without any file or grid at all:

```python
from kube import Cell

cell = Cell(states=4)
print(cell.faces)
# [0, 0, 0, 0, 0, 0]
```

A fresh cell starts with every face at `0`. Set a couple of faces (face
index `0` is `top`, index `4` is `front`):

```python
cell.set_face(0, 3)   # top = 3
cell.set_face(4, 1)   # front = 1
print(cell.faces)
# [3, 0, 0, 0, 1, 0]

print(cell.get_face(0))
# 3
```

Internally, Kube doesn't store those six values as six separate numbers — it
packs them into one integer, with each face occupying a fixed-width slice of
bits:

```python
packed = cell.to_packed()
print(packed)
# 259

restored = Cell.from_packed(packed, states=4)
print(restored.faces)
# [3, 0, 0, 0, 1, 0]
```

That packing is the whole reason Kube can store huge grids compactly and
still memory-map them — see [Chapter 2](02-storage-model.md). You almost
never need `to_packed()`/`from_packed()` directly; they exist so you can see
what's actually happening under `get_face`/`set_face`.

## A grid of cells

A `Kube` object is a 3D grid of `Cell`s. This is the same `set_face`/
`get_cell` API, just addressed by `(x, y, z)` instead of working with one
loose `Cell`:

```python
from kube import Kube

kube = Kube(2, 2, 2, states=4)
kube.set_face(0, 0, 0, 0, 3)   # cell (0,0,0), top = 3
kube.set_face(0, 0, 0, 4, 1)   # cell (0,0,0), front = 1

cell = kube.get_cell(0, 0, 0)
print(cell.faces)
# [3, 0, 0, 0, 1, 0]
```

That's the entire mental model: **a grid of cells, six faces per cell, a
small integer per face.** Everything else in Kube — the `.kube` file format,
the Polars/NumPy/Arrow export, the CLI, the 3D viewer — is built on top of
exactly this.

Next: [Chapter 2 — The Storage Model](02-storage-model.md), where this same
cell/face/state idea becomes something you can memory-map from disk.
