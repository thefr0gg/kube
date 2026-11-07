# 6. The Viewer

Tables and histograms get you most of the way to understanding a kube, but
sometimes you just want to look at it. `kube view` opens a 3D viewer/editor
for exactly that.

## Starting it

```bash
kube view data/sample.kube
```

This starts a small FastAPI server on `127.0.0.1` at an automatically picked
port, and opens your default browser to it. The command blocks — leave the
terminal running while you look at the viewer, and press Ctrl+C there when
you're done.

Don't have a file handy yet? `kube demo` builds a small in-memory kube,
fills it with sample bytes, and opens the viewer on it directly — no file
needed:

```bash
kube demo --shape 12x12x12 --states 4
```

## What it's for

The viewer has two render modes:

- **Shell mode** (default) — shows only the outer boundary faces of the
  cube. Good for seeing the overall shape and spotting patterns at a
  glance.
- **Slice mode** — shows every face of the cells on one axis-aligned plane.
  Good for inspecting the interior, one layer at a time.

```bash
kube view data/sample.kube --slice z=4
```

`--lod` (level of detail) skips interior cells on a stride, keeping
boundaries intact — useful once a kube is large enough that rendering every
cell gets slow:

```bash
kube view data/sample.kube --lod 4
```

## Interacting with it

- **Left drag** — orbit the camera.
- **Scroll** — zoom.
- **Double-click a face** — select it; an inspector panel opens showing the
  cell's coordinates, the selected face index, and its current value.
- **Inspector panel** — type a new value and click Apply. The edit is sent
  over a WebSocket, written through the same `set_face`/`flush()` path you'd
  use from Python, and the mesh refreshes to show it.
- The side panel also has controls for view mode, slice axis/index, and LOD
  — the same options available as CLI flags, adjustable live.

## Read-only mode

```bash
kube view data/sample.kube --read-only
```

Opens the file in `mode="r"` (see [Chapter 2](02-storage-model.md)) and
rejects every edit attempt from the viewer with an error message instead of
touching the file — useful when you want to look without any risk of
accidental changes, or when another process already has the file open for
writing.

## A note on scope

The viewer is a FastAPI server bound to `127.0.0.1` only. It has no
authentication and isn't meant to be reachable from anywhere but your own
machine — don't put it behind a reverse proxy or expose the port externally.

Next: [Chapter 7 — Sharded Datasets](07-datasets.md).
