# 5. The CLI

Everything so far has been the Python API. Kube also ships a `kube` command
for everyday work without writing a script: creating files, importing data,
inspecting what you have, and checking distributions.

All examples below use `--quiet` to suppress the log lines Kube normally
prints to stderr (handy for scripting or for docs; drop it to see what's
happening as commands run).

## Create

```bash
kube create data/sample.kube --shape 4x4x4 --states 4
```

```
Created data/sample.kube
Shape: 4x4x4  States: 4  DType: UInt16  Capacity: 96 bytes
```

`--shape` accepts either one number (a cube) or `WxHxD`. Add `--force` to
overwrite an existing file.

## Import

Stream an ordinary file into a `.kube` container:

```bash
kube import input.txt data/sample.kube --states 4
```

```
Imported input.txt -> data/sample.kube
Shape: 4x4x4  States: 4  Packed: 128 bytes  Source: 46 bytes
```

`import` figures out a shape that fits the source file if you don't pass
`--shape` explicitly. `export` does the reverse: unpack a `.kube` file back
to raw bytes.

## Inspect

```bash
kube inspect data/sample.kube
```

```
path               data/sample.kube
storage            mapped
acceleration       native
grid_shape         4x4x4
states_per_face    4
bits_per_face      2
dtype              UInt16
packed_bytes       128
unpacked_bytes     384
ratio              3.000
capacity_bytes     96
source_byte_count  0
```

This is a table by default. If you want the same data as JSON for scripting,
add the global `--output json` flag before the subcommand:

```bash
kube --output json inspect data/sample.kube
```

## Stats

Per-face value histograms, for the whole kube or one slice:

```bash
kube stats data/sample.kube
```

```
Shape: 4x4x4  States: 4

face    value  count
------  -----  -----
top     0      64
top     1      0
top     2      0
top     3      0
...
```

Add `--normalize` for proportions instead of raw counts, or `--slice z=0` to
restrict to one 2D plane — useful once a kube is too large to eyeball in
full.

## Exporting to Arrow/Parquet

```bash
kube export-arrow data/sample.kube out.arrow
kube export-parquet data/sample.kube out.parquet --schema-mode long_faces --compression zstd
```

Same schema modes as the Python API (`wide_faces`, `long_faces`,
`cell_packed`) — see [Chapter 4](04-tables-and-interop.md).

## The rest of the global flags

- `--output text|json` — pick the result format for commands that support
  both (default `text`).
- `--quiet` — suppress all but error-level log lines.
- `--log-level debug|info|warning|error` — set verbosity directly.
- `--log-file <path>` — also write the same log lines to a file.

## What's not in this chapter

The `dataset-*` subcommands (for `.kubed` sharded datasets) get their own
chapter — [Chapter 7](07-datasets.md). `view` and `demo` get
[Chapter 6](06-viewer.md). `benchmark` is covered in the main
[README.md](../../README.md#benchmark-results).

Next: [Chapter 6 — The Viewer](06-viewer.md).
