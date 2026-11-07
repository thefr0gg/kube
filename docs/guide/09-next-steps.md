# 9. Next Steps

You've gone from "what is a cell/face/state" to wiring Kube into a Polars,
Pandas, PyTorch, scikit-learn, or HuggingFace project. From here:

- **[docs/reference.md](../reference.md)** — the deep reference. The exact
  `.kube` file format (header layout, packing math), the native storage
  engine, the web viewer's architecture, native build internals. Reach for
  this once you know what you're looking for and want the precise mechanism.

- **[docs/library-integration.md](../library-integration.md)** — a condensed
  cookbook covering the same integration surface as
  [Chapter 8](08-integrating.md), plus a quick-reference API summary table.
  Good as a cheat sheet once the concepts in this guide are familiar.

- **[examples/](../../examples/)** — full, runnable scripts: dataset QA
  report export, text-corpus ingestion, tiled region processing, the binary
  round-trip pattern, and a complete tiny LM training loop.

- **[README.md](../../README.md)** — the project pitch and install
  instructions, plus a benchmark-results table if you want performance
  numbers before committing to anything.

- **[CONTRIBUTING.md](../../CONTRIBUTING.md)** — if you want to change Kube
  itself rather than just use it: build setup, testing expectations, and
  where the native storage engine vs. the web viewer code lives.

## If you're still deciding where to start

- Prototyping or experimenting → `Kube(...)`, no file needed
  ([Chapter 2](02-storage-model.md)).
- Something to persist or that might outgrow RAM → `create(...)` /
  `open(...)`.
- A pipeline step that already speaks Polars/Pandas/NumPy/Arrow → the
  conversion methods in [Chapter 4](04-tables-and-interop.md).
- Token-sequence data at training scale → `.kubed` and
  [Chapter 7](07-datasets.md).
- Wiring into an ML framework you already use →
  [Chapter 8](08-integrating.md).

That's the whole guide. Welcome to Kube.
