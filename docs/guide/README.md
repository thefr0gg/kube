# The Kube Guide

This is a step-by-step guide to Kube: what it is, how it works, and how to
put it to use in a project you already have. Read it top to bottom the first
time; come back to individual chapters later as a reference.

This guide is not the only doc in the repo. Here's how it relates to the
others:

- **This guide** — learn the concept from zero, then apply it. Start here if
  you're new to Kube.
- **[../../README.md](../../README.md)** — the project pitch: what Kube is,
  install instructions, a tour of the full CLI and API. Good for a quick
  overview or as an install reference.
- **[../reference.md](../reference.md)** — the deep reference: file format
  internals, the native storage engine, the web viewer's architecture, native
  build details. Good once you know what you're looking for and want the
  exact mechanism.
- **[../library-integration.md](../library-integration.md)** — a condensed
  cookbook of ML/data integration patterns with an API summary table. Good as
  a quick-lookup cheat sheet once you've been through this guide.

## Reading order

1. **[01 — The Concept](01-concept.md)** — what a cell, a face, and a state
   are, and why Kube is built around them.
2. **[02 — The Storage Model](02-storage-model.md)** — in-memory kubes vs.
   memory-mapped `.kube` files, and how `states` determines the packed layout.
3. **[03 — The Python API](03-python-api.md)** — creating, opening, reading,
   writing, and iterating over a kube.
4. **[04 — Tables and Interop](04-tables-and-interop.md)** — moving data
   between Kube and Polars, Pandas, NumPy, Arrow, and Parquet.
5. **[05 — The CLI](05-cli.md)** — the command-line tools for everyday work.
6. **[06 — The Viewer](06-viewer.md)** — the localhost 3D viewer/editor.
7. **[07 — Sharded Datasets](07-datasets.md)** — the `.kubed` format for
   large, sharded, token-sequence corpora.
8. **[08 — Integrating Kube](08-integrating.md)** — concrete recipes for
   dropping Kube into a Polars, PyTorch, scikit-learn, or HuggingFace project
   you already have.
9. **[09 — Next Steps](09-next-steps.md)** — where to go from here.

If you only have ten minutes, read chapters 1, 2, and 8.
