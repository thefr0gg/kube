# Contributing To Kube

Kube is a mixed Python + native project. Contributions usually touch one or more
of these layers:

- the `.kube` storage/runtime model in Python
- the `.kubed` dataset path
- the CLI and logging surface
- the web viewer/editor (FastAPI backend + WebGL frontend)
- docs and examples

This guide exists to make that workflow predictable so contributors can ship
changes without guessing how the repo is supposed to be built or tested.

## What To Know First

Kube is not only a viewer. The project identity spans:

- face-addressable storage
- memory-mapped data access
- DataFrame and dataset workflows
- interactive debugging via the web viewer

That means changes should preserve consistency across those surfaces when
possible. If you update the storage model, also think about:

- CLI behavior
- examples
- docs
- web viewer assumptions

## Development Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd kube
poetry install
```

Optional extras:

```bash
poetry install -E hf
poetry install -E torch
poetry install -E examples
```

### 2. Native prerequisites

The storage acceleration library is built with CMake.

You will typically need:

- `cmake >= 3.20`
- a working C toolchain

That project install path attempts to build the native library automatically
through the Poetry build hook.

## Building The Native Library

Kube builds one native shared library:

- `kube_storage`: storage hot paths for random IO

The default target is `kube_native`, which builds it.

Examples:

```bash
python scripts/build.py
python scripts/build.py --target kube_storage
python scripts/build.py --cmake-only
```

Windows MinGW example:

```powershell
python scripts/build.py --generator "MinGW Makefiles"
```

Useful flags:

- `--clean`
- `--build-dir <path>`

## Testing

Run the Python test suite from Poetry so dependencies are available:

```bash
poetry run python -m unittest tests.test_kube -q
```

Useful validation commands:

```bash
poetry run python -m py_compile kube\model.py kube\main.py kube\web\server.py
poetry run kube benchmark --size 16 --iterations 1
```

For native changes, also verify the shared library still builds:

```bash
python scripts/build.py
```

## Contribution Expectations

### Keep public behavior stable

Kube has multiple user-facing entry points:

- Python API
- CLI
- `.kube` file format
- web viewer/editor

Unless the change is intentionally breaking, keep those interfaces stable.

### Prefer compatibility over cleverness

Kube is used as both a storage layer and a debugging tool. Optimizations are
welcome, but they should be:

- measurable
- optional when appropriate
- safe to fall back from

That is especially important for native acceleration work.

### Update docs with behavior changes

When a contribution changes behavior, update the relevant docs in the same PR:

- [README.md](https://github.com/thefr0gg/kube/blob/main/README.md)
- [docs/reference.md](https://github.com/thefr0gg/kube/blob/main/docs/reference.md)
- examples in `examples/`

If a feature adds a new workflow, include at least one concrete example.

## Coding Notes

### Python

- keep the public API explicit and readable
- preserve `.kube` format compatibility unless the change is intentionally a
  format revision
- prefer small helpers when storage logic becomes hard to follow

### Native code

- the native library is storage-only (`kube_storage`); keep it free of
  UI/rendering dependencies
- preserve Windows build compatibility, especially MinGW-based setups

### Web viewer

- keep mesh-building logic in `kube/web/mesh.py` vectorized (numpy), not
  per-cell Python loops
- the server in `kube/web/server.py` is localhost-only by design — do not add
  auth scaffolding or externally-facing defaults
- the frontend in `kube/web/static/` has no build step; keep it that way

### Docs

- explain why a method or feature exists, not only what it does
- keep examples runnable
- prefer concrete commands over abstract instructions

## Pull Requests

A good PR usually includes:

- a short summary of the problem
- the implementation approach
- verification commands you ran
- docs updates when user-facing behavior changed

If something could not be tested in your environment, say that clearly in the
PR description.

## Areas That Benefit From Contributions

- storage and dataset performance
- editor UX and inspection workflows
- Windows and Linux build reliability
- examples for real use cases
- clearer docs around the data model

## Questions and Scope

If you are making a larger architectural change, keep the first PR narrow when
possible. For example:

- add the storage primitive first
- then wire it into the CLI or examples
- then document the broader workflow

That makes the mixed Python/native review surface much easier to reason about.
