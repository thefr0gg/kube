# Contributing To Kube

Kube is a mixed Python + native project. Contributions usually touch one or more
of these layers:

- the `.kube` storage/runtime model in Python
- the `.kubed` dataset path
- the CLI and logging surface
- the native editor and renderer
- docs and examples

This guide exists to make that workflow predictable so contributors can ship
changes without guessing how the repo is supposed to be built or tested.

## What To Know First

Kube is not only a viewer. The project identity spans:

- face-addressable storage
- memory-mapped data access
- DataFrame and dataset workflows
- native interactive debugging

That means changes should preserve consistency across those surfaces when
possible. If you update the storage model, also think about:

- CLI behavior
- examples
- docs
- native/editor assumptions

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

The editor and native acceleration libraries are built with CMake.

You will typically need:

- `cmake >= 3.20`
- a working C toolchain
- `raylib` available either through CMake, through fetch, or through
  `KUBE_RAYLIB_ROOT`

Windows example:

```powershell
$env:KUBE_RAYLIB_ROOT="C:\Users\you\Documents\Repos\raylib-5.5_win64_mingw-w64"
poetry install
```

That project install path now attempts to build the native libraries
automatically through the Poetry build hook.

## Building Native Libraries

Kube currently builds two native shared libraries:

- `kube_renderer`: the editor/viewer bridge
- `kube_storage`: storage hot paths for random IO

The default target is `kube_native`, which builds both.

Examples:

```bash
python build.py
python build.py --target kube_storage
python build.py --cmake-only
```

Windows MinGW example:

```powershell
python build.py --generator "MinGW Makefiles"
```

Useful flags:

- `--clean`
- `--build-dir <path>`
- `--use-system-raylib` / `--no-use-system-raylib`
- `--raylib-root <path>`

## Testing

Run the Python test suite from Poetry so dependencies are available:

```bash
poetry run python -m unittest tests.test_kube -q
```

Useful validation commands:

```bash
poetry run python -m py_compile kube\model.py kube\main.py kube\renderer.py
poetry run python examples\benchmark_kube_vs_binary.py --shape 8x8x8 --states 4 --ops 1000 --repeats 1
```

For native changes, also verify the shared libraries still build:

```bash
python build.py
```

## Contribution Expectations

### Keep public behavior stable

Kube has multiple user-facing entry points:

- Python API
- CLI
- `.kube` file format
- native viewer/editor

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

- [README.md](/c:/Users/0rams/Documents/Coding/Projects/kube/README.md)
- [WIKI.md](/c:/Users/0rams/Documents/Coding/Projects/kube/WIKI.md)
- examples in `examples/`

If a feature adds a new workflow, include at least one concrete example.

## Coding Notes

### Python

- keep the public API explicit and readable
- preserve `.kube` format compatibility unless the change is intentionally a
  format revision
- prefer small helpers when storage logic becomes hard to follow

### Native code

- keep renderer concerns separate from storage acceleration concerns
- avoid introducing UI dependencies into non-UI native libraries
- preserve Windows build compatibility, especially MinGW-based setups

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
