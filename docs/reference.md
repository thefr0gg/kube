# Kube Reference

This is the deep-reference document: exact file format, native storage
engine internals, the web viewer's architecture, and native build/loading
details. It assumes you've already been through
[the guide](guide/README.md) — this doc explains *how* things work, not
*how to use* them.

For task-oriented usage, see:

- **[the guide](guide/README.md)** — step-by-step, concept to integration.
- **[library-integration.md](library-integration.md)** — a condensed
  ML/data integration cookbook with an API summary table.

## Repository Map

- `kube/model.py`: core storage model, file IO, packing/unpacking, Polars export, region and batch iteration
- `kube/dataset.py`: `ShardedKubeDataset`, the `.kubed` sharded dataset layer
- `kube/web/serve.py`: blocking launcher — picks a port, starts uvicorn, opens the browser
- `kube/web/server.py`: FastAPI app (REST endpoints + the edit WebSocket)
- `kube/web/mesh.py`: shell/slice mesh construction over `faces_to_numpy()`
- `kube/web/static/`: WebGL (three.js) frontend — `index.html` + `viewer.js`, no build step
- `kube/_dll.py`: native storage library discovery and `ctypes` signature setup
- `kube/main.py`: CLI entry points
- `kube/integrations/`: numpy, sklearn, HuggingFace, and torch adapters
- `src/kube_storage.c`: native packed-cell storage engine for random IO, face batches, contiguous packed writes, and ingest hot paths
- `scripts/build.py`: configure/build helper for the native library
- `CMakeLists.txt`: native build definition
- `tests/`: `test_kube.py` (core storage), `test_dataset.py` (sharded datasets), `test_interop.py` (universal data backend), `test_web.py` (mesh + FastAPI app)

## Native Storage Engine

Kube has one native layer: the storage engine, `src/kube_storage.c`.

It exists so packed-cell access, face batches, clears, and ingest writes do
not have to route through Python loops once the buffer is addressable. The
current exported responsibilities are:

- single packed-value reads and writes
- single decoded face reads and writes
- batched face reads and writes
- contiguous packed-value batch writes
- byte-range zeroing for clears and trailing cleanup

This is the source that backs:

- native-only `Kube.get_packed(...)`
- native-only `Kube.get_face(...)`
- native-only `Kube.set_face(...)`
- `get_faces_batch(...)` and `set_faces_batch(...)`
- batched `encode(...)`, `_stream_import(...)`, and token-shard ingest
- the web viewer's mesh building and edit path (through the same Python API, not a separate bridge)

## Core Data Model

### Cell

A `Cell` is the logical unit of data:

- it always contains exactly 6 faces
- it validates that every face value is within the configured state range
- it can convert to and from a packed integer representation

The face order is fixed by index:

- `0`: top
- `1`: bottom
- `2`: left
- `3`: right
- `4`: front
- `5`: back

### Packed representation

`states` determines `bits_per_face`:

- `states=2` -> `1` bit per face
- `states=4` -> `2` bits per face
- `states=256` -> `8` bits per face

Each cell uses:

```text
bits_per_cell = 6 * log2(states)
```

Faces are packed into one integer, with face `0` occupying the lowest bits, then face `1`, and so on.

The storage item size is chosen automatically:

- up to 8 total bits -> `UInt8`
- up to 16 total bits -> `UInt16`
- up to 32 total bits -> `UInt32`
- otherwise -> `UInt64`

This logic lives in `_itemsize_for_states(...)` in `kube/model.py`.

### Grid indexing

Cells are stored in a flat contiguous buffer using:

```text
flat_index = x * (H * D) + y * D + z
```

That means:

- `x` is the outermost stride
- `z` is the fastest-changing coordinate

The same layout is used by Python and the native storage engine, which is why
the storage accelerator can read the mapped buffer directly without copying
it. The web viewer reads through the same `Kube` API (`faces_to_numpy()`), not
a raw buffer handoff.

## The `.kube` File Format

### Header

Every `.kube` file starts with a fixed 4096-byte header.

`kube/model.py` defines:

- `MAGIC = b"KUBE"`
- `FORMAT_VERSION = 1`
- `HEADER_SIZE = 4096`

The header includes:

- magic and format version
- shape: `W`, `H`, `D`
- `states`
- `itemsize`
- payload offset
- cell count
- source byte count
- metadata length
- JSON metadata blob

The packed header prefix uses `HEADER_STRUCT = struct.Struct("<4sHHQQQIIQQQI")`.

Important consequence:

- metadata is stored in the fixed header area
- metadata must fit inside the 4 KiB header
- `update_metadata(...)` rewrites the header for mapped files

### Payload

After the header, the payload is a contiguous array of packed cell values.

Kube uses a `PackedBuffer` abstraction so the payload can be backed by:

- a Python `bytearray` for in-memory kubes
- an `mmap` for disk-backed kubes

### Read/write modes

`Kube.open(path, mode=...)` supports:

- `mode="r"`: safe read-only access
- `mode="r+"`: read/write mapped access

When opened with `mode="r"`:

- Python mutation APIs raise `PermissionError`
- the file is mapped with copy-on-write semantics for the buffer view
- `flush()` becomes a no-op

When created or opened for writing:

- face edits modify the mapped payload directly
- `flush()` persists mapped updates and refreshes the header when needed

## Import/Export Model

Kube can treat ordinary byte streams as a packed sequence of face states.

The core helpers are:

- `_pack_stream_to_values(...)`
- `_unpack_values_to_bytes(...)`
- `_packed_value_to_stream_chunk(...)`

This is what powers:

- `Kube.from_bytes(...)`
- `Kube.from_file(...)`
- `Kube.import_file(...)`
- `Kube.export_file(...)`
- `Kube.to_bytes()`

There are two supported import styles:

- eager import for in-memory payloads
- streaming import for large files

`Kube.import_file(...)` streams from disk, packs incrementally, and clears unused trailing cells if the target cube is larger than the source payload.

## Web Viewer Architecture

The viewer is not a separate file parser. It is a live view over the current
`Kube` object, served over HTTP/WebSocket instead of a native window.

The call chain is:

```text
Kube.view(...)
  -> kube.web.serve.serve(...)
  -> uvicorn.Server(create_app(kube, ...)).run()
  -> FastAPI REST endpoints + /ws/edit WebSocket (kube/web/server.py)
  -> browser opens http://127.0.0.1:<port>/, loads kube/web/static/viewer.js
  -> three.js renders the mesh fetched from /api/mesh
```

`Kube.view(...)` blocks the calling thread until the server stops (Ctrl+C in
the terminal).

### What the server exposes

`create_app(kube, ...)` builds a `ViewerState` (current view mode, slice
axis/index, LOD) closed over one live `Kube` instance, and registers:

- `GET /api/state`: shape, states, labels, current view config
- `POST /api/view`: update view mode / slice axis / slice index / LOD
- `GET /api/mesh`: build and return the current mesh as flat JSON arrays
  (`positions`, `colors`, `indices`)
- `GET /api/cell?x=&y=&z=`: decoded faces + packed value for one cell
- `WS /ws/edit`: client sends `{x, y, z, face, value}`; the handler calls
  `kube.set_face(...)` then `kube.flush()` and echoes back `{ok, ...}` or an
  error
- static files under `/` (`index.html`, `viewer.js`) via `StaticFiles`

This is why edits made in the browser change the real `Kube` object
immediately — there is no separate renderer-side copy of the data.

### Mesh construction

`kube/web/mesh.py` builds visible-face geometry vectorized over numpy:

- `build_shell_mesh(faces, lod, visible_faces)`: only boundary faces, with a
  per-face-index visibility toggle (the client does not do camera-facing
  culling — all shell faces are visible by default)
- `build_slice_mesh(faces, slice_axis, slice_index, lod)`: all 6 faces of
  every cell on one axis-aligned plane

Both take the `(W, H, D, 6)` array from `kube.faces_to_numpy()` as input and
return `{positions, colors, indices, states}`, where `positions`/`colors` are
flat per-vertex arrays and `indices` are triangle indices (2 triangles per
face quad).

LOD sampling: boundary coordinates are always kept, interior coordinates are
kept only when `coord % lod == 0`.

Coloring rules (`_color_for_states` in `mesh.py`):

- for `states <= 8`, a fixed palette is used
- for larger state counts, colors are generated from an HSV sweep

### Frontend

`kube/web/static/viewer.js` is a single ES module, no build step, loading
three.js and `OrbitControls` from a CDN via plain `<script type="module">`
imports. It:

- fetches `/api/state` and `/api/mesh` on load and builds a `THREE.Mesh` from
  the returned flat buffers
- orbits the camera with `OrbitControls`
- on double-click, raycasts into the mesh, resolves the hit back to a
  `(x, y, z, face)` triple using the index buffer and the hit face normal,
  fetches `/api/cell` for the current value, and renders the inspector panel
- on "Apply" in the inspector, sends `{x, y, z, face, value}` over the
  `/ws/edit` WebSocket and refetches the mesh on a successful reply
- re-POSTs `/api/view` and refetches the mesh whenever the view-mode/slice/LOD
  controls change

### Editing model

Edits are WebSocket round-trips:

1. user double-clicks a face to select it
2. user edits the value field and clicks "Apply"
3. the client sends `{x, y, z, face, value}` over `/ws/edit`
4. the server calls `kube.set_face(...)` then `kube.flush()`
5. the server replies `{ok: true, ...}` (or `{ok: false, error: ...}`)
6. on success, the client refetches `/api/mesh` to show the change

If the viewer was started with `read_only=True`, the WebSocket handler
rejects every edit with `{ok: false, error: "viewer is read-only"}` before
touching the `Kube` object.

### Autosave semantics

Every successful edit flushes immediately (step 4 above) — there is no
unsaved/batched edit state on the server side.

## Build and Native Library Loading

### Build flow

The native storage library is built through `scripts/build.py`, which wraps CMake.

```bash
python scripts/build.py
```

On Windows with MinGW:

```powershell
python scripts/build.py --generator "MinGW Makefiles"
```

Useful flags: `--clean`, `--cmake-only`, `--reconfigure`, `--build-dir`.

### CMake behavior

`CMakeLists.txt`:

- builds `src/kube_storage.c` as the storage shared library
- exposes a `kube_native` target (an alias for `kube_storage`, kept for
  compatibility with existing build invocations)

The built shared library is written directly into the Python package directory:

- Windows: `kube/kube_storage.dll`
- Linux: `kube/libkube_storage.so`
- macOS: `kube/libkube_storage.dylib`

### Runtime loading

`kube/_dll.py` chooses the correct library name for the platform and loads
it with `ctypes.CDLL(...)`.

If the library is missing, Kube raises a `FileNotFoundError` with build
instructions instead of failing silently.

The web viewer itself needs no native library at all — `kube/web/server.py`
and `kube/web/mesh.py` are pure Python (FastAPI + numpy) talking to the same
`Kube` object through its normal Python API.

## Practical Constraints and Gotchas

- `states` must be a power of two. Many packing assumptions depend on this.
- Metadata must fit in the fixed 4096-byte header.
- The viewer requires only the `kube_storage` native library (not a separate renderer library) to be built first.
- The viewer is a shell/slice inspector, not a full volumetric renderer.
- Viewer edits flush immediately in writable mode.
- `Kube.set_face(...)` mutates the buffer but does not flush by itself.
- `Kube.view(...)` blocks the calling thread until the server is stopped (Ctrl+C).
- The viewer is localhost-only by design — there is no auth and it should never be exposed beyond the local machine.
- Mesh requests rebuild the full visible geometry server-side on each `/api/mesh` call, so very large cubes still depend heavily on LOD for interactivity.
