"""Scenario definitions for the Kube benchmark suite.

Unlike a micro-benchmark of raw random-access ops/sec, each scenario here
times a realistic task from the guide (docs/guide/): streaming a large file
in, converting a stored kube to a table for a pipeline, importing/iterating
an LLM token dataset, and the web viewer's edit + mesh-build hot paths.

Each workload is a zero-argument callable returning an integer checksum (row
count, byte count, etc.) so the runner can time it and prove it actually did
work. Workloads are grouped by scenario:

- stream_import   — import_file() streaming vs. from_file() eager-load
- table_conversion — Kube.open() -> to_polars/to_numpy/to_pandas/to_arrow
- lm_dataset      — import_token_ids() + iter_sequence_batches() (tokens/s)
- viewer_ops      — set_face()+flush() edit round trip, mesh build timing
"""

from __future__ import annotations

import shutil
from typing import Any, Callable

try:
    import pyarrow as _pa  # noqa: F401
    _HAS_ARROW = True
except ImportError:
    _HAS_ARROW = False

try:
    import pandas as _pd  # noqa: F401
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

_Suite = list[dict[str, Any]]


def build_suite(fixtures: dict[str, Any], config: Any) -> _Suite:
    """Return all available scenario workloads given the prepared fixtures."""
    suite: _Suite = []
    suite.extend(_stream_import_workloads(fixtures["stream_import"]))
    suite.extend(_table_conversion_workloads(fixtures["table_conversion"]))
    suite.extend(_lm_dataset_workloads(fixtures["lm_dataset"]))
    suite.extend(_viewer_ops_workloads(fixtures["viewer_ops"]))
    return suite


# -------------------------------------------------------------------- #
# stream_import — the "larger than RAM" storage use case
# -------------------------------------------------------------------- #
def _stream_import_workloads(fx: dict[str, Any]) -> _Suite:
    from kube.model import Kube

    suite: _Suite = []

    def _streaming() -> int:
        shutil.rmtree(fx["stream_dst"], ignore_errors=True)
        fx["stream_dst"].unlink(missing_ok=True)
        kube = Kube.import_file(fx["src_path"], fx["stream_dst"], states=fx["states"])
        byte_count = kube.memory_stats()["source_byte_count"]
        kube.close()
        return byte_count

    suite.append({
        "scenario": "stream_import",
        "backend": "import_file (streaming)",
        "fn": _streaming,
        "ops": fx["byte_count"],
    })

    def _eager() -> int:
        kube = Kube.from_file(fx["src_path"], states=fx["states"])
        byte_count = kube.memory_stats()["source_byte_count"]
        kube.close()
        return byte_count

    suite.append({
        "scenario": "stream_import",
        "backend": "from_file (load into RAM first)",
        "fn": _eager,
        "ops": fx["byte_count"],
    })

    return suite


# -------------------------------------------------------------------- #
# table_conversion — handing a stored kube to the rest of a pipeline
# -------------------------------------------------------------------- #
def _table_conversion_workloads(fx: dict[str, Any]) -> _Suite:
    from kube.model import Kube

    suite: _Suite = []
    shape = fx["shape"]
    total_cells = shape[0] * shape[1] * shape[2]

    def _to_polars() -> int:
        kube = Kube.open(fx["kube_path"], mode="r")
        try:
            frame = kube.to_polars()
            return frame.height
        finally:
            kube.close()

    suite.append({"scenario": "table_conversion", "backend": "to_polars", "fn": _to_polars, "ops": total_cells})

    try:
        import numpy  # noqa: F401
        _has_numpy = True
    except ImportError:
        _has_numpy = False

    if _has_numpy:
        def _to_numpy() -> int:
            kube = Kube.open(fx["kube_path"], mode="r")
            try:
                arr = kube.faces_to_numpy()
                return int(arr.size)
            finally:
                kube.close()

        suite.append({"scenario": "table_conversion", "backend": "faces_to_numpy", "fn": _to_numpy, "ops": total_cells})

    if _HAS_PANDAS and _HAS_ARROW:
        def _to_pandas() -> int:
            kube = Kube.open(fx["kube_path"], mode="r")
            try:
                df = kube.to_pandas()
                return len(df)
            finally:
                kube.close()

        suite.append({"scenario": "table_conversion", "backend": "to_pandas", "fn": _to_pandas, "ops": total_cells})

    if _HAS_ARROW:
        def _to_arrow() -> int:
            kube = Kube.open(fx["kube_path"], mode="r")
            try:
                table = kube.to_arrow()
                return table.num_rows
            finally:
                kube.close()

        suite.append({"scenario": "table_conversion", "backend": "to_arrow", "fn": _to_arrow, "ops": total_cells})

        def _to_parquet() -> int:
            kube = Kube.open(fx["kube_path"], mode="r")
            try:
                kube.to_parquet(fx["parquet_path"])
                return total_cells
            finally:
                kube.close()

        suite.append({"scenario": "table_conversion", "backend": "to_parquet", "fn": _to_parquet, "ops": total_cells})

    return suite


# -------------------------------------------------------------------- #
# lm_dataset — the LLM pretraining-corpus use case
# -------------------------------------------------------------------- #
def _lm_dataset_workloads(fx: dict[str, Any]) -> _Suite:
    from kube.dataset import create_sharded, import_token_ids, open_dataset

    suite: _Suite = []
    seq_len = fx["sequence_length"]

    def _import() -> int:
        shutil.rmtree(fx["dataset_root"], ignore_errors=True)
        dataset = create_sharded(
            fx["dataset_root"],
            shards=4,
            tokens_per_shard=max(1024, fx["total_tokens"]),
            sequence_length=seq_len,
        )
        dataset.close()
        dataset = import_token_ids(fx["tokens_path"], fx["dataset_root"])
        total = dataset.inspect()["total_token_count"]
        dataset.close()
        return total

    suite.append({
        "scenario": "lm_dataset",
        "backend": "import_token_ids",
        "fn": _import,
        "ops": fx["total_tokens"],
    })

    # Importing is a prerequisite for iteration below; run it once up front
    # so the iteration workload measures iteration, not import, cost.
    _import()

    def _iterate() -> int:
        dataset = open_dataset(fx["dataset_root"], mode="r")
        try:
            total = 0
            for batch in dataset.iter_sequence_batches(8, seq_len, drop_last=False):
                total += sum(len(ids) for ids in batch["input_ids"])
            return total
        finally:
            dataset.close()

    suite.append({
        "scenario": "lm_dataset",
        "backend": "iter_sequence_batches",
        "fn": _iterate,
        "ops": fx["total_tokens"],
        "unit": "tok/s",
    })

    return suite


# -------------------------------------------------------------------- #
# viewer_ops — the web viewer's edit round trip and mesh-build hot paths
# -------------------------------------------------------------------- #
def _viewer_ops_workloads(fx: dict[str, Any]) -> _Suite:
    from kube.web.mesh import build_shell_mesh, build_slice_mesh

    suite: _Suite = []
    kube = fx["kube"]
    edit_ops = fx["edit_ops"]
    shape = fx["shape"]

    def _edit_round_trip() -> int:
        # Mirrors the viewer's /ws/edit handler: one set_face + flush per op.
        for x, y, z, face, value in edit_ops:
            kube.set_face(x, y, z, face, value)
            kube.flush()
        return len(edit_ops)

    suite.append({
        "scenario": "viewer_ops",
        "backend": "set_face+flush (edit round trip)",
        "fn": _edit_round_trip,
        "ops": len(edit_ops),
        "unit": "edits/s",
    })

    def _mesh_shell() -> int:
        faces = kube.faces_to_numpy()
        mesh = build_shell_mesh(faces, lod=1)
        return int(mesh["positions"].shape[0])

    suite.append({
        "scenario": "viewer_ops",
        "backend": "build_shell_mesh",
        "fn": _mesh_shell,
        "ops": shape[0] * shape[1] * shape[2],
        "unit": "cells/s",
    })

    def _mesh_slice() -> int:
        faces = kube.faces_to_numpy()
        mesh = build_slice_mesh(faces, slice_axis=2, slice_index=shape[2] // 2, lod=1)
        return int(mesh["positions"].shape[0])

    suite.append({
        "scenario": "viewer_ops",
        "backend": "build_slice_mesh",
        "fn": _mesh_slice,
        "ops": shape[0] * shape[1],
        "unit": "cells/s",
    })

    return suite
