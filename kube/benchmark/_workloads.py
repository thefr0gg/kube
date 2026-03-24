"""Workload definitions for the Kube benchmark suite.

Each workload is a zero-argument callable that returns an integer checksum.
The runner times repeated calls and records throughput.

Workloads are organized into groups:

- random_read / random_write — face-level random access
- neighborhood — gather all 6-direction neighbours for a set of cells
- region_extract — materialize a bounded subgrid
- full_scan — iterate every cell and reduce
- histogram — compute per-face value counts
- polars_pipeline — filter + group_by via feature_frame
- batch_iter — streaming minibatch iteration
- lm_sequence_iter — causal LM token-sequence streaming
- mutation — interleaved read-write pairs
- export_arrow — to_arrow() export timings (NEW: interop layer)
- export_parquet — to_parquet() export timings (NEW: interop layer)
- export_numpy — faces_to_numpy() / to_numpy() (NEW: interop layer)
- export_pandas — to_pandas() (NEW: interop layer)
- sklearn_xy — to_sklearn_xy() (NEW: interop layer)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import polars as pl

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    import pyarrow as _pa
    _HAS_ARROW = True
except ImportError:
    _pa = None  # type: ignore[assignment]
    _HAS_ARROW = False

# A suite is a list of workload entry dicts
_Suite = list[dict[str, Any]]

_BATCH_SIZES = ((1_000, "1k"), (64_000, "64k"), (512_000, "512k"))
_LM_SEQ_LEN = 512
_LM_CAP = 1_000_000


def build_suite(fixtures: dict[str, Any], config: Any) -> _Suite:
    """Return all available workloads given the prepared fixtures."""
    kube = fixtures["kube"]  if "kube" in fixtures else fixtures["kube_instance"]
    ops = fixtures["ops"]
    compiled_batch = fixtures["compiled_batch"]
    compiled_neighbor_batch = fixtures["compiled_neighbor_batch"]
    n_neighbor_cells = fixtures["n_neighbor_cells"]
    binary_grid = fixtures["binary_grid"]
    region_bounds = fixtures["region_bounds"]
    W = fixtures["W"]
    H = fixtures["H"]
    D = fixtures["D"]
    S = fixtures["S"]
    n_ops = len(ops)
    parquet_path = fixtures.get("parquet_path", "/tmp/_kube_bench.parquet")
    arrow_path = fixtures.get("arrow_path", "/tmp/_kube_bench.arrow")

    numpy_arr = fixtures.get("numpy_arr")
    xs = fixtures.get("xs")
    ys = fixtures.get("ys")
    zs = fixtures.get("zs")
    face_idxs = fixtures.get("face_idxs")
    values = fixtures.get("values")
    nb_xs = fixtures.get("nb_xs")
    nb_ys = fixtures.get("nb_ys")
    nb_zs = fixtures.get("nb_zs")

    suite: _Suite = []

    # ------------------------------------------------------------------ #
    # random_read
    # ------------------------------------------------------------------ #
    def _kube_random_read() -> int:
        return sum(kube.get_faces_batch(compiled_batch))

    suite.append({"workload": "random_read", "backend": "kube", "fn": _kube_random_read, "ops": n_ops})

    if _HAS_NUMPY and numpy_arr is not None:
        def _numpy_random_read() -> int:
            return int(numpy_arr[xs, ys, zs, face_idxs].sum())
        suite.append({"workload": "random_read", "backend": "numpy", "fn": _numpy_random_read, "ops": n_ops})

    def _binary_random_read() -> int:
        total = 0
        for x, y, z, _f, _v in ops:
            total += binary_grid.get_cell(x, y, z)
        return total

    suite.append({"workload": "random_read", "backend": "binary 1 bit/cell", "fn": _binary_random_read, "ops": n_ops})

    # ------------------------------------------------------------------ #
    # random_write
    # ------------------------------------------------------------------ #
    def _kube_random_write() -> int:
        kube.set_faces_batch(compiled_batch)
        return 0

    suite.append({"workload": "random_write", "backend": "kube", "fn": _kube_random_write, "ops": n_ops})

    if _HAS_NUMPY and numpy_arr is not None:
        def _numpy_random_write() -> int:
            numpy_arr[xs, ys, zs, face_idxs] = values
            return 0
        suite.append({"workload": "random_write", "backend": "numpy", "fn": _numpy_random_write, "ops": n_ops})

    def _binary_random_write() -> int:
        for x, y, z, _f, v in ops:
            binary_grid.set_cell(x, y, z, v & 1)
        return 0

    suite.append({"workload": "random_write", "backend": "binary 1 bit/cell", "fn": _binary_random_write, "ops": n_ops})

    # ------------------------------------------------------------------ #
    # neighborhood
    # ------------------------------------------------------------------ #
    def _kube_neighborhood() -> int:
        return sum(kube.get_faces_batch(compiled_neighbor_batch))

    suite.append({"workload": "neighborhood", "backend": "kube", "fn": _kube_neighborhood, "ops": n_neighbor_cells})

    if _HAS_NUMPY and numpy_arr is not None and nb_xs is not None:
        def _numpy_neighborhood() -> int:
            return int(numpy_arr[nb_xs, nb_ys, nb_zs, 0].sum())
        suite.append({"workload": "neighborhood", "backend": "numpy", "fn": _numpy_neighborhood, "ops": n_neighbor_cells})

    # ------------------------------------------------------------------ #
    # region_extract
    # ------------------------------------------------------------------ #
    region_cells = (
        (region_bounds[0][1] - region_bounds[0][0])
        * (region_bounds[1][1] - region_bounds[1][0])
        * (region_bounds[2][1] - region_bounds[2][0])
    )

    def _kube_region_extract() -> int:
        frame = kube.read_region(region_bounds)
        return frame.height

    suite.append({
        "workload": "region_extract",
        "backend": "kube lower octant",
        "fn": _kube_region_extract,
        "ops": region_cells,
    })

    if _HAS_NUMPY and numpy_arr is not None:
        x0, x1 = region_bounds[0]
        y0, y1 = region_bounds[1]
        z0, z1 = region_bounds[2]

        def _numpy_region_extract() -> int:
            sub = numpy_arr[x0:x1, y0:y1, z0:z1, :]
            return int(sub.sum())

        suite.append({
            "workload": "region_extract",
            "backend": "numpy",
            "fn": _numpy_region_extract,
            "ops": region_cells,
        })

    # ------------------------------------------------------------------ #
    # full_scan
    # ------------------------------------------------------------------ #
    total_cells = W * H * D

    def _kube_full_scan() -> int:
        frame = kube.to_polars(include_coords=False, include_packed=False, include_faces=True)
        return frame.height

    suite.append({"workload": "full_scan", "backend": "kube", "fn": _kube_full_scan, "ops": total_cells})

    if _HAS_NUMPY and numpy_arr is not None:
        def _numpy_full_scan() -> int:
            return int(numpy_arr.reshape(-1, 6).sum())
        suite.append({"workload": "full_scan", "backend": "numpy reshape+sum", "fn": _numpy_full_scan, "ops": total_cells})

    # ------------------------------------------------------------------ #
    # histogram
    # ------------------------------------------------------------------ #
    def _kube_histogram() -> int:
        frame = kube.face_histogram()
        return frame.height

    suite.append({"workload": "histogram", "backend": "kube", "fn": _kube_histogram, "ops": total_cells})

    if _HAS_NUMPY and numpy_arr is not None:
        def _numpy_histogram() -> int:
            counts = np.bincount(numpy_arr.reshape(-1, 6)[:, 0].astype(np.intp), minlength=S)
            return int(counts.sum())
        suite.append({"workload": "histogram", "backend": "numpy np.bincount", "fn": _numpy_histogram, "ops": total_cells})

    def _polars_histogram() -> int:
        frame = kube.to_polars(include_coords=False, include_packed=False, include_faces=True)
        result = frame.select(pl.col("face_0").value_counts())
        return result.height

    suite.append({"workload": "histogram", "backend": "polars", "fn": _polars_histogram, "ops": total_cells})

    # ------------------------------------------------------------------ #
    # polars_pipeline
    # ------------------------------------------------------------------ #
    def _kube_polars_pipeline() -> int:
        frame = kube.feature_frame(include_coords=False, include_packed=False)
        result = (
            frame
            .filter(pl.col("face_0") > 0)
            .group_by("face_0")
            .agg(pl.col("face_1").mean())
        )
        return result.height

    suite.append({"workload": "polars_pipeline", "backend": "kube filter+group_by", "fn": _kube_polars_pipeline, "ops": total_cells})

    if _HAS_NUMPY and numpy_arr is not None:
        def _polars_pipeline_baseline() -> int:
            flat = numpy_arr.reshape(-1, 6)
            frame = pl.DataFrame({f"face_{i}": flat[:, i] for i in range(6)})
            result = (
                frame
                .filter(pl.col("face_0") > 0)
                .group_by("face_0")
                .agg(pl.col("face_1").mean())
            )
            return result.height
        suite.append({"workload": "polars_pipeline", "backend": "numpy→polars→filter+group_by", "fn": _polars_pipeline_baseline, "ops": total_cells})

    # ------------------------------------------------------------------ #
    # batch_iter
    # ------------------------------------------------------------------ #
    for bsz, bsz_label in _BATCH_SIZES:
        if bsz > total_cells * 4:
            continue

        def _make_batch_fn(batch_size: int) -> Callable[[], int]:
            def _fn() -> int:
                total = 0
                for batch in kube.iter_batches(batch_size, include_packed=False):
                    total += batch.height
                return total
            return _fn

        suite.append({
            "workload": "batch_iter",
            "backend": "kube",
            "fn": _make_batch_fn(bsz),
            "ops": total_cells,
            "size_label": bsz_label,
        })

    # ------------------------------------------------------------------ #
    # lm_sequence_iter
    # ------------------------------------------------------------------ #
    dataset_path = getattr(config, "dataset_path", None)
    if dataset_path is not None and Path(dataset_path).exists():
        from kube.dataset import open_dataset

        def _kube_lm_iter() -> int:
            total = 0
            ds = open_dataset(dataset_path, mode="r")
            try:
                for sample in ds.iter_sequences(_LM_SEQ_LEN):
                    total += len(sample["input_ids"])
                    if total >= _LM_CAP:
                        break
            finally:
                ds.close()
            return total

        suite.append({
            "workload": "lm_sequence_iter",
            "backend": "kube",
            "fn": _kube_lm_iter,
            "ops": _LM_CAP,
            "size_label": f"seq={_LM_SEQ_LEN}",
        })

    # ------------------------------------------------------------------ #
    # mutation (interleaved read+write)
    # ------------------------------------------------------------------ #
    half = len(ops) // 2
    read_half = kube.compile_face_batch(ops[:half])
    write_half = kube.compile_face_batch(ops[half:])

    def _kube_mutation() -> int:
        kube.set_faces_batch(write_half)
        return sum(kube.get_faces_batch(read_half))

    suite.append({"workload": "mutation", "backend": "kube interleaved r+w", "fn": _kube_mutation, "ops": n_ops})

    if _HAS_NUMPY and numpy_arr is not None:
        rh = len(xs) // 2

        def _numpy_mutation() -> int:
            numpy_arr[xs[rh:], ys[rh:], zs[rh:], face_idxs[rh:]] = values[rh:]
            return int(numpy_arr[xs[:rh], ys[:rh], zs[:rh], face_idxs[:rh]].sum())

        suite.append({"workload": "mutation", "backend": "numpy", "fn": _numpy_mutation, "ops": n_ops})

    def _binary_mutation() -> int:
        for x, y, z, _f, v in ops[half:]:
            binary_grid.set_cell(x, y, z, v & 1)
        total = 0
        for x, y, z, _f, _v in ops[:half]:
            total += binary_grid.get_cell(x, y, z)
        return total

    suite.append({"workload": "mutation", "backend": "binary 1 bit/cell", "fn": _binary_mutation, "ops": n_ops})

    # ------------------------------------------------------------------ #
    # export_arrow  (interop layer)
    # ------------------------------------------------------------------ #
    if _HAS_ARROW:
        def _kube_export_arrow_wide() -> int:
            table = kube.to_arrow(schema_mode="wide_faces")
            return table.num_rows

        suite.append({
            "workload": "export_arrow",
            "backend": "kube wide_faces",
            "fn": _kube_export_arrow_wide,
            "ops": total_cells,
        })

        def _kube_export_arrow_long() -> int:
            table = kube.to_arrow(schema_mode="long_faces")
            return table.num_rows

        suite.append({
            "workload": "export_arrow",
            "backend": "kube long_faces",
            "fn": _kube_export_arrow_long,
            "ops": total_cells * 6,
        })

        def _kube_export_arrow_packed() -> int:
            table = kube.to_arrow(schema_mode="cell_packed")
            return table.num_rows

        suite.append({
            "workload": "export_arrow",
            "backend": "kube cell_packed",
            "fn": _kube_export_arrow_packed,
            "ops": total_cells,
        })

    # ------------------------------------------------------------------ #
    # export_parquet  (interop layer)
    # ------------------------------------------------------------------ #
    if _HAS_ARROW:
        def _kube_export_parquet() -> int:
            kube.to_parquet(parquet_path)
            return total_cells

        suite.append({
            "workload": "export_parquet",
            "backend": "kube wide_faces",
            "fn": _kube_export_parquet,
            "ops": total_cells,
        })

    # ------------------------------------------------------------------ #
    # export_numpy  (interop layer)
    # ------------------------------------------------------------------ #
    if _HAS_NUMPY:
        def _kube_export_numpy_spatial() -> int:
            arr = kube.faces_to_numpy()
            return arr.size

        suite.append({
            "workload": "export_numpy",
            "backend": "kube faces_to_numpy",
            "fn": _kube_export_numpy_spatial,
            "ops": total_cells,
        })

        def _kube_export_numpy_flat() -> int:
            arr = kube.to_numpy()
            return arr.size

        suite.append({
            "workload": "export_numpy",
            "backend": "kube to_numpy flat",
            "fn": _kube_export_numpy_flat,
            "ops": total_cells,
        })

    # ------------------------------------------------------------------ #
    # export_pandas  (interop layer)
    # ------------------------------------------------------------------ #
    try:
        import pandas as _pd
        _HAS_PANDAS = True
    except ImportError:
        _HAS_PANDAS = False

    if _HAS_PANDAS:
        def _kube_export_pandas() -> int:
            df = kube.to_pandas()
            return len(df)

        suite.append({
            "workload": "export_pandas",
            "backend": "kube to_pandas",
            "fn": _kube_export_pandas,
            "ops": total_cells,
        })

    # ------------------------------------------------------------------ #
    # sklearn_xy  (interop layer)
    # ------------------------------------------------------------------ #
    if _HAS_NUMPY:
        def _kube_sklearn_xy() -> int:
            X, y = kube.to_sklearn_xy(target_face=0)
            return X.size + y.size

        suite.append({
            "workload": "sklearn_xy",
            "backend": "kube to_sklearn_xy",
            "fn": _kube_sklearn_xy,
            "ops": total_cells,
        })

    return suite
