"""Tests for the Kube universal data backend interop layer.

Covers: schemas, metadata, protocol, model interop methods, dataset interop
methods, CLI export commands, and integration modules.

Optional-dependency tests (Arrow, NumPy, pandas) are skipped when the dep is
not installed so the test suite works in minimal environments.
"""

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import polars as pl

from kube import Cell, Kube, create_sharded, import_token_ids
from kube.dataset import KubeDataCollatorForCausalLM
from kube.main import main
from kube.metadata import (
    FEATURE_SCHEMA,
    PIPELINE_VERSION,
    SEQUENCE_LENGTH,
    SOURCE_DATASET,
    SPLIT,
    TOKENIZER_NAME,
    TOKENIZER_PATH,
)
from kube.protocol import KubeBackend
from kube.schemas import (
    SCHEMA_CELL_PACKED,
    SCHEMA_LONG_FACES,
    SCHEMA_SUPERVISED,
    SCHEMA_TOKEN_SEQUENCES,
    SCHEMA_WIDE_FACES,
    schema_columns,
)

try:
    import pyarrow as pa
    _ARROW = True
except ImportError:
    _ARROW = False

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False


# ---------------------------------------------------------------------------
# schemas.py
# ---------------------------------------------------------------------------

class SchemaConstantsTests(unittest.TestCase):
    def test_constant_values(self):
        self.assertEqual(SCHEMA_WIDE_FACES, "wide_faces")
        self.assertEqual(SCHEMA_LONG_FACES, "long_faces")
        self.assertEqual(SCHEMA_CELL_PACKED, "cell_packed")
        self.assertEqual(SCHEMA_SUPERVISED, "supervised")
        self.assertEqual(SCHEMA_TOKEN_SEQUENCES, "token_sequences")

    def test_schema_columns_wide_faces_uint8(self):
        cols = schema_columns("wide_faces", states=4)
        self.assertIn("x", cols)
        self.assertIn("face_0", cols)
        self.assertEqual(cols["face_0"], "UInt8")

    def test_schema_columns_wide_faces_uint16(self):
        cols = schema_columns("wide_faces", states=1024)
        self.assertEqual(cols["face_0"], "UInt16")

    def test_schema_columns_wide_faces_uint32(self):
        cols = schema_columns("wide_faces", states=2 ** 17)
        self.assertEqual(cols["face_0"], "UInt32")

    def test_schema_columns_long_faces(self):
        cols = schema_columns("long_faces", states=4)
        self.assertIn("face", cols)
        self.assertIn("value", cols)
        self.assertNotIn("face_0", cols)

    def test_schema_columns_cell_packed(self):
        cols = schema_columns("cell_packed", states=4)
        self.assertIn("packed", cols)
        self.assertNotIn("face_0", cols)

    def test_schema_columns_supervised(self):
        cols = schema_columns("supervised", states=4)
        self.assertIn("target", cols)
        self.assertIn("face_0", cols)

    def test_schema_columns_token_sequences(self):
        cols = schema_columns("token_sequences", states=256)
        self.assertIn("input_ids", cols)
        self.assertIn("target_ids", cols)

    def test_schema_columns_unknown_raises(self):
        with self.assertRaises(ValueError):
            schema_columns("not_a_mode", states=4)


# ---------------------------------------------------------------------------
# metadata.py
# ---------------------------------------------------------------------------

class MetadataConstantsTests(unittest.TestCase):
    def test_values(self):
        self.assertEqual(SPLIT, "split")
        self.assertEqual(TOKENIZER_NAME, "tokenizer_name")
        self.assertEqual(TOKENIZER_PATH, "tokenizer_path")
        self.assertEqual(SEQUENCE_LENGTH, "sequence_length")
        self.assertEqual(FEATURE_SCHEMA, "feature_schema")
        self.assertEqual(SOURCE_DATASET, "source_dataset")
        self.assertEqual(PIPELINE_VERSION, "pipeline_version")

    def test_usable_as_dict_keys(self):
        meta = {SPLIT: "train", SEQUENCE_LENGTH: 1024}
        self.assertEqual(meta["split"], "train")


# ---------------------------------------------------------------------------
# protocol.py
# ---------------------------------------------------------------------------

class ProtocolTests(unittest.TestCase):
    def test_kube_is_instance_of_kube_backend(self):
        kube = Kube(4, 4, 4, states=4)
        self.assertIsInstance(kube, KubeBackend)

    def test_plain_object_is_not_instance(self):
        self.assertNotIsInstance(object(), KubeBackend)

    def test_partial_implementation_is_not_instance(self):
        class Stub:
            def inspect(self): ...

        self.assertNotIsInstance(Stub(), KubeBackend)


# ---------------------------------------------------------------------------
# Kube.schema()
# ---------------------------------------------------------------------------

class KubeSchemaMethodTests(unittest.TestCase):
    def test_schema_returns_wide_faces_by_default(self):
        kube = Kube(4, 4, 4, states=4)
        s = kube.schema()
        self.assertIn("face_0", s)
        self.assertEqual(s["face_0"], "UInt8")

    def test_schema_cell_packed_mode(self):
        kube = Kube(4, 4, 4, states=4)
        s = kube.schema("cell_packed")
        self.assertIn("packed", s)
        self.assertNotIn("face_0", s)

    def test_schema_long_faces_mode(self):
        kube = Kube(4, 4, 4, states=4)
        s = kube.schema("long_faces")
        self.assertIn("face", s)
        self.assertIn("value", s)


# ---------------------------------------------------------------------------
# Arrow interop
# ---------------------------------------------------------------------------

@unittest.skipUnless(_ARROW, "pyarrow not installed")
class ArrowInteropTests(unittest.TestCase):
    def _make_kube(self, shape=(4, 4, 4), states=4):
        kube = Kube(*shape, states=states)
        kube.set_cell(0, 0, 0, Cell(states, [1, 2, 3, 0, 1, 2]))
        return kube

    def test_to_arrow_returns_pa_table(self):
        kube = self._make_kube()
        table = kube.to_arrow()
        self.assertIsInstance(table, pa.Table)

    def test_to_arrow_wide_faces_has_face_columns(self):
        kube = self._make_kube()
        table = kube.to_arrow(schema_mode="wide_faces")
        self.assertIn("face_0", table.schema.names)
        self.assertIn("x", table.schema.names)

    def test_to_arrow_cell_packed_has_packed_column(self):
        kube = self._make_kube()
        table = kube.to_arrow(schema_mode="cell_packed")
        self.assertIn("packed", table.schema.names)
        self.assertNotIn("face_0", table.schema.names)

    def test_to_arrow_long_faces_has_face_and_value(self):
        kube = self._make_kube()
        table = kube.to_arrow(schema_mode="long_faces")
        self.assertIn("face", table.schema.names)
        self.assertIn("value", table.schema.names)
        # long_faces has 6× the rows
        self.assertEqual(table.num_rows, 4 * 4 * 4 * 6)

    def test_to_arrow_row_count_matches_cell_count(self):
        kube = self._make_kube()
        table = kube.to_arrow()
        self.assertEqual(table.num_rows, kube.cell_count)

    def test_to_arrow_with_bounds(self):
        kube = self._make_kube((8, 8, 8))
        bounds = ((0, 4), (0, 4), (0, 4))
        table = kube.to_arrow(bounds)
        self.assertEqual(table.num_rows, 64)

    def test_region_to_arrow_alias(self):
        kube = self._make_kube()
        bounds = ((0, 2), (0, 2), (0, 2))
        self.assertEqual(kube.region_to_arrow(bounds).num_rows, 8)

    def test_from_arrow_round_trip(self):
        kube = self._make_kube()
        kube.set_cell(1, 2, 3, Cell(4, [3, 2, 1, 0, 3, 2]))
        table = kube.to_arrow()
        restored = Kube.from_arrow(table, shape=(4, 4, 4), states=4)
        self.assertEqual(restored.get_cell(1, 2, 3).faces, [3, 2, 1, 0, 3, 2])
        self.assertEqual(restored.get_cell(0, 0, 0).faces, [1, 2, 3, 0, 1, 2])

    def test_to_arrow_unknown_schema_mode_raises(self):
        kube = self._make_kube()
        with self.assertRaises(ValueError):
            kube.to_arrow(schema_mode="bad_mode")


# ---------------------------------------------------------------------------
# NumPy interop
# ---------------------------------------------------------------------------

@unittest.skipUnless(_NUMPY, "numpy not installed")
class NumpyInteropTests(unittest.TestCase):
    def test_faces_to_numpy_shape(self):
        kube = Kube(3, 4, 5, states=4)
        arr = kube.faces_to_numpy()
        self.assertEqual(arr.shape, (3, 4, 5, 6))

    def test_faces_to_numpy_dtype_uint8_for_small_states(self):
        kube = Kube(2, 2, 2, states=4)
        arr = kube.faces_to_numpy()
        self.assertEqual(arr.dtype, np.uint8)

    def test_faces_to_numpy_dtype_uint16_for_medium_states(self):
        kube = Kube(2, 2, 2, states=1024)
        arr = kube.faces_to_numpy()
        self.assertEqual(arr.dtype, np.uint16)

    def test_faces_to_numpy_reflects_cell_values(self):
        kube = Kube(2, 2, 2, states=4)
        kube.set_cell(0, 0, 0, Cell(4, [1, 2, 3, 0, 1, 2]))
        arr = kube.faces_to_numpy()
        self.assertEqual(list(arr[0, 0, 0]), [1, 2, 3, 0, 1, 2])

    def test_to_numpy_flat_shape(self):
        kube = Kube(2, 3, 4, states=4)
        arr = kube.to_numpy()
        self.assertEqual(arr.shape, (24, 6))

    def test_to_numpy_packed_shape(self):
        kube = Kube(2, 3, 4, states=4)
        arr = kube.to_numpy(packed=True)
        self.assertEqual(arr.shape, (24,))

    def test_region_to_numpy(self):
        kube = Kube(4, 4, 4, states=4)
        arr = kube.region_to_numpy(((0, 2), (0, 2), (0, 2)))
        self.assertEqual(arr.shape, (8, 6))

    def test_array_protocol(self):
        kube = Kube(2, 2, 2, states=4)
        arr = np.array(kube)
        self.assertEqual(arr.shape, (2, 2, 2, 6))

    def test_array_protocol_with_dtype(self):
        kube = Kube(2, 2, 2, states=4)
        arr = np.array(kube, dtype=np.float32)
        self.assertEqual(arr.dtype, np.float32)

    def test_numpy_compat_module(self):
        from kube.integrations.numpy_compat import (
            faces_to_numpy,
            packed_to_numpy,
            region_to_numpy,
        )
        kube = Kube(2, 2, 2, states=4)
        self.assertEqual(faces_to_numpy(kube).shape, (2, 2, 2, 6))
        self.assertEqual(packed_to_numpy(kube).shape, (8,))
        self.assertEqual(region_to_numpy(kube, ((0, 1), (0, 1), (0, 1))).shape, (1, 6))

    def test_kube_from_numpy_round_trip(self):
        from kube.integrations.numpy_compat import kube_from_numpy
        kube = Kube(2, 2, 2, states=4)
        kube.set_cell(0, 0, 0, Cell(4, [1, 2, 3, 0, 1, 2]))
        arr = kube.faces_to_numpy()
        restored = kube_from_numpy(arr, states=4)
        self.assertEqual(restored.shape, (2, 2, 2))
        self.assertEqual(restored.get_cell(0, 0, 0).faces, [1, 2, 3, 0, 1, 2])

    def test_kube_from_numpy_rejects_wrong_ndim(self):
        from kube.integrations.numpy_compat import kube_from_numpy
        with self.assertRaises(ValueError):
            kube_from_numpy(np.zeros((2, 2, 2)), states=4)


# ---------------------------------------------------------------------------
# Pandas interop
# ---------------------------------------------------------------------------

@unittest.skipUnless(_PANDAS, "pandas not installed")
class PandasInteropTests(unittest.TestCase):
    def test_to_pandas_returns_dataframe(self):
        kube = Kube(2, 2, 2, states=4)
        df = kube.to_pandas()
        self.assertIsInstance(df, pd.DataFrame)

    def test_to_pandas_has_face_columns(self):
        kube = Kube(2, 2, 2, states=4)
        df = kube.to_pandas()
        for col in (f"face_{i}" for i in range(6)):
            self.assertIn(col, df.columns)

    def test_to_pandas_row_count(self):
        kube = Kube(2, 3, 4, states=4)
        df = kube.to_pandas()
        self.assertEqual(len(df), kube.cell_count)

    def test_from_pandas_round_trip(self):
        kube = Kube(2, 2, 2, states=4)
        kube.set_cell(1, 1, 1, Cell(4, [3, 2, 1, 0, 3, 2]))
        df = kube.to_pandas()
        restored = Kube.from_pandas(df, shape=(2, 2, 2), states=4)
        self.assertEqual(restored.get_cell(1, 1, 1).faces, [3, 2, 1, 0, 3, 2])


# ---------------------------------------------------------------------------
# Parquet interop
# ---------------------------------------------------------------------------

@unittest.skipUnless(_ARROW, "pyarrow not installed")
class ParquetInteropTests(unittest.TestCase):
    def test_to_parquet_writes_file(self):
        kube = Kube(4, 4, 4, states=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = Path(tmpdir) / "out.parquet"
            result = kube.to_parquet(dst)
            self.assertEqual(result, dst)
            self.assertTrue(dst.exists())

    def test_to_parquet_readable_by_polars(self):
        kube = Kube(4, 4, 4, states=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = Path(tmpdir) / "out.parquet"
            kube.to_parquet(dst)
            frame = pl.read_parquet(dst)
            self.assertEqual(frame.height, kube.cell_count)
            self.assertIn("face_0", frame.columns)

    def test_to_parquet_cell_packed_schema(self):
        kube = Kube(4, 4, 4, states=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = Path(tmpdir) / "packed.parquet"
            kube.to_parquet(dst, schema_mode="cell_packed")
            frame = pl.read_parquet(dst)
            self.assertIn("packed", frame.columns)
            self.assertNotIn("face_0", frame.columns)

    def test_to_parquet_with_bounds(self):
        kube = Kube(8, 8, 8, states=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = Path(tmpdir) / "region.parquet"
            kube.to_parquet(dst, bounds=((0, 4), (0, 4), (0, 4)))
            frame = pl.read_parquet(dst)
            self.assertEqual(frame.height, 64)

    def test_region_to_parquet_alias(self):
        kube = Kube(4, 4, 4, states=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = Path(tmpdir) / "region.parquet"
            kube.region_to_parquet(((0, 2), (0, 2), (0, 2)), dst)
            frame = pl.read_parquet(dst)
            self.assertEqual(frame.height, 8)

    def test_to_parquet_creates_parent_dirs(self):
        kube = Kube(2, 2, 2, states=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = Path(tmpdir) / "a" / "b" / "out.parquet"
            kube.to_parquet(dst)
            self.assertTrue(dst.exists())


# ---------------------------------------------------------------------------
# sklearn interop
# ---------------------------------------------------------------------------

@unittest.skipUnless(_NUMPY, "numpy not installed")
class SklearnInteropTests(unittest.TestCase):
    def test_to_sklearn_xy_shapes(self):
        kube = Kube(8, 8, 8, states=4)
        X, y = kube.to_sklearn_xy(target_face=0)
        self.assertEqual(X.shape, (512, 5))
        self.assertEqual(y.shape, (512,))

    def test_to_sklearn_xy_each_target_face(self):
        kube = Kube(2, 2, 2, states=4)
        for face in range(6):
            X, y = kube.to_sklearn_xy(target_face=face)
            self.assertEqual(X.shape[1], 5)
            self.assertEqual(y.shape[0], kube.cell_count)

    def test_to_sklearn_xy_with_bounds(self):
        kube = Kube(8, 8, 8, states=4)
        X, y = kube.to_sklearn_xy(target_face=0, bounds=((0, 4), (0, 4), (0, 4)))
        self.assertEqual(X.shape, (64, 5))
        self.assertEqual(y.shape, (64,))

    def test_sklearn_module_to_sklearn_xy(self):
        from kube.integrations.sklearn import to_sklearn_xy
        kube = Kube(2, 2, 2, states=4)
        X, y = to_sklearn_xy(kube, target_face=2)
        self.assertEqual(X.shape, (8, 5))

    def test_spatial_train_test_split_shapes(self):
        from kube.integrations.sklearn import spatial_train_test_split
        kube = Kube(8, 8, 8, states=4)
        X_tr, X_te, y_tr, y_te = spatial_train_test_split(
            kube, target_face=0, test_axis=2, test_fraction=0.25
        )
        total = X_tr.shape[0] + X_te.shape[0]
        self.assertEqual(total, kube.cell_count)
        self.assertEqual(X_tr.shape[1], 5)

    def test_spatial_train_test_split_no_overlap(self):
        from kube.integrations.sklearn import spatial_train_test_split
        kube = Kube(4, 4, 4, states=4)
        X_tr, X_te, y_tr, y_te = spatial_train_test_split(
            kube, target_face=0, test_axis=0, test_fraction=0.5
        )
        # train + test = full cube
        self.assertEqual(X_tr.shape[0] + X_te.shape[0], kube.cell_count)

    def test_spatial_train_test_split_rejects_invalid_axis(self):
        from kube.integrations.sklearn import spatial_train_test_split
        kube = Kube(4, 4, 4, states=4)
        with self.assertRaises(ValueError):
            spatial_train_test_split(kube, 0, test_axis=3)

    def test_spatial_train_test_split_rejects_invalid_fraction(self):
        from kube.integrations.sklearn import spatial_train_test_split
        kube = Kube(4, 4, 4, states=4)
        with self.assertRaises(ValueError):
            spatial_train_test_split(kube, 0, test_fraction=1.5)


# ---------------------------------------------------------------------------
# HuggingFace interop (no deps: just import guards)
# ---------------------------------------------------------------------------

class HuggingFaceImportGuardTests(unittest.TestCase):
    """Verify graceful ImportError when 'datasets' is not installed."""

    def _datasets_present(self):
        try:
            import datasets  # noqa: F401
            return True
        except ImportError:
            return False

    def test_to_hf_dataset_raises_without_datasets(self):
        if self._datasets_present():
            self.skipTest("datasets is installed — skipping guard test")
        from kube.integrations.hf import to_hf_dataset
        kube = Kube(2, 2, 2, states=4)
        with self.assertRaises(ImportError):
            to_hf_dataset(kube)

    def test_to_hf_iterable_dataset_raises_without_datasets(self):
        if self._datasets_present():
            self.skipTest("datasets is installed — skipping guard test")
        from kube.integrations.hf import to_hf_iterable_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "ds.kubed"
            ds = create_sharded(root, shards=1, states=256, tokens_per_shard=64, sequence_length=8)
            with self.assertRaises(ImportError):
                to_hf_iterable_dataset(ds, sequence_length=8)


# ---------------------------------------------------------------------------
# Dataset interop: to_arrow, to_parquet, iter_lm_batches
# ---------------------------------------------------------------------------

class _FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class DatasetInteropTests(unittest.TestCase):
    def _make_dataset(self, tmpdir, token_count=64, sequence_length=8):
        root = Path(tmpdir) / "test.kubed"
        ds = create_sharded(
            root,
            shards=2,
            states=256,
            token_bytes=2,
            tokens_per_shard=64,
            sequence_length=sequence_length,
        )
        rows = pl.DataFrame({"input_ids": [[i for i in range(token_count)]]})
        src = Path(tmpdir) / "tokens.parquet"
        rows.write_parquet(src)
        return import_token_ids(src, root, tokens_column="input_ids")

    @unittest.skipUnless(_ARROW, "pyarrow not installed")
    def test_to_arrow_returns_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = self._make_dataset(tmpdir)
            table = ds.to_arrow(sequence_length=8, count=4)
            self.assertIsInstance(table, pa.Table)
            self.assertIn("input_ids", table.schema.names)
            self.assertIn("target_ids", table.schema.names)
            self.assertLessEqual(table.num_rows, 4)

    @unittest.skipUnless(_ARROW, "pyarrow not installed")
    def test_to_parquet_writes_readable_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = self._make_dataset(tmpdir)
            out = Path(tmpdir) / "seqs.parquet"
            result = ds.to_parquet(out, sequence_length=8, count=4)
            self.assertTrue(result.exists())
            frame = pl.read_parquet(result)
            self.assertIn("input_ids", frame.columns)

    def test_iter_lm_batches_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = self._make_dataset(tmpdir)
            batch = next(ds.iter_lm_batches(batch_size=2, sequence_length=8))
            self.assertIn("input_ids", batch)
            self.assertIn("attention_mask", batch)
            self.assertIn("labels", batch)

    def test_iter_lm_batches_padding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = self._make_dataset(tmpdir)
            for batch in ds.iter_lm_batches(batch_size=2, sequence_length=8):
                # All sequences in a batch should have the same length
                lengths = [len(row) for row in batch["input_ids"]]
                self.assertEqual(len(set(lengths)), 1)
                break

    def test_iter_lm_batches_attention_mask_matches_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = self._make_dataset(tmpdir)
            for batch in ds.iter_lm_batches(batch_size=2, sequence_length=8):
                for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
                    self.assertEqual(len(ids), len(mask))
                break


# ---------------------------------------------------------------------------
# KubeDataCollatorForCausalLM
# ---------------------------------------------------------------------------

class DataCollatorTests(unittest.TestCase):
    def _samples(self):
        return [
            {"input_ids": [1, 2, 3], "labels": [2, 3, 4]},
            {"input_ids": [5, 6], "labels": [6, 7]},
        ]

    def test_returns_padded_equal_length_sequences(self):
        collator = KubeDataCollatorForCausalLM(pad_token_id=0)
        result = collator(self._samples())
        # Determine if result contains lists or tensors
        ids = result["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        lengths = [len(row) for row in ids]
        self.assertEqual(len(set(lengths)), 1)

    def test_pads_with_configured_token_id(self):
        collator = KubeDataCollatorForCausalLM(pad_token_id=99)
        result = collator(self._samples())
        ids = result["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        # The shorter sequence should be padded with 99
        self.assertIn(99, ids[1])

    def test_labels_padded_with_label_pad_id(self):
        collator = KubeDataCollatorForCausalLM(pad_token_id=0, label_pad_id=-100)
        result = collator(self._samples())
        labels = result["labels"]
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        self.assertIn(-100, labels[1])

    def test_attention_mask_has_ones_and_zeros(self):
        collator = KubeDataCollatorForCausalLM(pad_token_id=0)
        result = collator(self._samples())
        mask = result["attention_mask"]
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        # Shorter sequence gets a trailing 0
        self.assertIn(0, mask[1])
        self.assertEqual(mask[0], [1, 1, 1])

    def test_uses_target_ids_when_labels_missing(self):
        collator = KubeDataCollatorForCausalLM(pad_token_id=0)
        samples = [
            {"input_ids": [1, 2], "target_ids": [2, 3]},
            {"input_ids": [4, 5, 6], "target_ids": [5, 6, 7]},
        ]
        result = collator(samples)
        self.assertIn("labels", result)


# ---------------------------------------------------------------------------
# integrations/__init__.py
# ---------------------------------------------------------------------------

class IntegrationsInitTests(unittest.TestCase):
    def test_available_backends_returns_list(self):
        from kube.integrations import available_backends
        backends = available_backends()
        self.assertIsInstance(backends, list)
        # All elements should be strings
        for b in backends:
            self.assertIsInstance(b, str)

    def test_check_helpers_return_bool(self):
        from kube.integrations import (
            _check_arrow,
            _check_datasets,
            _check_numpy,
            _check_pandas,
            _check_sklearn,
            _check_torch,
        )
        for fn in (_check_arrow, _check_numpy, _check_pandas, _check_torch, _check_datasets, _check_sklearn):
            result = fn()
            self.assertIsInstance(result, bool)

    @unittest.skipUnless(_ARROW, "pyarrow not installed")
    def test_arrow_appears_in_available_backends(self):
        from kube.integrations import available_backends
        self.assertIn("arrow", available_backends())

    @unittest.skipUnless(_NUMPY, "numpy not installed")
    def test_numpy_appears_in_available_backends(self):
        from kube.integrations import available_backends
        self.assertIn("numpy", available_backends())


# ---------------------------------------------------------------------------
# __init__.py public exports
# ---------------------------------------------------------------------------

class PublicExportsTests(unittest.TestCase):
    def test_schema_constants_importable_from_kube(self):
        from kube import (
            SCHEMA_CELL_PACKED,
            SCHEMA_LONG_FACES,
            SCHEMA_SUPERVISED,
            SCHEMA_TOKEN_SEQUENCES,
            SCHEMA_WIDE_FACES,
        )
        self.assertEqual(SCHEMA_WIDE_FACES, "wide_faces")

    def test_metadata_constants_importable_from_kube(self):
        from kube import (
            FEATURE_SCHEMA,
            PIPELINE_VERSION,
            SEQUENCE_LENGTH,
            SOURCE_DATASET,
            SPLIT,
            TOKENIZER_NAME,
            TOKENIZER_PATH,
        )
        self.assertEqual(SPLIT, "split")


# ---------------------------------------------------------------------------
# CLI: export-arrow and export-parquet subcommands
# ---------------------------------------------------------------------------

@unittest.skipUnless(_ARROW, "pyarrow not installed")
class CLIExportArrowTests(unittest.TestCase):
    def _make_kube_file(self, tmpdir):
        path = Path(tmpdir) / "test.kube"
        kube = Kube.create(path, (4, 4, 4), states=4)
        kube.set_cell(0, 0, 0, Cell(4, [1, 2, 3, 0, 1, 2]))
        kube.close()
        return path

    def test_export_arrow_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = self._make_kube_file(tmpdir)
            dst = Path(tmpdir) / "out.arrow"
            with patch("sys.argv", ["kube", "export-arrow", str(src), str(dst)]):
                main()
            self.assertTrue(dst.exists())

    def test_export_arrow_file_is_readable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = self._make_kube_file(tmpdir)
            dst = Path(tmpdir) / "out.arrow"
            with patch("sys.argv", ["kube", "export-arrow", str(src), str(dst)]):
                main()
            import pyarrow.ipc as ipc
            with ipc.open_file(str(dst)) as reader:
                table = reader.read_all()
            self.assertIn("face_0", table.schema.names)
            self.assertEqual(table.num_rows, 64)

    def test_export_arrow_cell_packed_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = self._make_kube_file(tmpdir)
            dst = Path(tmpdir) / "packed.arrow"
            with patch("sys.argv", [
                "kube", "export-arrow", str(src), str(dst),
                "--schema-mode", "cell_packed",
            ]):
                main()
            import pyarrow.ipc as ipc
            with ipc.open_file(str(dst)) as reader:
                table = reader.read_all()
            self.assertIn("packed", table.schema.names)
            self.assertNotIn("face_0", table.schema.names)

    def test_export_arrow_json_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = self._make_kube_file(tmpdir)
            dst = Path(tmpdir) / "out.arrow"
            buf = io.StringIO()
            with patch("sys.argv", [
                "kube", "--output", "json",
                "export-arrow", str(src), str(dst),
            ]):
                with redirect_stdout(buf):
                    main()
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["operation"], "export-arrow")
            self.assertEqual(payload["rows"], 64)


@unittest.skipUnless(_ARROW, "pyarrow not installed")
class CLIExportParquetTests(unittest.TestCase):
    def _make_kube_file(self, tmpdir):
        path = Path(tmpdir) / "test.kube"
        kube = Kube.create(path, (4, 4, 4), states=4)
        kube.set_cell(0, 0, 0, Cell(4, [1, 2, 3, 0, 1, 2]))
        kube.close()
        return path

    def test_export_parquet_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = self._make_kube_file(tmpdir)
            dst = Path(tmpdir) / "out.parquet"
            with patch("sys.argv", ["kube", "export-parquet", str(src), str(dst)]):
                main()
            self.assertTrue(dst.exists())

    def test_export_parquet_is_readable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = self._make_kube_file(tmpdir)
            dst = Path(tmpdir) / "out.parquet"
            with patch("sys.argv", ["kube", "export-parquet", str(src), str(dst)]):
                main()
            frame = pl.read_parquet(dst)
            self.assertIn("face_0", frame.columns)
            self.assertEqual(frame.height, 64)

    def test_export_parquet_json_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = self._make_kube_file(tmpdir)
            dst = Path(tmpdir) / "out.parquet"
            buf = io.StringIO()
            with patch("sys.argv", [
                "kube", "--output", "json",
                "export-parquet", str(src), str(dst),
            ]):
                with redirect_stdout(buf):
                    main()
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["operation"], "export-parquet")
            self.assertIn("rows", payload)

    def test_export_parquet_long_faces_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = self._make_kube_file(tmpdir)
            dst = Path(tmpdir) / "long.parquet"
            with patch("sys.argv", [
                "kube", "export-parquet", str(src), str(dst),
                "--schema-mode", "long_faces",
            ]):
                main()
            frame = pl.read_parquet(dst)
            self.assertIn("face", frame.columns)
            self.assertIn("value", frame.columns)
            self.assertEqual(frame.height, 64 * 6)

    def test_export_parquet_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = self._make_kube_file(tmpdir)
            dst = Path(tmpdir) / "nested" / "out.parquet"
            with patch("sys.argv", ["kube", "export-parquet", str(src), str(dst)]):
                main()
            self.assertTrue(dst.exists())


if __name__ == "__main__":
    unittest.main()
