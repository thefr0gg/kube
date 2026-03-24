import argparse
import tempfile
import unittest
from pathlib import Path

from kube import Cell, Kube, from_polars
from kube.main import _parse_window_size


class KubeTests(unittest.TestCase):
    def test_in_memory_round_trip(self):
        payload = b"hello kube" * 32
        kube = Kube.from_bytes(payload, states=4)
        self.assertEqual(kube.to_bytes(), payload)

    def test_mapped_create_open_and_edit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.kube"
            kube = Kube.create(path, (4, 4, 4), states=4)
            kube.set_face(0, 0, 0, 0, 3)
            kube.set_cell(1, 1, 1, Cell(4, [1, 2, 3, 0, 1, 2]))
            kube.flush()
            kube.close()

            reopened = Kube.open(path)
            self.assertEqual(reopened.get_face(0, 0, 0, 0), 3)
            self.assertEqual(reopened.get_cell(1, 1, 1).faces, [1, 2, 3, 0, 1, 2])
            reopened.close()

    def test_create_rejects_existing_file_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.kube"
            kube = Kube.create(path, (2, 2, 2), states=4)
            kube.close()

            with self.assertRaises(FileExistsError):
                Kube.create(path, (2, 2, 2), states=4)

    def test_create_can_overwrite_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.kube"
            kube = Kube.create(path, (2, 2, 2), states=4)
            kube.set_face(0, 0, 0, 0, 3)
            kube.flush()
            kube.close()

            replaced = Kube.create(path, (3, 3, 3), states=4, overwrite=True)
            self.assertEqual(replaced.shape, (3, 3, 3))
            self.assertEqual(replaced.get_face(0, 0, 0, 0), 0)
            replaced.close()

    def test_stream_import_export(self):
        payload = bytes((index * 17) % 256 for index in range(4097))
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "input.bin"
            dst = Path(tmpdir) / "output.kube"
            restored = Path(tmpdir) / "restored.bin"
            src.write_bytes(payload)

            kube = Kube.import_file(src, dst, states=4, chunk_size=37)
            kube.export_file(restored, chunk_cells=19)
            kube.close()

            self.assertEqual(restored.read_bytes(), payload)

    def test_read_only_open_blocks_mutation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.kube"
            kube = Kube.create(path, (2, 2, 2), states=4)
            kube.close()

            read_only = Kube.open(path, mode="r")
            with self.assertRaises(PermissionError):
                read_only.set_face(0, 0, 0, 0, 1)
            read_only.close()

    def test_memory_stats_reports_acceleration_mode(self):
        kube = Kube(2, 2, 2, states=4)
        self.assertEqual(kube.memory_stats()["acceleration"], "native")
        self.assertEqual(kube.inspect()["acceleration"], "native")

    def test_native_random_io_works_for_memory_and_mapped_storage(self):
        states_values = (2, 4, 16, 256)
        shape = (3, 3, 3)
        operations = [
            (0, 0, 0, 0, 1),
            (2, 1, 1, 5, 1),
            (1, 2, 0, 3, 2),
            (2, 2, 2, 1, 3),
            (0, 1, 2, 4, 7),
            (1, 1, 1, 2, 15),
            (2, 0, 2, 0, 31),
            (1, 0, 1, 5, 63),
            (0, 2, 1, 3, 127),
            (2, 2, 0, 4, 255),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            for states in states_values:
                max_value = states - 1
                filtered_ops = [
                    (x, y, z, face, value % states)
                    for x, y, z, face, value in operations
                ]
                cell = Cell(states, [value % states for value in (0, 1, 2, 3, 4, 5)])

                cases: list[Kube] = []
                try:
                    cases.append(Kube(*shape, states=states))

                    native_path = tmpdir_path / f"native-{states}.kube"
                    cases.append(Kube.create(native_path, shape, states=states))

                    for kube_instance in cases:
                        compiled = kube_instance.compile_face_batch(filtered_ops)

                        for x, y, z, face, value in filtered_ops:
                            kube_instance.set_face(x, y, z, face, value)
                            self.assertEqual(
                                kube_instance.get_face(x, y, z, face),
                                value,
                            )

                        kube_instance.data.clear(accelerator=kube_instance._storage_accel)
                        kube_instance.set_faces_batch(compiled)
                        self.assertEqual(kube_instance.get_faces_batch(compiled), [op[4] for op in filtered_ops])

                        kube_instance.set_cell(1, 1, 1, cell)
                        self.assertEqual(kube_instance.get_cell(1, 1, 1).faces, cell.faces)

                        kube_instance.data.clear_from(5, accelerator=kube_instance._storage_accel)
                        for index in range(5, kube_instance.cell_count):
                            self.assertEqual(kube_instance.data.get_flat(index), 0)

                        kube_instance.set_face(0, 0, 0, 0, max_value)
                        kube_instance.data.clear(accelerator=kube_instance._storage_accel)
                        self.assertTrue(
                            all(kube_instance.data.get_flat(index) == 0 for index in range(kube_instance.cell_count))
                        )
                finally:
                    for kube_instance in cases:
                        kube_instance.close()

    def test_region_iteration(self):
        kube = Kube(6, 6, 6, states=4)
        kube.set_face(2, 2, 2, 4, 3)
        region = kube.read_region(((1, 4), (1, 4), (1, 4)))
        self.assertEqual(region.height, 27)
        self.assertIn("packed", region.columns)
        tiles = list(kube.iter_region(((0, 4), (0, 4), (0, 4)), tile_shape=(2, 2, 2)))
        self.assertTrue(tiles)

    def test_to_polars_and_feature_frame(self):
        kube = Kube(2, 2, 2, states=4)
        kube.set_cell(0, 0, 0, Cell(4, [0, 1, 2, 3, 0, 1]))
        frame = kube.to_polars()
        features = kube.feature_frame(include_coords=False)
        self.assertEqual(frame.height, 8)
        self.assertIn("packed", frame.columns)
        self.assertEqual(features.columns, [f"face_{index}" for index in range(6)])

    def test_iter_batches(self):
        kube = Kube(3, 2, 2, states=4)
        batches = list(kube.iter_batches(3, include_packed=False))
        self.assertEqual(sum(batch.height for batch in batches), len(kube))
        self.assertTrue(all("packed" not in batch.columns for batch in batches))

    def test_from_polars_round_trip(self):
        kube = Kube(2, 2, 2, states=4)
        kube.set_cell(1, 0, 1, Cell(4, [1, 2, 3, 0, 1, 2]))
        frame = kube.to_polars(include_packed=False)
        restored = from_polars(frame, shape=(2, 2, 2), states=4)
        self.assertEqual(restored.get_cell(1, 0, 1).faces, [1, 2, 3, 0, 1, 2])

    def test_slice_bounds_and_face_histogram(self):
        kube = Kube(2, 2, 2, states=4)
        kube.set_cell(0, 0, 0, Cell(4, [1, 1, 1, 1, 1, 1]))
        bounds = kube.slice_bounds("x", 0)
        histogram = kube.face_histogram(bounds=bounds, normalize=True)
        self.assertEqual(bounds, ((0, 1), (0, 2), (0, 2)))
        self.assertEqual(histogram.height, 24)
        rows = histogram.to_dicts()
        self.assertTrue(any(row["face"] == 0 and row["value"] == 1 for row in rows))

    def test_supervised_frame(self):
        kube = Kube(2, 2, 2, states=4)
        kube.set_cell(0, 0, 0, Cell(4, [1, 2, 3, 0, 1, 2]))
        frame = kube.supervised_frame(4, include_coords=False)
        self.assertIn("target", frame.columns)
        self.assertNotIn("face_4", frame.columns)

        full = kube.supervised_frame(4, include_coords=False, drop_target_face=False)
        self.assertIn("target", full.columns)
        self.assertIn("face_4", full.columns)

    def test_iter_supervised_batches(self):
        kube = Kube(3, 2, 2, states=4)
        batches = list(kube.iter_supervised_batches(4, target_face=2, include_packed=False))
        self.assertEqual(sum(batch.height for batch in batches), len(kube))
        self.assertTrue(all("target" in batch.columns for batch in batches))
        self.assertTrue(all("face_2" not in batch.columns for batch in batches))

    def test_view_rejects_invalid_window_size(self):
        kube = Kube(2, 2, 2, states=4)
        with self.assertRaises(ValueError):
            kube.view(window_size=(1024,))

    def test_labels_reflect_storage_mode(self):
        kube = Kube(2, 2, 2, states=4)
        self.assertEqual(kube.storage_label, "memory")
        self.assertEqual(kube.file_label, "in-memory kube")

    def test_parse_window_size(self):
        self.assertEqual(_parse_window_size("1440x900"), (1440, 900))
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_window_size("bad-size")


if __name__ == "__main__":
    unittest.main()
