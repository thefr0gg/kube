import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import polars as pl

from kube import create_sharded, import_token_ids, open_dataset
from kube.dataset import import_arrow_text, import_parquet_text
from kube.main import main


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class _FakeTokenizer:
    def encode(self, text: str):
        ids = [len(part) for part in text.split() if part]
        return _FakeEncoding(ids)

    def token_to_id(self, token: str):
        if token == "</s>":
            return 99
        return None


class ShardedDatasetTests(unittest.TestCase):
    def test_create_and_open_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "train.kubed"
            with create_sharded(
                root,
                shards=2,
                states=256,
                token_bytes=2,
                tokens_per_shard=4,
                sequence_length=8,
            ) as dataset:
                self.assertEqual(dataset.inspect()["shard_count"], 2)

            with open_dataset(root, mode="r") as reopened:
                self.assertEqual(reopened.default_sequence_length, 8)
                self.assertEqual(reopened.shard_count, 2)

    def test_import_token_ids_and_iter_sequences(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "tokens.kubed"
            src = Path(tmpdir) / "tokens.parquet"
            create_sharded(root, shards=2, states=256, token_bytes=2, tokens_per_shard=4)
            pl.DataFrame({"input_ids": [[1, 2, 3, 4, 5], [6, 7, 8]]}).write_parquet(src)

            dataset = import_token_ids(src, root, tokens_column="input_ids")
            self.assertEqual(dataset.total_document_count, 2)
            self.assertEqual(dataset.total_token_count, 8)

            samples = list(dataset.iter_sequences(3, stride=2))
            self.assertEqual(samples[0]["input_ids"], [1, 2, 3])
            self.assertEqual(samples[0]["target_ids"], [2, 3, 4])
            self.assertEqual(samples[1]["input_ids"], [3, 4, 5])
            self.assertEqual(samples[1]["target_ids"], [4, 5, 6])
            self.assertEqual(samples[2]["input_ids"], [5, 6, 7])
            self.assertEqual(samples[2]["target_ids"], [6, 7, 8])

    def test_resume_iterator_and_prefetch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "resume.kubed"
            src = Path(tmpdir) / "tokens.parquet"
            create_sharded(root, shards=3, states=256, token_bytes=2, tokens_per_shard=3)
            pl.DataFrame({"input_ids": [[1, 2, 3], [4, 5, 6], [7, 8]]}).write_parquet(src)
            dataset = import_token_ids(src, root)

            baseline = list(dataset.iter_sequences(2, stride=1, prefetch_shards=0))
            prefetched = list(dataset.iter_sequences(2, stride=1, prefetch_shards=1))
            self.assertEqual(
                [(sample["input_ids"], sample["target_ids"]) for sample in baseline],
                [(sample["input_ids"], sample["target_ids"]) for sample in prefetched],
            )

            resumed = list(dataset.resume_iterator(baseline[0]["resume_state"]))
            self.assertEqual(resumed[0]["input_ids"], baseline[1]["input_ids"])
            self.assertEqual(resumed[0]["target_ids"], baseline[1]["target_ids"])

    def test_text_import_and_document_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "text.kubed"
            src = Path(tmpdir) / "text.parquet"
            stats_out = Path(tmpdir) / "shards.parquet"
            create_sharded(root, shards=2, states=256, token_bytes=2, tokens_per_shard=6)
            pl.DataFrame({"text": ["hello world", "tiny dataset"]}).write_parquet(src)

            dataset = import_parquet_text(src, root, tokenizer=_FakeTokenizer(), text_column="text")
            self.assertEqual(dataset.total_document_count, 2)
            self.assertEqual(dataset.document_frame().height, 2)
            self.assertIn("eos_token_id", dataset.inspect()["tokenizer_metadata"])

            dataset.export_stats_parquet(stats_out)
            exported = pl.read_parquet(stats_out)
            self.assertEqual(exported.height, 2)
            self.assertIn("tokenizer_metadata", exported.columns)

    def test_arrow_text_import(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "arrow.kubed"
            src = Path(tmpdir) / "text.arrow"
            create_sharded(root, shards=2, states=256, token_bytes=2, tokens_per_shard=6)
            pl.DataFrame({"text": ["alpha beta", "gamma"]}).write_ipc(src)

            dataset = import_arrow_text(src, root, tokenizer=_FakeTokenizer(), text_column="text")
            self.assertEqual(dataset.total_document_count, 2)
            self.assertGreater(dataset.total_token_count, 0)

    def test_sequence_batch_and_sample_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "sample.kubed"
            src = Path(tmpdir) / "tokens.parquet"
            create_sharded(root, shards=2, states=256, token_bytes=2, tokens_per_shard=4)
            pl.DataFrame({"input_ids": [[10, 11, 12, 13], [14, 15, 16]]}).write_parquet(src)
            dataset = import_token_ids(src, root)

            batches = list(dataset.iter_sequence_batches(2, 2, stride=1))
            self.assertEqual(len(batches[0]["input_ids"]), 2)
            self.assertEqual(len(batches[0]["shard_indices"]), 2)

            frame = dataset.sample_sequences_frame(sequence_length=2, count=2, stride=1)
            self.assertEqual(frame.height, 2)
            self.assertIn("input_ids", frame.columns)

            frame_batches = list(dataset.iter_sequence_frame_batches(2, 2, stride=1))
            self.assertEqual(frame_batches[0].height, 2)
            self.assertIn("shard_index", frame_batches[0].columns)

    def test_resume_state_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "resume-file.kubed"
            src = Path(tmpdir) / "tokens.parquet"
            state_path = Path(tmpdir) / "resume-state.json"
            create_sharded(root, shards=2, states=256, token_bytes=2, tokens_per_shard=4)
            pl.DataFrame({"input_ids": [[1, 2, 3, 4], [5, 6, 7]]}).write_parquet(src)
            dataset = import_token_ids(src, root)

            sample = next(dataset.iter_sequences(2, stride=2))
            dataset.save_resume_state(state_path, sample["resume_state"])
            loaded = dataset.load_resume_state(state_path)

            self.assertEqual(loaded, sample["resume_state"])
            resumed = next(dataset.resume_iterator(loaded))
            self.assertEqual(resumed["input_ids"], [3, 4])

    def test_cli_dataset_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "cli.kubed"
            src = Path(tmpdir) / "tokens.parquet"
            docs_out = Path(tmpdir) / "documents.parquet"
            resume_out = Path(tmpdir) / "resume-state.json"
            pl.DataFrame({"input_ids": [[1, 2, 3], [4, 5, 6]]}).write_parquet(src)

            stdout = io.StringIO()
            with patch("sys.argv", ["kube", "dataset-create", str(root), "--shards", "2", "--token-bytes", "2", "--tokens-per-shard", "4"]):
                with redirect_stdout(stdout):
                    main()

            stdout = io.StringIO()
            with patch("sys.argv", ["kube", "dataset-import-tokens", str(src), str(root), "--tokens-column", "input_ids", "--sequence-length", "2"]):
                with redirect_stdout(stdout):
                    main()
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["total_document_count"], 2)

            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "kube",
                    "dataset-sample",
                    str(root),
                    "--sequence-length",
                    "2",
                    "--count",
                    "2",
                    "--resume-state-out",
                    str(resume_out),
                ],
            ):
                with redirect_stdout(stdout):
                    main()
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["count"], 2)
                self.assertEqual(payload["resume_state_out"], str(resume_out))
                self.assertTrue(resume_out.exists())

            stdout = io.StringIO()
            with patch("sys.argv", ["kube", "dataset-stats", str(root), "--export-documents", str(docs_out)]):
                with redirect_stdout(stdout):
                    main()
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["documents"]["count"], 2)
                self.assertEqual(payload["export_documents"], str(docs_out))
                self.assertTrue(docs_out.exists())


if __name__ == "__main__":
    unittest.main()
