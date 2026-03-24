"""Create a sharded dataset and ingest text rows from Parquet or Arrow.

This example exists to show the higher-level text-ingestion workflow for
`.kubed` datasets when your source corpus already lives in a tabular file.

Workflow:

1. Create a sharded dataset root with the desired capacity.
2. Read a Parquet or Arrow/IPC table that contains a text column.
3. Tokenize that text with a tokenizer JSON file.
4. Import the tokenized documents into dataset shards.
5. Write provenance metadata (split, source) via creation_parameters using the metadata key constants.

Typical usage:

    python examples/import_text_corpus.py news.parquet corpora/news.kubed tokenizer.json

    python examples/import_text_corpus.py news.parquet corpora/news.kubed tokenizer.json \
        --split train \
        --dataset-name news-corpus
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from kube import import_arrow_text, import_parquet_text
from kube.dataset import ShardedKubeDataset
from kube.metadata import SOURCE_DATASET, SPLIT


def parse_args() -> argparse.Namespace:
    """Define the CLI for the text-ingestion example."""
    parser = argparse.ArgumentParser(
        description="Create a sharded Kube dataset and ingest a text table into it"
    )
    parser.add_argument("src", type=Path, help="Source Parquet or Arrow/IPC table")
    parser.add_argument("root", type=Path, help="Destination dataset root ending in `.kubed`")
    parser.add_argument("tokenizer", type=Path, help="Tokenizer JSON file used for text tokenization")
    parser.add_argument("--text-column", default="text", help="Column in the source table that contains raw text")
    parser.add_argument("--shards", type=int, default=4, help="Number of dataset shards to create")
    parser.add_argument("--token-bytes", type=int, default=4, help="Bytes per token stored in each shard")
    parser.add_argument(
        "--tokens-per-shard",
        type=int,
        default=1 << 18,
        help="Token capacity per shard",
    )
    parser.add_argument("--sequence-length", type=int, default=1024, help="Default sequence length stored in the dataset manifest")
    parser.add_argument(
        "--dataset-name",
        help="Optional dataset name written into the dataset manifest",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split label written into shard metadata (e.g. train, val, test)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing dataset root before recreating it",
    )
    return parser.parse_args()


def main() -> None:
    """Create the dataset root, import text rows, and print the resulting summary."""
    args = parse_args()

    if args.overwrite and args.root.exists():
        shutil.rmtree(args.root)

    dataset_name = args.dataset_name or args.root.stem

    ShardedKubeDataset.create(
        args.root,
        shards=args.shards,
        token_bytes=args.token_bytes,
        tokens_per_shard=args.tokens_per_shard,
        sequence_length=args.sequence_length,
        dataset_name=dataset_name,
        creation_parameters={
            SPLIT: args.split,
            SOURCE_DATASET: str(args.src),
        },
    ).close()

    if args.src.suffix.lower() == ".parquet":
        dataset = import_parquet_text(
            args.src,
            args.root,
            text_column=args.text_column,
            tokenizer_path=args.tokenizer,
        )
    elif args.src.suffix.lower() in {".arrow", ".ipc", ".feather"}:
        dataset = import_arrow_text(
            args.src,
            args.root,
            text_column=args.text_column,
            tokenizer_path=args.tokenizer,
        )
    else:
        raise SystemExit("source table must be .parquet, .arrow, .ipc, or .feather")

    with dataset:
        info = dataset.inspect()
        meta = dataset.manifest.get("creation_parameters", {})

        print("dataset summary")
        print("===============")
        print(json.dumps(info, indent=2))
        print()
        print("metadata")
        print("--------")
        print(f"  {SPLIT}          : {meta.get(SPLIT, '—')}")
        print(f"  {SOURCE_DATASET} : {meta.get(SOURCE_DATASET, '—')}")
        print()
        print("shard preview")
        print("=============")
        print(dataset.shard_frame().head(5))


if __name__ == "__main__":
    main()
