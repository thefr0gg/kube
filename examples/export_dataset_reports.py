"""Export shard, document, and sequence-sample QA reports for a `.kubed` dataset.

This example exists to show how Kube datasets can produce lightweight QA
artifacts for notebooks, dashboards, or external tooling without requiring a
long-running Python session.

Workflow:

1. Open a sharded dataset root.
2. Print a high-level dataset summary plus small shard/document previews.
3. Export shard stats, document stats, and a sampled sequence table to Parquet.

Typical usage:

    python examples/export_dataset_reports.py corpora/wiki.kubed reports/wiki
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kube import open_dataset


def parse_args() -> argparse.Namespace:
    """Define the CLI for the dataset QA export example."""
    parser = argparse.ArgumentParser(
        description="Export a few useful QA reports from a sharded `.kubed` dataset"
    )
    parser.add_argument("root", type=Path, help="Dataset root ending in `.kubed`")
    parser.add_argument("output_dir", type=Path, help="Directory that will receive report files")
    parser.add_argument("--sequence-length", type=int, default=512, help="Window length for the sampled sequence report")
    parser.add_argument("--count", type=int, default=16, help="Number of sample windows to export")
    parser.add_argument("--stride", type=int, help="Optional stride override for the sampled sequence report")
    return parser.parse_args()


def main() -> None:
    """Open the dataset, print a summary, and export report artifacts."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open_dataset(args.root, mode="r") as dataset:
        summary = dataset.inspect()
        shard_preview = dataset.shard_frame().head(5)
        document_preview = dataset.document_frame().head(5)

        stats_path = dataset.export_stats_parquet(args.output_dir / "shard-stats.parquet")
        documents_path = dataset.export_document_parquet(args.output_dir / "documents.parquet")
        sample_path = dataset.export_sequence_sample_parquet(
            args.output_dir / "sequence-sample.parquet",
            sequence_length=args.sequence_length,
            count=args.count,
            stride=args.stride,
        )

    print("dataset summary")
    print("===============")
    print(json.dumps(summary, indent=2))
    print()
    print("shard preview")
    print("=============")
    print(shard_preview)
    print()
    print("document preview")
    print("================")
    print(document_preview)
    print()
    print(f"wrote shard stats : {stats_path}")
    print(f"wrote documents   : {documents_path}")
    print(f"wrote seq sample  : {sample_path}")


if __name__ == "__main__":
    main()
