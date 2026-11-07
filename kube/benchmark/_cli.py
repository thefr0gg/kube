"""CLI argument definitions and handler for `kube benchmark`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kube.benchmark._reporter import format_table, plot_results, to_json_dict
from kube.benchmark._runner import BenchmarkConfig, run_benchmarks

_ALL_SCENARIOS = [
    "stream_import",
    "table_conversion",
    "lm_dataset",
    "viewer_ops",
]


def add_benchmark_parser(parser: argparse.ArgumentParser) -> None:
    """Attach benchmark-specific arguments to *parser*."""
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        metavar="N",
        help="scenario size scale (cube side length / token row count, scenario-dependent)",
    )
    parser.add_argument(
        "--states",
        type=int,
        default=4,
        help="face state count for kube-backed scenarios",
    )
    parser.add_argument(
        "--scenarios",
        type=_parse_list,
        default=None,
        metavar="S,...",
        help=f"comma-separated scenario names; available: {', '.join(_ALL_SCENARIOS)}",
    )
    parser.add_argument(
        "--backends",
        type=_parse_list,
        default=None,
        metavar="B,...",
        help="comma-separated backend labels to restrict to (see table output for exact names)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="timing repetitions per scenario",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="save throughput bar chart to benchmark_results.png",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        dest="tmp_dir",
        metavar="PATH",
        help="directory for temporary fixture files (default: system tmp)",
    )


def handle_benchmark(args: argparse.Namespace) -> None:
    """Build config from parsed args, run benchmarks, and emit results."""
    config = BenchmarkConfig(
        size=args.size,
        states=args.states,
        scenarios=args.scenarios,
        backends=args.backends,
        iterations=args.iterations,
        seed=args.seed,
        tmp_dir=args.tmp_dir,
        plot=args.plot,
    )

    result = run_benchmarks(config)

    if getattr(args, "output", "text") == "json":
        print(json.dumps(to_json_dict(result), indent=2))
    else:
        print(format_table(result))

    if args.plot:
        out_path = "benchmark_results.png"
        plot_results(result, out_path)
        print(f"\nChart saved to {out_path}")


def _parse_list(text: str) -> list[str]:
    """Parse a comma-separated option value into a list of stripped strings."""
    return [item.strip() for item in text.split(",") if item.strip()]
