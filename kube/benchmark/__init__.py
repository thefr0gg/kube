"""Public API for the Kube benchmark suite.

Run the full benchmark suite programmatically::

    from kube.benchmark import run_benchmarks, BenchmarkConfig

    config = BenchmarkConfig(size=64, states=4, iterations=5)
    result = run_benchmarks(config)

Or from the CLI::

    kube benchmark --size 64 --iterations 5
"""

from __future__ import annotations

from kube.benchmark._runner import (
    BenchmarkConfig,
    BenchmarkResult,
    TimingResult,
    WorkloadResult,
    run_benchmarks,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "TimingResult",
    "WorkloadResult",
    "run_benchmarks",
]
