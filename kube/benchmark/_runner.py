"""Timing harness and top-level orchestration for the benchmark suite."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass
class TimingResult:
    """Timing samples for one benchmark case."""

    samples: list[float]
    checksum: int
    ops: int

    @property
    def best_s(self) -> float:
        """Fastest sample in seconds."""
        return min(self.samples)

    @property
    def mean_s(self) -> float:
        """Mean sample in seconds."""
        return statistics.mean(self.samples)

    @property
    def throughput(self) -> float:
        """ops/s based on the best sample (0 if best_s is 0)."""
        return self.ops / self.best_s if self.best_s > 0 else 0.0


@dataclass
class WorkloadResult:
    """Single (workload, backend) measurement."""

    workload: str
    backend: str
    timing: TimingResult
    size_label: str = ""
    notes: str = ""


@dataclass
class BenchmarkResult:
    """Full benchmark run output."""

    config: dict[str, Any]
    timestamp: str
    results: list[WorkloadResult]
    kube_version: str


@dataclass
class BenchmarkConfig:
    """Input parameters for one benchmark run."""

    size: int = 64
    states: int = 4
    workloads: list[str] | None = None
    backends: list[str] | None = None
    iterations: int = 5
    seed: int = 1337
    dataset_path: Path | None = None
    tmp_dir: str | None = None
    plot: bool = False


def _xorshift32(state: int) -> int:
    """Return the next deterministic pseudo-random 32-bit integer."""
    state &= 0xFFFF_FFFF
    state ^= (state << 13) & 0xFFFF_FFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFF_FFFF
    return state & 0xFFFF_FFFF


def build_random_ops(
    W: int,
    H: int,
    D: int,
    S: int,
    count: int,
    seed: int,
) -> list[tuple[int, int, int, int, int]]:
    """Build a deterministic list of (x, y, z, face, value) operations."""
    state = int(seed) or 1
    ops: list[tuple[int, int, int, int, int]] = []
    for _ in range(int(count)):
        state = _xorshift32(state)
        x = state % W
        state = _xorshift32(state)
        y = state % H
        state = _xorshift32(state)
        z = state % D
        state = _xorshift32(state)
        face = state % 6
        state = _xorshift32(state)
        value = state % S
        ops.append((x, y, z, face, value))
    return ops


def _warm_up(fn: Callable) -> None:
    """Run one un-timed warm-up iteration."""
    fn()


def _measure(fn: Callable, iterations: int, ops: int) -> TimingResult:
    """Time `fn` for `iterations` repetitions and return aggregated results."""
    _warm_up(fn)
    samples: list[float] = []
    checksum = 0
    for _ in range(iterations):
        t0 = time.perf_counter()
        checksum = int(fn())
        samples.append(time.perf_counter() - t0)
    return TimingResult(samples=samples, checksum=checksum, ops=ops)


def run_benchmarks(config: BenchmarkConfig) -> BenchmarkResult:
    """Run the selected workloads and return a :class:`BenchmarkResult`."""
    import importlib.metadata

    try:
        kube_version = importlib.metadata.version("kube")
    except importlib.metadata.PackageNotFoundError:
        kube_version = "0.0.0+local"

    from kube.benchmark._backends import build_fixtures
    from kube.benchmark._workloads import build_suite

    W = H = D = config.size
    S = config.states
    ops_list = build_random_ops(W, H, D, S, count=200_000, seed=config.seed)
    fixtures = build_fixtures(W, H, D, S, ops_list, config.seed, tmp_dir=config.tmp_dir)

    suite = build_suite(fixtures, config)

    want_workloads = set(config.workloads) if config.workloads else None
    want_backends = set(config.backends) if config.backends else None

    results: list[WorkloadResult] = []
    for entry in suite:
        wname = entry["workload"]
        bname = entry["backend"]
        if want_workloads is not None and wname not in want_workloads:
            continue
        if want_backends is not None and bname not in want_backends:
            continue
        fn = entry["fn"]
        n_ops = entry.get("ops", len(ops_list))
        timing = _measure(fn, config.iterations, n_ops)
        results.append(
            WorkloadResult(
                workload=wname,
                backend=bname,
                timing=timing,
                size_label=entry.get("size_label", ""),
                notes=entry.get("notes", ""),
            )
        )

    ts = datetime.now(timezone.utc).isoformat()
    cfg_dict = {
        "size": config.size,
        "states": config.states,
        "iterations": config.iterations,
        "seed": config.seed,
        "dataset_path": str(config.dataset_path) if config.dataset_path else None,
    }
    return BenchmarkResult(
        config=cfg_dict,
        timestamp=ts,
        results=results,
        kube_version=kube_version,
    )
