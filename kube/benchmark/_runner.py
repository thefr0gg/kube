"""Timing harness and top-level orchestration for the benchmark suite."""

from __future__ import annotations

import shutil
import statistics
import time
from dataclasses import dataclass
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
class ScenarioResult:
    """Single (scenario, backend) measurement."""

    scenario: str
    backend: str
    timing: TimingResult
    unit: str = "ops/s"
    notes: str = ""


@dataclass
class BenchmarkResult:
    """Full benchmark run output."""

    config: dict[str, Any]
    timestamp: str
    results: list[ScenarioResult]
    kube_version: str


@dataclass
class BenchmarkConfig:
    """Input parameters for one benchmark run."""

    size: int = 64
    states: int = 4
    scenarios: list[str] | None = None
    backends: list[str] | None = None
    iterations: int = 3
    seed: int = 1337
    tmp_dir: str | None = None
    plot: bool = False


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
    """Run the selected scenarios and return a :class:`BenchmarkResult`."""
    import importlib.metadata

    try:
        kube_version = importlib.metadata.version("kube")
    except importlib.metadata.PackageNotFoundError:
        kube_version = "0.0.0+local"

    from kube.benchmark._backends import build_fixtures
    from kube.benchmark._workloads import build_suite

    fixtures = build_fixtures(config.size, config.states, config.seed, tmp_dir=config.tmp_dir)
    tmp_root: Path = fixtures["tmp_root"]

    try:
        suite = build_suite(fixtures, config)

        want_scenarios = set(config.scenarios) if config.scenarios else None
        want_backends = set(config.backends) if config.backends else None

        results: list[ScenarioResult] = []
        for entry in suite:
            sname = entry["scenario"]
            bname = entry["backend"]
            if want_scenarios is not None and sname not in want_scenarios:
                continue
            if want_backends is not None and bname not in want_backends:
                continue
            fn = entry["fn"]
            n_ops = entry.get("ops", 0)
            timing = _measure(fn, config.iterations, n_ops)
            results.append(
                ScenarioResult(
                    scenario=sname,
                    backend=bname,
                    timing=timing,
                    unit=entry.get("unit", "ops/s"),
                    notes=entry.get("notes", ""),
                )
            )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    ts = datetime.now(timezone.utc).isoformat()
    cfg_dict = {
        "size": config.size,
        "states": config.states,
        "iterations": config.iterations,
        "seed": config.seed,
    }
    return BenchmarkResult(
        config=cfg_dict,
        timestamp=ts,
        results=results,
        kube_version=kube_version,
    )
