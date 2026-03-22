"""Native build helper for Kube's native libraries.

This script exists for two related reasons:

- developers can run it directly to configure and compile the native libraries
- Poetry can call it as a build hook when packaging the project

Keeping the native build logic here avoids burying CMake-specific behavior in
the package runtime code.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_BUILD_DIR = ROOT / "build"
MULTI_CONFIG_GENERATORS = {
    "Visual Studio 16 2019",
    "Visual Studio 17 2022",
    "Xcode",
}


def _bool_flag(value: bool) -> str:
    """Convert a Python boolean into the CMake `ON`/`OFF` convention."""
    return "ON" if value else "OFF"


def _run(command: list[str], *, cwd: Path, log_file: Path | None = None) -> None:
    """Run one subprocess and optionally tee output into a build log.

    This helper exists so configure/build commands behave the same whether the
    caller wants console-only output or a persisted log file.
    """
    print("+ " + " ".join(command))
    if log_file is None:
        subprocess.run(command, cwd=cwd, check=True)
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        return_code = process.wait()
        handle.write(f"[exit code: {return_code}]\n\n")
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def _cmake_executable() -> str:
    """Locate the `cmake` executable or fail with a clear message."""
    cmake = shutil.which("cmake")
    if cmake is None:
        raise SystemExit("cmake was not found on PATH")
    return cmake


def _default_parallel() -> int:
    """Choose a conservative default parallelism based on CPU count."""
    return max(1, (os.cpu_count() or 1))


def _cache_generator(build_dir: Path) -> str | None:
    """Read the configured CMake generator from an existing cache.

    This helper exists because a stale cache created with the wrong generator
    can break all future builds until the build directory is reset.
    """
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return None
    for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("CMAKE_GENERATOR:INTERNAL="):
            return line.partition("=")[2].strip() or None
    return None


def _cache_exists(build_dir: Path) -> bool:
    """Return whether CMake has already configured the selected build directory."""
    return (build_dir / "CMakeCache.txt").exists()


def _source_stamp(root: Path) -> float:
    """Compute the latest source mtime among files that affect native builds.

    The function exists so the build script can skip unnecessary reconfigure
    work when inputs have not changed.
    """
    paths = [
        root / "CMakeLists.txt",
        root / "build.py",
    ]
    paths.extend((root / "src").glob("*.c"))
    paths.extend((root / "src").glob("*.h"))
    return max((path.stat().st_mtime for path in paths if path.exists()), default=0.0)


def _configure_stamp(build_dir: Path) -> float:
    """Return the mtime of the CMake cache for the build directory."""
    cache = build_dir / "CMakeCache.txt"
    return cache.stat().st_mtime if cache.exists() else 0.0


def _should_configure(args: argparse.Namespace) -> bool:
    """Decide whether the CMake configure step should run again."""
    if args.cmake_only or args.reconfigure:
        return True
    if not _cache_exists(args.build_dir):
        return True
    return _configure_stamp(args.build_dir) < _source_stamp(ROOT)


def _has_msvc_tools() -> bool:
    """Return whether a usable MSVC command-line toolchain appears available."""
    return shutil.which("cl") is not None and shutil.which("nmake") is not None


def _resolve_windows_defaults(args: argparse.Namespace) -> None:
    """Pick a sane Windows generator/compiler default when the user omitted one.

    CMake can default to `NMake Makefiles` on Windows, which fails outside a
    Visual Studio developer shell. This helper exists to choose a working
    default based on the tools that are actually installed.
    """
    if os.name != "nt":
        return

    if args.generator is None:
        if shutil.which("ninja") is not None:
            args.generator = "Ninja"
        elif shutil.which("mingw32-make") is not None and shutil.which("gcc") is not None:
            args.generator = "MinGW Makefiles"
        elif _has_msvc_tools():
            args.generator = "NMake Makefiles"

    cmake_args = list(args.cmake_arg)
    generator = args.generator
    if generator in {"Ninja", "MinGW Makefiles"} and not _has_msvc_tools():
        gcc = shutil.which("gcc")
        gxx = shutil.which("g++")
        if gcc is not None and not any(arg.startswith("-DCMAKE_C_COMPILER=") for arg in cmake_args):
            cmake_args.append(f"-DCMAKE_C_COMPILER={gcc}")
        if gxx is not None and not any(arg.startswith("-DCMAKE_CXX_COMPILER=") for arg in cmake_args):
            cmake_args.append(f"-DCMAKE_CXX_COMPILER={gxx}")
    args.cmake_arg = cmake_args


def _reset_incompatible_cache(args: argparse.Namespace) -> None:
    """Delete the build directory when its cached generator conflicts.

    This exists because CMake caches the generator choice. If the cache says
    `NMake Makefiles` but the environment only has Ninja/MinGW tools, every
    rebuild will keep failing until the cache is removed.
    """
    cached_generator = _cache_generator(args.build_dir)
    if cached_generator is None:
        return
    requested_generator = args.generator
    if requested_generator is None:
        return
    if cached_generator == requested_generator:
        return
    print(
        f"+ resetting build directory because cached generator '{cached_generator}' "
        f"does not match requested generator '{requested_generator}'"
    )
    shutil.rmtree(args.build_dir)
    args.build_dir.mkdir(parents=True, exist_ok=True)


def _reset_incompatible_nested_caches(args: argparse.Namespace) -> None:
    """Delete nested CMake sub-builds that were configured with another generator.

    FetchContent dependencies such as raylib can create their own CMake cache
    directories under the main build tree. When the top-level build switches
    from `MinGW Makefiles` to `Ninja` (or the reverse), those nested caches can
    still fail configure even after the main cache is valid. This helper exists
    to clean only the stale nested sub-builds instead of forcing a full rebuild
    every time.
    """
    requested_generator = args.generator
    if requested_generator is None or not args.build_dir.exists():
        return

    for cache_path in args.build_dir.rglob("CMakeCache.txt"):
        if cache_path.parent == args.build_dir:
            continue
        cached_generator = None
        for line in cache_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("CMAKE_GENERATOR:INTERNAL="):
                cached_generator = line.partition("=")[2].strip() or None
                break
        if cached_generator is None or cached_generator == requested_generator:
            continue
        stale_dir = cache_path.parent
        print(
            f"+ resetting nested build directory because cached generator '{cached_generator}' "
            f"does not match requested generator '{requested_generator}': {stale_dir}"
        )
        shutil.rmtree(stale_dir)


def _configure_command(args: argparse.Namespace, cmake: str) -> list[str]:
    """Assemble the `cmake -S ... -B ...` configure command."""
    command = [
        cmake,
        "-S",
        str(ROOT),
        "-B",
        str(args.build_dir),
        f"-DKUBE_USE_SYSTEM_RAYLIB={_bool_flag(args.use_system_raylib)}",
    ]
    if args.generator:
        command.extend(["-G", args.generator])
    if args.config and args.generator not in MULTI_CONFIG_GENERATORS:
        command.append(f"-DCMAKE_BUILD_TYPE={args.config}")
    if args.verbose:
        command.append("--log-level=VERBOSE")
    if getattr(args, "raylib_root", None):
        command.append(f"-DKUBE_RAYLIB_ROOT={args.raylib_root}")
    command.extend(args.cmake_arg)
    return command


def _build_command(args: argparse.Namespace, cmake: str) -> list[str]:
    """Assemble the `cmake --build ...` command for the requested native target."""
    command = [cmake, "--build", str(args.build_dir)]
    if args.target:
        command.extend(["--target", args.target])
    if args.config:
        command.extend(["--config", args.config])
    if args.parallel:
        command.extend(["--parallel", str(args.parallel)])
    if args.verbose:
        command.append("--verbose")
    if args.build_arg:
        command.append("--")
        command.extend(args.build_arg)
    return command


def parse_args() -> argparse.Namespace:
    """Define the command-line interface for manual native builds."""
    parser = argparse.ArgumentParser(description="Configure and build Kube's native libraries")
    parser.add_argument("--generator", help='CMake generator, for example "MinGW Makefiles"')
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--config", default="Release", help="Build configuration")
    parser.add_argument("--target", default="kube_native", help="CMake target to build")
    parser.add_argument("--parallel", type=int, default=_default_parallel(), help="Parallel native build jobs")
    parser.add_argument("--clean", action="store_true", help="Delete the build directory first")
    parser.add_argument("--cmake-only", action="store_true", help="Stop after the configure step")
    parser.add_argument("--reconfigure", action="store_true", help="Force the CMake configure step")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose native build output")
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Write full configure/build logs to this file while still streaming them to the console",
    )
    parser.add_argument(
        "--use-system-raylib",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer an installed raylib package before FetchContent downloads it",
    )
    parser.add_argument(
        "--raylib-root",
        type=Path,
        help="path to an unpacked raylib folder containing include/ and lib/",
    )
    parser.add_argument(
        "--cmake-arg",
        action="append",
        default=[],
        help="Extra argument to pass to the CMake configure step",
    )
    parser.add_argument(
        "--build-arg",
        action="append",
        default=[],
        help="Extra argument to pass to the native build tool",
    )
    return parser.parse_args()


def build() -> None:
    """Poetry build hook for compiling Kube's native libraries.

    Poetry can invoke this module during `poetry install` and `poetry build`
    when `[tool.poetry.build] script = "build.py"` is configured in
    `pyproject.toml`.

    The hook intentionally uses conservative defaults so that a normal package
    install will attempt to build the editor without requiring CLI arguments.
    A few environment variables are supported for CI or local overrides:

    - `KUBE_BUILD_GENERATOR`
    - `KUBE_BUILD_DIR`
    - `KUBE_BUILD_CONFIG`
    - `KUBE_BUILD_TARGET`
    - `KUBE_BUILD_PARALLEL`
     - `KUBE_BUILD_VERBOSE`
     - `KUBE_USE_SYSTEM_RAYLIB`
     - `KUBE_RAYLIB_ROOT`
     - `KUBE_BUILD_CMAKE_ONLY`
     - `KUBE_BUILD_RECONFIGURE`
     """
    generator = os.getenv("KUBE_BUILD_GENERATOR")
    build_dir_value = os.getenv("KUBE_BUILD_DIR")
    config = os.getenv("KUBE_BUILD_CONFIG", "Release")
    target = os.getenv("KUBE_BUILD_TARGET", "kube_native")
    parallel_raw = os.getenv("KUBE_BUILD_PARALLEL")
    verbose = os.getenv("KUBE_BUILD_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}
    cmake_only = os.getenv("KUBE_BUILD_CMAKE_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}
    reconfigure = os.getenv("KUBE_BUILD_RECONFIGURE", "").strip().lower() in {"1", "true", "yes", "on"}
    use_system_raylib_raw = os.getenv("KUBE_USE_SYSTEM_RAYLIB", "1").strip().lower()
    use_system_raylib = use_system_raylib_raw not in {"0", "false", "no", "off"}
    raylib_root_value = os.getenv("KUBE_RAYLIB_ROOT")

    args = argparse.Namespace(
        generator=generator,
        build_dir=Path(build_dir_value).resolve() if build_dir_value else DEFAULT_BUILD_DIR.resolve(),
        config=config,
        target=target,
        parallel=int(parallel_raw) if parallel_raw else _default_parallel(),
        clean=False,
        cmake_only=cmake_only,
        reconfigure=reconfigure,
        verbose=verbose,
        log_file=None,
        use_system_raylib=use_system_raylib,
        raylib_root=Path(raylib_root_value).resolve() if raylib_root_value else None,
        cmake_arg=[],
        build_arg=[],
    )

    _resolve_windows_defaults(args)
    cmake = _cmake_executable()
    args.build_dir.mkdir(parents=True, exist_ok=True)
    _reset_incompatible_cache(args)
    _reset_incompatible_nested_caches(args)

    if _should_configure(args):
        _run(_configure_command(args, cmake), cwd=ROOT, log_file=None)
    if not args.cmake_only:
        _run(_build_command(args, cmake), cwd=ROOT, log_file=None)


def main() -> None:
    """Run the direct-command build flow used by developers and CI."""
    args = parse_args()
    cmake = _cmake_executable()
    if args.raylib_root is None:
        raylib_root_value = os.getenv("KUBE_RAYLIB_ROOT")
        if raylib_root_value:
            args.raylib_root = Path(raylib_root_value)
    args.build_dir = args.build_dir.resolve()
    if args.raylib_root is not None:
        args.raylib_root = args.raylib_root.resolve()
    if args.log_file is not None:
        args.log_file = args.log_file.resolve()

    if args.clean and args.build_dir.exists():
        shutil.rmtree(args.build_dir)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    _resolve_windows_defaults(args)
    _reset_incompatible_cache(args)
    _reset_incompatible_nested_caches(args)

    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        args.log_file.write_text("", encoding="utf-8")

    if _should_configure(args):
        _run(_configure_command(args, cmake), cwd=ROOT, log_file=args.log_file)
    else:
        print(f"+ skipping configure (up to date): {args.build_dir}")
        if args.log_file is not None:
            with args.log_file.open("a", encoding="utf-8") as handle:
                handle.write(f"+ skipping configure (up to date): {args.build_dir}\n")
    if args.cmake_only:
        return
    _run(_build_command(args, cmake), cwd=ROOT, log_file=args.log_file)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
