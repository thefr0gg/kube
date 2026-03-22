#!/usr/bin/env sh
set -eu

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "python3 or python is required to run build.py" >&2
    exit 1
fi

mkdir -p build
"$PYTHON" build.py --verbose --log-file build/build.log "$@"
echo "Full log written to build/build.log"
