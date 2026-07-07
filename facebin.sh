#!/bin/sh
# Backward-compatible launcher. Deprecated: install the package
# (`pip install -e .[ml,ui]`) and run `facebin` directly.

FCBF=$(dirname "$(realpath "$0")")
ENV="$FCBF/env/bin"
[ -d "$FCBF/.venv/bin" ] && ENV="$FCBF/.venv/bin"

LOGDIR="$FCBF/logs/$(date +"%F-%H-%M-%S")"
mkdir -p "$LOGDIR"

exec "$ENV/python3" -m facebin run \
    1>"$LOGDIR/facebin.out" 2>"$LOGDIR/facebin.err"
