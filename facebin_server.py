#!/usr/bin/env python3
"""Backward-compatible wrapper for the headless server.

Deprecated: use ``facebin server`` (or ``python -m facebin server``)
instead.  Configuration now lives in ``facebin.toml``; see
docs/CONFIGURATION.md.
"""

import sys

from facebin.cli import main

if __name__ == '__main__':
    print("facebin_server.py is deprecated; use `facebin server` instead.",
          file=sys.stderr)
    sys.exit(main(["server"] + sys.argv[1:]))
