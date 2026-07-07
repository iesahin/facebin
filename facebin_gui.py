#!/usr/bin/env python3
"""Backward-compatible wrapper for the desktop GUI.

Deprecated: use `facebin run` (server + GUI) or `facebin gui` instead.
The implementation moved to facebin/ui/main_window.py.
"""

import sys

from facebin.ui.main_window import main

if __name__ == '__main__':
    sys.exit(main())
