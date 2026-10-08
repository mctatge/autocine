#!/usr/bin/env python3
"""AutoCine — entry point.

Usage:
    python3 studio.py devices
    python3 studio.py record --display 2 --render
    python3 studio.py render recordings/<session>
    python3 studio.py mcp --print-config
"""

import sys

from autocine.cli import main

if __name__ == "__main__":
    sys.exit(main())
