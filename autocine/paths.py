"""Portable paths shared by the CLI, Studio app, and MCP server.

The source checkout keeps recordings beside ``studio.py``.  That preserves the
existing developer workflow while allowing a clone to live anywhere instead of
assuming a particular username or ``~/Projects`` layout.  Packaged builds can
set ``AUTOCINE_RECORDINGS_ROOT`` to an Application Support or Movies location
without giving the three entry points separate defaults again.
"""

import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def recordings_root():
    """Return AutoCine's recordings directory.

    ``AUTOCINE_RECORDINGS_ROOT`` is intentionally the only override.  Expand
    ``~`` and normalize relative values so every surface reports and uses the
    same absolute path.
    """
    configured = os.environ.get("AUTOCINE_RECORDINGS_ROOT")
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return os.path.join(ROOT, "recordings")


def studio_entrypoint():
    """Absolute path to the source checkout's public CLI entry point."""
    return os.path.join(ROOT, "studio.py")
