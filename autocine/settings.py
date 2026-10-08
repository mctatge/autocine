"""settings.py — the few preferences that must outlive a process.

Deliberately tiny, and deliberately NOT edits.json: that file describes one
recording and is compare-and-swapped between the web editor and MCP. This is
app-level state, it has no `rev`, and losing it costs a preference rather
than someone's edit.

`settings.json` sits beside `bar-pos.json` at the repo root and follows the
same rules as that file: unknown keys are ignored, any read failure is an
empty dict, and a write failure is swallowed. A preferences file must never
be able to stop someone recording.

Today it holds one key. `capture_backend` exists because the alternative was
an environment variable that has to be retyped on every launch, with nothing
in the UI to show which one is active — and the difference it makes (whether
the recording bar is burned into the video) is only discovered afterwards,
in the finished take.
"""

import json
import os
import threading

_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "settings.json"))

# Keys we will persist. Anything else is dropped on read AND on write, so a
# hand-edited or future-version file can't smuggle state into the app.
#
# `auto_add_windows` is here for the same reason `capture_backend` is: it
# changes what ends up in the recording, and the difference is only
# discoverable AFTERWARDS, in the finished take. It lived in bar.js memory
# only, so every bar relaunch silently reverted it -- a user who had turned it
# on got a take with none of the windows they expected and no way to tell why
# (reported 2026-08-31). Tri-state on purpose: absent means "never chosen",
# which the bar reads as its default-ON, and only an explicit uncheck writes
# False.
_ALLOWED = ("capture_backend", "auto_add_windows")

_lock = threading.Lock()


def path():
    return _PATH


def load(from_path=None):
    """Persisted settings as a dict. {} on any failure."""
    try:
        with open(from_path or _PATH) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if k in _ALLOWED}
    except Exception:
        return {}


def save(values, to_path=None):
    """Merge `values` into the file. Returns the new dict; never raises.

    Merge rather than replace so two callers writing different keys don't
    erase each other, and so a key this version doesn't know about survives
    a downgrade... which is exactly why `_ALLOWED` filters reads too: the
    file is a message from a possibly-different version of this app.
    """
    target = to_path or _PATH
    with _lock:
        current = load(target)
        for key, value in (values or {}).items():
            if key in _ALLOWED:
                current[key] = value
        try:
            with open(target, "w") as f:
                json.dump(current, f, indent=2)
        except Exception:
            pass
        return dict(current)


def get(key, default=None, from_path=None):
    return load(from_path).get(key, default)
