"""Auto-restart supervisor for `studio app --reload`.

Watches Python and static-asset trees (mtime poll, no watchdog dep). On
change, SIGTERMs the child server subprocess and respawns it. The child
runs with AUTOCINE_LIVE=1 in its environment so the frontend's live-
reload poller knows the server is dev-mode; when the boot id changes, open
tabs auto-reload. Ctrl+C in the parent terminates the child and exits.

The supervisor deliberately never opens the browser -- the parent opens
exactly one tab at startup (see cli._cmd_app), so restarts don't clutter
Chrome. If the child dies on its own (e.g. import error), the supervisor
prints the exit code and waits for the next file change before respawning
-- avoids a hot restart loop when the code is broken.
"""

import os
import signal
import subprocess
import sys
import time


DEFAULT_PATTERNS = (".py", ".js", ".html", ".css")


def latest_mtime(dirs, patterns=DEFAULT_PATTERNS):
    """Highest mtime across matching files under any of `dirs`. Missing
    dirs/files are silently skipped. Returns 0.0 when nothing matches."""
    latest = 0.0
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, dnames, files in os.walk(d):
            # skip dotdirs and __pycache__ for speed + noise reduction
            dnames[:] = [x for x in dnames
                         if not x.startswith(".") and x != "__pycache__"]
            for f in files:
                if not any(f.endswith(p) for p in patterns):
                    continue
                try:
                    m = os.stat(os.path.join(root, f)).st_mtime
                except OSError:
                    continue
                if m > latest:
                    latest = m
    return latest


def _terminate(proc, timeout=5):
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=timeout)
        except Exception:
            pass


def run(argv, watch_dirs, patterns=DEFAULT_PATTERNS,
        poll_sec=0.3, coalesce_sec=0.25):
    """Fork `argv` as a subprocess, watch `watch_dirs` for changes,
    restart the child on change. Blocks until Ctrl+C. Returns the last
    child's exit code (0 on clean shutdown)."""
    env = dict(os.environ)
    env["AUTOCINE_LIVE"] = "1"
    print("[reload] watching {}".format(", ".join(watch_dirs)),
          file=sys.stderr)
    proc = subprocess.Popen(argv, env=env)
    baseline = latest_mtime(watch_dirs, patterns)
    last_restart = time.monotonic()
    try:
        while True:
            time.sleep(poll_sec)
            if proc.poll() is not None:
                code = proc.returncode
                print("[reload] child exited (code {}); waiting for a "
                      "file change before restarting...".format(code),
                      file=sys.stderr)
                while True:
                    time.sleep(poll_sec)
                    m = latest_mtime(watch_dirs, patterns)
                    if m > baseline:
                        baseline = m
                        print("[reload] change detected; restarting",
                              file=sys.stderr)
                        proc = subprocess.Popen(argv, env=env)
                        last_restart = time.monotonic()
                        break
                continue
            latest = latest_mtime(watch_dirs, patterns)
            now = time.monotonic()
            if latest > baseline and (now - last_restart) > coalesce_sec:
                print("[reload] change detected; restarting",
                      file=sys.stderr)
                _terminate(proc)
                proc = subprocess.Popen(argv, env=env)
                baseline = latest
                last_restart = time.monotonic()
    except KeyboardInterrupt:
        print("\n[reload] shutting down...", file=sys.stderr)
        _terminate(proc)
        return 0
