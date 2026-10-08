"""Standalone child process: kills the recorder's children if the parent
Python process disappears without cleaning up after itself.

WHY THIS EXISTS. `Recorder.start()` launches ffmpeg (and, on the non-shared
facecam path, a second ffmpeg, and the key-activity child) with
`start_new_session=True` -- deliberately, so a terminal Ctrl+C doesn't kill
ffmpeg mid-write and corrupt the file; the recorder stops it explicitly
(SIGINT, then wait for a clean trailer) in `_finalize()`. That's the right
design for a NORMAL stop. It has one real gap: if the PARENT process itself
dies WITHOUT running `_finalize()` -- a native crash (the SIGTRAP class of
bug `_key_worker.py` exists to contain, or any other), a SIGKILL, a hard
crash of any kind -- nothing ever tells ffmpeg to stop. `start_new_session`
means it isn't even in the parent's process group any more. It just keeps
recording, silently, for as long as the machine stays on. This was found the
hard way: after a run of crashes, NINE orphaned ffmpeg processes were still
running hours later, one for over six hours, collectively pinning real CPU
and writing gigabytes into (by then unlinked, invisible-to-`du`) disk space.
A background screen recorder nobody can see running is a resource problem
AND a privacy one -- it should not be possible for a crash to leave one
behind unnoticed.

This process's entire job: poll whether the parent PID is still alive; the
moment it isn't, stop every child PID it was told to watch (SIGINT, then
SIGKILL after a short grace period), then exit. It takes no other action and
depends on nothing from the parent after launch -- no pipes, no shared
state -- specifically so a parent that dies in an arbitrarily bad way still
leaves this behind to clean up. Deliberately NOT itself supervised by
anything else: it does one small thing and is cheap enough (~1 poll/second)
that the recursion stops here.

USAGE: `python3 _watchdog.py <parent_pid> <child_pid> [<child_pid> ...]`
Exits 0 on its own once EITHER the parent is confirmed gone and the children
have been signaled, OR it receives SIGINT/SIGTERM itself -- which is exactly
what `Recorder._finalize()` sends it on every NORMAL stop, so a successful
take leaves nothing behind either.
"""
import os
import signal
import sys
import time

POLL_SEC = 1.0
KILL_GRACE_SEC = 5.0


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists, just not ours to signal-probe further
    return True


def _stop_children(pids):
    living = [p for p in pids if _alive(p)]
    for pid in living:
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + KILL_GRACE_SEC
    while time.monotonic() < deadline and any(_alive(p) for p in living):
        time.sleep(0.2)
    for pid in living:
        if _alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main():
    if len(sys.argv) < 3:
        return
    parent_pid = int(sys.argv[1])
    child_pids = [int(a) for a in sys.argv[2:]]

    stood_down = []

    def _stand_down(_signum, _frame):
        # The parent stopped normally and told us to go away -- its own
        # _finalize() already owns shutting the children down cleanly.
        stood_down.append(True)

    signal.signal(signal.SIGINT, _stand_down)
    signal.signal(signal.SIGTERM, _stand_down)
    # Undo any inherited SIG_BLOCK on our stand-down signals. A parent that
    # blocks SIGINT/SIGTERM process-wide (cli._install_signal_quit, for the
    # bar's sigwait quit) leaks that blocked mask across fork+exec, and
    # `signal.signal` sets only the disposition -- so `stop_watchdog`'s SIGINT
    # would stay pending and we would never stand down. Every normal bar take
    # would then leave this poller lingering (watching now-stale pids) until
    # the bar itself exits -- exactly the orphaned-process problem this file
    # exists to prevent. Unblock so the handlers above can fire.
    if hasattr(signal, "pthread_sigmask"):
        try:
            signal.pthread_sigmask(
                signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
        except (ValueError, OSError):
            pass

    while not stood_down:
        if not _alive(parent_pid):
            _stop_children(child_pids)
            return
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
