"""Unit tests for autocine/_watchdog.py's pure logic.

_watchdog.py is a standalone script (not part of the autocine package -- it
runs as its own process, deliberately with no import dependency on
anything that could be mid-crash in the parent), so it's loaded directly
from its file path rather than imported as autocine._watchdog.

The end-to-end behavior (kill the child once the parent is confirmed gone,
respecting the grace period; stand down without touching children on
SIGINT/SIGTERM) was verified against real processes during development --
that's not repeatable here without spawning real subprocesses in CI, so
this file pins the PURE decision functions (_alive, _stop_children) against
fakes, which is what actually varies across changes.
"""
import importlib.util
import os
import sys
import time
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(_HERE, "..", "autocine", "_watchdog.py")
_spec = importlib.util.spec_from_file_location("_watchdog_under_test", _PATH)
watchdog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(watchdog)


class AliveCheck(unittest.TestCase):
    def test_a_running_process_is_alive(self):
        # Our own PID always exists and we always have permission to probe it.
        self.assertTrue(watchdog._alive(os.getpid()))

    def test_a_nonexistent_pid_is_not_alive(self):
        with mock.patch.object(watchdog.os, "kill",
                               side_effect=ProcessLookupError):
            self.assertFalse(watchdog._alive(123456))

    def test_permission_denied_counts_as_alive(self):
        # The PID exists (a real process just isn't ours to signal) -- that
        # is NOT the same as gone, and must not be misread as "safe to
        # orphan-reap".
        with mock.patch.object(watchdog.os, "kill",
                               side_effect=PermissionError):
            self.assertTrue(watchdog._alive(1))


class StopChildren(unittest.TestCase):
    """_stop_children: SIGINT first, SIGKILL only after the grace period,
    and only for children that are actually still alive."""

    def test_sigint_then_returns_if_the_child_exits_promptly(self):
        alive = {42: True}
        sent = []
        def fake_alive(pid):
            return alive.get(pid, False)
        def fake_kill(pid, sig):
            sent.append((pid, sig))
            if sig == watchdog.signal.SIGINT:
                alive[pid] = False   # the child responds and exits
        with mock.patch.object(watchdog, "_alive", side_effect=fake_alive), \
             mock.patch.object(watchdog.os, "kill", side_effect=fake_kill), \
             mock.patch.object(watchdog.time, "sleep"):
            watchdog._stop_children([42])
        self.assertEqual(sent, [(42, watchdog.signal.SIGINT)])   # no SIGKILL needed

    def test_escalates_to_sigkill_after_the_grace_period(self):
        sent = []
        # Never responds to SIGINT -- stays "alive" forever from _alive's
        # perspective until the loop gives up and SIGKILLs it.
        with mock.patch.object(watchdog, "_alive", return_value=True), \
             mock.patch.object(watchdog.os, "kill",
                               side_effect=lambda pid, sig: sent.append((pid, sig))), \
             mock.patch.object(watchdog.time, "sleep"), \
             mock.patch.object(watchdog.time, "monotonic",
                               side_effect=[0.0, 0.0, 10.0]):
            # deadline = 0.0 + KILL_GRACE_SEC; the loop's own `while` check
            # reads monotonic() again and sees we're already past it.
            watchdog._stop_children([42])
        self.assertIn((42, watchdog.signal.SIGINT), sent)
        self.assertIn((42, watchdog.signal.SIGKILL), sent)

    def test_already_dead_children_are_never_signaled(self):
        sent = []
        with mock.patch.object(watchdog, "_alive", return_value=False), \
             mock.patch.object(watchdog.os, "kill",
                               side_effect=lambda pid, sig: sent.append((pid, sig))):
            watchdog._stop_children([42])
        self.assertEqual(sent, [])

    def test_a_pid_that_vanishes_mid_signal_does_not_raise(self):
        with mock.patch.object(watchdog, "_alive", return_value=True), \
             mock.patch.object(watchdog.os, "kill",
                               side_effect=ProcessLookupError), \
             mock.patch.object(watchdog.time, "sleep"), \
             mock.patch.object(watchdog.time, "monotonic",
                               side_effect=[0.0, 10.0]):
            watchdog._stop_children([42])   # must not raise

    def test_multiple_children_all_get_signaled(self):
        sent = []
        with mock.patch.object(watchdog, "_alive", return_value=False), \
             mock.patch.object(watchdog.os, "kill",
                               side_effect=lambda pid, sig: sent.append(pid)):
            watchdog._stop_children([1, 2, 3])
        # none "alive" -> none signaled at all; re-run with all alive+responsive
        sent2 = []
        alive = {1: True, 2: True, 3: True}
        def fake_kill(pid, sig):
            sent2.append(pid)
            alive[pid] = False
        with mock.patch.object(watchdog, "_alive", side_effect=lambda p: alive[p]), \
             mock.patch.object(watchdog.os, "kill", side_effect=fake_kill):
            watchdog._stop_children([1, 2, 3])
        self.assertEqual(sorted(sent2), [1, 2, 3])


class MainArgParsing(unittest.TestCase):
    def test_too_few_arguments_returns_without_doing_anything(self):
        with mock.patch.object(sys, "argv", ["_watchdog.py", "111"]), \
             mock.patch.object(watchdog, "_alive") as m:
            watchdog.main()   # must not raise or block
        m.assert_not_called()


if __name__ == "__main__":
    unittest.main()
