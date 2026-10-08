"""The capture/key workers must stop on the parent's SIGINT even when they
inherit a BLOCKED SIGINT/SIGTERM mask.

`cli._install_signal_quit` blocks SIGINT+SIGTERM process-wide so a `sigwait`
thread can quit the bar's Cocoa-owned main thread. A blocked signal mask is
inherited across fork+exec, and `signal.signal` sets only the disposition --
never the mask -- so every worker the bar spawns started with its stop signals
permanently pending. `record._harvest_fleet`'s SIGINT was then ignored until the
40s SIGKILL, and the whole take was discarded (it lost real recordings on
2026-08-31). Each worker now unblocks its own stop signals; these tests pin both
the MECHANISM (behaviourally, no macOS permissions) and that the workers keep
the call.

Permission-free: a throwaway python child, not a real SCStream.
"""
import os
import signal
import subprocess
import sys
import unittest

from autocine import record


# A child that mirrors the worker's stop wiring: install a SIGINT handler, then
# (only with --unblock) clear any inherited block, exactly as the workers do.
_CHILD = r"""
import signal, sys, time
stopped = {"v": False}
signal.signal(signal.SIGINT, lambda *a: stopped.__setitem__("v", True))
signal.signal(signal.SIGTERM, lambda *a: stopped.__setitem__("v", True))
if "--unblock" in sys.argv:
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
sys.stdout.write("READY\n"); sys.stdout.flush()
t0 = time.time()
while not stopped["v"] and time.time() - t0 < 10.0:
    time.sleep(0.02)
sys.stdout.write("STOPPED\n" if stopped["v"] else "TIMEOUT\n")
sys.stdout.flush()
"""


@unittest.skipUnless(hasattr(signal, "pthread_sigmask"),
                     "needs pthread_sigmask (POSIX)")
class InheritedBlockedSigint(unittest.TestCase):
    """Spawn the child with SIGINT+SIGTERM blocked in the parent -- the exact
    state the bar leaves -- and check whether a parent SIGINT stops it."""

    def _run_child(self, unblock):
        # Block on THIS (spawning) thread so the child inherits the block, then
        # restore the mask no matter what so no other test is affected.
        old = signal.pthread_sigmask(
            signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        args = [sys.executable, "-c", _CHILD]
        if unblock:
            args.append("--unblock")
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, start_new_session=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(), "READY")
            proc.send_signal(signal.SIGINT)
            try:
                # Generous vs the ~0.05s real response; still far under the
                # child's 10s self-timeout, so a hang here means SIGINT lost.
                rc = proc.wait(timeout=3.0)
                line = proc.stdout.readline().strip()
                return "responded", rc, line
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3.0)
                return "ignored", proc.returncode, None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=3.0)
            proc.stdout.close()
            signal.pthread_sigmask(signal.SIG_SETMASK, old)

    def test_unblock_makes_sigint_stop_the_child(self):
        outcome, rc, line = self._run_child(unblock=True)
        self.assertEqual(outcome, "responded")   # SIGINT was delivered
        self.assertEqual(line, "STOPPED")        # ...and ran the handler
        self.assertEqual(rc, 0)                  # clean exit, not SIGKILL

    def test_without_unblock_the_child_ignores_sigint(self):
        # The negative control: this is the bug. Proves the block really does
        # swallow SIGINT, so the positive test above is meaningful and not a
        # no-op.
        outcome, rc, _ = self._run_child(unblock=False)
        self.assertEqual(outcome, "ignored")
        self.assertEqual(rc, -signal.SIGKILL)    # only SIGKILL could stop it


class WorkersKeepTheUnblock(unittest.TestCase):
    """Source pins: the fix lives in standalone worker scripts (spawned as
    `python3 _*_worker.py`, not imported), so a plain behavioural import test
    cannot reach it. Pin that the call stays."""

    def _src(self, path):
        with open(path) as f:
            return f.read()

    def test_sck_worker_unblocks_stop_signals(self):
        src = self._src(record._SCK_WORKER_PATH)
        self.assertIn("SIG_UNBLOCK", src)
        self.assertIn("pthread_sigmask", src)

    def test_key_worker_unblocks_stop_signals(self):
        src = self._src(record._KEY_WORKER_PATH)
        self.assertIn("SIG_UNBLOCK", src)
        self.assertIn("pthread_sigmask", src)

    def test_watchdog_unblocks_stand_down_signals(self):
        # The watchdog stands down on the parent's SIGINT; an inherited block
        # would leave it polling stale pids for the bar's whole lifetime.
        src = self._src(record._WATCHDOG_PATH)
        self.assertIn("SIG_UNBLOCK", src)
        self.assertIn("pthread_sigmask", src)


if __name__ == "__main__":
    unittest.main()
