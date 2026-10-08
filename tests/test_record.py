"""Unit tests for Recorder construction logic that doesn't need ffmpeg/permissions."""

import contextlib
import inspect
import io
import json
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

from autocine import record
from autocine import devices
# Safe to import: pynput is imported inside main(), never at module scope.
from autocine import _key_worker


class FacecamCapture(unittest.TestCase):
    """The best-effort webcam capture: device detection, ffmpeg command
    construction, and the no-face default -- all without real hardware."""

    def test_find_camera_skips_screen_devices(self):
        devs = {"video": [(0, "FaceTime HD Camera"), (2, "Capture screen 0")],
                "audio": []}
        self.assertEqual(devices.find_camera_device(devs), 0)

    def test_find_camera_when_screen_is_listed_first(self):
        devs = {"video": [(1, "Capture screen 0"), (0, "Studio Camera")],
                "audio": []}
        self.assertEqual(devices.find_camera_device(devs), 0)

    def test_find_camera_none_when_only_screens(self):
        devs = {"video": [(2, "Capture screen 0")], "audio": []}
        self.assertIsNone(devices.find_camera_device(devs))

    def test_no_face_by_default(self):
        r = record.Recorder("/tmp/x", 0)
        self.assertIsNone(r.face_idx)
        self.assertIsNone(r._face_capture)

    def test_face_cmd_captures_webcam_to_face_mov(self):
        r = record.Recorder("/tmp/sess", 2, face_idx=0, face_fps=30)
        cmd = r._face_cmd()
        self.assertIn("avfoundation", cmd)
        # video-only input (no audio -- the mic rides the screen capture)
        self.assertIn("0:none", cmd)
        self.assertEqual(cmd[-1], r.face_path)
        self.assertTrue(r.face_path.endswith("face.mov"))
        # CFR + zerolatency so it shares the screen track's monotonic sync
        self.assertIn("zerolatency", cmd)
        i = cmd.index("-framerate")
        self.assertEqual(cmd[i + 1], "30")


class FacecamSerialization(unittest.TestCase):
    """The webcam capture must start ONLY after the screen session is up --
    concurrent avfoundation session setup kills the screen capture. These
    pin the two serialization primitives without ffmpeg/permissions."""

    class _FakeProc:
        def __init__(self, rc):
            self._rc = rc

        def poll(self):
            return self._rc

    def test_wait_returns_true_once_screen_has_first_frame(self):
        r = record.Recorder("/tmp/x", 1, face_idx=0)
        r._t0 = 123.0   # screen encoded a frame -> session is fully up
        self.assertTrue(r._wait_screen_capturing(timeout=1.0))

    def test_wait_times_out_when_screen_never_produces_a_frame(self):
        r = record.Recorder("/tmp/x", 1, face_idx=0)
        r.proc = self._FakeProc(None)   # alive but never emits _t0
        self.assertFalse(r._wait_screen_capturing(timeout=0.05))

    def test_wait_returns_false_when_screen_proc_already_died(self):
        r = record.Recorder("/tmp/x", 1, face_idx=0)
        r.proc = self._FakeProc(1)      # ffmpeg already exited
        self.assertFalse(r._wait_screen_capturing(timeout=1.0))

    def test_wait_returns_false_on_stop_request(self):
        r = record.Recorder("/tmp/x", 1, face_idx=0)
        r._stop_requested.set()
        self.assertFalse(r._wait_screen_capturing(timeout=1.0))

    def test_head_start_is_short_but_nonzero(self):
        # Zero would reintroduce the concurrent-session race that kills the
        # screen capture; a long delay would visibly stall the bubble.
        self.assertGreater(record.FACE_START_DELAY_SEC, 0.0)
        self.assertLessEqual(record.FACE_START_DELAY_SEC, 1.0)

    def test_wait_ends_early_and_does_not_burn_the_full_delay(self):
        # Once the screen confirms a frame the webcam starts immediately,
        # so the head start is a ceiling rather than a fixed sleep.
        r = record.Recorder("/tmp/x", 1, face_idx=0)
        r._t0 = 5.0
        t = time.monotonic()
        self.assertTrue(r._wait_screen_capturing(
            timeout=record.FACE_START_DELAY_SEC))
        self.assertLess(time.monotonic() - t, record.FACE_START_DELAY_SEC)

    def test_start_face_survives_a_launch_failure(self):
        r = record.Recorder("/tmp/x", 1, face_idx=0)
        with mock.patch.object(record.subprocess, "Popen",
                               side_effect=OSError("boom")):
            fp, fe = r._start_face()
        self.assertIsNone(fp)
        self.assertIsNone(fe)
        self.assertIsNone(r.face_proc)   # never abort the screen recording


class _FakeSimpleProc(object):
    """A minimal fake for plain subprocess.Popen handles this module only
    polls/signals (the watchdog process, or a stand-in ffmpeg/facecam proc)
    -- deliberately not _FakeKeyProc's readline-capable stdout, since the
    watchdog process's stdout is /dev/null in real use."""
    def __init__(self, pid, alive=True):
        self.pid = pid
        self.returncode = None if alive else 0
        self.signals = []

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)


class ProcessDeathWatchdog(unittest.TestCase):
    """Recorder._start_watchdog/_stop_watchdog: launching autocine/_watchdog.py
    to reap ffmpeg (and the key-activity child) if THIS process disappears
    without running _finalize() -- found necessary the hard way, as real
    orphaned ffmpeg processes still running hours after a crash. All
    exercised with a fake Popen; no real subprocess, no permissions."""

    def test_watches_the_screen_ffmpeg_pid_by_default(self):
        r = record.Recorder("/tmp/x", 1)
        r.proc = _FakeSimpleProc(pid=111)
        calls = []
        with mock.patch.object(
                record.subprocess, "Popen",
                side_effect=lambda cmd, **kw: calls.append(cmd) or _FakeSimpleProc(999)):
            r._start_watchdog(kb_listener=None)
        cmd = calls[0]
        self.assertEqual(cmd[0], record.sys.executable)
        self.assertTrue(cmd[1].endswith("_watchdog.py"))
        self.assertEqual(cmd[2], str(os.getpid()))
        self.assertIn("111", cmd)          # the screen ffmpeg pid
        self.assertIsNotNone(r._watchdog_proc)

    def test_also_watches_the_key_worker_and_facecam_pids_when_present(self):
        r = record.Recorder("/tmp/x", 1)
        r.proc = _FakeSimpleProc(pid=111)
        r.face_proc = _FakeSimpleProc(pid=333)
        kb = mock.Mock()
        kb.proc = _FakeSimpleProc(pid=222)
        calls = []
        with mock.patch.object(
                record.subprocess, "Popen",
                side_effect=lambda cmd, **kw: calls.append(cmd) or _FakeSimpleProc(999)):
            r._start_watchdog(kb_listener=kb)
        cmd = calls[0]
        self.assertIn("111", cmd)
        self.assertIn("222", cmd)
        self.assertIn("333", cmd)

    def test_no_face_proc_means_no_facecam_pid_in_the_command(self):
        r = record.Recorder("/tmp/x", 1)
        r.proc = _FakeSimpleProc(pid=111)
        calls = []
        with mock.patch.object(
                record.subprocess, "Popen",
                side_effect=lambda cmd, **kw: calls.append(cmd) or _FakeSimpleProc(999)):
            r._start_watchdog(kb_listener=None)
        # exactly [python, script, our_pid, screen_pid] -- nothing extra
        self.assertEqual(len(calls[0]), 4)

    def test_survives_a_launch_failure(self):
        r = record.Recorder("/tmp/x", 1)
        r.proc = _FakeSimpleProc(pid=111)
        with mock.patch.object(record.subprocess, "Popen",
                               side_effect=OSError("boom")):
            r._start_watchdog(kb_listener=None)   # must not raise
        self.assertIsNone(r._watchdog_proc)

    def test_stop_signals_a_live_watchdog(self):
        r = record.Recorder("/tmp/x", 1)
        proc = _FakeSimpleProc(pid=999, alive=True)
        r._watchdog_proc = proc
        r._stop_watchdog()
        self.assertEqual(proc.signals, [record.signal.SIGINT])
        self.assertIsNone(r._watchdog_proc)   # cleared either way

    def test_stop_is_a_noop_when_never_started(self):
        r = record.Recorder("/tmp/x", 1)
        r._stop_watchdog()   # must not raise
        self.assertIsNone(r._watchdog_proc)

    def test_stop_does_not_signal_an_already_dead_watchdog(self):
        r = record.Recorder("/tmp/x", 1)
        proc = _FakeSimpleProc(pid=999, alive=False)
        r._watchdog_proc = proc
        r._stop_watchdog()
        self.assertEqual(proc.signals, [])


class ScreenFailureDiagnosis(unittest.TestCase):
    """The avfoundation 'screen input could not be configured -> fell back to
    a camera' fingerprint gets a specific, actionable failure message."""

    DENIED_TAIL = (
        "[AVFoundation indev] Configuration of video device failed, "
        "falling back to default.\n"
        "[in#0] Selected pixel format (yuv420p) is not supported by the "
        "input device.\n[in#0] Supported pixel formats:\n  uyvy422\n  nv12\n")

    def test_signature_detects_screen_fallback(self):
        self.assertTrue(record.Recorder._screen_denied_signature(self.DENIED_TAIL))

    def test_signature_ignores_unrelated_errors(self):
        self.assertFalse(record.Recorder._screen_denied_signature(
            "Input/output error\nStream mapping failed"))
        self.assertFalse(record.Recorder._screen_denied_signature(""))

    def test_failure_message_names_permission_and_reboot(self):
        r = record.Recorder("/tmp/x", 2)
        r._stderr_tail.append(self.DENIED_TAIL)
        msg = r._failure_message(rc=1, raw_ok=False)
        low = msg.lower()
        self.assertIn("screen recording", low)
        self.assertIn("reboot", low)          # the wedged-TCC remedy
        self.assertIn("continuity camera", low)

    def test_generic_message_when_no_signature(self):
        r = record.Recorder("/tmp/x", 2)
        r._stderr_tail.append("some other ffmpeg failure")
        msg = r._failure_message(rc=1, raw_ok=False)
        self.assertIn("Common causes", msg)   # falls back to the generic list

    def test_signature_message_mentions_face_only_when_facecam_on(self):
        # Screen-only recording: no facecam note in the diagnosis.
        r = record.Recorder("/tmp/x", 2)
        r._stderr_tail.append(self.DENIED_TAIL)
        self.assertNotIn("--face", r._failure_message(rc=1, raw_ok=False))
        # --face recording: the diagnosis calls out the concurrent capture.
        rf = record.Recorder("/tmp/x", 2, face_idx=0)
        rf._stderr_tail.append(self.DENIED_TAIL)
        self.assertIn("--face", rf._failure_message(rc=1, raw_ok=False))


class CursorModeValidation(unittest.TestCase):
    def test_defaults_to_system(self):
        r = record.Recorder("/tmp/does-not-matter", 0)
        self.assertEqual(r.cursor_mode, "system")

    def test_synthetic_is_accepted(self):
        r = record.Recorder("/tmp/does-not-matter", 0, cursor_mode="synthetic")
        self.assertEqual(r.cursor_mode, "synthetic")

    def test_unknown_value_falls_back_to_system(self):
        r = record.Recorder("/tmp/does-not-matter", 0, cursor_mode="bogus")
        self.assertEqual(r.cursor_mode, "system")


class KeyLogging(unittest.TestCase):
    """The privacy contract of _on_key: bare quantized activity ticks, one
    per bucket, never any key identity -- exercised without a listener."""

    def _recorder_with_open_events(self, tmpdir, **kw):
        r = record.Recorder(tmpdir, 0, **kw)
        r._ev_file = open(r.events_path, "w")
        return r

    def _lines(self, r):
        r._ev_file.close()
        r._ev_file = None
        with open(r.events_path) as f:
            return [json.loads(l) for l in f if l.strip()]

    def test_log_keys_defaults_on_and_flag_disables(self):
        self.assertTrue(record.Recorder("/tmp/x", 0).log_keys)
        self.assertFalse(record.Recorder("/tmp/x", 0, log_keys=False).log_keys)

    def test_on_key_writes_bare_quantized_ticks_and_coalesces(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._recorder_with_open_events(td)
            with mock.patch("autocine.record.time.monotonic",
                            side_effect=[10.01, 10.04, 10.12]):
                r._on_key(object())   # bucket 100
                r._on_key(object())   # same bucket -> coalesced away
                r._last_xy = (123.0, 456.0)
                r._on_key(object())   # bucket 101
            lines = self._lines(r)
            self.assertEqual(len(lines), 2)
            for e in lines:
                # exactly these keys -- nothing that could identify a key
                self.assertEqual(sorted(e.keys()), ["t", "type", "x", "y"])
                self.assertEqual(e["type"], "key")
                # quantized to the bucket grid
                self.assertAlmostEqual(
                    e["t"] / record.KEY_BUCKET_SEC,
                    round(e["t"] / record.KEY_BUCKET_SEC), places=6)
            self.assertAlmostEqual(lines[0]["t"], 10.0, places=6)
            self.assertAlmostEqual(lines[1]["t"], 10.1, places=6)
            # x/y padding: 0.0 before any move, then the last cursor pos
            self.assertEqual((lines[0]["x"], lines[0]["y"]), (0.0, 0.0))
            self.assertEqual((lines[1]["x"], lines[1]["y"]), (123.0, 456.0))

    def test_on_key_never_serializes_the_key_object(self):
        # A key whose repr/str would leak must never appear in the file.
        class LoudKey(object):
            char = "s"
            def __repr__(self):
                return "SECRET-KEY-REPR"
            def __str__(self):
                return "s"
        with tempfile.TemporaryDirectory() as td:
            r = self._recorder_with_open_events(td)
            with mock.patch("autocine.record.time.monotonic", return_value=20.05):
                r._on_key(LoudKey())
            r._ev_file.close()
            r._ev_file = None
            with open(r.events_path) as f:
                raw = f.read()
            self.assertNotIn("SECRET", raw)
            self.assertNotIn('"s"', raw)

    def test_on_key_noop_when_events_file_closed(self):
        with tempfile.TemporaryDirectory() as td:
            r = record.Recorder(td, 0)
            with mock.patch("autocine.record.time.monotonic", return_value=30.0):
                r._on_key(object())   # must not raise, nothing to write to
            self.assertFalse(os.path.exists(r.events_path))


class ScrollLogging(unittest.TestCase):
    """_on_scroll: rate-limited activity ticks at the real cursor position,
    real timestamps (no grid quantization -- scrolls have no keystroke-privacy
    concern), deltas never stored -- exercised without a listener."""

    def _recorder_with_open_events(self, tmpdir, **kw):
        r = record.Recorder(tmpdir, 0, **kw)
        r._ev_file = open(r.events_path, "w")
        return r

    def _lines(self, r):
        r._ev_file.close()
        r._ev_file = None
        with open(r.events_path) as f:
            return [json.loads(l) for l in f if l.strip()]

    def test_on_scroll_writes_rate_limited_position_ticks(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._recorder_with_open_events(td)
            # 4 scrolls in 90ms, then one 110ms later: only the 1st and last
            # survive the SCROLL_MIN_INTERVAL_SEC rate limit. Each _on_scroll
            # that passes the limit calls monotonic twice (limiter + write).
            with mock.patch("autocine.record.time.monotonic",
                            side_effect=[10.00, 10.00,   # logged
                                         10.03, 10.06, 10.09,  # suppressed
                                         10.11, 10.11]):  # logged
                r._on_scroll(100.0, 200.0, 0, -2)
                r._on_scroll(101.0, 201.0, 0, -2)
                r._on_scroll(102.0, 202.0, 0, -2)
                r._on_scroll(103.0, 203.0, 0, -2)
                r._on_scroll(104.0, 204.0, 0, -3)
            lines = self._lines(r)
            self.assertEqual(len(lines), 2)
            for e in lines:
                # exactly these keys -- in particular no dx/dy delta payload
                self.assertEqual(sorted(e.keys()), ["t", "type", "x", "y"])
                self.assertEqual(e["type"], "scroll")
            # real cursor positions, real (unquantized) timestamps
            self.assertEqual((lines[0]["x"], lines[0]["y"]), (100.0, 200.0))
            self.assertEqual((lines[1]["x"], lines[1]["y"]), (104.0, 204.0))
            self.assertAlmostEqual(lines[0]["t"], 10.00, places=6)
            self.assertAlmostEqual(lines[1]["t"], 10.11, places=6)

    def test_zero_delta_scroll_events_are_ignored(self):
        # macOS trackpad gesture phases can emit dx == dy == 0 bookkeeping
        # events; they are not scroll activity.
        with tempfile.TemporaryDirectory() as td:
            r = self._recorder_with_open_events(td)
            with mock.patch("autocine.record.time.monotonic", return_value=10.0):
                r._on_scroll(100.0, 200.0, 0, 0)
            lines = self._lines(r)
            self.assertEqual(lines, [])

    def test_on_scroll_noop_when_events_file_closed(self):
        with tempfile.TemporaryDirectory() as td:
            r = record.Recorder(td, 0)
            with mock.patch("autocine.record.time.monotonic", return_value=30.0):
                r._on_scroll(1.0, 2.0, 0, -1)   # must not raise
            self.assertFalse(os.path.exists(r.events_path))

    def test_mouse_listener_registers_all_callbacks(self):
        # A correct-but-unwired _on_scroll logs nothing: pin the actual
        # Listener wiring (mutation-proven gap: dropping on_scroll from
        # the construction survived the old suite).
        r = record.Recorder("/tmp/does-not-matter", 0)
        kw = r._mouse_listener_kwargs()
        self.assertEqual(kw, {"on_move": r._on_move,
                              "on_click": r._on_click,
                              "on_scroll": r._on_scroll})


class _FakeListener(object):
    def __init__(self, alive, raise_on_start=False):
        self._alive = alive
        self._raise = raise_on_start
    def start(self):
        if self._raise:
            raise RuntimeError("boom")
    def wait(self):
        pass
    def is_alive(self):
        return self._alive


class _FakeKeyboard(object):
    def __init__(self, alive, raise_on_start=False):
        self._alive = alive
        self._raise = raise_on_start
    def Listener(self, on_press=None):
        return _FakeListener(self._alive, self._raise)


class _FakeKeyProc(object):
    """A fake subprocess.Popen for _KeyWorker: readline()s from a canned
    string (mirrors exactly what _key_worker.py would have written) and
    implements enough of Popen's surface for stop()."""
    def __init__(self, output=""):
        self.stdout = io.StringIO(output)
        self.returncode = None
        self.killed = False
        self.signal_sent = None

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.signal_sent = sig
        self.returncode = -sig

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


class KeyWorkerSubprocessIsolation(unittest.TestCase):
    """The keyboard listener is now a disposable CHILD PROCESS
    (_key_worker.py), not an in-process pynput.keyboard.Listener -- see
    record.py's _start_key_listener docstring for why (a real, reproducible
    native SIGTRAP crash in pynput's macOS TIS decoding, confirmed 6/6 on
    real hardware, that no try/except can survive). All exercised with a
    fake Popen; no subprocess, no pynput, no permissions."""

    def _ticks(self, r):
        out = []
        r._on_key = lambda _key, t=None: out.append(t)
        return out

    def test_ready_reports_activity_and_launches_the_worker_script(self):
        r = record.Recorder("/tmp/x", 0)
        calls = []
        def fake_popen(cmd, **kw):
            calls.append((cmd, kw))
            return _FakeKeyProc("READY\n")
        worker, state = r._start_key_listener(popen=fake_popen)
        self.assertEqual(state, "activity")
        self.assertIsNotNone(worker)
        # the exact child: this interpreter, running _key_worker.py
        cmd = calls[0][0]
        self.assertEqual(cmd[0], record.sys.executable)
        self.assertTrue(cmd[1].endswith("_key_worker.py"))
        worker.stop()

    def test_silent_exit_reports_failed(self):
        # The macOS Input Monitoring denial path: the child's own tap check
        # fails, it exits with NO output at all (see _key_worker.py).
        r = record.Recorder("/tmp/x", 0)
        worker, state = r._start_key_listener(
            popen=lambda *a, **k: _FakeKeyProc(""))
        self.assertEqual(state, "failed")
        self.assertIsNone(worker)

    def test_launch_failure_reports_failed(self):
        r = record.Recorder("/tmp/x", 0)
        def raising_popen(*a, **k):
            raise OSError("boom")
        worker, state = r._start_key_listener(popen=raising_popen)
        self.assertEqual(state, "failed")
        self.assertIsNone(worker)

    def test_a_child_offering_no_clock_passes_ticks_straight_through(self):
        # A bare "READY" -- an older _key_worker.py, or any caller that
        # doesn't offer a clock -- leaves the pairing offset at 0.0, which is
        # exactly the pre-pairing behavior, kept bit-for-bit.
        r = record.Recorder("/tmp/x", 0)
        ticks = self._ticks(r)
        worker, state = r._start_key_listener(
            popen=lambda *a, **k: _FakeKeyProc("READY\n12.5\n13.75\n"))
        self.assertEqual(state, "activity")
        worker._handshake_thread.join(timeout=2.0)
        self.assertEqual(ticks, [12.5, 13.75])

    def test_ready_clock_pairing_translates_ticks_into_our_timeframe(self):
        # time.monotonic()'s reference point is PER-PROCESS on this stack
        # (measured 2026-07-30: a child 0.5 s into a parent's life read
        # 0.0058 against the parent's 0.55), so a child's raw reading is not
        # in our timeframe. Untranslated, every tick lands one whole
        # parent-process-age in the past -- and the parent is usually the
        # long-lived `studio.py bar`, so they fall clean off the front of the
        # take. READY carries the child's own clock so we can pair them.
        r = record.Recorder("/tmp/x", 0)
        ticks = self._ticks(r)
        before = time.monotonic()
        worker, state = r._start_key_listener(
            popen=lambda *a, **k: _FakeKeyProc("READY 0.0\n0.25\n"))
        worker._handshake_thread.join(timeout=2.0)
        after = time.monotonic()
        self.assertEqual(state, "activity")
        # The child sent 0.25 with its clock reading 0.0 at READY, so the
        # press belongs 0.25 s after the instant we read READY -- an instant
        # that necessarily lies within [before, after]. Bounded on both
        # sides, so this is exact rather than a tolerance.
        self.assertEqual(len(ticks), 1)
        self.assertGreaterEqual(ticks[0], before + 0.25)
        self.assertLessEqual(ticks[0], after + 0.25)

    def test_a_malformed_ready_clock_still_starts_the_worker(self):
        # Mistimed key ticks are worth far more than no key capture at all,
        # so an unparseable clock degrades to the pass-through offset rather
        # than failing the handshake.
        r = record.Recorder("/tmp/x", 0)
        ticks = self._ticks(r)
        worker, state = r._start_key_listener(
            popen=lambda *a, **k: _FakeKeyProc("READY not-a-float\n7.0\n"))
        self.assertEqual(state, "activity")
        worker._handshake_thread.join(timeout=2.0)
        self.assertEqual(ticks, [7.0])

    def test_the_child_really_sends_its_clock_on_the_ready_line(self):
        # The two halves of this handshake live in different files, and a
        # mismatch is SILENT: the parent would just see no clock, fall back
        # to a 0.0 offset, and reintroduce the exact bug the pairing exists
        # to fix. So pin the child's half of the contract too.
        src = inspect.getsource(_key_worker)
        self.assertIn('"READY {}\\n".format(time.monotonic())', src)

    def test_malformed_lines_are_skipped_not_fatal(self):
        r = record.Recorder("/tmp/x", 0)
        ticks = self._ticks(r)
        worker, state = r._start_key_listener(
            popen=lambda *a, **k: _FakeKeyProc("READY\nnot-a-float\n5.0\n"))
        self.assertEqual(state, "activity")
        worker._handshake_thread.join(timeout=2.0)
        self.assertEqual(ticks, [5.0])

    def test_stop_signals_and_reaps_the_child(self):
        r = record.Recorder("/tmp/x", 0)
        worker, _ = r._start_key_listener(
            popen=lambda *a, **k: _FakeKeyProc("READY\n"))
        proc = worker.proc
        worker.stop()
        self.assertEqual(proc.signal_sent, record.signal.SIGINT)

    def test_on_key_uses_live_clock_when_no_child_timestamp_is_given(self):
        # Every existing caller (tests, and the pre-child-process code path)
        # passes no `t` -- must be unaffected by the new optional parameter.
        with tempfile.TemporaryDirectory() as td:
            r = record.Recorder(td, 0)
            r._ev_file = open(r.events_path, "w")
            with mock.patch("autocine.record.time.monotonic", return_value=40.0):
                r._on_key(object())
            r._ev_file.close()
            with open(r.events_path) as f:
                line = json.loads(f.readline())
            self.assertAlmostEqual(line["t"], 40.0, places=6)


class CaptureWindowMeta(unittest.TestCase):
    """Record-time window capture. avfoundation can't target a window, so the
    Recorder captures the whole display as always and only WRITES the window's
    rect (points, global top-left origin) into meta.json for render.py to crop
    to. All exercised without ffmpeg, Quartz or permissions."""

    # A devices.window_rect_points entry as the picker hands it over.
    PICK = {"id": 9856, "app": "Google Chrome", "title": "Docs — Chrome",
            "label": "Google Chrome — Docs",
            "x": 100.0, "y": 60.0, "w": 900.0, "h": 600.0,
            "display_id": 1, "display_origin": [0.0, 0.0],
            "main_display": True}

    def _meta(self, r):
        return r._meta_dict(1440.0, 900.0, "quartz", 1234.5, False)

    @staticmethod
    def _quiet(fn, *args):
        """Run a step that may print for the user and return what it said."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(*args)
        return buf.getvalue()

    # -- the "existing sessions are unchanged" pin --------------------------
    def test_normal_session_meta_is_exactly_what_it_was(self):
        # Dict equality, not a subset check: a session recorded without a
        # capture window must produce the pre-feature meta.json verbatim --
        # no capture_window key, nothing renamed, nothing added.
        r = record.Recorder("/tmp/sess", 2)
        self.assertEqual(self._meta(r), {
            "fps": 60,
            "logical_w": 1440.0, "logical_h": 900.0, "geom_source": "quartz",
            "t0_monotonic": 1234.5,
            "video_index": 2, "mic_index": None,
            "raw": "raw.mov", "events": "events.jsonl",
            "cursor_mode": "system", "key_capture": None,
            "face": None, "face_index": None, "face_fps": None,
            "face_t0_monotonic": None, "face_capture": None,
        })

    def test_no_window_means_no_quartz_calls_at_all(self):
        r = record.Recorder("/tmp/sess", 2)
        with mock.patch.object(record.dev, "window_rect_points") as m:
            r._resolve_capture_window()
            r._finalize()
        self.assertFalse(m.called)
        self.assertNotIn("capture_window", self._meta(r))

    def test_unusable_capture_window_omits_the_key(self):
        # Fail safe: a malformed entry renders full-frame rather than cropping
        # to garbage. (Non-dict is rejected outright at construction.)
        self.assertIsNone(record.Recorder("/tmp/s", 2, capture_window=7).capture_window)
        self.assertNotIn("capture_window",
                         self._meta(record.Recorder("/tmp/s", 2, capture_window=7)))
        broken = record.Recorder("/tmp/s", 2, capture_window={"id": 5, "x": 1.0})
        self.assertNotIn("capture_window", self._meta(broken))

    # -- the meta.json contract --------------------------------------------
    def test_meta_block_has_the_documented_shape_and_units(self):
        r = record.Recorder("/tmp/sess", 2, capture_window=self.PICK)
        fresh = dict(self.PICK, x=220.0, y=140.0, title="Docs — moved")
        with mock.patch.object(record.dev, "window_rect_points",
                               return_value=fresh):
            r._resolve_capture_window()
        meta = self._meta(r)
        # exactly one new top-level key vs. a normal session
        self.assertEqual(set(meta) - set(self._meta(record.Recorder("/tmp/sess", 2))),
                         {"capture_window"})
        cw = meta["capture_window"]
        self.assertEqual(sorted(cw), ["app", "display_origin", "end_rect", "id",
                                      "rect", "resnapshot", "source", "title",
                                      "track", "units"])
        self.assertEqual(cw["id"], 9856)
        self.assertEqual(cw["app"], "Google Chrome")
        self.assertEqual(cw["title"], "Docs — moved")
        self.assertEqual(cw["units"], "points")     # POINTS, never pixels
        self.assertEqual(cw["source"], "quartz")
        # the post-countdown rect, not the pick-time one
        self.assertEqual(cw["rect"], [220.0, 140.0, 900.0, 600.0])
        self.assertEqual(cw["display_origin"], [0.0, 0.0])
        self.assertIs(cw["resnapshot"], True)
        self.assertIsNone(cw["end_rect"])
        # it goes straight into meta.json, so it must be JSON round-trippable
        self.assertEqual(json.loads(json.dumps(meta)), meta)

    def test_rect_is_floats_even_from_an_int_entry(self):
        r = record.Recorder("/tmp/sess", 2, capture_window=self.PICK)
        ints = dict(self.PICK, x=0, y=25, w=800, h=500, display_origin=[0, 0])
        with mock.patch.object(record.dev, "window_rect_points",
                               return_value=ints):
            r._resolve_capture_window()
        cw = self._meta(r)["capture_window"]
        self.assertEqual(cw["rect"], [0.0, 25.0, 800.0, 500.0])
        for v in cw["rect"] + cw["display_origin"]:
            self.assertIsInstance(v, float)

    # -- the post-countdown re-snapshot ------------------------------------
    def test_snapshot_asks_quartz_for_the_picked_window(self):
        r = record.Recorder("/tmp/sess", 2, capture_window=self.PICK)
        with mock.patch.object(record.dev, "window_rect_points",
                               return_value=None) as m:
            r._snapshot_capture_window()
        self.assertEqual(m.call_args[0][0], 9856)
        # our own windows (the bar) must never shadow the pick
        self.assertIn(os.getpid(), tuple(m.call_args[1]["exclude_pids"]))

    def test_resnapshot_happens_after_the_countdown_just_before_ffmpeg(self):
        # Wiring + ORDER pin: the countdown is exactly when the user brings
        # the window forward, so a pick-time-only rect would make the
        # countdown useless. Order is what makes the feature work and it
        # isn't observable without ffmpeg + permissions.
        src = inspect.getsource(record.Recorder.start)
        self.assertLess(src.index("range(countdown"),
                        src.index("_resolve_capture_window()"))
        self.assertLess(src.index("_resolve_capture_window()"),
                        src.index("subprocess.Popen("))

    def test_failed_resnapshot_falls_back_to_the_pick_time_rect(self):
        # Window minimized/gone/on another display at start: keep the rect we
        # already had and say so. Never abort a take that's already counting.
        r = record.Recorder("/tmp/sess", 2, capture_window=self.PICK)
        with mock.patch.object(record.dev, "window_rect_points",
                               return_value=None):
            said = self._quiet(r._resolve_capture_window)
        cw = self._meta(r)["capture_window"]
        self.assertEqual(cw["rect"], [100.0, 60.0, 900.0, 600.0])
        self.assertIs(cw["resnapshot"], False)
        self.assertEqual(cw["title"], "Docs — Chrome")
        self.assertIn("picked", said)     # and the user is told why

    def test_resnapshot_never_raises_when_quartz_blows_up(self):
        r = record.Recorder("/tmp/sess", 2, capture_window=self.PICK)
        with mock.patch.object(record.dev, "window_rect_points",
                               side_effect=RuntimeError("no Quartz here")):
            self._quiet(r._resolve_capture_window)   # must not propagate
        cw = self._meta(r)["capture_window"]
        self.assertIs(cw["resnapshot"], False)
        self.assertEqual(cw["rect"], [100.0, 60.0, 900.0, 600.0])

    # -- end_rect (the drift warning's input) ------------------------------
    def test_end_rect_records_a_window_that_moved_during_the_take(self):
        r = record.Recorder("/tmp/sess", 2, capture_window=self.PICK)
        with mock.patch.object(record.dev, "window_rect_points",
                               return_value=self.PICK):
            r._resolve_capture_window()
        moved = dict(self.PICK, x=300.0, y=180.0)
        with mock.patch.object(record.dev, "window_rect_points",
                               return_value=moved):
            r._finalize()          # no proc, no events file: pure stop path
        cw = self._meta(r)["capture_window"]
        self.assertEqual(cw["rect"], [100.0, 60.0, 900.0, 600.0])
        self.assertEqual(cw["end_rect"], [300.0, 180.0, 900.0, 600.0])

    def test_end_rect_is_none_when_the_window_is_gone_at_stop(self):
        r = record.Recorder("/tmp/sess", 2, capture_window=self.PICK)
        with mock.patch.object(record.dev, "window_rect_points",
                               return_value=None):
            self._quiet(r._resolve_capture_window)
            r._finalize()
        self.assertIsNone(self._meta(r)["capture_window"]["end_rect"])

    def test_finalize_survives_a_quartz_failure_at_stop(self):
        r = record.Recorder("/tmp/sess", 2, capture_window=self.PICK)
        with mock.patch.object(record.dev, "window_rect_points",
                               side_effect=RuntimeError("boom")):
            r._finalize()          # a geometry read must never break the stop
        self.assertIsNone(self._meta(r)["capture_window"]["end_rect"])


class WindowNativeCapture(unittest.TestCase):
    """Occlusion-free capture: the SCK worker records the window's OWN buffer,
    so meta marks the take `mode: window_native` and render must NOT crop. The
    OFF state -- a display-crop capture_window take, SCK or avfoundation -- must
    stay byte-identical. All pure: no ObjC, no permissions. See
    docs/architecture.md."""

    # A top-left window (0,25 720x875 pts) -- the P0 spike's real geometry.
    PICK = {"id": 4242, "app": "Safari", "title": "Docs", "label": "Safari — Docs",
            "x": 0.0, "y": 25.0, "w": 720.0, "h": 875.0,
            "display_id": 1, "display_origin": [0.0, 0.0], "main_display": True}

    def _meta(self, r):
        return r._meta_dict(1440.0, 900.0, "quartz", 1234.5, False)

    def _native(self, **kw):
        return record.Recorder("/tmp/sess", 2, capture_window=self.PICK,
                               backend="sck", window_native=True, **kw)

    # -- the single fail-safe guard ----------------------------------------
    def test_guard_requires_sck_a_window_and_the_flag(self):
        self.assertTrue(self._native()._is_window_native())
        # avfoundation can't capture a window's own buffer, even if asked
        self.assertFalse(record.Recorder(
            "/tmp/s", 2, capture_window=self.PICK, backend="avfoundation",
            window_native=True)._is_window_native())
        # no window -> nothing to capture natively
        self.assertFalse(record.Recorder(
            "/tmp/s", 2, backend="sck", window_native=True)._is_window_native())
        # flag off -> the display-crop path, even on SCK with a window
        self.assertFalse(record.Recorder(
            "/tmp/s", 2, capture_window=self.PICK, backend="sck")._is_window_native())

    # -- the OFF switch: a display-crop take is byte-identical --------------
    def test_display_crop_sck_take_gains_no_window_native_keys(self):
        r = record.Recorder("/tmp/sess", 2, capture_window=self.PICK, backend="sck")
        cw = self._meta(r)["capture_window"]
        for k in ("mode", "logical_w", "logical_h", "buffer_w", "buffer_h"):
            self.assertNotIn(k, cw)

    def test_display_crop_config_has_no_capture_window_id(self):
        r = record.Recorder("/tmp/sess", 2, capture_window=self.PICK, backend="sck")
        self.assertNotIn("capture_window_id", r._sck_config())

    # -- the window-native meta marker -------------------------------------
    def test_native_meta_marks_mode_and_the_window_point_size(self):
        r = self._native()
        meta = self._meta(r)
        cw = meta["capture_window"]
        self.assertEqual(cw["mode"], "window_native")
        # the WINDOW's own point size (the event-mapping denominator)...
        self.assertEqual(cw["logical_w"], 720.0)
        self.assertEqual(cw["logical_h"], 875.0)
        # ...distinct from the top-level logical_w/h, which stay the display's
        self.assertEqual(meta["logical_w"], 1440.0)
        self.assertEqual(meta["logical_h"], 900.0)
        # goes straight to disk, so it must round-trip through JSON
        self.assertEqual(json.loads(json.dumps(meta)), meta)

    def test_native_meta_omits_buffer_dims_until_the_worker_reports_size(self):
        cw = self._meta(self._native())["capture_window"]
        self.assertNotIn("buffer_w", cw)
        self.assertNotIn("buffer_h", cw)

    def test_native_meta_records_the_encoded_buffer_when_size_arrives(self):
        r = self._native()
        r._sck_size = {"width": 1440, "height": 1750, "fps": 60}
        cw = self._meta(r)["capture_window"]
        self.assertEqual(cw["buffer_w"], 1440)
        self.assertEqual(cw["buffer_h"], 1750)

    # -- the worker config -------------------------------------------------
    def test_native_config_carries_the_target_window_id(self):
        self.assertEqual(self._native()._sck_config()["capture_window_id"], 4242)

    # -- the SIZE line is consumed (and only meta-gated, not dropped) -------
    def test_read_sck_stdout_stores_the_size_line(self):
        r = self._native()
        r.proc = mock.Mock()
        r.proc.stdout = io.BytesIO(b"SIZE 1440 1750 60\nT0 5.0\n")
        r._read_sck_stdout()
        self.assertEqual(r._sck_size["width"], 1440)
        self.assertEqual(r._sck_size["height"], 1750)


class MultiNativeDetection(unittest.TestCase):
    """Recorder detects multi-window native (P3.1) at construction. Empty
    `_sck_workers` is the fail-safe -- every consumer keys off the list, so
    a partial or inconsistent config never half-enters the mode."""

    def _picks(self, n):
        # Windows shaped like devices.window_rect_points entries. Aspects and
        # origins vary so a bug that mixed up indices would land on a wrong
        # rect the manifest check would catch.
        return [{"id": 100 + i, "app": "App{}".format(i), "title": "t",
                 "x": float(i * 100), "y": 0.0, "w": 400.0, "h": 300.0}
                for i in range(n)]

    def test_single_window_rides_a_one_worker_fleet(self):
        # Scene takes (docs/architecture.md): a mic-less, face-less
        # single occlusion-free pick rides the FLEET as one worker, so
        # pause/re-pick works from a 1-window opening scene. The on-disk
        # contract is preserved by `_finalize_fleet_single` (raw.mov + the
        # pinned capture_window block); `_is_window_native` stays true, which
        # is what lets that down-conversion reuse the singleton meta path.
        r = record.Recorder("/tmp/x", 1, backend="sck", window_native=True,
                            capture_window=self._picks(1)[0])
        self.assertTrue(r._is_multi_window_native())
        self.assertTrue(r._fleet_single)
        self.assertEqual(len(r._sck_workers), 1)
        self.assertTrue(r._is_window_native())
        self.assertTrue(r.pause_supported)

    def test_single_window_with_mic_or_face_keeps_the_singleton_path(self):
        # Mic and facecam are unsupported on the fleet; losing them would be
        # a worse regression than an unpausable take, so those configs stay
        # on the proven single-native path -- and pause is refused there
        # (the P1 machinery would drop the capture_window coordinate space).
        for kw in ({"mic_idx": 0}, {"face_idx": 0}):
            r = record.Recorder("/tmp/x", 1, backend="sck",
                                window_native=True,
                                capture_window=self._picks(1)[0], **kw)
            self.assertFalse(r._is_multi_window_native(), kw)
            self.assertEqual(r._sck_workers, [], kw)
            self.assertFalse(r._fleet_single, kw)
            self.assertTrue(r._is_window_native(), kw)
            self.assertFalse(r.pause_supported, kw)

    def test_multi_native_with_mic_refuses_pause(self):
        # A mic take stays single-scene: the scene (paused) render is
        # video-only today, so pausing would silently drop the voiceover
        # across the seam. Audio-across-a-scene-seam is the next milestone.
        r = record.Recorder("/tmp/x", 1, backend="sck", window_native=True,
                            capture_windows=self._picks(2), mic_idx=0)
        self.assertTrue(r._is_multi_window_native())
        self.assertFalse(r.pause_supported)
        # ...and a mic-LESS fleet still pauses (the scene-take capability).
        r2 = record.Recorder("/tmp/x", 1, backend="sck", window_native=True,
                             capture_windows=self._picks(2))
        self.assertTrue(r2.pause_supported)

    def test_multi_native_populates_a_worker_per_window(self):
        picks = self._picks(3)
        r = record.Recorder("/tmp/x", 1, backend="sck", window_native=True,
                            capture_windows=picks)
        self.assertTrue(r._is_multi_window_native())
        self.assertEqual(len(r._sck_workers), 3)
        for i, worker in enumerate(r._sck_workers):
            self.assertEqual(worker.index, i)
            self.assertEqual(worker.window["id"], picks[i]["id"])
            # Per-worker raw_i.mov, distinct from every other worker and from
            # the (unused) top-level raw.mov. A bug that collided two workers
            # on one file would make the second overwrite the first.
            self.assertTrue(worker.raw_path.endswith("raw_{}.mov".format(i)))
        # capture_window (singular) cleared -- the two modes are mutually
        # exclusive in the manifest, and leaving both set would let a meta
        # consumer see both keys and have to guess which is the truth.
        self.assertIsNone(r.capture_window)

    def test_multi_native_requires_sck_backend(self):
        # avfoundation cannot target a window buffer at all -- silently
        # dropping into that mode with capture_windows set would produce a
        # display-crop MULTI take, not the native one asked for. Guard:
        # window_native=True + avfoundation + capture_windows -> NOT multi
        # native (the workers list stays empty).
        r = record.Recorder("/tmp/x", 1, backend="avfoundation",
                            window_native=True,
                            capture_windows=self._picks(2))
        self.assertFalse(r._is_multi_window_native())
        self.assertEqual(r._sck_workers, [])

    def test_multi_native_requires_window_native_true(self):
        # Multi capture_windows without --occlusion-free is the pre-existing
        # display-crop MULTI-window composite, NOT native. Must not populate
        # workers.
        r = record.Recorder("/tmp/x", 1, backend="sck", window_native=False,
                            capture_windows=self._picks(2))
        self.assertFalse(r._is_multi_window_native())
        self.assertEqual(r._sck_workers, [])
        # And the existing display-crop MULTI state is preserved untouched.
        self.assertEqual(len(r.capture_windows), 2)


class RaiseTick(unittest.TestCase):
    """`_raise_tick` -- the "brought to the front" detector that decides which
    already-open windows can still become JOIN candidates (docs/architecture.md
    M2.4c). Pure bookkeeping over the geometry poll's front-to-back list, so
    it is testable without Quartz."""

    def setUp(self):
        self.r = record.Recorder("/tmp/x", 0)
        self.now = 1000.0

    def _tick(self, *ids):
        with mock.patch.object(record.time, "monotonic",
                               side_effect=lambda: self.now):
            self.r._raise_tick([{"id": i} for i in ids])

    def test_window_frontmost_at_start_is_never_a_raise(self):
        # The first reading is the baseline front -- typically the terminal
        # the server was launched from. Holding it forever is not "bringing
        # it up", so it must never become a candidate.
        for _ in range(20):
            self._tick(5, 6, 7)
            self.now += record.WINDOW_POLL_SEC
        self.assertEqual(self.r.raised_window_ids, frozenset())

    def test_raise_held_past_the_dwell_is_promoted(self):
        self._tick(5, 6, 7)                     # 5 is the start front
        self._tick(6, 5, 7)                     # 6 takes the front
        self.assertEqual(self.r.raised_window_ids, frozenset())
        self.now += record.WINDOW_RAISE_DWELL_SEC
        self._tick(6, 5, 7)
        self.assertEqual(self.r.raised_window_ids, frozenset({6}))

    def test_flyby_shorter_than_the_dwell_is_not_promoted(self):
        self._tick(5, 6, 7)
        self._tick(6, 5, 7)                     # 6 flashes to the front...
        self.now += record.WINDOW_RAISE_DWELL_SEC / 2.0
        self._tick(7, 5, 6)                     # ...and 7 takes it before the dwell
        self.now += record.WINDOW_RAISE_DWELL_SEC / 2.0
        self._tick(7, 5, 6)                     # 7's own dwell is not up either
        self.assertEqual(self.r.raised_window_ids, frozenset())

    def test_second_raise_accumulates(self):
        # The real reported take: two windows raised in turn, both wanted.
        self._tick(5, 6, 7)
        self._tick(6, 5, 7)
        self.now += record.WINDOW_RAISE_DWELL_SEC
        self._tick(6, 5, 7)
        self._tick(7, 6, 5)
        self.now += record.WINDOW_RAISE_DWELL_SEC
        self._tick(7, 6, 5)
        self.assertEqual(self.r.raised_window_ids, frozenset({6, 7}))

    def test_returning_to_the_start_front_clears_the_dwell(self):
        self._tick(5, 6, 7)
        self._tick(6, 5, 7)
        self.now += record.WINDOW_RAISE_DWELL_SEC / 2.0
        self._tick(5, 6, 7)                     # back to the baseline front
        self.now += record.WINDOW_RAISE_DWELL_SEC / 2.0
        self._tick(6, 5, 7)                     # 6 restarts its dwell here
        self.assertEqual(self.r.raised_window_ids, frozenset())
        self.now += record.WINDOW_RAISE_DWELL_SEC
        self._tick(6, 5, 7)
        self.assertEqual(self.r.raised_window_ids, frozenset({6}))

    def test_empty_or_unreadable_list_holds_the_dwell(self):
        # Quartz returning nothing (or an entry with no id) must neither crash
        # nor credit/reset a dwell in progress -- a thinned poll should cost a
        # chip at worst, never a wrong join.
        self._tick(5, 6)
        self._tick(6, 5)
        self._tick()                            # nothing pickable
        self._tick({})                          # unreadable entry -> id None
        self.assertEqual(self.r.raised_window_ids, frozenset())
        self.now += record.WINDOW_RAISE_DWELL_SEC
        self._tick(6, 5)
        self.assertEqual(self.r.raised_window_ids, frozenset({6}))

    def test_own_chrome_never_occupies_the_front_slot(self):
        # The pill / notes overlay are always-on-top. Normally they are not on
        # layer 0 so `entries` never carries them, but if one ever did, it
        # must not mask the real front for the whole take.
        self.r.exclude_windows = [99]
        self._tick(99, 5, 6)                    # 5 is the start front, not 99
        self._tick(99, 6, 5)                    # 6 takes it
        self.now += record.WINDOW_RAISE_DWELL_SEC
        self._tick(99, 6, 5)
        self.assertEqual(self.r.raised_window_ids, frozenset({6}))

    def test_raised_ids_are_rebound_not_mutated(self):
        # The HTTP thread reads this property without a lock; the poller must
        # never mutate the set a reader is holding.
        self._tick(5, 6)
        self._tick(6, 5)
        before = self.r.raised_window_ids
        self.now += record.WINDOW_RAISE_DWELL_SEC
        self._tick(6, 5)
        self.assertEqual(before, frozenset())
        self.assertEqual(self.r.raised_window_ids, frozenset({6}))


class WindowNativeCoordinateSpace(unittest.TestCase):
    """Which coordinate space a take records window rects in.

    `devices` clips every window to its display by default -- right for a
    display-crop pick (an avfoundation crop cannot reach pixels outside the
    captured display), wrong for occlusion-free, where SCK records the
    window's own surface INCLUDING the part hanging off the screen. A clipped
    rect there describes less than the file contains: the card renders
    stretched and `buffer/logical` -- the event transform's scale -- is
    inflated on the clipped axis, so clicks land off-target inside that card.
    (docs/architecture.md, "The display-CLIPPED rect".)
    """

    PICK = {"id": 42, "app": "Microsoft Word", "title": "doc",
            "x": 723.0, "y": 57.0, "w": 717.0, "h": 835.0,
            "display_id": 1, "display_origin": [0.0, 0.0],
            "main_display": True}

    def _clip_arg(self, **kw):
        """What the recorder asks `devices` for when it re-snapshots."""
        r = record.Recorder("/tmp/x", 0, **kw)
        with mock.patch.object(record.dev, "window_rect_points",
                               return_value=None) as m:
            r._snapshot_window(self.PICK)
        self.assertTrue(m.called)
        return m.call_args[1]["clip_to_display"]

    def test_occlusion_free_asks_for_unclipped_rects(self):
        self.assertFalse(self._clip_arg(capture_window=self.PICK,
                                        window_native=True, backend="sck"))

    def test_a_display_crop_window_pick_still_clips(self):
        # The default pick captures the DISPLAY and crops, so the visible part
        # remains the honest answer. Bit-exact with what shipped before.
        self.assertTrue(self._clip_arg(capture_window=self.PICK))
        self.assertTrue(self._clip_arg(capture_window=self.PICK,
                                       backend="sck"))

    def test_window_native_without_sck_still_clips(self):
        # window_native only takes effect on the SCK backend (__init__ builds
        # the fleet under exactly that condition), so the space must follow
        # the same condition or the two disagree.
        self.assertTrue(self._clip_arg(capture_window=self.PICK,
                                       window_native=True,
                                       backend="avfoundation"))

    def test_the_space_is_known_before_any_worker_exists(self):
        """The reason `_window_native_space` is looser than the fleet guards:
        the rects that matter are read during the pick and the countdown,
        when `_sck_workers` is still empty and `_is_multi_window_native()` is
        false. Keying the space off those would clip exactly the reads that
        end up in meta.json."""
        r = record.Recorder("/tmp/x", 0, capture_windows=[self.PICK, dict(
            self.PICK, id=43)], window_native=True, backend="sck")
        r._sck_workers = []
        self.assertFalse(r._is_multi_window_native())
        self.assertTrue(r._window_native_space())

    def test_the_geometry_poller_reads_the_same_space_as_the_meta_rect(self):
        """A track in a different space than the rect it is scaled against is
        wrong every frame instead of once, so these two reads must never
        drift apart."""
        for kw, want_clip in (
                (dict(capture_window=self.PICK, window_native=True,
                      backend="sck"), False),
                (dict(capture_window=self.PICK), True),
                (dict(), True)):
            r = record.Recorder("/tmp/x", 0, **kw)
            # The loop tests the stop event FIRST, so stop it from inside the
            # call under test: exactly one pass, no sleeping, no thread.
            def one_pass(*a, **k):
                r._win_stop.set()
                return []
            with mock.patch.object(record.dev, "list_windows",
                                   side_effect=one_pass) as m:
                with mock.patch.object(record.dev, "displays_points",
                                       return_value=[]):
                    r._poll_window_geometry()
            self.assertTrue(m.called, "the poller never enumerated")
            self.assertEqual(m.call_args[1]["clip_to_display"], want_clip,
                             "poller space for {}".format(kw))


class PlanJoinScenes(unittest.TestCase):
    """`_plan_join_scenes` -- the pure core of the seamless window-join. The
    strongest invariant: every SURVIVOR's per-scene sub-ranges are contiguous
    and cover its whole continuous file (no gap, no overlap at the seam)."""

    def test_no_join_returns_empty(self):
        # n_original == len -> a plain fleet, no scenes (off-switch: caller
        # writes the flat capture_channels manifest, byte-identical).
        self.assertEqual(
            record._plan_join_scenes([100.0, 100.01], [180, 180], 2, 60), [])

    def test_single_join_two_scenes(self):
        # 2 survivors (continuous 180-frame files) + 1 joiner at t=101.0 on its
        # own 90-frame file.
        t0s = [100.0, 100.01, 101.0]
        counts = [180, 180, 90]
        scenes = record._plan_join_scenes(t0s, counts, 2, 60)
        self.assertEqual(len(scenes), 2)
        self.assertEqual(len(scenes[0]["channels"]), 2)   # before: 2 cards
        self.assertEqual(len(scenes[1]["channels"]), 3)   # after: 3 cards
        # joiner rides scene 1 from its own frame 0
        joiner = scenes[1]["channels"][2]
        self.assertEqual((joiner["frame_start"], joiner["frame_count"]), (0, 90))
        # scene 1 re-anchors every channel's t0 to the seam
        for ch in scenes[1]["channels"]:
            self.assertAlmostEqual(ch["t0"], 101.0)

    def test_survivor_subranges_are_contiguous_and_cover_the_file(self):
        for t0s, counts, n0 in (
            ([100.0, 100.01, 101.0], [180, 180, 90], 2),        # 1 join
            ([100.0, 100.0, 101.0, 102.5], [300, 300, 210, 90], 2),  # 2 joins
            ([50.0, 50.02, 50.7, 51.9], [240, 240, 240, 60], 3),     # 1 join, 3 survivors
        ):
            scenes = record._plan_join_scenes(t0s, counts, n0, 60)
            self.assertEqual(len(scenes), len(t0s) - n0 + 1)
            # For each ORIGINAL survivor channel i, walk its scenes in order:
            # frame_start must chain (fs_next == fs+fc) and end at counts[i].
            for i in range(n0):
                cursor = 0
                for s, scene in enumerate(scenes):
                    ch = next((c for c in scene["channels"]
                               if c["index"] == i), None)
                    self.assertIsNotNone(ch, "survivor {} missing in scene {}"
                                         .format(i, s))
                    self.assertEqual(ch["frame_start"], cursor,
                                     "channel {} scene {} not contiguous"
                                     .format(i, s))
                    cursor += ch["frame_count"]
                self.assertEqual(cursor, counts[i],
                                 "channel {} sub-ranges don't cover the file"
                                 .format(i))

    def test_scene_start_times_are_strictly_ascending(self):
        # SegmentClock.owner/media partition PARENT TIME via ascending t0s;
        # a non-ascending seam would break the clock.
        scenes = record._plan_join_scenes(
            [100.0, 100.0, 101.0, 102.5], [300, 300, 210, 90], 2, 60)
        starts = [sc["start_t"] for sc in scenes]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(len(set(starts)), len(starts))


class PlanFleetScenes(unittest.TestCase):
    """`_plan_fleet_scenes` -- the M3.1 generalization of the join planner to
    EXIT events (the card shrink, docs/architecture.md Milestone 3). The two
    load-bearing contracts: the exits={} path is byte-identical to the
    pre-M3.1 `_plan_join_scenes` (the off-switch), and presence is COMPUTED,
    never fabricated -- a departing channel's sub-ranges end at its seam and
    its dup-fill tail goes unreferenced."""

    # -- off-switch ---------------------------------------------------------

    def test_no_events_returns_empty(self):
        self.assertEqual(
            record._plan_fleet_scenes([100.0, 100.01], [180, 180], 2, 60),
            ([], []))

    def test_join_only_branch_pins_the_exact_pre_m31_manifest(self):
        # NOT a wrapper-equality check (that would be a tautology now that
        # _plan_join_scenes delegates here) -- the expected output is
        # HARD-CODED from the pre-M3.1 algorithm, pinning the verbatim
        # branch itself. Scene 0 keeps ACTUAL per-channel t0s (100.0 vs
        # 100.4 -- distinct, so a t0s[0] mutation cannot hide).
        scenes, demoted = record._plan_fleet_scenes(
            [100.0, 100.4, 103.0], [600, 576, 300], 2, 60, exits={})
        self.assertEqual(demoted, [])
        self.assertEqual(scenes, [
            {"start_t": 100.0, "end_t": 103.0, "channels": [
                {"index": 0, "frame_start": 0, "frame_count": 180,
                 "t0": 100.0},
                {"index": 1, "frame_start": 0, "frame_count": 156,
                 "t0": 100.4},
            ]},
            {"start_t": 103.0, "end_t": None, "channels": [
                {"index": 0, "frame_start": 180, "frame_count": 420,
                 "t0": 103.0},
                {"index": 1, "frame_start": 156, "frame_count": 420,
                 "t0": 103.0},
                {"index": 2, "frame_start": 0, "frame_count": 300,
                 "t0": 103.0},
            ]},
        ])

    def test_scene_zero_keeps_actual_t0s_on_the_exit_path(self):
        # The general path shares the contract: scene 0 = actual per-channel
        # t0s + frame_start 0 (the whole naive-presence correction hinges on
        # it -- docs/architecture.md M3.1).
        scenes, _ = record._plan_fleet_scenes(
            [100.0, 100.4], [600, 576], 2, 60, exits={1: 105.0})
        self.assertEqual([c["t0"] for c in scenes[0]["channels"]],
                         [100.0, 100.4])
        self.assertEqual([c["frame_start"] for c in scenes[0]["channels"]],
                         [0, 0])

    # -- exit-only ----------------------------------------------------------

    def test_exit_only_two_scenes_and_unreferenced_tail(self):
        # 2 originals, ch1 minimized at t=105.0 (back-dated seam); both files
        # run the full 600 frames (ch1's post-hide frames are dup-fill).
        scenes, demoted = record._plan_fleet_scenes(
            [100.0, 100.01], [600, 600], 2, 60, exits={1: 105.0})
        self.assertEqual(demoted, [])
        self.assertEqual(len(scenes), 2)
        s0, s1 = scenes
        self.assertEqual([c["index"] for c in s0["channels"]], [0, 1])
        self.assertEqual([c["index"] for c in s1["channels"]], [0])
        # Scene 0 keeps actual t0s + frame_start 0 (today's contract).
        for c in s0["channels"]:
            self.assertEqual(c["frame_start"], 0)
        self.assertEqual(s0["channels"][0]["t0"], 100.0)
        # The departing channel's slice ENDS at the seam; the ~301 dup-fill
        # frames after it are simply never referenced.
        dep = s0["channels"][1]
        self.assertEqual(dep["frame_count"], 299)    # round((105-100.01)*60)
        self.assertLess(dep["frame_count"], 600)
        # The survivor's sub-ranges chain across the seam and cover its file.
        srv0, srv1 = s0["channels"][0], s1["channels"][0]
        self.assertEqual(srv1["frame_start"],
                         srv0["frame_start"] + srv0["frame_count"])
        self.assertEqual(srv0["frame_count"] + srv1["frame_count"], 600)
        # Scene 1 re-anchors to the seam.
        self.assertEqual(s1["start_t"], 105.0)
        self.assertEqual(srv1["t0"], 105.0)
        self.assertIsNone(s1["end_t"])

    # -- join and exit compose ---------------------------------------------

    def test_join_then_exit_three_scenes(self):
        # 2 originals; joiner (ch2) at 101.0; the joiner departs at 103.0.
        scenes, demoted = record._plan_fleet_scenes(
            [100.0, 100.01, 101.0], [600, 600, 480], 2, 60,
            exits={2: 103.0})
        self.assertEqual(demoted, [])
        self.assertEqual([sc["start_t"] for sc in scenes],
                         [100.0, 101.0, 103.0])
        self.assertEqual([[c["index"] for c in sc["channels"]]
                          for sc in scenes], [[0, 1], [0, 1, 2], [0, 1]])
        # The joiner rides its scene from its own frame 0 and ends at its
        # exit seam: round((103-101)*60) = 120 of its 480 frames.
        joiner = scenes[1]["channels"][2]
        self.assertEqual((joiner["frame_start"], joiner["frame_count"]),
                         (0, 120))

    def test_exit_then_join_three_scenes(self):
        # ch1 departs at 103.0, then a new window (ch2) joins at 106.0.
        scenes, demoted = record._plan_fleet_scenes(
            [100.0, 100.01, 106.0], [600, 600, 240], 2, 60,
            exits={1: 103.0})
        self.assertEqual(demoted, [])
        self.assertEqual([[c["index"] for c in sc["channels"]]
                          for sc in scenes], [[0, 1], [0], [0, 2]])
        # Survivor ch0 chains through all three scenes and covers its file.
        cursor = 0
        for sc in scenes:
            ch = next(c for c in sc["channels"] if c["index"] == 0)
            self.assertEqual(ch["frame_start"], cursor)
            cursor += ch["frame_count"]
        self.assertEqual(cursor, 600)

    def test_scene_starts_strictly_ascending_with_mixed_events(self):
        scenes, _ = record._plan_fleet_scenes(
            [100.0, 100.01, 102.0, 108.0], [900, 900, 700, 400], 2, 60,
            exits={1: 105.0, 2: 110.0})
        starts = [sc["start_t"] for sc in scenes]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(len(set(starts)), len(starts))

    # -- the clamp and the sub-floor merge ---------------------------------

    def test_backdated_exit_just_before_join_coalesces(self):
        # ch1's hide back-dates to 4.8s; a joiner lands at 5.0s. The 0.2s
        # in-between scene is sub-floor (12 < 15 frames @60), so the exit
        # slides forward onto the join seam: ONE boundary, no fabricated
        # slice for the in-flight joiner.
        scenes, demoted = record._plan_fleet_scenes(
            [100.0, 100.01, 105.0], [600, 600, 300], 2, 60,
            exits={1: 104.8})
        self.assertEqual(demoted, [])
        self.assertEqual(len(scenes), 2)
        self.assertEqual([[c["index"] for c in sc["channels"]]
                          for sc in scenes], [[0, 1], [0, 2]])
        self.assertEqual(scenes[1]["start_t"], 105.0)

    def test_join_then_immediate_hide_demotes_entirely(self):
        # Decision 7: a joiner that hides within the sub-floor of joining
        # demotes outright -- BOTH its seams vanish and the plan collapses
        # to no scenes at all (caller writes the flat manifest).
        scenes, demoted = record._plan_fleet_scenes(
            [100.0, 100.01, 101.0], [600, 600, 30], 2, 60,
            exits={2: 101.05})
        self.assertEqual(scenes, [])
        self.assertEqual(demoted, [2])

    def test_original_hidden_at_start_drops_from_manifest(self):
        # An original minimized almost immediately: its exit slides back to
        # the take start and the channel vanishes from the manifest (the
        # caller must fire the departed hint -- silent drop is a broken
        # promise, pinned at the M3.2 layer).
        scenes, demoted = record._plan_fleet_scenes(
            [100.0, 100.01], [600, 600], 2, 60, exits={1: 100.05})
        self.assertEqual(scenes, [])
        self.assertEqual(demoted, [1])

    def test_file_exhausted_channel_is_dropped_not_fabricated(self):
        # ch1's file died at 100 frames (t=101.67) but its exit is at 103.0:
        # present-by-time in the [102.0, 103.0) scene with an EMPTY slice.
        # It must be dropped from that scene, never given a fabricated
        # max(1,...) frame past its EOF (kills the fabrication mutant).
        scenes, demoted = record._plan_fleet_scenes(
            [100.0, 100.0, 102.0], [600, 100, 400], 2, 60,
            exits={1: 103.0})
        self.assertEqual(demoted, [])
        self.assertEqual([[c["index"] for c in sc["channels"]]
                          for sc in scenes], [[0, 1], [0, 2], [0, 2]])
        for sc in scenes:
            for c in sc["channels"]:
                self.assertLessEqual(
                    c["frame_start"] + c["frame_count"],
                    [600, 100, 400][c["index"]])

    def test_exact_exit_join_tie_is_clamped_apart(self):
        # An exit and a join stamped at the SAME instant: the join's seam is
        # clamped one frame later (strictly-ascending starts), the 1-frame
        # scene merges, and the joiner's frame_start reflects the clamped
        # seam -- fs=1, not 0 (kills the dropped-clamp mutant).
        scenes, demoted = record._plan_fleet_scenes(
            [100.0, 100.01, 105.0], [600, 600, 300], 2, 60,
            exits={1: 105.0})
        self.assertEqual(demoted, [])
        self.assertEqual(len(scenes), 2)
        self.assertAlmostEqual(scenes[1]["start_t"], 105.0 + 1.0 / 60)
        joiner = next(c for c in scenes[1]["channels"] if c["index"] == 2)
        self.assertEqual(joiner["frame_start"], 1)
        self.assertEqual([c["index"] for c in scenes[1]["channels"]], [0, 2])

    def test_scene_just_above_the_sub_floor_survives(self):
        # 18 frames @60 is ABOVE max(2, 60//4)=15: the middle scene must
        # survive the merge (kills the enlarged-min_frames mutant).
        scenes, demoted = record._plan_fleet_scenes(
            [100.0, 100.01, 105.0], [900, 900, 400], 2, 60,
            exits={1: 105.3})
        self.assertEqual(demoted, [])
        self.assertEqual(len(scenes), 3)
        mid = next(c for c in scenes[1]["channels"] if c["index"] == 0)
        self.assertEqual(mid["frame_count"], 18)

    def test_zero_frame_file_demotes_upfront_no_spurious_seam(self):
        # A 0-frame file can appear in no scene; its event must demote up
        # front rather than leave a no-op seam (or a silent absence the
        # demoted list never reports).
        scenes, demoted = record._plan_fleet_scenes(
            [100.0, 100.0], [600, 0], 2, 60, exits={1: 104.0})
        self.assertEqual((scenes, demoted), ([], [1]))

    def test_short_file_never_discards_another_channels_footage(self):
        # The adversarial repro that broke the first cut of the merge loop:
        # the JOINER's file is short (190 frames -- e.g. it closed after the
        # take's end was trimmed), which once made the final scene read
        # "sub-floor" and slide ch1's healthy exit back 3 SECONDS. The final
        # scene must never merge for file-exhaustion: ch1 keeps every real
        # frame up to its seam.
        scenes, demoted = record._plan_fleet_scenes(
            [100.0, 100.0, 103.0], [900, 900, 190], 2, 60,
            exits={1: 106.0})
        self.assertEqual(demoted, [])
        self.assertEqual([[c["index"] for c in sc["channels"]]
                          for sc in scenes], [[0, 1], [0, 1, 2], [0, 2]])
        ch1_total = sum(c["frame_count"] for sc in scenes
                        for c in sc["channels"] if c["index"] == 1)
        self.assertEqual(ch1_total, 360)     # every frame up to t=106.0

    def test_exit_near_the_take_end_keeps_a_tiny_trailing_scene(self):
        # An exit landing within the sub-floor of EOF keeps its seam and a
        # tiny trailing scene -- it must NOT demote the channel and discard
        # its 9.9s of real footage (the adversarial contract repro).
        scenes, demoted = record._plan_fleet_scenes(
            [100.0, 100.01], [600, 600], 2, 60, exits={1: 109.9})
        self.assertEqual(demoted, [])
        self.assertEqual(len(scenes), 2)
        dep = next(c for c in scenes[0]["channels"] if c["index"] == 1)
        self.assertEqual(dep["frame_count"], 593)   # round((109.9-100.01)*60)
        self.assertEqual([c["index"] for c in scenes[1]["channels"]], [0])

    def test_stranded_joiner_is_demoted_never_silently_absent(self):
        # A joiner whose scene merges away while an exit survives: it must
        # land in `demoted` (the adversarial contract repro -- a joiner
        # absent from every scene but unreported would poison the caller's
        # flat fallback / hint accounting).
        scenes, demoted = record._plan_fleet_scenes(
            [0.0, 0.01, 5.0], [318, 318, 18], 2, 60, exits={1: 5.1})
        self.assertEqual(demoted, [2])
        for sc in scenes:
            self.assertNotIn(2, [c["index"] for c in sc["channels"]])
        # ch1's real footage up to its seam is intact.
        ch1_total = sum(c["frame_count"] for sc in scenes
                        for c in sc["channels"] if c["index"] == 1)
        self.assertEqual(ch1_total, 305)

    def test_join_right_after_start_keeps_a_short_scene_zero(self):
        # A joiner landing within the sub-floor of the take start makes
        # scene 0 sub-floor. Demoting it would discard its WHOLE file
        # (adversarially measured at 30+ seconds); the planner keeps the
        # few-frame opening scene instead -- lossless beats pretty.
        scenes, demoted = record._plan_fleet_scenes(
            [100.0, 100.05, 100.2], [600, 597, 588], 2, 60,
            exits={1: 105.0})
        self.assertEqual(demoted, [])
        self.assertEqual(scenes[0]["start_t"], 100.0)
        self.assertEqual(scenes[0]["end_t"], 100.2)   # 12-frame opener, kept
        self.assertEqual([c["index"] for c in scenes[1]["channels"]],
                         [0, 1, 2])
        ch2_total = sum(c["frame_count"] for sc in scenes
                        for c in sc["channels"] if c["index"] == 2)
        self.assertEqual(ch2_total, 588)              # nothing discarded

    def test_dense_event_cascade_reports_not_corrupts(self):
        # The documented planner bound: a fuzz-only cascade of sub-floor-
        # spaced events (impossible from the serialized grow feed + the
        # exit debounce) may DEMOTE a stranded joiner -- but it must be
        # REPORTED in demoted, survivors keep every frame, and the result
        # still satisfies every structural invariant. Distilled from fuzz
        # seed 5771.
        t0s = [100.0, 100.0232, 100.0054, 100.0249, 100.8585, 103.167,
               105.0989, 105.1121, 105.2045]
        counts = [548, 547, 548, 547, 497, 358, 242, 242, 236]
        exits = {4: 104.6373, 8: 105.2311, 1: 99.9871, 5: 105.1262,
                 6: 101.6746, 2: 99.9802}
        scenes, demoted = record._plan_fleet_scenes(
            t0s, counts, 4, 60, exits=exits)
        # Every channel is either in a scene or reported demoted -- never
        # silently absent.
        in_scenes = {c["index"] for sc in scenes for c in sc["channels"]}
        self.assertEqual(sorted(in_scenes | set(demoted)), list(range(9)))
        self.assertEqual(in_scenes & set(demoted), set())
        # Scene starts strictly ascending, all slices real.
        starts = [sc["start_t"] for sc in scenes]
        self.assertEqual(starts, sorted(set(starts)))
        for sc in scenes:
            for c in sc["channels"]:
                self.assertGreater(c["frame_count"], 0)
                self.assertLessEqual(
                    c["frame_start"] + c["frame_count"], counts[c["index"]])
        # The never-exiting survivors keep their whole files.
        for i in (0, 3):
            total = sum(c["frame_count"] for sc in scenes
                        for c in sc["channels"] if c["index"] == i)
            self.assertEqual(total, counts[i])
        # The heir rule: ch7's entrance boundary belonged to ch8, which the
        # cascade demoted -- the seam must survive under ch7 (losing only
        # the bounded few-frame cascade head), not vanish and strand it.
        ch7_total = sum(c["frame_count"] for sc in scenes
                        for c in sc["channels"] if c["index"] == 7)
        self.assertEqual(ch7_total, 235)

    def test_empty_plan_demotes_every_joiner(self):
        # When no seams survive at all, a flat manifest can only express the
        # originals -- every joiner must be in `demoted`, not silently
        # dropped.
        scenes, demoted = record._plan_fleet_scenes(
            [0.0, 0.01, 0.04], [600, 600, 60], 2, 60, exits={1: 0.12})
        self.assertEqual((scenes, demoted), ([], [1, 2]))

    def test_exit_before_channel_born_demotes_upfront(self):
        # An exit stamped at/before the joiner's own t0 (degenerate input):
        # demoted up front, never a seam.
        scenes, demoted = record._plan_fleet_scenes(
            [100.0, 100.01, 104.0], [600, 600, 300], 2, 60,
            exits={2: 104.0})
        self.assertEqual(scenes, [])
        self.assertEqual(demoted, [2])

    # -- guards -------------------------------------------------------------

    def test_zero_channel_scene_raises(self):
        # Every channel exiting mid-take would leave a channel-less tail
        # scene; the detector's last-card guard makes this unreachable, so
        # the pure function refuses loudly instead of emitting it.
        with self.assertRaises(ValueError):
            record._plan_fleet_scenes(
                [100.0, 100.01], [600, 600], 2, 60,
                exits={0: 104.0, 1: 105.0})

    def test_exit_for_unknown_channel_raises(self):
        with self.assertRaises(ValueError):
            record._plan_fleet_scenes(
                [100.0, 100.01], [600, 600], 2, 60, exits={5: 104.0})


class WorkerFrameCount(unittest.TestCase):
    """`_worker_frame_count` -- the pinned count-source contract (scene-takes
    M3.0b): PREFER the worker's exact DONE count (`finish()` appends the idle
    tail AFTER the last <=1Hz STAT, so STAT can undercount the file by up to
    ~1s), FALL BACK to STAT when DONE never drained (slow pipe at finalize),
    0 when neither arrived. No re-probe of the file either way."""

    def _count(self, stat=None, done=None):
        r = record.Recorder("/tmp/x", 1, backend="sck", window_native=True,
                            capture_windows=[
                                {"id": 100, "app": "A", "title": "t",
                                 "x": 0.0, "y": 0.0, "w": 400.0, "h": 300.0}])
        w = record._NativeWorker(r.session_dir, 0,
                                 {"id": 100, "app": "A", "title": "t",
                                  "x": 0.0, "y": 0.0, "w": 400.0, "h": 300.0})
        w.stat = stat
        w.done = done
        return r._worker_frame_count(w)

    def test_done_wins_over_stale_stat(self):
        self.assertEqual(
            self._count(stat={"appended": 180}, done={"frames": 204}), 204)

    def test_stat_is_the_fallback(self):
        self.assertEqual(self._count(stat={"appended": 180}, done=None), 180)

    def test_zero_when_neither_arrived(self):
        self.assertEqual(self._count(stat=None, done=None), 0)


class JoinFinalize(unittest.TestCase):
    """`_finalize_join_take` -- assembles the sub-range `capture_scenes`
    manifest from live worker state (no decode; counts come off DONE, STAT
    as the fallback -- WorkerFrameCount pins the source contract)."""

    def _picks(self, n):
        return [{"id": 100 + i, "app": "App{}".format(i), "title": "t",
                 "x": float(i * 420), "y": 0.0, "w": 400.0, "h": 300.0}
                for i in range(n)]

    def _rec_with_join(self, joiner_frames=90, mic=0):
        r = record.Recorder("/tmp/x", 1, backend="sck", window_native=True,
                             capture_windows=self._picks(2), mic_idx=mic)
        # Splice in a joiner exactly as _try_grow would.
        joiner = record._NativeWorker(r.session_dir, 2, self._picks(3)[2])
        r._sck_workers.append(joiner)
        for i, w in enumerate(r._sck_workers):
            w.entry = w.window
            w.resnapshot = True
            w.t0 = 100.0 + (0.0 if i < 2 else 1.0) + i * 0.001
            w.size = {"width": 800, "height": 600}
            w.stat = {"appended": 180 if i < 2 else joiner_frames,
                      "dup": 0, "dropped": 0, "notready": 0, "idle": 0}
        r._n_original = 2
        r._join_marks = [2]
        return r

    def _finalize(self, r):
        d = tempfile.mkdtemp(prefix="join_fin_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        r.session_dir = d
        r.meta_path = os.path.join(d, "meta.json")
        r._finalize_join_take(1440.0, 900.0, "quartz")
        with open(r.meta_path) as f:
            return json.load(f)

    def test_manifest_is_a_two_scene_subrange(self):
        meta = self._finalize(self._rec_with_join())
        self.assertIn("capture_scenes", meta)
        self.assertNotIn("capture_channels", meta)
        self.assertEqual(meta["mic_index"], 0)      # join can carry a mic
        s0, s1 = meta["capture_scenes"]
        self.assertEqual(len(s0["channels"]), 2)    # 2 cards before the join
        self.assertEqual(len(s1["channels"]), 3)    # 3 after
        # Survivors are ONE continuous file, sub-ranged contiguously.
        for i in range(2):
            a, b = s0["channels"][i], s1["channels"][i]
            self.assertEqual(a["file"], b["file"])          # same file
            self.assertEqual(a["frame_start"], 0)
            self.assertEqual(b["frame_start"], a["frame_count"])
            self.assertEqual(a["frame_count"] + b["frame_count"], 180)
        # Joiner rides scene 1 from its own frame 0.
        self.assertEqual(s1["channels"][2]["frame_start"], 0)
        self.assertEqual(s1["channels"][2]["frame_count"], 90)
        # Scene 1 re-anchors t0 to the seam (ascending scene t0s).
        self.assertLess(s0["t0_monotonic"], s1["t0_monotonic"])

    def test_done_counts_reach_the_manifest(self):
        # The exact DONE tail (up to ~1s past the last STAT) must land in the
        # sub-ranges: survivors' per-scene counts sum to DONE, not STAT.
        r = self._rec_with_join()
        for i, w in enumerate(r._sck_workers):
            if i < 2:
                w.done = {"frames": 204, "media": "3.4"}   # STAT said 180
        meta = self._finalize(r)
        s0, s1 = meta["capture_scenes"]
        for i in range(2):
            self.assertEqual(s0["channels"][i]["frame_count"]
                             + s1["channels"][i]["frame_count"], 204)

    def test_short_joiner_is_demoted_to_flat_manifest(self):
        # A window that closes right after joining must NOT define its scene
        # (n_out=min would truncate the survivors). It is dropped, leaving a
        # plain fleet manifest over the survivors.
        meta = self._finalize(self._rec_with_join(joiner_frames=3))
        self.assertNotIn("capture_scenes", meta)
        self.assertIn("capture_channels", meta)
        self.assertEqual(len(meta["capture_channels"]), 2)

    def test_micless_join_manifest_is_silent(self):
        meta = self._finalize(self._rec_with_join(mic=None))
        self.assertIsNone(meta["mic_index"])
        self.assertIn("capture_scenes", meta)


class GrowDuplicateGuard(unittest.TestCase):
    """`_try_grow` is AUTHORITATIVE against a DUPLICATE join (docs/architecture.md
    M3.2/M2.4). studio_app's `captured_window_ids` guard is checked at request
    time, but a joiner isn't in `_sck_workers` until its ~2s T0 wait finishes
    -- so a manual chip RE-CLICK for the same window, landing in that
    spawn-not-yet-appended gap (with the last-wins `_grow_pending` already
    drained), used to spawn a SECOND worker on a window already in the fleet.
    Pure: `_spawn_fleet` is stubbed, so a leaked spawn fails loudly instead of
    launching a subprocess."""

    def _picks(self, n):
        return [{"id": 100 + i, "app": "App{}".format(i), "title": "t",
                 "x": float(i * 420), "y": 0.0, "w": 400.0, "h": 300.0}
                for i in range(n)]

    def _fleet(self, n=2):
        r = record.Recorder("/tmp/x", 1, backend="sck", window_native=True,
                             capture_windows=self._picks(n))
        for w in r._sck_workers:
            w.entry = w.window
        # Records which worker sets reached the spawn; raising keeps it pure
        # (no subprocess) and exercises the surrounding abort path.
        self.spawned = []

        def _stub(workers):
            self.spawned.append([wk.index for wk in workers])
            raise RuntimeError("stubbed spawn")

        r._spawn_fleet = _stub
        return r

    def test_same_wid_grow_spawns_no_second_worker(self):
        r = self._fleet(2)
        n_before = len(r._sck_workers)
        dup = dict(self._picks(2)[0])       # window id 100 == worker 0's window
        ok = r._try_grow(dup, [], [], None, lambda *a, **k: None)
        self.assertFalse(ok)
        self.assertEqual(len(r._sck_workers), n_before)     # no second worker
        self.assertEqual(self.spawned, [])                  # never reached spawn
        self.assertEqual(r.captured_window_ids, {100, 101})

    def test_in_flight_wid_is_refused_before_spawn(self):
        # The append-not-yet-done case AT THE RECORDER: a worker is mid-spawn
        # (marked in `_growing_wid`) but not yet in `_sck_workers`, so
        # `captured_window_ids` can't see it -- a re-request for that wid must
        # still refuse without spawning.
        r = self._fleet(2)
        r._growing_wid = 555
        n_before = len(r._sck_workers)
        entry = {"id": 555, "app": "New", "title": "t",
                 "x": 0.0, "y": 0.0, "w": 400.0, "h": 300.0}
        ok = r._try_grow(entry, [], [], None, lambda *a, **k: None)
        self.assertFalse(ok)
        self.assertEqual(len(r._sck_workers), n_before)
        self.assertEqual(self.spawned, [])

    def test_a_genuinely_new_wid_is_not_blocked(self):
        # Counter-pin: the guard is a same-wid filter, not a blanket refusal.
        # A fresh window passes it and reaches the spawn (stubbed to raise, so
        # the abort path runs and the fleet keeps its two originals), and the
        # in-flight marker is cleared even on that abort so a later legitimate
        # re-add of the same wid is never wedged.
        r = self._fleet(2)
        notes = []
        entry = {"id": 999, "app": "New", "title": "t",
                 "x": 0.0, "y": 0.0, "w": 400.0, "h": 300.0}
        ok = r._try_grow(entry, [], [], None,
                         lambda st, **k: notes.append(k))
        self.assertFalse(ok)                    # spawn stub raised -> aborted
        self.assertEqual(self.spawned, [[2]])   # but it DID reach the spawn
        self.assertEqual(len(r._sck_workers), 2)
        self.assertIsNone(r._growing_wid)
        self.assertTrue(any("grow_error" in k for k in notes))


class MultiNativeWorkerConfig(unittest.TestCase):
    """Per-worker JSON handed to `_sck_worker.py` on stdin. Same shape as the
    single-window `_sck_config`, differing in exactly two keys -- a
    rewrite that quietly diverged the fleet would fail here."""

    def _picks(self, n):
        return [{"id": 100 + i, "app": "App{}".format(i), "title": "t",
                 "x": 0.0, "y": 0.0, "w": 400.0, "h": 300.0}
                for i in range(n)]

    def _rec(self, n):
        return record.Recorder("/tmp/x", 1, backend="sck", window_native=True,
                               capture_windows=self._picks(n))

    def test_every_worker_has_a_distinct_out_and_window_id(self):
        r = self._rec(3)
        cfgs = [r._multi_native_worker_cfg(w) for w in r._sck_workers]
        outs = [c["out"] for c in cfgs]
        wids = [c["capture_window_id"] for c in cfgs]
        # Distinct outputs (one raw_i.mov per worker) and distinct window
        # ids (one capture per worker). Collisions here would produce
        # two workers racing on one file, or two capturing the same window.
        self.assertEqual(len(set(outs)), 3)
        self.assertEqual(len(set(wids)), 3)
        for i, cfg in enumerate(cfgs):
            self.assertTrue(cfg["out"].endswith("raw_{}.mov".format(i)))
            self.assertEqual(cfg["capture_window_id"], 100 + i)

    def test_only_out_and_window_id_differ_across_the_fleet(self):
        # Every non-per-worker key must match across the fleet. Movie
        # fragments, cursor mode, fps, excludes -- if any of these drifted
        # per-worker, a multi-window take would decode inconsistently between
        # its channels (or worse, break on one).
        r = self._rec(4)
        cfgs = [r._multi_native_worker_cfg(w) for w in r._sck_workers]
        shared_keys = set(cfgs[0].keys()) - {"out", "capture_window_id"}
        for cfg in cfgs[1:]:
            for k in shared_keys:
                self.assertEqual(cfg.get(k), cfgs[0].get(k),
                                 "key {!r} drifted across workers".format(k))

    def test_movie_fragment_env_flows_into_every_worker(self):
        r = self._rec(2)
        with mock.patch.dict(os.environ,
                             {"AUTOCINE_MOVIE_FRAGMENT_INTERVAL_SEC": "0"},
                             clear=False):
            cfgs = [r._multi_native_worker_cfg(w) for w in r._sck_workers]
        # 0 = byte-exact off switch for Phase C's fragments. Every worker
        # must honour it or a single monolithic-layout raw_i.mov in the
        # fleet would break the "all-or-nothing on has_moov" invariant.
        for cfg in cfgs:
            self.assertEqual(cfg["movie_fragment_interval"], 0.0)

    def test_mic_off_and_exclude_are_not_smuggled_in(self):
        # Off switch: with NO mic requested, no worker carries a mic (byte-exact
        # with the pre-audio fleet) and the fleet passes no exclusions.
        r = self._rec(2)
        for cfg in [r._multi_native_worker_cfg(w) for w in r._sck_workers]:
            self.assertNotIn("mic_unique_id", cfg)
            self.assertEqual(cfg["exclude"], [])

    def test_mic_rides_channel_zero_only(self):
        # Audio goes on EXACTLY one worker -- channel 0, the session anchor.
        # Mic on every worker would open N concurrent mic sessions, which macOS
        # refuses (a silent capture failure for the whole fleet). Resolution is
        # mocked so the test is device-free.
        r = record.Recorder("/tmp/x", 1, backend="sck", window_native=True,
                             capture_windows=self._picks(3), mic_idx=0)
        with mock.patch.object(r, "_resolve_mic_uid", return_value="UID:mic0"):
            cfgs = [r._multi_native_worker_cfg(w) for w in r._sck_workers]
        self.assertEqual(cfgs[0].get("mic_unique_id"), "UID:mic0")
        for cfg in cfgs[1:]:
            self.assertNotIn("mic_unique_id", cfg)

    def test_unresolvable_mic_leaves_the_fleet_silent(self):
        # If the mic uid cannot be resolved SAFELY (None), no worker carries a
        # mic -- recording the wrong input is worse than a silent take, exactly
        # the single-window contract.
        r = record.Recorder("/tmp/x", 1, backend="sck", window_native=True,
                             capture_windows=self._picks(2), mic_idx=7)
        with mock.patch.object(r, "_resolve_mic_uid", return_value=None):
            cfgs = [r._multi_native_worker_cfg(w) for w in r._sck_workers]
        for cfg in cfgs:
            self.assertNotIn("mic_unique_id", cfg)


class MultiNativeMeta(unittest.TestCase):
    """The `capture_channels` manifest, and its mutual exclusion with the
    single-window `capture_window` block. Off-switch: a single-window native
    take's meta is untouched."""

    def _picks(self, n):
        return [{"id": 100 + i, "app": "App{}".format(i), "title": "t{}".format(i),
                 "x": float(i * 200), "y": 0.0, "w": 400.0, "h": 300.0,
                 "display_origin": [0.0, 0.0]}
                for i in range(n)]

    def _multi(self, n):
        r = record.Recorder("/tmp/x", 1, backend="sck", window_native=True,
                            capture_windows=self._picks(n))
        # Simulate a completed take: entry re-snapshot done, T0 pairing
        # arrived, SIZE and STAT emitted.
        for i, w in enumerate(r._sck_workers):
            w.entry = w.window
            w.resnapshot = True
            w.end_rect = [float(i * 200), 0.0, 400.0, 300.0]
            w.t0 = 12.34 + i * 0.001
            w.size = {"width": 800, "height": 600}
            w.stat = {"appended": 300, "dup": 60, "dropped": 0, "notready": 3}
        return r

    def test_manifest_has_one_entry_per_worker(self):
        r = self._multi(3)
        meta = r._multi_native_meta_dict(1440.0, 900.0, "quartz", 12.34)
        self.assertEqual(len(meta["capture_channels"]), 3)
        for i, ch in enumerate(meta["capture_channels"]):
            self.assertEqual(ch["role"], "screen_window")
            self.assertEqual(ch["mode"], "window_native")
            self.assertEqual(ch["id"], 100 + i)
            self.assertEqual(ch["file"], "raw_{}.mov".format(i))
            self.assertEqual(ch["logical_w"], 400.0)
            self.assertEqual(ch["logical_h"], 300.0)
            self.assertEqual(ch["buffer_w"], 800)
            self.assertEqual(ch["buffer_h"], 600)
            self.assertAlmostEqual(ch["t0_monotonic"], 12.34 + i * 0.001)
            self.assertEqual(ch["capture_stats"]["appended"], 300)

    def test_manifest_meta_omits_single_window_keys(self):
        # `raw`, `capture_window`, `capture_windows` -- each implies a
        # single-file OR display-crop story that doesn't apply to the manifest.
        # A renderer that saw both `raw` and `capture_channels` couldn't tell
        # which to trust. mic_index is None here as an off-switch (no mic
        # requested), NOT because it is forbidden -- see the mic-on test below.
        r = self._multi(2)
        meta = r._multi_native_meta_dict(1440.0, 900.0, "quartz", 12.34)
        self.assertNotIn("raw", meta)
        self.assertNotIn("capture_window", meta)
        self.assertNotIn("capture_windows", meta)
        self.assertIsNone(meta["mic_index"])

    def test_manifest_meta_carries_mic_index_when_recording_audio(self):
        # mic_index records that a mic was requested; the audio track itself
        # lives inside channel 0's raw_0.mov (render probes it). A no-audio
        # channel 0 still renders silently, but the manifest must not lie about
        # intent.
        r = record.Recorder("/tmp/x", 1, backend="sck", window_native=True,
                             capture_windows=self._picks(2), mic_idx=3)
        for i, w in enumerate(r._sck_workers):
            w.entry = w.window
            w.t0 = 12.34 + i * 0.001
        meta = r._multi_native_meta_dict(1440.0, 900.0, "quartz", 12.34)
        self.assertEqual(meta["mic_index"], 3)

    def test_single_window_meta_is_untouched(self):
        # Off-switch: a single-window (native or display-crop) take's meta
        # must NOT gain a `capture_channels` key. The pinned key sets on
        # avfoundation and single SCK stay exactly what they were.
        r = record.Recorder("/tmp/x", 1, backend="sck", window_native=True,
                            capture_window={"id": 44041, "app": "A", "title": "t",
                                            "x": 0.0, "y": 0.0, "w": 700.0,
                                            "h": 800.0,
                                            "display_origin": [0.0, 0.0]})
        r._cw_entry = r.capture_window
        r._cw_end_rect = [0.0, 0.0, 700.0, 800.0]
        r._sck_size = {"width": 1400, "height": 1600, "fps": 60}
        r._sck_stat = {"appended": 300, "dup": 0, "dropped": 0, "notready": 0}
        meta = r._meta_dict(1440.0, 900.0, "quartz", 12.34, False)
        self.assertNotIn("capture_channels", meta)
        self.assertIn("capture_window", meta)
        self.assertEqual(meta["capture_window"]["mode"], "window_native")


if __name__ == "__main__":
    unittest.main()


class KeyCaptureSettling(unittest.TestCase):
    """`key_capture` must describe the END of a take, not just its start.

    The decode runs in a child process precisely because it sits on a call
    with a native-crash history (see _key_worker.py), so "READY arrived" is
    emphatically not "it survived the take". Before this settled, a worker
    that trapped one second in still left meta.json claiming "activity"
    while events.jsonl held zero key ticks -- which is indistinguishable
    from a take where the user simply never typed, and is exactly how a
    keyboard-capture regression stays invisible.
    """

    class _Worker(object):
        """Stands in for _KeyWorker: just the two methods start() drives."""

        def __init__(self, alive):
            self._alive = alive
            self.stopped = False

        def is_alive(self):
            return self._alive

        def stop(self):
            self.stopped = True

    def _rec(self, state):
        r = record.Recorder("/tmp/x", 0)
        r._key_capture = state
        return r

    def test_worker_alive_at_stop_stays_activity(self):
        r = self._rec("activity")
        r._settle_key_capture(self._Worker(True))
        self.assertEqual(r._key_capture, "activity")

    def test_dead_worker_downgrades_to_failed(self):
        r = self._rec("activity")
        r._settle_key_capture(self._Worker(False))
        self.assertEqual(r._key_capture, "failed")

    def test_disabled_is_never_touched(self):
        # --no-key-log: there was no worker to die, and "disabled" must not
        # be rewritten into a failure the user didn't have.
        r = self._rec("disabled")
        r._settle_key_capture(self._Worker(False))
        self.assertEqual(r._key_capture, "disabled")

    def test_failed_start_stays_failed(self):
        r = self._rec("failed")
        r._settle_key_capture(self._Worker(True))
        self.assertEqual(r._key_capture, "failed")

    def test_no_worker_is_a_no_op(self):
        r = self._rec("activity")
        r._settle_key_capture(None)
        self.assertEqual(r._key_capture, "activity")

    def test_is_alive_raising_never_breaks_the_take(self):
        class _Boom(object):
            def is_alive(self):
                raise OSError("gone")

        r = self._rec("activity")
        r._settle_key_capture(_Boom())
        self.assertEqual(r._key_capture, "activity")

    def test_meta_reports_the_settled_value(self):
        r = self._rec("activity")
        r._settle_key_capture(self._Worker(False))
        meta = r._meta_dict(1440.0, 900.0, "quartz", 0.0, False)
        self.assertEqual(meta["key_capture"], "failed")

    def test_key_worker_really_exposes_is_alive(self):
        # The fake above would happily answer a method _KeyWorker doesn't
        # have; pin the real seam so it can't rot.
        self.assertTrue(callable(getattr(record._KeyWorker, "is_alive", None)))

    def test_settle_runs_before_stop_in_the_record_loop(self):
        """Ordering guard. stop() kills the child, so a settle placed after
        it would read every take -- successful ones included -- as failed."""
        src = inspect.getsource(record.Recorder.start)
        self.assertLess(src.index("_settle_key_capture"),
                        src.index("kb_listener.stop()"))


class FailureErrorLog(unittest.TestCase):
    """A failed take must leave its own diagnosis on disk.

    A failure writes no meta.json, and the RecordError text only ever
    reached the console of whatever launched the recorder -- which, for a
    take started from the bar, is a window nobody is watching. Sessions
    really were found holding events.jsonl and a face.mov with nothing at
    all to say why the screen capture died. error.log is the durable copy.
    """

    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_error_path_sits_in_the_session_dir(self):
        r = record.Recorder(self.td, 0)
        self.assertEqual(r.error_path, os.path.join(self.td, "error.log"))

    def test_message_lands_on_disk(self):
        r = record.Recorder(self.td, 0)
        r._write_error_log("ffmpeg screen capture failed (exit code 1)")
        with open(r.error_path) as f:
            self.assertIn("exit code 1", f.read())

    def test_it_captures_the_ffmpeg_tail_and_diagnosis(self):
        # The whole point: the thing that identifies the failure mode has to
        # survive, not just the headline.
        r = record.Recorder(self.td, 0)
        r._stderr_tail.append("Configuration of video device failed\n")
        r._write_error_log(r._failure_message(1, False))
        with open(r.error_path) as f:
            body = f.read()
        self.assertIn("Configuration of video device failed", body)

    def test_missing_session_dir_is_created(self):
        # A take can die before anything else creates the directory.
        sub = os.path.join(self.td, "never-made")
        r = record.Recorder(sub, 0)
        r._write_error_log("boom")
        self.assertTrue(os.path.exists(os.path.join(sub, "error.log")))

    def test_a_write_failure_never_masks_the_real_error(self):
        r = record.Recorder(self.td, 0)
        with mock.patch("autocine.record.open", side_effect=OSError("nope")):
            r._write_error_log("boom")   # must not raise

    def test_written_before_the_raise(self):
        """Pinned at the call site: the log has to exist by the time the
        RecordError propagates, or a caller that exits on it loses the
        evidence again."""
        src = inspect.getsource(record.Recorder.start)
        self.assertLess(src.index("self._write_error_log(message)"),
                        src.index("raise RecordError(message)"))


class ScreenDeviceValidation(unittest.TestCase):
    """avfoundation video indices are POSITIONAL -- webcams and screens share
    one numbering, so a device connecting or disconnecting renumbers
    everything after it. An index remembered from an older listing can come
    to name a camera, and recording it yields video of the user's face
    instead of their screen."""

    DEVS = {"video": [(0, "FaceTime HD Camera"),
                      (1, "Capture screen 0"),
                      (2, "Capture screen 1")],
            "audio": []}

    def test_screen_indices_are_accepted(self):
        self.assertTrue(devices.is_screen_device(self.DEVS, 1))
        self.assertTrue(devices.is_screen_device(self.DEVS, 2))

    def test_camera_index_is_rejected(self):
        self.assertFalse(devices.is_screen_device(self.DEVS, 0))

    def test_unknown_index_is_rejected(self):
        self.assertFalse(devices.is_screen_device(self.DEVS, 7))

    def test_none_is_rejected(self):
        self.assertFalse(devices.is_screen_device(self.DEVS, None))

    def test_empty_device_list_is_rejected(self):
        self.assertFalse(
            devices.is_screen_device({"video": [], "audio": []}, 0))

    def test_the_renumbering_that_causes_the_bug(self):
        """Index 1 is the screen until a virtual camera appears ahead of it,
        after which the very same number is a CAMERA. This is the shift that
        turns a remembered pick into a recording of the user's face."""
        before = {"video": [(0, "FaceTime HD Camera"),
                            (1, "Capture screen 0")], "audio": []}
        after = {"video": [(0, "FaceTime HD Camera"),
                           (1, "OBS Virtual Camera"),
                           (2, "Capture screen 0")], "audio": []}
        self.assertTrue(devices.is_screen_device(before, 1))
        self.assertFalse(devices.is_screen_device(after, 1))
        self.assertTrue(devices.is_screen_device(after, 2))

    def test_never_disagrees_with_find_screen_device(self):
        # Auto-detect and validation must apply the same test, or "Auto" and
        # an explicit pick of the same device could resolve differently.
        for devs in (self.DEVS,
                     {"video": [(0, "Capture screen 0")], "audio": []},
                     {"video": [(0, "Studio Camera")], "audio": []},
                     {"video": [], "audio": []}):
            found = devices.find_screen_device(devs)
            if found is None:
                self.assertFalse(any(devices.is_screen_device(devs, i)
                                     for i, _ in devs["video"]))
            else:
                self.assertTrue(devices.is_screen_device(devs, found))


class _StdoutProc(object):
    """A fake capture process exposing only the stdout `_read_sck_stdout` reads."""
    def __init__(self, data):
        self.stdout = io.BytesIO(data)


class _WaitProc(object):
    """A fake process that records the timeouts `_wait_clean` waits with and
    exits immediately, so the grace budget can be pinned without waiting."""
    def __init__(self):
        self.wait_timeouts = []
        self.returncode = 0
    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        return 0
    def terminate(self):
        pass
    def kill(self):
        pass


class SckFinalization(unittest.TestCase):
    """An SCK take must not be filed as a success unless it actually finalized.

    A worker stopped mid-tail leaves every captured frame in raw.mov but no
    `moov` index -- the AVAssetWriter tail-loss risk (docs/architecture.md).
    The recorder catches that, and gives the worker enough grace not to cause
    it in the first place. avfoundation is deliberately unaffected."""

    def test_read_sck_stdout_records_the_done_signal(self):
        r = record.Recorder("/tmp/x", 1, backend="sck")
        r.proc = _StdoutProc(b"STAT 4 5 1 0 0\nDONE 105 1.75\n")
        r._read_sck_stdout()
        self.assertEqual(r._sck_done, {"frames": 105, "media": 1.75})
        self.assertIsNotNone(r._sck_stat)          # unrelated lines still parse

    def test_done_stays_none_when_the_worker_never_finalizes(self):
        r = record.Recorder("/tmp/x", 1, backend="sck")
        r.proc = _StdoutProc(b"STAT 4 5 1 0 0\n")   # stops before DONE
        r._read_sck_stdout()
        self.assertIsNone(r._sck_done)

    def test_wait_clean_outwaits_the_sck_worker_finish_budget(self):
        # The worker's finish() runs up to ~35s (stopCapture 5s + finishWriting
        # 30s); the parent must outwait it or its SIGKILL is what loses the
        # tail. ffmpeg's trailer is sub-second, so it keeps the tight budget.
        r = record.Recorder("/tmp/x", 1, backend="sck")
        r.proc = _WaitProc()
        r._wait_clean()
        self.assertGreaterEqual(r.proc.wait_timeouts[0], 35)

    def test_wait_clean_keeps_ffmpeg_on_the_tight_budget(self):
        r = record.Recorder("/tmp/x", 1, backend="avfoundation")
        r.proc = _WaitProc()
        r._wait_clean()
        self.assertEqual(r.proc.wait_timeouts[0], 8)

    def test_unfinalized_message_explains_tail_loss_not_a_capture_failure(self):
        r = record.Recorder("/tmp/x", 1, backend="sck")
        r._sck_stat = {"appended": 3787, "dup": 0, "dropped": 0, "notready": 0}
        r._sck_done = None
        msg = r._sck_unfinalized_message(rc=-9)
        low = msg.lower()
        self.assertIn("finalize", low)
        self.assertIn("moov", low)
        self.assertIn("3787", msg)          # names the frames that ARE there
        self.assertIn("done", low)          # calls out the missing DONE signal
        # Must NOT read like the capture/permission failure -- capture worked.
        self.assertNotIn("screen recording", low)

    def test_unfinalized_message_omits_the_done_note_when_done_was_seen(self):
        r = record.Recorder("/tmp/x", 1, backend="sck")
        r._sck_stat = {"appended": 10}
        r._sck_done = {"frames": 10, "media": 0.16}
        self.assertNotIn("never reported DONE", r._sck_unfinalized_message(rc=0))
