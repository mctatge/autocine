"""The CFR gate, and the pinned screen-capture argv.

`autocine/sck.py` imports no ScreenCaptureKit and no PyObjC on purpose, so all
of this runs anywhere: the rate parsing and verdicts go through an injected
probe, and only the two end-to-end cases at the bottom touch ffmpeg (skipped
when it isn't installed). No permissions, no recording, no display.
"""
import contextlib
import io
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from autocine import record
from autocine import sck


def _probe(avg, base, nb=None, codec="video"):
    """A stand-in for ffprobe returning one video stream."""
    stream = {"codec_type": codec, "avg_frame_rate": avg, "r_frame_rate": base}
    if nb is not None:
        stream["nb_frames"] = nb
    return lambda path: {"streams": [stream]}


def _grid(hz, n, start=0.0, jitter=None):
    """n frame times on a perfect `hz` grid, optionally perturbed."""
    out = []
    for i in range(n):
        t = start + i / float(hz)
        if jitter:
            t += jitter(i)
        out.append(t)
    return out


class InferGridHz(unittest.TestCase):
    """Which grid did the frames REALLY arrive on?"""

    def test_a_clean_60_grid_reads_as_60(self):
        self.assertEqual(sck.infer_grid_hz(_grid(60.0, 200)), 60.0)

    def test_59_94_is_not_mistaken_for_60(self):
        # THE case this function exists for. The two differ by one part in a
        # thousand; a nominal-60 grid on a 59.94 panel drifts about a frame
        # every 16 seconds, silently.
        self.assertEqual(sck.infer_grid_hz(_grid(59.94, 400)), 59.94)

    def test_60_is_not_mistaken_for_59_94_either(self):
        # The mirror case matters just as much: 59.94 sorts first, so a
        # sloppy tolerance would claim it for genuinely-60 data.
        self.assertEqual(sck.infer_grid_hz(_grid(60.0, 400)), 60.0)

    def test_gaps_from_idle_frames_do_not_confuse_it(self):
        # SCK emits nothing while the screen is static, so real input is a
        # sparse subset of the grid, not every slot.
        full = _grid(60.0, 600)
        sparse = [t for i, t in enumerate(full) if i % 37 == 0]
        self.assertEqual(sck.infer_grid_hz(sparse), 60.0)

    def test_120_hz_samples_are_not_flattened_to_60(self):
        # The odd frames land halfway between 60 Hz slots, so 120 really is
        # the coarsest grid that explains them. (The reverse DOES collapse:
        # see the next test -- that asymmetry is why candidates are tried in
        # ascending order.)
        self.assertEqual(sck.infer_grid_hz(_grid(120.0, 400)), 120.0)

    def test_60_hz_samples_are_not_promoted_to_120(self):
        # They fit a 120 grid perfectly (every frame on an even slot), so
        # only the ascending order stops us claiming a finer grid than we
        # can support and padding the file with duplicates.
        self.assertEqual(sck.infer_grid_hz(_grid(60.0, 400)), 60.0)

    def test_30_hz_samples_read_as_30(self):
        self.assertEqual(sck.infer_grid_hz(_grid(30.0, 200)), 30.0)

    def test_junk_timings_are_refused_not_guessed(self):
        import random
        rnd = random.Random(7)
        times = sorted(rnd.uniform(0, 10) for _ in range(300))
        self.assertIsNone(sck.infer_grid_hz(times))

    def test_a_short_sample_refuses_rather_than_coin_flips(self):
        # 0.1s of frames cannot separate 59.94 from 60 however many there
        # are -- only elapsed time helps. Refusing is the honest answer.
        self.assertIsNone(sck.infer_grid_hz(_grid(60.0, 6)))

    def test_too_few_samples_is_none(self):
        self.assertIsNone(sck.infer_grid_hz([]))
        self.assertIsNone(sck.infer_grid_hz([0.0]))

    def test_small_jitter_still_resolves(self):
        # Real timestamps are not exact; a little noise must not flip it to
        # a refusal.
        times = _grid(60.0, 400, jitter=lambda i: (0.0002 if i % 2 else -0.0002))
        self.assertEqual(sck.infer_grid_hz(times), 60.0)


class PlanSlots(unittest.TestCase):
    """Frame arrival -> CFR slots. The whole constant-rate story."""

    STEP = 1.0 / 60.0

    def test_the_first_frame_defines_the_origin(self):
        self.assertEqual(sck.plan_slots(123.456, 123.456, 60, None),
                         ([], 0, False))

    def test_the_first_frame_is_slot_zero_whatever_its_pts(self):
        # pts0 is whatever the machine's clock said; only offsets matter.
        self.assertEqual(sck.plan_slots(9999.0, 0.0, 60, None), ([], 0, False))

    def test_a_consecutive_frame_needs_no_fill(self):
        self.assertEqual(sck.plan_slots(self.STEP, 0.0, 60, 0),
                         ([], 1, False))

    def test_an_idle_gap_is_filled_with_the_previous_frame(self):
        # SCK delivers nothing while the screen is static. Writing these as
        # they arrive is exactly what makes the file variable-rate.
        repeats, slot, dropped = sck.plan_slots(5 * self.STEP, 0.0, 60, 0)
        self.assertEqual(repeats, [1, 2, 3, 4])
        self.assertEqual(slot, 5)
        self.assertFalse(dropped)

    def test_two_seconds_of_stillness_fills_the_whole_span(self):
        repeats, slot, _ = sck.plan_slots(2.0, 0.0, 60, 0)
        self.assertEqual(slot, 120)
        self.assertEqual(len(repeats), 119)
        self.assertEqual(repeats[0], 1)
        self.assertEqual(repeats[-1], 119)

    def test_a_frame_in_an_already_written_slot_is_dropped(self):
        # Two frames inside one 16ms slot means the display showed one of
        # them. Nudging the second forward would push everything after it
        # off its true time, permanently.
        self.assertEqual(sck.plan_slots(self.STEP * 1.4, 0.0, 60, 1),
                         ([], None, True))

    def test_an_out_of_order_frame_is_dropped(self):
        self.assertEqual(sck.plan_slots(0.5 * self.STEP, 0.0, 60, 3),
                         ([], None, True))

    def test_slots_round_to_nearest_not_down(self):
        # A frame 0.9 of a step late belongs in the NEXT slot; truncating
        # would systematically pull the whole take earlier.
        _, slot, _ = sck.plan_slots(0.9 * self.STEP, 0.0, 60, None)
        self.assertEqual(slot, 0)      # first frame is always the origin
        _, slot, _ = sck.plan_slots(1.9 * self.STEP, 0.0, 60, 0)
        self.assertEqual(slot, 2)

    def test_max_fill_clamps_an_absurd_gap(self):
        # A slept machine otherwise asks for a fill of hundreds of thousands
        # of frames, turning a hiccup into an out-of-disk.
        repeats, slot, _ = sck.plan_slots(3600.0, 0.0, 60, 0, max_fill=10)
        self.assertEqual(len(repeats), 10)
        self.assertEqual(slot, 11)

    def test_no_clamp_by_default(self):
        repeats, _, _ = sck.plan_slots(10.0, 0.0, 60, 0)
        self.assertEqual(len(repeats), 599)

    def test_a_bad_fps_drops_rather_than_dividing_by_zero(self):
        for bad in (0, None, "x"):
            self.assertEqual(sck.plan_slots(1.0, 0.0, bad, 0), ([], None, True))

    def test_a_whole_take_stays_on_the_grid(self):
        """End to end: sparse arrivals -> a dense, gapless, correct timeline."""
        arrivals = [t for i, t in enumerate(_grid(60.0, 600)) if i % 7 == 0]
        written = []
        last = None
        for t in arrivals:
            repeats, slot, dropped = sck.plan_slots(t, arrivals[0], 60, last)
            if dropped:
                continue
            written.extend(repeats)
            written.append(slot)
            last = slot
        # Every slot exactly once, in order, with no holes -- which is what
        # "constant frame rate" means to everything downstream.
        self.assertEqual(written, list(range(len(written))))
        # ...and the timeline still ends where the frames really ended.
        self.assertAlmostEqual((len(written) - 1) / 60.0,
                               arrivals[-1] - arrivals[0], places=6)


class PlanIdleFill(unittest.TestCase):
    """Wall-clock fill: the half of CFR `plan_slots` structurally cannot do.

    `plan_slots` only runs when a frame ARRIVES, and SCK delivers nothing
    while the screen is still -- so a motionless window used to stop the
    timeline dead. Measured before this existed: a 10.0s idle window wrote
    1.08s, a 26.0s three-window take wrote 1.28s per channel, both with
    `dropped: 0`.
    """

    STEP = 1.0 / 60.0

    def test_nothing_to_fill_before_the_first_frame(self):
        # No pts0/last_slot yet: there is no held frame to duplicate, and the
        # first arrival defines slot 0 whenever it comes.
        self.assertEqual(sck.plan_idle_fill(100.0, None, 60, None), [])
        self.assertEqual(sck.plan_idle_fill(100.0, 0.0, 60, None), [])

    def test_a_still_second_is_filled_up_to_the_lag(self):
        # 1.0s of silence at 60fps, lagging 0.5s -> slots 1..30.
        got = sck.plan_idle_fill(1.0, 0.0, 60, 0, lag_sec=0.5)
        self.assertEqual(got, list(range(1, 31)))

    def test_the_lag_holds_the_fill_behind_the_clock(self):
        # Inside the lag window nothing is written -- that headroom is what
        # keeps an in-flight frame from being displaced onto a written slot.
        self.assertEqual(sck.plan_idle_fill(0.4, 0.0, 60, 0, lag_sec=0.5), [])

    def test_zero_lag_closes_the_tail_exactly(self):
        # What `finish` uses: the stream is stopped, nothing is in flight.
        got = sck.plan_idle_fill(1.0, 0.0, 60, 0, lag_sec=0.0)
        self.assertEqual(got, list(range(1, 61)))

    def test_it_never_rewrites_a_written_slot(self):
        # The timeline is already ahead of the lagged target: stay put, or
        # `plan_slots` would later drop a real frame as a duplicate.
        self.assertEqual(sck.plan_idle_fill(1.0, 0.0, 60, 200, lag_sec=0.5), [])

    def test_max_fill_bounds_a_stalled_stream(self):
        # A machine that slept must not ask for a million duplicate frames.
        got = sck.plan_idle_fill(10_000.0, 0.0, 60, 0, lag_sec=0.0, max_fill=90)
        self.assertEqual(got, list(range(1, 91)))

    def test_it_composes_with_plan_slots(self):
        # After a fill, a real arrival still lands on its OWN true slot --
        # the fill advances the timeline, it does not retime the footage.
        filled = sck.plan_idle_fill(1.0, 0.0, 60, 0, lag_sec=0.5)
        last = filled[-1]                       # slot 30
        repeats, slot, dropped = sck.plan_slots(2.0, 0.0, 60, last)
        self.assertFalse(dropped)
        self.assertEqual(slot, 120)             # its true time, not 31
        self.assertEqual(repeats, list(range(31, 120)))

    def test_a_frame_still_in_flight_survives_the_lag(self):
        # Arrival stamped 0.75s reaching us while the clock says 1.0s: the
        # 0.5s lag means we only filled to 0.5s, so it is not dropped.
        filled = sck.plan_idle_fill(1.0, 0.0, 60, 0, lag_sec=0.5)
        _r, slot, dropped = sck.plan_slots(0.75, 0.0, 60, filled[-1])
        self.assertFalse(dropped)
        self.assertEqual(slot, 45)

    def test_bad_input_never_raises(self):
        self.assertEqual(sck.plan_idle_fill(1.0, 0.0, 0, 0), [])
        self.assertEqual(sck.plan_idle_fill("x", 0.0, 60, 0), [])

    def test_now_host_pts_is_the_uptime_clock(self):
        # Same timebase as CMClockGetHostTimeClock -- see host_pts_to_monotonic.
        a = time.clock_gettime(time.CLOCK_UPTIME_RAW)
        b = sck.now_host_pts()
        self.assertLess(abs(b - a), 0.5)


class ResolveBackend(unittest.TestCase):
    def test_defaults_to_avfoundation(self):
        # "Every off switch must be bit-exact" cannot survive a backend
        # change -- no SCK file is byte-identical to an ffmpeg one -- so the
        # only honest off switch is the old path still being the default.
        self.assertEqual(sck.resolve_backend(), "avfoundation")
        self.assertEqual(sck.resolve_backend(None, {}), "avfoundation")

    def test_explicit_wins_over_env(self):
        self.assertEqual(
            sck.resolve_backend("avfoundation",
                                {"AUTOCINE_CAPTURE_BACKEND": "sck"}),
            "avfoundation")

    def test_env_selects_sck(self):
        self.assertEqual(
            sck.resolve_backend(None, {"AUTOCINE_CAPTURE_BACKEND": "sck"}),
            "sck")

    def test_case_and_whitespace_are_forgiven(self):
        self.assertEqual(sck.resolve_backend("  SCK "), "sck")

    def test_a_typo_falls_back_rather_than_raising(self):
        # A typo in an env var must not be able to stop someone recording.
        # The CLI validates its own flag up front, where a typo can still be
        # reported to the person who typed it.
        self.assertEqual(
            sck.resolve_backend(None, {"AUTOCINE_CAPTURE_BACKEND": "skc"}),
            "avfoundation")


class WorkerWindowNativeConfig(unittest.TestCase):
    """The child worker reads `capture_window_id` off its JSON config -- the one
    field that flips it from display capture to the window's own buffer. Pure:
    `_sck_worker.Capture.__init__` imports no ObjC (SCK loads inside build())."""

    def _capture(self, **extra):
        from autocine import _sck_worker
        cfg = dict({"out": "/tmp/x.mov", "fps": 60}, **extra)
        return _sck_worker.Capture(cfg)

    def test_capture_window_id_parsed_when_present(self):
        self.assertEqual(self._capture(capture_window_id=4242).capture_window_id,
                         4242)

    def test_capture_window_id_none_when_absent(self):
        self.assertIsNone(self._capture().capture_window_id)

    def test_apply_pending_exclusions_is_inert_in_window_native(self):
        # The exclude path would rebuild a DISPLAY filter and break a
        # window-native capture; it must no-op when a target window is set.
        cap = self._capture(capture_window_id=4242)
        cap.request_exclusions([1, 2, 3])
        cap.apply_pending_exclusions()          # must not touch _want_exclude
        self.assertEqual(cap._want_exclude, [1, 2, 3])


class PlanRestoreAction(unittest.TestCase):
    """The pure minimize/restore edge logic (no ObjC). See
    _sck_worker.check_window_restore and docs/architecture.md."""

    def test_no_edge_is_none(self):
        self.assertEqual(sck.plan_restore_action(True, True, 0, 6), "none")
        self.assertEqual(sck.plan_restore_action(False, False, 3, 6), "none")

    def test_unreadable_poll_changes_nothing(self):
        # A transient CGWindowList read failure must never trigger a rebuild.
        self.assertEqual(sck.plan_restore_action(True, None, 0, 6), "none")
        self.assertEqual(sck.plan_restore_action(False, None, 3, 6), "none")

    def test_going_offscreen_is_hidden(self):
        self.assertEqual(sck.plan_restore_action(True, False, 0, 6), "hidden")

    def test_coming_back_rebuilds(self):
        self.assertEqual(sck.plan_restore_action(False, True, 0, 6), "rebuild")
        self.assertEqual(sck.plan_restore_action(False, True, 5, 6), "rebuild")

    def test_gives_up_after_max_tries(self):
        self.assertEqual(sck.plan_restore_action(False, True, 6, 6), "giveup")
        self.assertEqual(sck.plan_restore_action(False, True, 9, 6), "giveup")


class WorkerRestoreStateMachine(unittest.TestCase):
    """`check_window_restore` drives the pure edge logic and the rebuild, with
    the two ObjC bits (_target_onscreen, _rebuild_stream) stubbed. No SCK, no
    display -- Capture.__init__ imports no ObjC."""

    def _cap(self, **extra):
        from autocine import _sck_worker
        cfg = dict({"out": "/tmp/x.mov", "fps": 60}, **extra)
        return _sck_worker.Capture(cfg)

    def _run(self, cap, readings):
        """One check_window_restore call per on-screen reading, poll-throttle
        disabled and the worker's stdout swallowed."""
        from autocine import _sck_worker
        seq = list(readings)
        cap._target_onscreen = lambda: seq.pop(0)
        with mock.patch.object(_sck_worker, "ONSCREEN_POLL_SEC", 0.0):
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(len(readings)):
                    cap._last_onscreen_check = 0.0
                    cap.check_window_restore()

    def test_inert_for_display_capture(self):
        cap = self._cap()                       # no capture_window_id
        cap._target_onscreen = lambda: self.fail("must not poll for display")
        cap._rebuild_stream = lambda: self.fail("must not rebuild for display")
        cap.check_window_restore()              # returns before touching either

    def test_minimize_then_restore_rebuilds_once(self):
        cap = self._cap(capture_window_id=4242)
        calls = []
        cap._rebuild_stream = lambda: (calls.append(1), True)[1]
        self._run(cap, [True, False, True])     # steady, minimize, restore
        self.assertEqual(len(calls), 1)
        self.assertTrue(cap._win_onscreen)
        self.assertEqual(cap._rebuild_fails, 0)

    def test_occlusion_never_rebuilds(self):
        # A covered window stays on-screen -> no False reading -> no rebuild.
        cap = self._cap(capture_window_id=4242)
        calls = []
        cap._rebuild_stream = lambda: (calls.append(1), True)[1]
        self._run(cap, [True, True, True, True])
        self.assertEqual(calls, [])

    def test_rebuild_retries_until_success(self):
        cap = self._cap(capture_window_id=4242)
        results, calls = [False, False, True], []

        def stub():
            calls.append(1)
            return results.pop(0)

        cap._rebuild_stream = stub
        self._run(cap, [False, True, True, True])   # hidden, then 3 restore reads
        self.assertEqual(len(calls), 3)
        self.assertTrue(cap._win_onscreen)
        self.assertEqual(cap._rebuild_fails, 0)

    def test_gives_up_after_max_tries(self):
        from autocine import _sck_worker
        cap = self._cap(capture_window_id=4242)
        calls = []
        cap._rebuild_stream = lambda: (calls.append(1), False)[1]
        reads = [False] + [True] * (_sck_worker.MAX_REBUILD_TRIES + 3)
        self._run(cap, reads)
        # Attempted exactly MAX_REBUILD_TRIES times, then giveup stops retrying.
        self.assertEqual(len(calls), _sck_worker.MAX_REBUILD_TRIES)
        self.assertTrue(cap._win_onscreen)


class ParseWorkerLine(unittest.TestCase):
    """The worker's stdout is a control channel for a live recording."""

    def test_ready(self):
        self.assertEqual(sck.parse_worker_line("READY"), ("ready", {}))

    def test_size(self):
        kind, p = sck.parse_worker_line("SIZE 2880 1800 60")
        self.assertEqual(kind, "size")
        self.assertEqual((p["width"], p["height"], p["fps"]), (2880, 1800, 60))

    def test_t0_keeps_full_precision(self):
        # This anchors every click to every frame; a float rounded here is a
        # recording whose zooms sit slightly off their clicks.
        kind, p = sck.parse_worker_line("T0 199502.502454125")
        self.assertEqual(kind, "t0")
        self.assertEqual(p["pts0"], 199502.502454125)

    def test_filter_reports_unresolved_ids(self):
        kind, p = sck.parse_worker_line("FILTER 2 1")
        self.assertEqual((kind, p["applied"], p["missing"]), ("filter", 2, 1))

    def test_stat(self):
        kind, p = sck.parse_worker_line("STAT 496 497 22 3 4")
        self.assertEqual(kind, "stat")
        self.assertEqual(p["slot"], 496)
        self.assertEqual(p["dropped"], 3)
        self.assertEqual(p["notready"], 4)

    def test_done(self):
        kind, p = sck.parse_worker_line("DONE 443 7.383333")
        self.assertEqual((kind, p["frames"]), ("done", 443))

    def test_warn_and_err_keep_their_text(self):
        kind, p = sck.parse_worker_line("WARN fill clamped at 1800 frames")
        self.assertEqual(kind, "warn")
        self.assertIn("clamped", p["detail"])
        kind, p = sck.parse_worker_line("ERR start denied by TCC")
        self.assertEqual((kind, p["domain"]), ("err", "start"))
        self.assertIn("denied", p["detail"])

    def test_junk_is_dropped_not_fatal(self):
        # A line we don't understand must never take down a recording in
        # progress.
        for bad in ("", "   ", "GARBAGE", "SIZE", "STAT 1 2", "T0 abc",
                    None, "SIZE a b c"):
            self.assertEqual(sck.parse_worker_line(bad), ("", {}), repr(bad))


class HostPtsToMonotonic(unittest.TestCase):
    def test_converts_into_our_own_timeframe(self):
        # Nothing about the child's clock is trusted: the conversion uses two
        # adjacent reads taken HERE. (time.monotonic() is per-process on this
        # stack -- the trap that put every key timestamp a process-age early.)
        got = sck.host_pts_to_monotonic(1000.0, mono=50.0, uptime=1005.0)
        self.assertAlmostEqual(got, 45.0)

    def test_a_pts_at_this_instant_maps_to_now(self):
        got = sck.host_pts_to_monotonic(2000.0, mono=7.5, uptime=2000.0)
        self.assertAlmostEqual(got, 7.5)

    def test_live_call_lands_near_now(self):
        import time as _t
        before = _t.monotonic()
        got = sck.host_pts_to_monotonic(_t.clock_gettime(_t.CLOCK_UPTIME_RAW))
        after = _t.monotonic()
        self.assertGreaterEqual(got, before - 0.05)
        self.assertLessEqual(got, after + 0.05)


class ParseRate(unittest.TestCase):
    def test_reads_a_rational(self):
        self.assertEqual(sck._parse_rate("60/1"), 60.0)
        self.assertAlmostEqual(sck._parse_rate("30000/1001"), 29.97, places=2)

    def test_zero_over_zero_is_unknown_not_zero(self):
        # ffprobe's "I don't know". Treating it as a number is how a missing
        # measurement becomes a confident wrong answer.
        self.assertIsNone(sck._parse_rate("0/0"))
        self.assertIsNone(sck._parse_rate("0/1"))

    def test_junk_is_none(self):
        for bad in (None, "", "N/A", "abc", "60/0"):
            self.assertIsNone(sck._parse_rate(bad), bad)


class VerifyCfr(unittest.TestCase):
    def test_matching_rates_at_the_requested_fps_are_ok(self):
        status, detail = sck.verify_cfr("x.mov", 60, run=_probe("60/1", "60/1"))
        self.assertEqual(status, "ok")
        self.assertIn("avg_frame_rate", detail)

    def test_disagreeing_rates_are_suspect(self):
        status, detail = sck.verify_cfr(
            "x.mov", 60, run=_probe("473/10", "60/1"))
        self.assertEqual(status, "suspect")
        self.assertIn("variable frame rate", detail)
        # The detail must name the numbers -- the first question anyone asks
        # about a drifting render is "was the file CFR?".
        self.assertIn("avg_frame_rate", detail)
        self.assertIn("r_frame_rate", detail)

    def test_constant_but_at_the_wrong_rate_is_suspect(self):
        # The realistic mismatch: a 59.94 Hz display against a nominal-60
        # grid accumulates about a frame every 16 seconds.
        status, detail = sck.verify_cfr(
            "x.mov", 60, run=_probe("30000/1001", "30000/1001"))
        self.assertEqual(status, "suspect")
        self.assertIn("not at the requested", detail)

    def test_59_94_is_caught_against_a_60_request(self):
        status, _ = sck.verify_cfr(
            "x.mov", 60, run=_probe("60000/1001", "60000/1001"))
        self.assertEqual(status, "suspect")

    def test_unreadable_is_unknown_not_ok(self):
        # "we didn't look" and "we looked and it's fine" must never collapse
        # into one value -- the same tri-state lesson as meta["key_capture"].
        status, _ = sck.verify_cfr("x.mov", 60, run=_probe(None, None))
        self.assertEqual(status, "unknown")

    def test_no_video_stream_is_unknown(self):
        status, _ = sck.verify_cfr(
            "x.mov", 60, run=lambda p: {"streams": [{"codec_type": "audio"}]})
        self.assertEqual(status, "unknown")

    def test_a_probe_that_raises_is_unknown_not_fatal(self):
        def boom(path):
            raise OSError("no ffprobe")
        status, _ = sck.verify_cfr("x.mov", 60, run=boom)
        self.assertEqual(status, "unknown")

    def test_a_wedged_probe_is_unknown_not_a_hang(self):
        # _run_ffprobe passes timeout=, so a wedged ffprobe raises
        # TimeoutExpired rather than blocking forever. That matters because
        # _check_cfr runs on EVERY backend and BEFORE meta.json is written:
        # a hang there would leave a complete raw.mov with no meta.json, which
        # this project reads as a failed take. TimeoutExpired is an Exception,
        # so it degrades to "unknown" like any other probe failure.
        def wedged(path):
            raise subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=20.0)
        status, _ = sck.verify_cfr("x.mov", 60, run=wedged)
        self.assertEqual(status, "unknown")

    def test_the_probe_command_actually_carries_a_timeout(self):
        # The guard above only holds if the real _run_ffprobe asks for it --
        # a mocked runner can't prove that. Pin the kwarg on the real call.
        seen = {}

        class _Proc(object):
            stdout = b"{}"

        def fake_run(cmd, **kw):
            seen.update(kw)
            seen["cmd"] = cmd
            return _Proc()

        with mock.patch.object(sck.subprocess, "run", fake_run):
            sck._run_ffprobe("x.mov")
        self.assertEqual(seen.get("timeout"), sck.PROBE_TIMEOUT_SEC)
        self.assertEqual(seen["cmd"][0], "ffprobe")

    def test_tiny_float_error_still_counts_as_ok(self):
        status, _ = sck.verify_cfr(
            "x.mov", 60, run=_probe("60000/1000", "60/1"))
        self.assertEqual(status, "ok")


class CheckCfrOnTheRecorder(unittest.TestCase):
    """`Recorder._check_cfr` — warns, records, and never fails a take."""

    def _rec(self):
        r = record.Recorder(tempfile.mkdtemp(), 0)
        r.fps = 60
        return r

    def test_suspect_is_stored_and_warned_but_does_not_raise(self):
        r = self._rec()
        orig = sck.verify_cfr
        sck.verify_cfr = lambda path, fps: ("suspect", "variable frame rate: x")
        try:
            self.assertEqual(r._check_cfr(), "suspect")
        finally:
            sck.verify_cfr = orig
        self.assertEqual(r._cfr, "suspect")
        self.assertIn("variable", r._cfr_detail)

    def test_a_probe_explosion_never_kills_a_finished_recording(self):
        # The file is written and watchable by this point. A measurement
        # blowing up must not turn a good take into a RecordError.
        r = self._rec()
        orig = sck.verify_cfr

        def boom(path, fps):
            raise RuntimeError("kaboom")
        sck.verify_cfr = boom
        try:
            self.assertEqual(r._check_cfr(), "unknown")
        finally:
            sck.verify_cfr = orig
        self.assertIn("kaboom", r._cfr_detail)


class ScreenCmdIsPinned(unittest.TestCase):
    """The avfoundation argv, frozen.

    Nothing pinned this command before it was extracted from `start()`.
    `-thread_queue_size` is load-bearing -- it must precede `-i` or ffmpeg
    silently ignores it, and a starved queue DROPS AUDIO PACKETS (heard as
    static, seen as picture sliding behind sound). A reordering that moved it
    after `-i` would still run, still record, and quietly degrade audio; only
    a literal comparison catches that.
    """

    def _rec(self, **kw):
        r = record.Recorder("/tmp/x", 1, **kw)
        return r

    def test_the_exact_command_for_a_plain_take(self):
        r = self._rec()
        r.fps, r.crf = 60, 18
        self.assertEqual(r._screen_cmd(), [
            "ffmpeg", "-y", "-hide_banner",
            "-f", "avfoundation", "-capture_cursor", "1",
            "-framerate", "60",
            "-thread_queue_size", str(record.THREAD_QUEUE_SIZE),
            # "none" rather than an empty slot: an avfoundation input with a
            # trailing colon would open the DEFAULT audio device, so a
            # no-mic take would silently record the room.
            "-i", "1:none",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-crf", "18", "-pix_fmt", "yuv420p",
            "-r", "60",
            "-progress", "pipe:1", "-nostats", r.raw_path,
        ])

    def test_thread_queue_size_precedes_the_input(self):
        # Stated separately from the frozen literal so the REASON survives
        # even if the literal is ever legitimately updated.
        cmd = self._rec()._screen_cmd()
        self.assertLess(cmd.index("-thread_queue_size"), cmd.index("-i"))

    def test_mic_adds_the_resampler_and_pins_48k(self):
        r = self._rec(mic_idx=0)
        cmd = r._screen_cmd()
        self.assertIn("-ar", cmd)
        self.assertEqual(cmd[cmd.index("-ar") + 1], "48000")
        self.assertIn("aresample=async=1:first_pts=0", cmd)

    def test_no_mic_means_no_audio_flags_at_all(self):
        cmd = self._rec()._screen_cmd()
        for flag in ("-c:a", "-ar", "-af", "-b:a"):
            self.assertNotIn(flag, cmd)

    def test_synthetic_cursor_turns_capture_cursor_off(self):
        r = self._rec(cursor_mode="synthetic")
        cmd = r._screen_cmd()
        self.assertEqual(cmd[cmd.index("-capture_cursor") + 1], "0")

    def test_cfr_is_forced(self):
        r = self._rec()
        r.fps = 30
        cmd = r._screen_cmd()
        self.assertEqual(cmd[cmd.index("-r") + 1], "30")


class MicResolution(unittest.TestCase):
    """avfoundation ordinal -> AVFoundation uniqueID.

    Recording the WRONG input is the failure that matters here: this app
    already warns that avfoundation indices are not stable (anything
    connecting or disconnecting renumbers everything after it), and the
    screen-device version of this mistake records the user's face.
    """

    def _devices(self, listing):
        from autocine import devices
        orig = devices.audio_unique_ids
        devices.audio_unique_ids = lambda: list(listing)
        self.addCleanup(lambda: setattr(devices, "audio_unique_ids", orig))
        return devices

    def test_resolves_by_position(self):
        d = self._devices([("uid-a", "MacBook Air Microphone"),
                           ("uid-b", "Microsoft Teams Audio")])
        self.assertEqual(d.mic_unique_id(0), "uid-a")
        self.assertEqual(d.mic_unique_id(1), "uid-b")

    def test_a_matching_name_confirms_the_choice(self):
        d = self._devices([("uid-a", "MacBook Air Microphone")])
        self.assertEqual(d.mic_unique_id(0, "MacBook Air Microphone"), "uid-a")

    def test_a_mismatched_name_refuses_rather_than_picking(self):
        # The numbering moved under us. Guessing here is how a take ends up
        # recording a different microphone than the one the user chose.
        d = self._devices([("uid-a", "MacBook Air Microphone"),
                           ("uid-b", "Microsoft Teams Audio")])
        self.assertIsNone(d.mic_unique_id(0, "Microsoft Teams Audio"))

    def test_out_of_range_and_junk_are_none(self):
        d = self._devices([("uid-a", "Mic")])
        for bad in (99, -1, None, "x"):
            self.assertIsNone(d.mic_unique_id(bad), repr(bad))

    def test_no_devices_is_none_not_a_default(self):
        d = self._devices([])
        self.assertIsNone(d.mic_unique_id(0))


class MicConfig(unittest.TestCase):
    def _rec(self, **kw):
        return record.Recorder(tempfile.mkdtemp(), 1, backend="sck", **kw)

    def _stub(self, uid):
        from autocine import devices
        orig_u, orig_l = devices.mic_unique_id, devices.list_avf_devices
        devices.mic_unique_id = lambda o, n=None: uid
        devices.list_avf_devices = lambda: {"audio": [(0, "Mic")]}
        self.addCleanup(lambda: (setattr(devices, "mic_unique_id", orig_u),
                                 setattr(devices, "list_avf_devices", orig_l)))

    def test_no_mic_means_no_mic_key(self):
        self.assertNotIn("mic_unique_id", self._rec()._sck_config())

    def test_a_resolved_mic_rides_the_config(self):
        self._stub("uid-a")
        self.assertEqual(self._rec(mic_idx=0)._sck_config()["mic_unique_id"],
                         "uid-a")

    def test_an_unresolvable_mic_records_without_audio(self):
        # Silently recording the wrong input would be worse than recording
        # none, so an unresolved ordinal drops the key entirely.
        self._stub(None)
        self.assertNotIn("mic_unique_id", self._rec(mic_idx=0)._sck_config())


class ExclusionPoller(unittest.TestCase):
    """Keeping the live filter in step with windows created mid-take.

    The facecam bubble pops out of the pill into its own NSWindow, and the
    picker opens — both AFTER recording starts, so neither is in a filter
    built at start time.
    """

    class _Proc(object):
        def __init__(self, alive=True):
            self.written = []
            self._alive = alive
            self.stdin = self
        def poll(self):
            return None if self._alive else 0
        def write(self, data):
            self.written.append(data.decode("utf-8").strip())
        def flush(self):
            pass

    def _rec(self, provider, proc=None):
        r = record.Recorder(tempfile.mkdtemp(), 1, backend="sck",
                            exclude_windows=[1], exclude_provider=provider)
        r.EXCLUDE_POLL_SEC = 0.01
        r.proc = proc or self._Proc()
        return r

    def _run(self, r, ticks=6):
        import threading
        t = threading.Thread(target=r._poll_exclusions, daemon=True)
        t.start()
        time.sleep(0.01 * ticks + 0.08)
        r._stop_exclude_poll.set()
        t.join(timeout=2.0)

    def test_an_unchanged_set_sends_nothing(self):
        # A still desktop must cost one list walk and no IPC at all.
        r = self._rec(lambda: [1])
        self._run(r)
        self.assertEqual(r.proc.written, [])

    def test_a_new_window_is_pushed_once(self):
        r = self._rec(lambda: [1, 99])
        self._run(r)
        self.assertEqual(r.proc.written, ["EXCLUDE 1 99"])
        self.assertEqual(r.exclude_windows, [1, 99])

    def test_a_closed_window_is_pushed_too(self):
        r = self._rec(lambda: [])
        self._run(r)
        self.assertEqual(r.proc.written, ["EXCLUDE"])

    def test_a_provider_that_raises_never_ends_the_take(self):
        def boom():
            raise RuntimeError("bar went away")
        r = self._rec(boom)
        self._run(r)
        self.assertEqual(r.proc.written, [])      # survived, sent nothing

    def test_a_dead_worker_stops_the_poller(self):
        r = self._rec(lambda: [1, 2], proc=self._Proc(alive=False))
        self._run(r)
        self.assertEqual(r.proc.written, [])

    def test_no_provider_means_no_poller_at_all(self):
        # The CLI has no bar to ask, so the set is fixed for the take.
        r = record.Recorder(tempfile.mkdtemp(), 1, backend="sck")
        self.assertIsNone(r._exclude_provider)


class BackendSeam(unittest.TestCase):
    """`Recorder` with backend="sck" — config, meta gating, failure text."""

    def _rec(self, **kw):
        return record.Recorder(tempfile.mkdtemp(), 1, **kw)

    def test_the_default_recorder_is_still_avfoundation(self):
        self.assertEqual(self._rec().backend, "avfoundation")

    def test_config_carries_the_exclusions_and_no_argv(self):
        # stdin, not argv: argv is visible in `ps` to every process on the
        # machine, and this carries the output path and the list of windows
        # the user is hiding.
        r = self._rec(backend="sck", exclude_windows=[11, 22])
        cfg = r._sck_config()
        self.assertEqual(cfg["exclude"], [11, 22])
        self.assertEqual(cfg["out"], r.raw_path)
        self.assertTrue(cfg["show_cursor"])

    def test_synthetic_cursor_turns_the_cursor_off_in_the_config(self):
        r = self._rec(backend="sck", cursor_mode="synthetic")
        self.assertFalse(r._sck_config()["show_cursor"])

    def test_t0_comes_from_the_worker_pts_converted_locally(self):
        # The worker's PTS is on the host clock; our t0 must be in OUR
        # monotonic domain. Nothing about the child's clock is trusted.
        r = self._rec(backend="sck")
        r.proc = type("P", (), {"stdout": io.BytesIO(
            b"READY\nT0 1000.0\nSTAT 5 6 1 0 0\nFILTER 2 0\n")})()
        before = time.monotonic()
        r._read_sck_stdout()
        after = time.monotonic()
        # pts0 is far in the past on the uptime clock, so t0 must be too --
        # and by the same amount, in our own timeframe.
        expected = sck.host_pts_to_monotonic(1000.0)
        self.assertAlmostEqual(r._t0, expected, delta=0.5)
        self.assertLess(r._t0, before)
        self.assertEqual(r._sck_stat["dup"], 1)
        self.assertEqual(r._sck_filter["applied"], 2)
        del after

    def test_an_unparseable_line_never_takes_down_a_recording(self):
        r = self._rec(backend="sck")
        r.proc = type("P", (), {"stdout": io.BytesIO(
            b"GARBAGE\n\xff\xfe bad utf8\nT0 5.0\nSTAT nope\n")})()
        r._read_sck_stdout()           # must not raise
        self.assertIsNotNone(r._t0)

    def test_meta_stays_byte_identical_on_the_avfoundation_path(self):
        # The normal-session key set is pinned by dict equality elsewhere;
        # this asserts the NEW keys are the reason that still passes.
        r = self._rec()
        meta = r._meta_dict(1440.0, 900.0, "test", 0.0, False)
        for key in ("capture_backend", "cfr", "excluded_windows",
                    "excluded_applied", "capture_stats"):
            self.assertNotIn(key, meta)

    def test_meta_records_the_backend_and_exclusions_on_sck(self):
        r = self._rec(backend="sck", exclude_windows=[7])
        r._cfr = "ok"
        r._sck_filter = {"applied": 1, "missing": 0}
        r._sck_stat = {"appended": 100, "dup": 20, "dropped": 0, "notready": 0}
        meta = r._meta_dict(1440.0, 900.0, "test", 0.0, False)
        self.assertEqual(meta["capture_backend"], "sck")
        self.assertEqual(meta["cfr"], "ok")
        self.assertEqual(meta["excluded_windows"], [7])
        self.assertEqual(meta["excluded_applied"], 1)
        self.assertEqual(meta["capture_stats"]["duplicated"], 20)

    def test_a_denied_sck_take_says_so_instead_of_going_quiet(self):
        # _screen_denied_signature works by string-matching ffmpeg's stderr,
        # and none of those strings exist here. Without its own branch, a
        # denied SCK take would fall through to a generic message.
        r = self._rec(backend="sck")
        r._sck_error = "start NSError TCC declined screen recording"
        msg = r._failure_message(2, False)
        self.assertIn("ScreenCaptureKit", msg)
        self.assertIn("Screen Recording", msg)
        self.assertIn("--capture-backend avfoundation", msg)

    def test_an_aborted_worker_is_explained_not_left_as_a_number(self):
        r = self._rec(backend="sck")
        msg = r._failure_message(134, False)
        self.assertIn("SIGABRT", msg)
        self.assertIn("app itself is unaffected", msg)

    def test_the_ffmpeg_failure_message_is_untouched(self):
        r = self._rec()
        r._stderr_tail.append("Configuration of video device failed")
        msg = r._failure_message(1, False)
        self.assertIn("ffmpeg screen capture failed", msg)
        self.assertNotIn("ScreenCaptureKit", msg)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "needs ffmpeg/ffprobe")
class RealFiles(unittest.TestCase):
    """End to end against files ffmpeg actually wrote. No permissions."""

    def setUp(self):
        self.td = tempfile.mkdtemp()

    def _make(self, name, extra):
        path = os.path.join(self.td, name)
        cmd = (["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                "-i", "testsrc2=size=320x180:rate=60", "-t", "1"]
               + extra + [path])
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return path

    def test_a_real_cfr_file_passes(self):
        path = self._make("cfr.mov", ["-r", "60", "-pix_fmt", "yuv420p"])
        status, detail = sck.verify_cfr(path, 60)
        self.assertEqual(status, "ok", detail)

    def test_a_real_vfr_file_is_caught(self):
        # Frames deliberately unevenly spaced, which is exactly the shape
        # SCK's own output was reported to have.
        path = self._make("vfr.mov", [
            "-vf", "select='not(mod(n,3))',setpts=N/60/TB*1.7",
            "-fps_mode", "passthrough", "-pix_fmt", "yuv420p"])
        status, detail = sck.verify_cfr(path, 60)
        self.assertEqual(status, "suspect", detail)
        self.assertIn("avg_frame_rate", detail)


def _box(typ, payload=b"", size=None):
    """One QuickTime/MP4 box: 4-byte big-endian size + 4-byte type + payload."""
    if size is None:
        size = 8 + len(payload)
    return size.to_bytes(4, "big") + typ + payload


def _box64(typ, payload=b""):
    """A 64-bit-size box: size field 1, then an 8-byte largesize."""
    return (1).to_bytes(4, "big") + typ + (16 + len(payload)).to_bytes(8, "big") + payload


def _bytes_opener(data):
    """A drop-in for `open` that hands `has_moov` an in-memory file, so these
    stay pure -- no disk, no ffmpeg, no permissions (test_sck's whole point)."""
    def _open(_path, _mode="rb"):
        return io.BytesIO(data)
    return _open


class HasMoov(unittest.TestCase):
    """`sck.has_moov` tells a finalized take (its `moov` index is present) from
    one whose writer was stopped mid-tail (frames in `mdat`, no `moov`)."""

    def _check(self, data):
        return sck.has_moov("ignored", _open=_bytes_opener(data))

    def test_a_finalized_file_with_a_trailing_moov_passes(self):
        data = (_box(b"ftyp", b"qt  " + b"\x00" * 8)
                + _box(b"mdat", b"\x00" * 64)
                + _box(b"moov", b"\x00" * 32))
        self.assertTrue(self._check(data))

    def test_the_field_shape_no_moov_after_a_size0_mdat_fails(self):
        # ftyp + wide + mdat(size 0 -> extends to EOF): the EXACT layout of the
        # broken takes -- frames present, index never written.
        data = (_box(b"ftyp", b"\x00" * 12)
                + _box(b"wide")
                + (0).to_bytes(4, "big") + b"mdat" + b"\x00" * 200)
        self.assertFalse(self._check(data))

    def test_a_file_truncated_before_its_moov_fails(self):
        data = _box(b"ftyp", b"\x00" * 12) + _box(b"mdat", b"\x00" * 100)
        self.assertFalse(self._check(data))

    def test_a_faststart_moov_before_mdat_passes(self):
        data = _box(b"moov", b"\x00" * 32) + _box(b"mdat", b"\x00" * 64)
        self.assertTrue(self._check(data))

    def test_it_skips_a_64bit_sized_box_to_find_a_later_moov(self):
        data = (_box(b"ftyp", b"\x00" * 12)
                + _box64(b"free", b"\x00" * 300)   # largesize path
                + _box(b"moov", b"\x00" * 16))
        self.assertTrue(self._check(data))

    def test_a_short_header_is_not_finalized(self):
        self.assertFalse(self._check(b"\x00\x00"))
        self.assertFalse(self._check(b""))

    def test_a_box_size_too_small_to_hold_its_header_stops_the_scan(self):
        # size 3 cannot even cover the 8-byte header: reject rather than loop.
        data = (3).to_bytes(4, "big") + b"junk"
        self.assertFalse(self._check(data))

    def test_a_missing_file_is_false_not_an_exception(self):
        self.assertFalse(sck.has_moov("/no/such/path/raw.mov"))


class HasMoovOnFragmentedLayout(unittest.TestCase):
    """Movie fragments (Phase C) put `moov` at the TOP of the file and append
    a stream of `moof`+`mdat` pairs after it. `has_moov` finds the moov early
    and returns True; a kill-9 truncating the trailing fragments has no way
    to lose it, which is the whole point of the layout change.

    Pinned here even though `test_a_faststart_moov_before_mdat_passes` above
    already exercises moov-before-mdat -- fragmented is the DEFAULT SCK layout
    from C onward, and pinning the layout explicitly documents the shape the
    single-stream path actually writes."""

    def _check(self, data):
        return sck.has_moov("ignored", _open=_bytes_opener(data))

    def _fragmented(self, n_fragments=3):
        # ftyp + moov (with mvex, signalling fragments follow) + N pairs of
        # moof + mdat. `has_moov` doesn't look inside `moov`, so the mvex
        # payload is a placeholder -- the shape that matters is the box
        # ordering.
        blob = _box(b"ftyp", b"qt  " + b"\x00" * 8)
        blob += _box(b"moov", _box(b"mvex", b"\x00" * 16) + b"\x00" * 16)
        for _ in range(n_fragments):
            blob += _box(b"moof", b"\x00" * 24)
            blob += _box(b"mdat", b"\x00" * 128)
        return blob

    def test_moov_at_head_followed_by_fragments_passes(self):
        self.assertTrue(self._check(self._fragmented(n_fragments=3)))

    def test_a_kill9_truncated_last_fragment_still_leaves_moov_findable(self):
        # A SIGKILL mid-fragment leaves the tail of the file cut off inside a
        # `mdat` (or partway through a `moof`), but the `moov` at the head
        # already landed at `startWriting`. has_moov must return True.
        full = self._fragmented(n_fragments=4)
        for cut in (len(full) - 60, len(full) - 200, len(full) - 400):
            if cut <= 0:
                continue
            self.assertTrue(
                self._check(full[:cut]),
                "truncated at {} bytes: moov still at head".format(cut))

    def test_a_zero_fragment_file_is_still_playable(self):
        # writer.startWriting() flushed the moov but no fragment ever landed:
        # the file has `ftyp` + `moov` and nothing else. Still playable (an
        # empty movie, one moov, zero samples), so has_moov -> True.
        data = (_box(b"ftyp", b"\x00" * 12)
                + _box(b"moov", b"\x00" * 16))
        self.assertTrue(self._check(data))


class MovieFragmentIntervalConfig(unittest.TestCase):
    """The parent resolves `AUTOCINE_MOVIE_FRAGMENT_INTERVAL_SEC` and folds
    it into the worker config -- so the value is testable without importing
    PyObjC or opening a real AVAssetWriter."""

    def _rec(self, **kw):
        return record.Recorder(tempfile.mkdtemp(), 1, backend="sck", **kw)

    def _cfg(self, env):
        with mock.patch.dict(os.environ, env, clear=False):
            return self._rec()._sck_config()

    def test_default_config_omits_the_key_and_worker_default_fires(self):
        # No env var -> key absent from the config. The worker reads its own
        # module constant (MOVIE_FRAGMENT_INTERVAL_SEC = 1.0) when the key is
        # missing, so C is still on -- proven by
        # `test_worker_uses_module_default_when_key_absent` below.
        env = {k: v for k, v in os.environ.items()
               if k != "AUTOCINE_MOVIE_FRAGMENT_INTERVAL_SEC"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = self._rec()._sck_config()
        self.assertNotIn("movie_fragment_interval", cfg)

    def test_env_var_flows_into_the_worker_config(self):
        cfg = self._cfg({"AUTOCINE_MOVIE_FRAGMENT_INTERVAL_SEC": "0.5"})
        self.assertEqual(cfg["movie_fragment_interval"], 0.5)

    def test_env_var_zero_is_the_byte_exact_off_switch(self):
        # 0 is a VALID float; the worker sees the value and skips
        # setMovieFragmentInterval_ (byte-exact pre-C monolithic-file layout).
        # The alternative -- treating 0 like "unset" and silently re-enabling
        # fragments -- would defeat the off switch a user's env var promised.
        cfg = self._cfg({"AUTOCINE_MOVIE_FRAGMENT_INTERVAL_SEC": "0"})
        self.assertEqual(cfg["movie_fragment_interval"], 0.0)

    def test_env_var_negative_disables_too(self):
        # Any non-positive value disables; both are unambiguous "off" in the
        # worker's `if self.movie_fragment_interval > 0:` guard.
        cfg = self._cfg({"AUTOCINE_MOVIE_FRAGMENT_INTERVAL_SEC": "-1"})
        self.assertEqual(cfg["movie_fragment_interval"], -1.0)

    def test_garbage_env_var_falls_back_to_the_worker_default(self):
        # A typo in an env var must not stop someone recording -- the parent
        # drops the key and the worker's module default fires.
        cfg = self._cfg({"AUTOCINE_MOVIE_FRAGMENT_INTERVAL_SEC": "not-a-number"})
        self.assertNotIn("movie_fragment_interval", cfg)

    def test_worker_uses_module_default_when_key_absent(self):
        # The other half of `default_config_omits_the_key`: without the key,
        # the worker's own default is what fires, and it is ON (Phase C).
        from autocine import _sck_worker
        cap = _sck_worker.Capture({"out": "/tmp/x", "fps": 60})
        self.assertEqual(cap.movie_fragment_interval,
                         _sck_worker.MOVIE_FRAGMENT_INTERVAL_SEC)
        self.assertGreater(cap.movie_fragment_interval, 0)

    def test_worker_honours_cfg_zero_as_the_off_switch(self):
        # The `if self.movie_fragment_interval > 0:` guard in build() is
        # keyed off this attribute; 0 must survive the round-trip through
        # __init__ intact rather than being coerced back to the default.
        from autocine import _sck_worker
        cap = _sck_worker.Capture(
            {"out": "/tmp/x", "fps": 60, "movie_fragment_interval": 0})
        self.assertEqual(cap.movie_fragment_interval, 0.0)

    def test_worker_ignores_garbage_cfg_value(self):
        from autocine import _sck_worker
        cap = _sck_worker.Capture(
            {"out": "/tmp/x", "fps": 60,
             "movie_fragment_interval": "not-a-number"})
        self.assertEqual(cap.movie_fragment_interval,
                         _sck_worker.MOVIE_FRAGMENT_INTERVAL_SEC)


if __name__ == "__main__":
    unittest.main()
