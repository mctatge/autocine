"""Pure tests for the pause/resume HTTP surface (segmented takes, Layer 5).

`pause_record` / `resume_record` mirror `stop_record`: state-gated calls into
the active recorder, flipping the status the bar polls. No live capture, no
permissions -- the recorder is a stub. The live click is an on-device gate
(see docs/architecture.md).
"""

import inspect
import shutil
import tempfile
import unittest
from unittest import mock

from autocine import studio_app


class _StubRecorder(object):
    """Just enough Recorder for the pause/resume surface: the two calls it
    receives, and the window-capture attributes the Phase-1 guard reads."""

    def __init__(self, capture_window=None, capture_windows=None):
        self.capture_window = capture_window
        self.capture_windows = capture_windows or []
        self.paused_calls = 0
        self.resumed_calls = 0
        self.stopped = 0

    def pause(self):
        self.paused_calls += 1

    def resume(self):
        self.resumed_calls += 1

    def stop(self):
        self.stopped += 1


class _FleetStub(_StubRecorder):
    """A scene-capable occlusion-free fleet take: pause_supported, and
    resume() accepts a re-picked window set (scene takes). Also carries the
    seamless-JOIN surface: grow(), grow_supported, captured_window_ids."""

    pause_supported = True

    def __init__(self):
        _StubRecorder.__init__(
            self, capture_windows=[{"id": 1}, {"id": 2}])
        self.resume_scene = "unset"
        self.grow_supported = True
        self.grown = []

    def _is_multi_window_native(self):
        return True

    def resume(self, scene=None):
        self.resumed_calls += 1
        self.resume_scene = scene

    def grow(self, entry):
        self.grown.append(entry)

    @property
    def captured_window_ids(self):
        return {int(w["id"]) for w in self.capture_windows}


class PauseResumeEndpointState(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.state = studio_app.StudioState(recordings_root=self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _recording(self, recorder=None):
        rec = recorder or _StubRecorder()
        self.state._active_recorder = rec
        self.state._record_status = {"status": "recording", "message": "",
                                     "session": "s"}
        return rec

    # -- pause --------------------------------------------------------------

    def test_pause_from_recording_calls_recorder_and_reports_pausing(self):
        rec = self._recording()
        status = self.state.pause_record()
        self.assertEqual(rec.paused_calls, 1)
        self.assertEqual(status["status"], "pausing")

    def test_pause_409_when_idle(self):
        with self.assertRaises(studio_app.StudioError) as ctx:
            self.state.pause_record()
        self.assertEqual(ctx.exception.status, 409)

    def test_pause_409_when_already_paused(self):
        rec = self._recording()
        self.state._record_status["status"] = "paused"
        with self.assertRaises(studio_app.StudioError) as ctx:
            self.state.pause_record()
        self.assertEqual(ctx.exception.status, 409)
        self.assertEqual(rec.paused_calls, 0)

    def test_pause_400_on_window_capture_take(self):
        # Phase-1 scope: whole-screen only. Refused loud -- the multi-native
        # run loop never consumes the request, so accepting would wedge the
        # status at "pausing" forever.
        rec = self._recording(_StubRecorder(capture_window={"id": 1}))
        with self.assertRaises(studio_app.StudioError) as ctx:
            self.state.pause_record()
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(rec.paused_calls, 0)

    def test_pause_400_on_multi_window_take(self):
        rec = self._recording(_StubRecorder(
            capture_windows=[{"id": 1}, {"id": 2}]))
        with self.assertRaises(studio_app.StudioError) as ctx:
            self.state.pause_record()
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(rec.paused_calls, 0)

    def test_pause_allowed_on_a_scene_capable_fleet_take(self):
        # Scene takes: the recorder's own pause_supported is the authority --
        # an occlusion-free fleet take pauses even though it IS a window
        # capture. The display-crop 400s above stay exactly as pinned.
        rec = self._recording(_FleetStub())
        status = self.state.pause_record()
        self.assertEqual(rec.paused_calls, 1)
        self.assertEqual(status["status"], "pausing")

    # -- resume -------------------------------------------------------------

    def test_resume_from_paused_calls_recorder_and_reports_resuming(self):
        rec = self._recording()
        self.state._record_status["status"] = "paused"
        status = self.state.resume_record()
        self.assertEqual(rec.resumed_calls, 1)
        self.assertEqual(status["status"], "resuming")

    def test_resume_409_when_recording(self):
        rec = self._recording()
        with self.assertRaises(studio_app.StudioError) as ctx:
            self.state.resume_record()
        self.assertEqual(ctx.exception.status, 409)
        self.assertEqual(rec.resumed_calls, 0)

    def test_resume_409_while_still_pausing(self):
        # The segment hasn't finalized yet; the bar retries a poll later.
        self._recording()
        self.state._record_status["status"] = "pausing"
        with self.assertRaises(studio_app.StudioError):
            self.state.resume_record()

    # -- resume with a window RE-PICK (scene takes) --------------------------

    def _paused_fleet(self):
        rec = self._recording(_FleetStub())
        self.state._record_status["status"] = "paused"
        return rec

    def test_repick_resume_passes_resolved_entries_to_the_recorder(self):
        rec = self._paused_fleet()
        real = studio_app.dev.window_rect_points
        studio_app.dev.window_rect_points = (
            lambda wid, exclude_pids=None: {"id": wid, "x": 0.0, "y": 0.0,
                                            "w": 100.0, "h": 100.0})
        try:
            status = self.state.resume_record({"window_ids": [7, 8, 9]})
        finally:
            studio_app.dev.window_rect_points = real
        self.assertEqual(rec.resumed_calls, 1)
        self.assertEqual([e["id"] for e in rec.resume_scene], [7, 8, 9])
        self.assertEqual(status["status"], "resuming")

    def test_repick_is_strict_a_stale_id_400s_and_stays_paused(self):
        # NOT `_resolve_capture_windows`' lenient drop-and-degrade: a silently
        # shrunk set betrays the pick the user just made. The take must stay
        # `paused` -- a refused pick is never a failure.
        rec = self._paused_fleet()
        real = studio_app.dev.window_rect_points
        studio_app.dev.window_rect_points = (
            lambda wid, exclude_pids=None: None if wid == 8 else
            {"id": wid, "x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0})
        try:
            with self.assertRaises(studio_app.StudioError) as ctx:
                self.state.resume_record({"window_ids": [7, 8]})
        finally:
            studio_app.dev.window_rect_points = real
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn("still", str(ctx.exception))
        self.assertEqual(rec.resumed_calls, 0)
        self.assertEqual(self.state._record_status["status"], "paused")

    def test_repick_rejects_unparseable_ids_outright(self):
        # Strict means strict: {"window_ids": ["oops", 7]} must 400 naming
        # the bad value, never resume with a silently shrunk set.
        rec = self._paused_fleet()
        real = studio_app.dev.window_rect_points
        studio_app.dev.window_rect_points = (
            lambda wid, exclude_pids=None: {"id": wid, "x": 0.0, "y": 0.0,
                                            "w": 100.0, "h": 100.0})
        try:
            with self.assertRaises(studio_app.StudioError) as ctx:
                self.state.resume_record({"window_ids": ["oops", 7]})
        finally:
            studio_app.dev.window_rect_points = real
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn("oops", str(ctx.exception))
        self.assertEqual(rec.resumed_calls, 0)
        self.assertEqual(self.state._record_status["status"], "paused")

    def test_repick_refused_on_a_non_fleet_take(self):
        self._recording()          # plain whole-screen stub
        self.state._record_status["status"] = "paused"
        with self.assertRaises(studio_app.StudioError) as ctx:
            self.state.resume_record({"window_ids": [7]})
        self.assertEqual(ctx.exception.status, 400)

    def test_plain_resume_still_takes_no_arguments(self):
        # The P1 contract: a bodyless resume calls resume() -- older stubs
        # (and the whole-screen recorder) accept no scene argument.
        rec = self._recording()
        self.state._record_status["status"] = "paused"
        self.state.resume_record()
        self.assertEqual(rec.resumed_calls, 1)

    # -- interactions with the rest of the surface --------------------------

    def test_stop_works_while_paused(self):
        rec = self._recording()
        self.state._record_status["status"] = "paused"
        status = self.state.stop_record()
        self.assertEqual(rec.stopped, 1)
        self.assertEqual(status["status"], "stopping")

    def test_delete_refuses_while_paused(self):
        import os
        name = "20260101-000000"
        os.makedirs(os.path.join(self.td, name))
        self.state._record_status = {"status": "paused", "session": name}
        with self.assertRaises(studio_app.StudioError) as ctx:
            self.state.delete_session(name)
        self.assertEqual(ctx.exception.status, 409)


class WaveformOnSegmentedTake(unittest.TestCase):
    """A segmented take has raw_path=None but has_audio=True (mic-on Phase 1).
    waveform() must degrade to empty peaks -- its own contract says the
    timeline UI can always render -- not TypeError on the missing raw."""

    def test_no_raw_path_yields_empty_peaks(self):
        import os
        td = tempfile.mkdtemp()
        try:
            state = studio_app.StudioState(recordings_root=td)
            name = "20260101-000000"
            os.makedirs(os.path.join(td, name))
            state._describe = lambda d: {
                "duration": 2.0, "has_audio": True, "raw_path": None}
            out = state.waveform(name)
            self.assertEqual(out, {"peaks": [], "duration": 2.0})
        finally:
            shutil.rmtree(td, ignore_errors=True)


class _StubBarState(object):
    def __init__(self, status):
        self._status = status
        self.stopped = 0

    def snapshot(self):
        return {"record": {"status": self._status}}

    def stop_record(self):
        self.stopped += 1
        self._status = "done"


class StopNativeBarPausedTake(unittest.TestCase):
    """Ctrl+C on the native pill during a PAUSED (or transitioning) take must
    still route through stop_record -- otherwise the hard exit skips the
    segmented finalize and the take is left with no meta.json (a failed take
    with not even an error.log)."""

    def _run_with(self, status):
        old_state = studio_app._ACTIVE_STATE
        old_api = studio_app._ACTIVE_BAR_API
        stub = _StubBarState(status)
        studio_app._ACTIVE_STATE = stub
        studio_app._ACTIVE_BAR_API = None
        try:
            studio_app.stop_native_bar()
        finally:
            studio_app._ACTIVE_STATE = old_state
            studio_app._ACTIVE_BAR_API = old_api
        return stub

    def test_paused_take_is_stopped(self):
        for status in ("pausing", "paused", "resuming"):
            self.assertEqual(self._run_with(status).stopped, 1,
                             "status %r must be stopped on quit" % status)

    def test_idle_is_untouched(self):
        self.assertEqual(self._run_with("done").stopped, 0)


class GrowEndpoint(unittest.TestCase):
    """`grow_record` -- the seamless mid-take window-join HTTP surface. Every
    guard rejects BEFORE `recorder.grow`, so a refused join leaves the live
    take untouched (it keeps recording)."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.state = studio_app.StudioState(recordings_root=self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _recording(self, recorder):
        self.state._active_recorder = recorder
        self.state._record_status = {"status": "recording", "message": "",
                                     "session": "s"}
        return recorder

    def test_409_when_not_recording(self):
        for st in ("idle", "paused", "countdown", "stopping"):
            self.state._active_recorder = _FleetStub()
            self.state._record_status = {"status": st, "session": "s"}
            with self.assertRaises(studio_app.StudioError) as ctx:
                self.state.grow_record({"window_id": 5})
            self.assertEqual(ctx.exception.status, 409, st)

    def test_400_on_non_fleet_take(self):
        self._recording(_StubRecorder(capture_window={"id": 1}))
        with self.assertRaises(studio_app.StudioError) as ctx:
            self.state.grow_record({"window_id": 5})
        self.assertEqual(ctx.exception.status, 400)

    def test_400_at_cap(self):
        rec = self._recording(_FleetStub())
        rec.grow_supported = False           # fleet full
        with self.assertRaises(studio_app.StudioError) as ctx:
            self.state.grow_record({"window_id": 5})
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(rec.grown, [])      # never reached the recorder

    def test_400_on_already_captured_window(self):
        # The duplicate-wid guard (M3.3, decision 10): after an auto-rejoin,
        # a user's in-flight chip click for the same window must not spawn a
        # SECOND worker on it. The resolver is patched to SUCCEED so only
        # the guard can produce the 400 (without the patch this test would
        # pass even with the guard neutered -- the resolver also 400s in
        # this stub environment).
        rec = self._recording(_FleetStub())  # capturing wids {1, 2}
        with mock.patch.object(
                studio_app.dev, "window_rect_points",
                return_value={"id": 2, "app": "A", "title": "t",
                              "x": 0.0, "y": 0.0, "w": 400.0, "h": 300.0}):
            with self.assertRaises(studio_app.StudioError) as ctx:
                self.state.grow_record({"window_id": 2})
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn("already being recorded", str(ctx.exception))
        self.assertEqual(rec.grown, [])

    def test_400_on_bad_window_id_keeps_recording(self):
        rec = self._recording(_FleetStub())
        for bad in ([1, 2], None, -3):
            with self.assertRaises(studio_app.StudioError) as ctx:
                self.state.grow_record({"window_id": bad})
            self.assertEqual(ctx.exception.status, 400, bad)
        self.assertEqual(rec.grown, [])
        # the take is still recording -- a bad pick never mutated the status
        self.assertEqual(self.state._record_status["status"], "recording")

    def test_400_when_window_vanished(self):
        rec = self._recording(_FleetStub())
        with mock.patch.object(studio_app.dev, "window_rect_points",
                               return_value=None):
            with self.assertRaises(studio_app.StudioError) as ctx:
                self.state.grow_record({"window_id": 999})
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(rec.grown, [])

    def test_success_calls_grow_and_leaves_status_recording(self):
        rec = self._recording(_FleetStub())
        entry = {"id": 999, "x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0}
        with mock.patch.object(studio_app.dev, "window_rect_points",
                               return_value=entry):
            status = self.state.grow_record({"window_id": 999})
        self.assertEqual(rec.grown, [entry])          # resolved entry, not id
        self.assertEqual(status["status"], "recording")   # seamless: unchanged
        self.assertNotIn("joined", status)            # the poll surfaces that

    def test_route_and_method_exist_in_source(self):
        import inspect
        src = inspect.getsource(studio_app)
        self.assertIn("/api/record/grow", src)        # the HTTP route
        self.assertIn("def grow_record", src)


class ShrinkStateFeed(unittest.TestCase):
    """The card-shrink surface (docs/architecture.md M3.4): the `departed` /
    `departed_app` / `mic_anchor_hidden` keys ride the same mutual-exclusion
    block as `joined`/`grow_error` (source-pinned -- the closure lives inside
    `start_record`, the SceneRunLoopOrdering idiom), and the baseline discard
    lets a from-start departed window come back as an ordinary chip."""

    def setUp(self):
        self.src = inspect.getsource(studio_app.StudioState._start_record)

    def test_departed_keys_ride_the_exclusion_block(self):
        block = self.src.split('payload.get("grow_error")')[1]
        for token in ('payload.get("departed")',
                      'payload.get("departed_app")',
                      'payload.get("mic_anchor_hidden")'):
            self.assertIn(token, block, token)
        # joined SUPERSEDES departed ("came back" beats "removed"): the
        # joined branch pops departed/departed_app.
        joined_branch = block.split("elif j is not None:")[1].split(
            "elif d is not None:")[0]
        self.assertIn('pop("departed"', joined_branch)
        self.assertIn('pop("departed_app"', joined_branch)
        # ...and the plain-notify branch clears everything, mic note included.
        tail = block.split("elif not mic_note")[1]
        for key in ("grow_error", "joined", "windows", "departed",
                    "departed_app", "mic_anchor_hidden"):
            self.assertIn('pop("{}"'.format(key), tail, key)

    def test_departed_branch_discards_the_wid_from_the_baseline(self):
        # Without this, a from-start window that departed stays hidden from
        # `new_windows` forever (the baseline subtraction) and its chip
        # fallback can never appear.
        branch = self.src.split("elif d is not None:")[1].split(
            'elif payload.get("windows")')[0]
        # The OPERATOR is the pin: a `-=` -> `-` typo (no-op expression)
        # survived token-presence mutation testing.
        self.assertIn("_grow_baseline_ids -= set(", branch)
        self.assertIn("departed_window_ids", branch)

    def test_windows_only_notify_never_clears_the_hint_keys(self):
        # A gate-failed shrink notifies {windows} alone; reading it as a
        # PLAIN notify wiped a standing mic hint / unread departed mid-take
        # (adversarially caught -- and the recorder never re-notes the same
        # hide episode, so the wipe was permanent). The windows-carrying
        # branch stores the count and touches nothing else.
        block = self.src.split('payload.get("grow_error")')[1]
        win_branch = block.split(
            'elif payload.get("windows") is not None:')[1].split("elif")[0]
        self.assertIn('self._record_status["windows"]', win_branch)
        self.assertNotIn(".pop(", win_branch)

    def test_mic_anchor_seen_retires_the_hint(self):
        block = self.src.split('payload.get("grow_error")')[1]
        self.assertIn('payload.get("mic_anchor_seen")', block)
        seen = block.split('payload.get("mic_anchor_seen"):')[1].split(
            "if ge:")[0]
        self.assertIn('pop("mic_anchor_hidden"', seen)


class GrowStateFeed(unittest.TestCase):
    """`snapshot()` advertises `grow_supported` + `new_windows` ONLY for a live
    fleet take. Bit-exact off-switch: every other take's record block is
    unchanged."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.state = studio_app.StudioState(recordings_root=self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_idle_snapshot_has_no_grow_keys(self):
        rec = self.state.snapshot()["record"]
        self.assertNotIn("grow_supported", rec)
        self.assertNotIn("new_windows", rec)

    def test_non_fleet_recording_has_no_grow_keys(self):
        # A method-less / non-fleet recorder must not raise and must not gain
        # the keys (the getattr default + try/except floor).
        self.state._active_recorder = _StubRecorder(capture_window={"id": 1})
        self.state._record_status = {"status": "recording", "session": "s"}
        rec = self.state.snapshot()["record"]
        self.assertNotIn("grow_supported", rec)
        self.assertNotIn("new_windows", rec)

    def test_fleet_recording_advertises_new_windows(self):
        self.state._active_recorder = _FleetStub()   # captures ids {1, 2}
        self.state._record_status = {"status": "recording", "session": "s"}
        self.state._grow_baseline_ids = {1, 2, 7}     # 7 was there at start
        live = [{"id": 1, "app": "A"}, {"id": 7, "app": "Old"},
                {"id": 9, "app": "New"}, {"id": 10, "app": "Newer"},
                {"id": 11, "app": "Newest"}]
        with mock.patch.object(studio_app.dev, "list_windows",
                               return_value=live):
            rec = self.state.snapshot()["record"]
        self.assertTrue(rec["grow_supported"])
        # 1,2 captured; 7 in baseline -> candidates are 9,10,11, capped at 2,
        # app-name-only (no title/rect).
        self.assertEqual(rec["new_windows"],
                         [{"id": 9, "app": "New"}, {"id": 10, "app": "Newer"}])

    def test_raised_window_escapes_the_baseline(self):
        """M2.4c: a window that was ALREADY on screen at t0 but which the user
        brought to the front mid-take IS a candidate. This is the flagship
        case -- opening Finder / a document raises an existing window rather
        than creating one, so the id-only baseline rule made it unreachable
        (recordings/20260831-092226)."""
        rec_stub = _FleetStub()
        rec_stub.raised_window_ids = frozenset({7})
        self.state._active_recorder = rec_stub
        self.state._record_status = {"status": "recording", "session": "s"}
        self.state._grow_baseline_ids = {1, 2, 7, 8}
        live = [{"id": 7, "app": "Finder"}, {"id": 8, "app": "Terminal"},
                {"id": 1, "app": "A"}, {"id": 9, "app": "New"}]
        with mock.patch.object(studio_app.dev, "list_windows",
                               return_value=live):
            rec = self.state.snapshot()["record"]
        # 7 raised -> offered despite the baseline; 8 baselined and never
        # raised -> still suppressed; 1 captured; 9 genuinely new.
        self.assertEqual(rec["new_windows"],
                         [{"id": 7, "app": "Finder"}, {"id": 9, "app": "New"}])

    def test_recorder_without_raised_ids_keeps_the_id_only_rule(self):
        """Off-switch: `_FleetStub` has no `raised_window_ids`, so the getattr
        default leaves the baseline an absolute wall -- byte-identical to the
        pre-M2.4c feed."""
        self.state._active_recorder = _FleetStub()
        self.state._record_status = {"status": "recording", "session": "s"}
        self.state._grow_baseline_ids = {7}
        live = [{"id": 7, "app": "Old"}, {"id": 9, "app": "New"}]
        with mock.patch.object(studio_app.dev, "list_windows",
                               return_value=live):
            rec = self.state.snapshot()["record"]
        self.assertEqual(rec["new_windows"], [{"id": 9, "app": "New"}])

    def test_full_fleet_offers_no_candidates(self):
        rec_stub = _FleetStub()
        rec_stub.grow_supported = False               # at the 4-card cap
        self.state._active_recorder = rec_stub
        self.state._record_status = {"status": "recording", "session": "s"}
        with mock.patch.object(studio_app.dev, "list_windows") as lw:
            rec = self.state.snapshot()["record"]
            self.assertEqual(rec["new_windows"], [])
            lw.assert_not_called()                    # skips the sweep entirely
        self.assertFalse(rec["grow_supported"])


if __name__ == "__main__":
    unittest.main()
