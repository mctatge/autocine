"""Pure (no ffmpeg / no permissions) tests for the SCENE-take record side
(docs/architecture.md): the `capture_scenes` manifest shape, the
scene lifecycle helpers, per-scene naming + the N-file rename, the 1-worker
fleet down-conversion, and the run-loop ordering pins. The live
pause -> re-pick -> resume of a real fleet is an on-device gate.
"""

import inspect
import json
import os
import tempfile
import unittest

from autocine import record
from autocine import segments


class _FakeProc(object):
    def __init__(self, rc=0):
        self.returncode = rc

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def send_signal(self, sig):
        pass


def _pick(i, wid=None):
    return {"id": wid if wid is not None else 100 + i,
            "app": "App{}".format(i), "title": "t{}".format(i),
            "x": float(i * 200), "y": 0.0, "w": 400.0, "h": 300.0,
            "display_origin": [0.0, 0.0]}


def _worker(session_dir, scene, idx, t0=10.0, rc=0, write=True):
    w = record._NativeWorker(session_dir, idx, _pick(idx))
    if scene > 0:
        w.raw_path = os.path.join(
            session_dir, "scene_{}_raw_{}.mov".format(scene, idx))
    if write:
        with open(w.raw_path, "wb") as f:
            f.write(b"\0" * 2048)
    w.proc = _FakeProc(rc)
    w.t0 = t0
    w.entry = w.window
    w.resnapshot = True
    w.size = {"width": 800, "height": 600}
    w.stat = {"appended": 300, "dup": 0, "dropped": 0, "notready": 0,
              "idle": 0}
    return w


class _MoovPatched(unittest.TestCase):
    """Patch the moov probe: these tests write throwaway bytes, not movies."""

    def setUp(self):
        self._real_has_moov = record.sck.has_moov
        record.sck.has_moov = lambda path: True

    def tearDown(self):
        record.sck.has_moov = self._real_has_moov


class SceneMetaDict(_MoovPatched):
    def _rec(self, td):
        r = record.Recorder(td, 1, backend="sck", window_native=True,
                            capture_windows=[_pick(0), _pick(1)])
        r._key_capture = "activity"
        s0 = record._SceneRecord(0, [_worker(td, 0, 0, t0=10.0)],
                                 wall_end=16.5)
        s1 = record._SceneRecord(
            1, [_worker(td, 1, 0, t0=20.0), _worker(td, 1, 1, t0=20.02)],
            wall_end=None)
        r._scenes = [s0, s1]
        return r

    def test_manifest_shape_and_discriminator(self):
        with tempfile.TemporaryDirectory() as td:
            meta = self._rec(td)._scene_meta_dict(1440.0, 900.0, "quartz")
            self.assertTrue(segments.is_scene_meta(meta))
            self.assertNotIn("raw", meta)
            self.assertNotIn("capture_channels", meta)
            self.assertNotIn("capture_segments", meta)
            scenes = meta["capture_scenes"]
            self.assertEqual([s["index"] for s in scenes], [0, 1])
            self.assertEqual(len(scenes[0]["channels"]), 1)
            self.assertEqual(len(scenes[1]["channels"]), 2)
            self.assertEqual(scenes[1]["channels"][1]["file"],
                             "scene_1_raw_1.mov")
            self.assertEqual(scenes[1]["channels"][0]["mode"],
                             "window_native")

    def test_session_origin_is_scene_zero_channel_zero(self):
        with tempfile.TemporaryDirectory() as td:
            meta = self._rec(td)._scene_meta_dict(1440.0, 900.0, "quartz")
            self.assertEqual(meta["t0_monotonic"], 10.0)

    def test_wall_end_recorded_only_when_known(self):
        # The shortfall check (segments.scene_clock_entries) reads it; a
        # scene without one simply gets no warning, never a crash.
        with tempfile.TemporaryDirectory() as td:
            scenes = self._rec(td)._scene_meta_dict(
                1440.0, 900.0, "quartz")["capture_scenes"]
            self.assertEqual(scenes[0]["wall_end_monotonic"], 16.5)
            self.assertNotIn("wall_end_monotonic", scenes[1])

    def test_mic_and_face_are_cut(self):
        with tempfile.TemporaryDirectory() as td:
            meta = self._rec(td)._scene_meta_dict(1440.0, 900.0, "quartz")
            self.assertIsNone(meta["mic_index"])
            self.assertIsNone(meta["face"])


class FinalizeSceneTake(_MoovPatched):
    def _rec(self, td):
        r = record.Recorder(td, 1, backend="sck", window_native=True,
                            capture_windows=[_pick(0), _pick(1)])
        r._scenes = [
            record._SceneRecord(0, [_worker(td, 0, 0), _worker(td, 0, 1)],
                                wall_end=16.0),
            record._SceneRecord(1, [_worker(td, 1, 0, t0=20.0)],
                                wall_end=26.0),
        ]
        return r

    def test_renames_scene_zero_files_and_writes_the_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            out = r._finalize_scene_take(1440.0, 900.0, "quartz", None)
            self.assertEqual(out, td)
            self.assertTrue(os.path.exists(
                os.path.join(td, "scene_0_raw_0.mov")))
            self.assertTrue(os.path.exists(
                os.path.join(td, "scene_0_raw_1.mov")))
            self.assertFalse(os.path.exists(os.path.join(td, "raw_0.mov")))
            with open(os.path.join(td, "meta.json")) as f:
                meta = json.load(f)
            self.assertTrue(segments.is_scene_meta(meta))
            self.assertEqual(
                meta["capture_scenes"][0]["channels"][0]["file"],
                "scene_0_raw_0.mov")

    def test_missing_moov_fails_the_take_with_error_log(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            record.sck.has_moov = lambda path: "scene_1" not in path
            with self.assertRaises(record.RecordError):
                r._finalize_scene_take(1440.0, 900.0, "quartz", None)
            self.assertTrue(os.path.exists(r.error_path))
            self.assertFalse(os.path.exists(r.meta_path))

    def test_mid_scene_death_fails_the_take(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            dead = _worker(td, 2, 0, rc=1)
            with self.assertRaises(record.RecordError) as ctx:
                r._finalize_scene_take(1440.0, 900.0, "quartz", dead)
            self.assertIn("exited during scene", str(ctx.exception))

    def test_null_channel_t0_fails_the_take(self):
        # A spawn-time fallback would silently misalign the whole scene --
        # the reader raises instead (docs/architecture.md, clock section).
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            r._scenes[1].workers[0].t0 = None
            with self.assertRaises(record.RecordError) as ctx:
                r._finalize_scene_take(1440.0, 900.0, "quartz", None)
            self.assertIn("no t0 pairing", str(ctx.exception))


class FleetSingleDownConversion(_MoovPatched):
    def test_never_paused_single_pick_writes_the_pinned_singleton_shape(self):
        with tempfile.TemporaryDirectory() as td:
            r = record.Recorder(td, 1, backend="sck", window_native=True,
                                capture_window=_pick(0, wid=777))
            self.assertTrue(r._fleet_single)
            w = r._sck_workers[0]
            with open(w.raw_path, "wb") as f:
                f.write(b"\0" * 2048)
            w.proc = _FakeProc(0)
            w.t0 = 42.5
            w.entry = w.window
            w.resnapshot = True
            w.end_rect = [0.0, 0.0, 400.0, 300.0]
            w.size = {"width": 800, "height": 600}
            w.stat = {"appended": 300, "dup": 0, "dropped": 0,
                      "notready": 0, "idle": 0}
            r._key_capture = "activity"
            out = r._finalize_fleet_single(1440.0, 900.0, "quartz", 1.0)
            self.assertEqual(out, td)
            self.assertTrue(os.path.exists(os.path.join(td, "raw.mov")))
            self.assertFalse(os.path.exists(os.path.join(td, "raw_0.mov")))
            with open(os.path.join(td, "meta.json")) as f:
                meta = json.load(f)
            # The single-native contract: raw + a window_native
            # capture_window block, and none of the manifest keys.
            self.assertEqual(meta["raw"], "raw.mov")
            self.assertEqual(meta["t0_monotonic"], 42.5)
            cw = meta["capture_window"]
            self.assertEqual(cw["mode"], "window_native")
            self.assertEqual(cw["id"], 777)
            self.assertEqual(cw["buffer_w"], 800)
            self.assertNotIn("capture_scenes", meta)
            self.assertNotIn("capture_channels", meta)
            self.assertNotIn("capture_segments", meta)

    def test_key_set_matches_a_singleton_native_take(self):
        # The down-converted meta must carry EXACTLY the singleton path's
        # keys -- a consumer must not be able to tell the fleet ran.
        with tempfile.TemporaryDirectory() as td:
            r = record.Recorder(td, 1, backend="sck", window_native=True,
                                capture_window=_pick(0))
            w = r._sck_workers[0]
            with open(w.raw_path, "wb") as f:
                f.write(b"\0" * 2048)
            w.proc = _FakeProc(0)
            w.t0 = 1.0
            w.entry = w.window
            r._finalize_fleet_single(1440.0, 900.0, "quartz", 1.0)
            with open(os.path.join(td, "meta.json")) as f:
                converted = json.load(f)
        # Reference: a singleton-configured recorder (mic keeps it off the
        # fleet) producing the same meta dict.
        ref = record.Recorder("/tmp/x", 1, backend="sck", window_native=True,
                              capture_window=_pick(0), mic_idx=0)
        ref.mic_idx = None      # key set only; mic_index is None either way
        ref._cw_entry = ref.capture_window
        ref_meta = ref._meta_dict(1440.0, 900.0, "quartz", 1.0, False)
        self.assertEqual(set(converted.keys()), set(ref_meta.keys()))


class BuildSceneWorkers(unittest.TestCase):
    def test_naming_and_fresh_snapshot(self):
        calls = []
        clips = []
        real = record.dev.window_rect_points

        # Accepts `clip_to_display` because the real one does: this is a
        # window-native take, so the re-snapshot must ask for UNCLIPPED rects
        # (docs/architecture.md, "The display-CLIPPED rect"). A double
        # that rejected the kwarg would raise into `_snapshot_window`'s
        # best-effort except and read as "the window vanished".
        def stub(wid, exclude_pids=None, main_only=True, clip_to_display=True):
            calls.append(wid)
            clips.append(clip_to_display)
            return _pick(0, wid)

        record.dev.window_rect_points = stub
        try:
            with tempfile.TemporaryDirectory() as td:
                r = record.Recorder(td, 1, backend="sck", window_native=True,
                                    capture_windows=[_pick(0), _pick(1)])
                workers = r._build_scene_workers(2, [_pick(0, 500),
                                                     _pick(1, 501)])
        finally:
            record.dev.window_rect_points = real
        self.assertEqual(
            [os.path.basename(w.raw_path) for w in workers],
            ["scene_2_raw_0.mov", "scene_2_raw_1.mov"])
        self.assertEqual(calls, [500, 501])
        self.assertEqual(clips, [False, False], "a scene re-pick on an "
                         "occlusion-free take must read unclipped rects")
        self.assertTrue(all(w.resnapshot for w in workers))


class WaitFleetT0s(unittest.TestCase):
    def _rec(self):
        return record.Recorder("/tmp/x", 1, backend="sck",
                               window_native=True,
                               capture_windows=[_pick(0), _pick(1)])

    def test_true_when_every_t0_lands(self):
        r = self._rec()
        for w in r._sck_workers:
            w.proc = _FakeProc(None)
            w.t0 = 5.0
        self.assertTrue(r._wait_fleet_t0s(r._sck_workers, timeout=0.2))

    def test_false_on_a_missing_t0(self):
        r = self._rec()
        for w in r._sck_workers:
            w.proc = _FakeProc(None)
        r._sck_workers[0].t0 = 5.0
        self.assertFalse(r._wait_fleet_t0s(r._sck_workers, timeout=0.1))

    def test_false_fast_on_child_death(self):
        r = self._rec()
        for w in r._sck_workers:
            w.proc = _FakeProc(1)
        self.assertFalse(r._wait_fleet_t0s(r._sck_workers, timeout=5.0))


class FinalizeActiveScene(_MoovPatched):
    def _rec(self, td):
        return record.Recorder(td, 1, backend="sck", window_native=True,
                               capture_windows=[_pick(0), _pick(1)])

    def test_happy_path_appends_a_scene_with_wall_end(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            workers = [_worker(td, 0, 0), _worker(td, 0, 1)]
            scene = r._finalize_active_scene(workers, [])
            self.assertIsNotNone(scene)
            self.assertEqual(scene.index, 0)
            self.assertEqual(len(r._scenes), 1)
            self.assertIsNotNone(scene.wall_end)

    def test_null_t0_fails_the_scene_never_falls_back(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            workers = [_worker(td, 0, 0), _worker(td, 0, 1, t0=None)]
            self.assertIsNone(r._finalize_active_scene(workers, []))
            self.assertEqual(r._scenes, [])

    def test_missing_moov_fails_the_scene(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            record.sck.has_moov = lambda path: False
            workers = [_worker(td, 0, 0)]
            self.assertIsNone(r._finalize_active_scene(workers, []))


class AbortSpawnedScene(unittest.TestCase):
    def test_partial_files_are_removed_and_scenes_kept(self):
        with tempfile.TemporaryDirectory() as td:
            r = record.Recorder(td, 1, backend="sck", window_native=True,
                                capture_windows=[_pick(0), _pick(1)])
            kept = record._SceneRecord(0, [])
            r._scenes = [kept]
            doomed = [_worker(td, 1, 0), _worker(td, 1, 1)]
            r._abort_spawned_scene(doomed, [])
            for w in doomed:
                self.assertFalse(os.path.exists(w.raw_path))
            self.assertEqual(r._scenes, [kept])


class SceneRunLoopOrdering(unittest.TestCase):
    """Source pins on `_start_multi_native`'s state machine -- the same
    inspect discipline `test_record_segmented` uses on `start()`."""

    def setUp(self):
        self.src = inspect.getsource(record.Recorder._start_multi_native)

    def test_pause_gates_events_before_the_scene_finalize(self):
        branch = self.src.split('notify("pausing")')[1].split(
            'notify("paused"')[0]
        self.assertIn("self._paused.set()", branch)
        self.assertLess(branch.index("self._paused.set()"),
                        branch.index("_finalize_active_scene"))

    def test_pause_clears_a_stale_resume_and_pending_pick(self):
        # A duplicate resume during `resuming` must not replay a STALE window
        # set at the next pause (docs/architecture.md, record lifecycle).
        branch = self.src.split('notify("pausing")')[1].split(
            "state = \"paused\"")[0]
        self.assertIn("_resume_requested.clear()", branch)
        self.assertIn("self._scene_pending = None", branch)

    def test_pause_clears_a_stale_grow_and_pending_entry(self):
        # A grow that raced the pause must not replay after resume -- the
        # scene finalize ignores `_join_marks`, so the replayed joiner's
        # file would be silently orphaned (docs/architecture.md M3.0c).
        branch = self.src.split('notify("pausing")')[1].split(
            "state = \"paused\"")[0]
        self.assertIn("_grow_requested.clear()", branch)
        self.assertIn("self._grow_pending = None", branch)

    def test_resume_failure_returns_to_paused_not_take_death(self):
        branch = self.src.split('notify("resuming")')[1]
        self.assertIn("_abort_spawned_scene", branch)
        self.assertIn('notify("paused"', branch)
        self.assertNotIn("died_early = ", branch)

    def test_paused_clears_only_after_every_t0(self):
        branch = self.src.split('notify("resuming")')[1]
        self.assertLess(branch.index("_wait_fleet_t0s"),
                        branch.index("self._paused.clear()"))


class WallEndStampsAtCaptureEnd(unittest.TestCase):
    def test_wall_end_taken_before_the_harvest(self):
        # The shortfall detector compares t0+D against wall_end; stamping it
        # AFTER the multi-second harvest/join/gate inflates every measurement
        # by teardown latency and makes the design's #1 named hazard warn on
        # every healthy pause (the review's top finding).
        src = inspect.getsource(record.Recorder._finalize_active_scene)
        self.assertIn("wall_end = time.monotonic()", src)
        self.assertLess(src.index("wall_end = time.monotonic()"),
                        src.index("self._harvest_fleet"))


class P1PauseClearsStaleResume(unittest.TestCase):
    def test_whole_screen_pause_branch_clears_the_replay(self):
        # Mirrored from the fleet loop: a duplicate resume during `resuming`
        # must not fire a ghost self-resume at the NEXT pause.
        src = inspect.getsource(record.Recorder.start)
        branch = src.split('notify("pausing")')[1].split('notify("paused"')[0]
        self.assertIn("_resume_requested.clear()", branch)


class GeometryCoalescingSeamFix(unittest.TestCase):
    def test_dropped_write_does_not_advance_the_coalescing_state(self):
        # A window that appears/moves during the paused gap must re-log on
        # the FIRST post-resume sample -- the old state update marked it
        # already-written and suppressed it until the next change/heartbeat.
        with tempfile.TemporaryDirectory() as td:
            r = record.Recorder(td, 1)
            r._ev_file = open(r.events_path, "w")
            try:
                r._paused.set()
                r._write_window_event(9, [0, 0, 50, 50], z=0)
                self.assertNotIn(9, r._win_last_rect)
                self.assertEqual(r._win_samples, 0)
                r._paused.clear()
                r._write_window_event(9, [0, 0, 50, 50], z=0)
                self.assertIn(9, r._win_last_rect)
                self.assertEqual(r._win_samples, 1)
            finally:
                r._ev_file.close()


class GrowSupportedPauseGate(unittest.TestCase):
    """An ever-paused take must refuse a join: finalize routes any 2+-scene
    take through the scene manifest, which ignores `_join_marks` -- the
    joiner would be silently dropped, its file orphaned, and
    `scene_channel_alignment`'s origin could truncate survivors
    (docs/architecture.md M3.0c)."""

    def _fleet(self):
        return record.Recorder(
            "/tmp/x", 1, backend="sck", window_native=True,
            capture_windows=[_pick(0), _pick(1)])

    def test_ever_paused_refuses_grow(self):
        r = self._fleet()
        self.assertTrue(r.grow_supported)
        r._ever_paused = True
        self.assertFalse(r.grow_supported)

    def test_try_grow_refuses_before_spawning_and_reports(self):
        r = self._fleet()
        r._ever_paused = True
        notes = []
        ok = r._try_grow(_pick(9), [], [], None,
                         lambda st, **kw: notes.append((st, kw)))
        self.assertFalse(ok)
        self.assertEqual(len(r._sck_workers), 2)   # nothing spliced in
        self.assertEqual(r._join_marks, [])
        self.assertTrue(any("grow_error" in kw for _, kw in notes))


class PauseSupported(unittest.TestCase):
    def test_whole_screen_and_fleet_yes_display_crop_no(self):
        self.assertTrue(record.Recorder("/tmp/x", 1).pause_supported)
        self.assertTrue(record.Recorder(
            "/tmp/x", 1, backend="sck", window_native=True,
            capture_windows=[_pick(0), _pick(1)]).pause_supported)
        self.assertFalse(record.Recorder(
            "/tmp/x", 1, capture_window=_pick(0)).pause_supported)
        self.assertFalse(record.Recorder(
            "/tmp/x", 1,
            capture_windows=[_pick(0), _pick(1)]).pause_supported)


if __name__ == "__main__":
    unittest.main()
