"""Pure (no ffmpeg / no permissions) tests for the segmented-take RECORD side.

Covers the parts of Layer 2 that don't need a live capture: the shared event
`_append` pause gate, the `capture_segments` manifest shape, segment file
naming, and the off-switch (a never-paused take stays the single-file path).
The live pause/resume of a real capture is an on-device gate (see
docs/architecture.md).
"""

import inspect
import json
import os
import tempfile
import unittest

from autocine import record
from autocine import segments


class AppendPauseGate(unittest.TestCase):
    """All three event writers route through `_append`, which drops writes
    while paused -- the deleted gap must contain no logged input."""

    def _rec_with_open_events(self, root):
        r = record.Recorder(root, 1)
        r._ev_file = open(r.events_path, "w")
        return r

    def _lines(self, path):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]

    def test_click_key_geometry_suppressed_while_paused(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec_with_open_events(td)
            try:
                r._write_event("down", 10, 20, "left")   # recorded
                r._write_window_event(7, [0, 0, 100, 100], z=0)  # recorded
                r._paused.set()
                r._write_event("down", 30, 40, "left")   # dropped (paused)
                r._on_key(None)                          # dropped (paused)
                r._write_window_event(7, [5, 5, 100, 100])  # dropped (paused)
                r._paused.clear()
                r._write_event("up", 50, 60, "left")     # recorded again
            finally:
                r._ev_file.close()
            recs = self._lines(r.events_path)
            types = [(rec["type"], rec.get("x")) for rec in recs]
            # Only the pre-pause and post-resume writes survive; nothing from
            # the paused window (the 30,40 down / key / moved rect) is present.
            self.assertIn(("down", 10.0), types)
            self.assertIn(("window", 50.0), types)   # centre of [0,0,100,100]
            self.assertIn(("up", 50.0), types)
            self.assertNotIn(("down", 30.0), types)
            self.assertFalse(any(t == "key" for t, _ in types))

    def test_never_paused_is_unchanged(self):
        # Off switch: with _paused never set, _append behaves like the old
        # inline writes -- every event lands.
        with tempfile.TemporaryDirectory() as td:
            r = self._rec_with_open_events(td)
            try:
                r._write_event("down", 1, 2, "left")
                r._write_event("up", 1, 2, "left")
            finally:
                r._ev_file.close()
            self.assertEqual(len(self._lines(r.events_path)), 2)


class SegmentedMetaDict(unittest.TestCase):
    """The capture_segments manifest, its shape, and mutual exclusion with the
    single-file `raw` key."""

    def _rec(self):
        r = record.Recorder("/tmp/x", 1, backend="sck", mic_idx=0)
        r._key_capture = "activity"
        r._segments = [
            record._Segment(0, "seg_0.mov", 100.0,
                            stat={"appended": 300, "dup": 60, "dropped": 0,
                                  "notready": 3, "idle": 12}),
            record._Segment(1, "seg_1.mov", 106.0, stat=None),
        ]
        return r

    def test_manifest_lists_every_segment_in_order(self):
        meta = self._rec()._segmented_meta_dict(1440.0, 900.0, "quartz")
        segs = meta["capture_segments"]
        self.assertEqual([s["file"] for s in segs], ["seg_0.mov", "seg_1.mov"])
        self.assertEqual([s["index"] for s in segs], [0, 1])
        self.assertEqual([s["t0_monotonic"] for s in segs], [100.0, 106.0])

    def test_no_top_level_raw(self):
        # The whole point: consumers that read meta["raw"] must fail loud, not
        # silently read segment 0.
        meta = self._rec()._segmented_meta_dict(1440.0, 900.0, "quartz")
        self.assertNotIn("raw", meta)

    def test_session_origin_is_segment_zero(self):
        meta = self._rec()._segmented_meta_dict(1440.0, 900.0, "quartz")
        self.assertEqual(meta["t0_monotonic"], 100.0)

    def test_mic_index_is_carried_face_is_cut(self):
        meta = self._rec()._segmented_meta_dict(1440.0, 900.0, "quartz")
        self.assertEqual(meta["mic_index"], 0)       # Phase 1 keeps mic ON
        self.assertIsNone(meta["face"])              # facecam is a Phase-1 cut

    def test_per_segment_capture_stats_only_when_present(self):
        meta = self._rec()._segmented_meta_dict(1440.0, 900.0, "quartz")
        s0, s1 = meta["capture_segments"]
        self.assertEqual(s0["capture_stats"]["idle_filled"], 12)
        self.assertNotIn("capture_stats", s1)

    def test_meta_is_recognized_as_segmented(self):
        meta = self._rec()._segmented_meta_dict(1440.0, 900.0, "quartz")
        self.assertTrue(segments.is_segmented_meta(meta))


class SegmentNaming(unittest.TestCase):
    def test_segment_zero_is_raw_mov_later_are_seg_i(self):
        # `_spawn_screen_segment` sets self.raw_path; segment 0 must stay
        # raw.mov (byte-identical off switch), later segments seg_i.mov. We
        # exercise the path-setting half without spawning a real process by
        # checking the name it targets.
        r = record.Recorder("/tmp/sess", 1)
        # index 0 -> raw.mov (the default already in the ctor)
        self.assertEqual(os.path.basename(r.raw_path), "raw.mov")
        # The naming rule the spawner uses:
        for i, want in ((0, "raw.mov"), (1, "seg_1.mov"), (2, "seg_2.mov")):
            name = ("raw.mov" if i == 0 else "seg_{:d}.mov".format(i))
            self.assertEqual(name, want)


class OffSwitchNoPause(unittest.TestCase):
    def test_fresh_recorder_has_no_segments(self):
        # A recorder that is never paused has an empty _segments list, so
        # start() takes the single-file path (len < 2) -- byte-identical.
        r = record.Recorder("/tmp/x", 1)
        self.assertEqual(r._segments, [])
        self.assertFalse(r._paused.is_set())
        self.assertFalse(r._ever_paused)


class AbortSpawnedSegment(unittest.TestCase):
    """Stop during a resume spin-up must DISCARD the frameless new segment and
    keep the take -- not fail it (the reviewed stop-during-resume defect)."""

    def test_restores_last_segment_and_removes_partial(self):
        with tempfile.TemporaryDirectory() as td:
            r = record.Recorder(td, 1)
            r._segments = [record._Segment(
                0, "raw.mov", 100.0, stat={"appended": 5}, filt="applied",
                size=[320, 180])]
            partial = os.path.join(r.session_dir, "seg_1.mov")
            with open(partial, "wb") as f:
                f.write(b"x")
            r.raw_path = partial
            with r._t0_lock:
                r._t0 = None
            r.proc = None
            r._abort_spawned_segment()
            # The single-file fallthrough (a pause-then-stop take with one
            # segment) must see exactly the stop-while-paused state: segment
            # 0's file, t0 and SCK snapshot restored, the partial gone.
            self.assertEqual(os.path.basename(r.raw_path), "raw.mov")
            self.assertEqual(r._t0, 100.0)
            self.assertEqual(r._sck_filter, "applied")
            self.assertEqual(r._sck_size, [320, 180])
            self.assertFalse(os.path.exists(partial))

    def test_reader_threads_drain_before_the_restore(self):
        # The dead worker's STAT/FILTER/SIZE setters are unconditional; the
        # readers must hit EOF before the snapshot restore, or a late line
        # from the discarded segment overwrites it.
        src = inspect.getsource(record.Recorder._abort_spawned_segment)
        self.assertLess(src.index("prog.join"),
                        src.index("self._sck_stat = last.stat"))

    def test_stop_during_resume_routes_to_abort_not_died_early(self):
        # Pin the run loop's routing: in the resume branch, a set
        # _stop_requested leads to _abort_spawned_segment (keep the take),
        # and only a REAL spin-up failure to died_early.
        src = inspect.getsource(record.Recorder.start)
        branch = src.split('notify("resuming")')[1]
        self.assertIn("_abort_spawned_segment", branch)
        self.assertLess(branch.index("_stop_requested.is_set()"),
                        branch.index("died_early = True"))
        self.assertLess(branch.index("_abort_spawned_segment"),
                        branch.index("died_early = True"))


class SpawnPublishOrdering(unittest.TestCase):
    def test_child_published_only_after_its_config_is_written(self):
        # The exclusion poller survives the pause and re-reads self.proc each
        # tick; publishing the child before its config line is written lets an
        # EXCLUDE line arrive as the worker's FIRST stdin line, which fails
        # its config json.loads and kills the take.
        src = inspect.getsource(record.Recorder._spawn_screen_segment)
        self.assertIn("self.proc = proc", src)
        self.assertLess(src.index("_sck_config"),
                        src.index("self.proc = proc"))


class PauseGateOrdering(unittest.TestCase):
    def test_paused_set_before_finalize_in_the_pause_branch(self):
        # The doc's pause step 1: gate the event writers FIRST. Finalize takes
        # seconds with no frames being produced -- input logged in that window
        # would clamp onto the seam as a phantom cluster.
        src = inspect.getsource(record.Recorder.start)
        branch = src.split('notify("pausing")')[1].split('notify("paused"')[0]
        self.assertIn("self._paused.set()", branch)
        self.assertLess(branch.index("self._paused.set()"),
                        branch.index("self._finalize_active_segment()"))


class FinalizeT0Fallback(unittest.TestCase):
    """A segment finalized before its first-frame pairing lands must never
    write a null t0 -- SegmentClock would coerce it to 0 and clamp every one
    of the segment's events onto the seam."""

    def _rec_with_raw(self, td):
        r = record.Recorder(td, 1)      # avfoundation: no moov gate
        with open(r.raw_path, "wb") as f:
            f.write(b"\0" * 2048)       # > 1024, passes the size gate
        r.proc = None
        return r

    def test_null_t0_falls_back_to_the_spawn_anchor(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec_with_raw(td)
            with r._t0_lock:
                r._t0 = None
            r._seg_spawn_mono = 123.25
            seg = r._finalize_active_segment()
            self.assertIsNotNone(seg)
            self.assertEqual(seg.t0, 123.25)

    def test_paired_t0_is_used_directly(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec_with_raw(td)
            with r._t0_lock:
                r._t0 = 50.5
            r._seg_spawn_mono = 1.0
            seg = r._finalize_active_segment()
            self.assertEqual(seg.t0, 50.5)


if __name__ == "__main__":
    unittest.main()
