"""The fleet card SHRINK, record side (docs/architecture.md Milestone 3, M3.2).

Covers the four layers the slice added:
  - `sck.plan_shrink_exits` -- the PURE per-poll exit decision (both-signals
    rule, back-dated seam, global-vanish guard, gaveup promotion)
  - `Recorder._try_shrink` -- the run-loop transition (remove-before-kill,
    departure gate, moov salvage, the exempt guards)
  - `Recorder._finalize_fleet_take` -- the exits-aware manifest
  - source-ordering pins on `_start_multi_native` (the SceneRunLoopOrdering
    idiom) and the structural off-switch

All without macOS permissions: fake procs, throwaway files, a patched moov
probe, and a patched `dev.window_onscreen`.
"""
import inspect
import os
import tempfile
import unittest

from autocine import record, sck


class _FakeProc(object):
    def __init__(self, rc=0, pid=4242):
        self.returncode = rc
        self.pid = pid
        self.signals = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)


def _pick(i, wid=None):
    return {"id": wid if wid is not None else 100 + i,
            "app": "App{}".format(i), "title": "t{}".format(i),
            "x": float(i * 200), "y": 0.0, "w": 400.0, "h": 300.0,
            "display_origin": [0.0, 0.0]}


def _worker(session_dir, idx, t0=10.0, rc=0, frames=600, write=True):
    w = record._NativeWorker(session_dir, idx, _pick(idx))
    if write:
        with open(w.raw_path, "wb") as f:
            f.write(b"\0" * 2048)
    w.proc = _FakeProc(rc)
    w.t0 = t0
    w.entry = w.window
    w.resnapshot = True
    w.size = {"width": 800, "height": 600}
    w.stat = {"appended": frames, "dup": 0, "dropped": 0, "notready": 0,
              "idle": 0}
    return w


class _MoovPatched(unittest.TestCase):
    """Patch the moov probe: these tests write throwaway bytes, not movies."""

    def setUp(self):
        self._real_has_moov = record.sck.has_moov
        record.sck.has_moov = lambda path: True
        self._real_onscreen = record.dev.window_onscreen
        record.dev.window_onscreen = lambda wid: False

    def tearDown(self):
        record.sck.has_moov = self._real_has_moov
        record.dev.window_onscreen = self._real_onscreen


def _fleet_rec(td, n=2, mic=None):
    r = record.Recorder(td, 1, backend="sck", window_native=True,
                        capture_windows=[_pick(i) for i in range(n)],
                        mic_idx=mic)
    r._key_capture = "activity"
    r._sck_workers = [_worker(td, i, t0=100.0 + i * 0.01) for i in range(n)]
    r._n_start = n
    r._next_channel_idx = n
    return r


# ---------------------------------------------------------------------------
# The pure decision
# ---------------------------------------------------------------------------
class PlanShrinkExits(unittest.TestCase):
    """`sck.plan_shrink_exits` -- decisions 1-3 as one pure function. The
    spine: an exit needs absent-from-list AND arrivals-dead, held for the
    debounce, and the seam back-dates to the hide start."""

    D = 5.0

    def _tick(self, tracker, onscreen, arrivals, now, gaveup=()):
        return sck.plan_shrink_exits(tracker, onscreen, arrivals, now,
                                     self.D, gaveup=gaveup)

    def test_arm_then_fire_with_backdated_seam(self):
        tr, ex = self._tick({}, {7: False}, {7: 500}, 100.0)
        self.assertEqual(ex, [])
        self.assertEqual(tr[7]["hidden_since"], 100.0)
        tr, ex = self._tick(tr, {7: False}, {7: 500}, 105.1)
        self.assertEqual(ex, [(7, 100.0)])       # seam = hide start, not now

    def test_debounce_holds_before_the_threshold(self):
        tr, _ = self._tick({}, {7: False}, {7: 500}, 100.0)
        tr, ex = self._tick(tr, {7: False}, {7: 500}, 104.9)
        self.assertEqual(ex, [])
        self.assertIsNotNone(tr[7]["hidden_since"])

    def test_seen_cancels_and_a_later_hide_rearms(self):
        tr, _ = self._tick({}, {7: False}, {7: 500}, 100.0)
        tr, ex = self._tick(tr, {7: True}, {7: 500}, 102.0)
        self.assertEqual(ex, [])
        self.assertIsNone(tr[7]["hidden_since"])
        tr, _ = self._tick(tr, {7: False}, {7: 500}, 103.0)
        tr, ex = self._tick(tr, {7: False}, {7: 500}, 108.1)
        self.assertEqual(ex, [(7, 103.0)])       # the NEW hide, not the old

    def test_unreadable_poll_counts_as_seen(self):
        tr, _ = self._tick({}, {7: False}, {7: 500}, 100.0)
        tr, ex = self._tick(tr, {7: None}, {7: 500}, 105.1)
        self.assertEqual(ex, [])                 # a Quartz hiccup never exits
        self.assertIsNone(tr[7]["hidden_since"])

    def test_arrivals_alive_blocks_and_rearms_the_baseline(self):
        # The measured Space case: off the list but still DELIVERING. The
        # baseline re-arms so the eventual seam sits where delivery stopped.
        tr, _ = self._tick({}, {7: False}, {7: 500}, 100.0)
        tr, ex = self._tick(tr, {7: False}, {7: 680}, 103.0)
        self.assertEqual(ex, [])
        self.assertEqual(tr[7]["hidden_since"], 103.0)
        tr, ex = self._tick(tr, {7: False}, {7: 680}, 107.9)
        self.assertEqual(ex, [])                 # only 4.9s delivery-dead
        tr, ex = self._tick(tr, {7: False}, {7: 680}, 108.1)
        self.assertEqual(ex, [(7, 103.0)])

    def test_global_vanish_suppresses_but_armed_wids_keep_their_clock(self):
        tr, _ = self._tick({}, {7: False, 8: True, 9: True},
                           {7: 1, 8: 1, 9: 1}, 100.0)
        # 8 and 9 vanish TOGETHER (Space switch): neither arms; 7 keeps its
        # earlier solo clock.
        tr, ex = self._tick(tr, {7: False, 8: False, 9: False},
                            {7: 1, 8: 1, 9: 1}, 102.0)
        self.assertEqual(ex, [])
        self.assertIsNone(tr[8].get("hidden_since"))
        self.assertIsNone(tr[9].get("hidden_since"))
        self.assertEqual(tr[7]["hidden_since"], 100.0)
        tr, ex = self._tick(tr, {7: False, 8: False, 9: False},
                            {7: 1, 8: 1, 9: 1}, 105.1)
        self.assertEqual(ex, [(7, 100.0)])

    def test_staggered_global_vanish_is_still_global(self):
        # MEASURED on-device (the gate's run 4): a Space switch staggers the
        # list-departures across ~50ms polls, so tick-coincidence read two
        # cards as two SOLO vanishes and one EXITED. Vanishes within
        # GLOBAL_VANISH_WINDOW_SEC of each other are one gesture: the second
        # is suppressed AND the first's arm is retro-cancelled.
        tr, ex = self._tick({}, {7: False, 8: True}, {7: 1, 8: 1}, 100.0)
        self.assertEqual(tr[7]["hidden_since"], 100.0)   # armed as solo...
        tr, ex = self._tick(tr, {7: False, 8: False}, {7: 1, 8: 1}, 100.05)
        self.assertEqual(ex, [])
        self.assertIsNone(tr[7].get("hidden_since"))     # ...retro-cancelled
        self.assertIsNone(tr[8].get("hidden_since"))
        # Both stay delivery-dead and hidden past the debounce: still no
        # exits -- the gesture window disarmed the pair for good.
        for t in (102.0, 104.0, 106.0):
            tr, ex = self._tick(tr, {7: False, 8: False}, {7: 1, 8: 1}, t)
        self.assertEqual(ex, [])

    def test_well_separated_solo_hides_still_exit(self):
        # Two deliberate minimizes >window apart are two per-window intents.
        tr, _ = self._tick({}, {7: False, 8: True}, {7: 1, 8: 1}, 100.0)
        tr, _ = self._tick(tr, {7: False, 8: False}, {7: 1, 8: 1}, 102.0)
        tr, ex = self._tick(tr, {7: False, 8: False}, {7: 1, 8: 1}, 105.1)
        self.assertEqual(ex, [(7, 100.0)])
        tr, ex = self._tick(tr, {8: False}, {8: 1}, 107.1)
        self.assertEqual(ex, [(8, 102.0)])

    def test_gaveup_promotes_with_the_original_hide_as_seam(self):
        # Restored on-screen but the stream is dead forever: the one case
        # the diagnostics channel earns a decision role. Seam = the REAL
        # hide start, retained across the restore cancel.
        tr, _ = self._tick({}, {7: False}, {7: 9}, 100.0)
        tr, _ = self._tick(tr, {7: True}, {7: 9}, 102.0)     # restore cancels
        tr, ex = self._tick(tr, {7: True}, {7: 9}, 104.0, gaveup={7})
        self.assertEqual(ex, [])
        self.assertEqual(tr[7]["hidden_since"], 100.0)
        tr, ex = self._tick(tr, {7: True}, {7: 9}, 105.1, gaveup={7})
        self.assertEqual(ex, [(7, 100.0)])

    def test_fired_wid_leaves_the_tracker(self):
        tr, _ = self._tick({}, {7: False}, {7: 0}, 100.0)
        tr, ex = self._tick(tr, {7: False}, {7: 0}, 105.1)
        self.assertEqual(len(ex), 1)
        self.assertNotIn(7, tr)


# ---------------------------------------------------------------------------
# Run-loop + source ordering pins
# ---------------------------------------------------------------------------
class ShrinkRunLoopOrdering(unittest.TestCase):
    """Source pins, the SceneRunLoopOrdering idiom."""

    def setUp(self):
        self.loop = inspect.getsource(record.Recorder._start_multi_native)
        self.shrink = inspect.getsource(record.Recorder._try_shrink)

    def test_shrink_drains_after_grow_before_liveness(self):
        self.assertLess(self.loop.index("_grow_requested.is_set()"),
                        self.loop.index("_shrink_requested.is_set()"))
        self.assertLess(self.loop.index("_shrink_requested.is_set()"),
                        self.loop.index("died_early = worker"))

    def test_pause_clears_the_shrink_mailbox(self):
        # An exit armed just before a pause must not replay after resume --
        # the scene machine has no exit support (decision 7).
        branch = self.loop.split('notify("pausing")')[1].split(
            'state = "paused"')[0]
        self.assertIn("_shrink_requested.clear()", branch)
        self.assertIn("self._shrink_pending = None", branch)

    def test_worker_is_removed_before_it_is_signalled(self):
        # Same thread as the liveness poll: removal-then-SIGINT is an
        # ordering pin, not a race guard -- but it is THE reason a departing
        # card can never be read as died_early.
        self.assertLess(self.shrink.index("_sck_workers.remove"),
                        self.shrink.index("send_signal"))
        self.assertLess(self.shrink.index("_sck_workers.remove"),
                        self.shrink.index("_respawn_watchdog"))
        self.assertLess(self.shrink.index("_respawn_watchdog"),
                        self.shrink.index("send_signal"))

    def test_exit_routing_precedes_the_join_routing(self):
        # A take with exits must reach _finalize_fleet_take; join-only takes
        # keep the untouched pre-M3.2 path (the byte-identity off-switch).
        self.assertLess(self.loop.index("_finalize_fleet_take"),
                        self.loop.index("_finalize_join_take"))


# ---------------------------------------------------------------------------
# The transition
# ---------------------------------------------------------------------------
class TryShrink(_MoovPatched):
    def _rec(self, td, n=2, mic=None):
        r = _fleet_rec(td, n=n, mic=mic)
        self.respawns = []
        r._respawn_watchdog = (
            lambda workers, kb: self.respawns.append(list(workers)))
        return r

    def test_success_retires_one_worker(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            notes = []
            gone = r._sck_workers[1]
            ok = r._try_shrink(101, 105.0, None,
                               lambda st, **kw: notes.append((st, kw)))
            self.assertTrue(ok)
            self.assertEqual([w.index for w in r._sck_workers], [0])
            self.assertEqual(r._exit_marks, [1])
            self.assertEqual(len(r._departed), 1)
            self.assertIs(r._departed[0].worker, gone)
            self.assertEqual(r._departed[0].seam_t, 105.0)
            self.assertTrue(r._departed[0].gated)
            # Watchdog re-armed over the SURVIVORS, before the SIGINT.
            self.assertEqual([[w.index for w in ws] for ws in self.respawns],
                             [[0]])
            self.assertIn(("recording", {"departed": 1,
                                         "departed_app": "App1",
                                         "windows": 1}),
                          notes)
            self.assertIsNotNone(gone.end_rect)

    def test_mic_anchor_never_shrinks(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td, mic=0)
            self.assertFalse(r._try_shrink(100, 105.0, None,
                                           lambda st, **kw: None))
            self.assertEqual(len(r._sck_workers), 2)
            self.assertEqual(r._exit_marks, [])

    def test_last_card_never_shrinks(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td, n=1)
            self.assertFalse(r._try_shrink(100, 105.0, None,
                                           lambda st, **kw: None))
            self.assertEqual(len(r._sck_workers), 1)

    def test_ever_paused_never_shrinks(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            r._ever_paused = True
            self.assertFalse(r._try_shrink(101, 105.0, None,
                                           lambda st, **kw: None))
            self.assertEqual(len(r._sck_workers), 2)

    def test_restore_racing_the_decision_cancels(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            record.dev.window_onscreen = lambda wid: True
            self.assertFalse(r._try_shrink(101, 105.0, None,
                                           lambda st, **kw: None))
            self.assertEqual(len(r._sck_workers), 2)

    def test_gate_failure_records_an_ungated_departure(self):
        # The departure is RECORDED (gated=False) even when the file is
        # unusable -- an unrecorded departure would make finalize treat the
        # reduced fleet as the whole take (adversarially measured: a
        # 1-channel flat manifest no dispatcher accepts, or a stale-prefix
        # join fallback truncating every survivor). The record lands BEFORE
        # the harvest so a Ctrl+C mid-wait leaves it too.
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            record.sck.has_moov = lambda path: False
            ok = r._try_shrink(101, 105.0, None, lambda st, **kw: None)
            self.assertTrue(ok)
            self.assertEqual([w.index for w in r._sck_workers], [0])
            self.assertEqual(r._exit_marks, [])      # nothing gated...
            self.assertEqual(len(r._departed), 1)    # ...but never invisible
            self.assertFalse(r._departed[0].gated)
            self.assertFalse(r.pause_supported)      # decision 8 holds

    def test_departure_record_lands_before_the_harvest(self):
        # Source pin for the Ctrl+C hole: the _DepartedChannel append must
        # precede the SIGINT/wait so an interrupt anywhere still routes.
        src = inspect.getsource(record.Recorder._try_shrink)
        self.assertLess(src.index("self._departed.append"),
                        src.index("send_signal"))

    def test_abnormal_exit_with_moov_is_salvaged(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            r._sck_workers[1].proc = _FakeProc(rc=1)
            ok = r._try_shrink(101, 105.0, None, lambda st, **kw: None)
            self.assertTrue(ok)
            self.assertEqual(r._exit_marks, [1])
            self.assertTrue(r._departed[0].gated)


class ShrinkGuards(unittest.TestCase):
    def test_env_kill_switch_disables_detection(self):
        old = os.environ.get("AUTOCINE_FLEET_SHRINK")
        try:
            os.environ["AUTOCINE_FLEET_SHRINK"] = "0"
            with tempfile.TemporaryDirectory() as td:
                r = record.Recorder(td, 1, backend="sck", window_native=True,
                                    capture_windows=[_pick(0), _pick(1)])
                self.assertFalse(r._shrink_enabled)
        finally:
            if old is None:
                os.environ.pop("AUTOCINE_FLEET_SHRINK", None)
            else:
                os.environ["AUTOCINE_FLEET_SHRINK"] = old

    def test_pause_supported_false_once_joined_or_shrunk(self):
        # Decision 8: the scene finalize ignores both mark sets, so pausing
        # after either would silently drop a channel. Live-read.
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            self.assertTrue(r.pause_supported)
            r._join_marks = [2]
            self.assertFalse(r.pause_supported)
            r._join_marks = []
            r._exit_marks = [1]
            self.assertFalse(r.pause_supported)

    def test_channel_indexes_are_monotone_across_a_shrink(self):
        # A rejoin must NEVER collide with a departed channel's raw_i.mov.
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            self.assertEqual(r._n_start, 2)
            self.assertEqual(r._alloc_channel_index(), 2)
            del r._sck_workers[0]                    # a card left
            self.assertEqual(r._alloc_channel_index(), 3)   # not 1, not 2


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------
class FinalizeFleetTake(_MoovPatched):
    def _load(self, r):
        import json
        with open(r.meta_path) as f:
            return json.load(f)

    def test_exit_only_two_scene_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            gone = r._sck_workers.pop(1)
            r._departed = [record._DepartedChannel(gone, 105.0)]
            r._exit_marks = [1]
            r._finalize_fleet_take(1440.0, 900.0, "quartz", 99.0)
            meta = self._load(r)
            self.assertIn("capture_scenes", meta)
            self.assertNotIn("capture_channels", meta)
            s0, s1 = meta["capture_scenes"]
            self.assertEqual(len(s0["channels"]), 2)
            self.assertEqual(len(s1["channels"]), 1)
            # Survivor: one continuous file, sub-ranged contiguously across
            # the seam, covering all 600 frames.
            a = next(c for c in s0["channels"] if c["file"] == "raw_0.mov")
            b = s1["channels"][0]
            self.assertEqual(b["file"], "raw_0.mov")
            self.assertEqual(b["frame_start"], a["frame_count"])
            self.assertEqual(a["frame_count"] + b["frame_count"], 600)
            # Departed: ends at the back-dated seam; its dup tail is simply
            # never referenced.
            d = next(c for c in s0["channels"] if c["file"] == "raw_1.mov")
            self.assertEqual(d["frame_count"], 299)  # round((105-100.01)*60)
            self.assertEqual(s1["t0_monotonic"], 105.0)

    def test_join_then_exit_three_scene_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            joiner = _worker(td, 2, t0=101.0, frames=480)
            r._join_marks = [2]
            r._departed = [record._DepartedChannel(joiner, 103.0)]
            r._exit_marks = [2]
            r._finalize_fleet_take(1440.0, 900.0, "quartz", 99.0)
            scenes = self._load(r)["capture_scenes"]
            self.assertEqual([len(s["channels"]) for s in scenes], [2, 3, 2])
            j = scenes[1]["channels"][2]
            self.assertEqual(j["file"], "raw_2.mov")
            self.assertEqual((j["frame_start"], j["frame_count"]), (0, 120))

    def test_early_exit_downconverts_to_single_native(self):
        # A card that departed almost at the start: the planner demotes it
        # and the lone survivor DOWN-CONVERTS to the pinned single-native
        # shape (render refuses <2 capture_channels), with the stale multi
        # pick state cleared -- capture_window / capture_windows are
        # mutually exclusive composition contracts.
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            gone = r._sck_workers.pop(1)
            r._departed = [record._DepartedChannel(gone, 100.05)]
            r._exit_marks = [1]
            r._finalize_fleet_take(1440.0, 900.0, "quartz", 99.0)
            meta = self._load(r)
            self.assertNotIn("capture_channels", meta)
            self.assertNotIn("capture_scenes", meta)
            self.assertNotIn("capture_windows", meta)
            self.assertEqual(meta["raw"], "raw.mov")
            self.assertEqual(meta["capture_window"]["id"], 100)
            self.assertEqual(meta["capture_window"]["mode"], "window_native")
            self.assertTrue(os.path.exists(
                os.path.join(td, "raw.mov")))       # raw_0.mov renamed

    def test_gate_failed_departure_leaves_survivors_flat(self):
        # Salvage failed at departure: no seam, no manifest entry -- the
        # survivors' continuous files span the moment.
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td, n=3)
            gone = r._sck_workers.pop(2)
            r._departed = [record._DepartedChannel(gone, 105.0, gated=False)]
            r._finalize_fleet_take(1440.0, 900.0, "quartz", 99.0)
            meta = self._load(r)
            self.assertIn("capture_channels", meta)
            self.assertEqual(
                [c["file"] for c in meta["capture_channels"]],
                ["raw_0.mov", "raw_1.mov"])


class _Clock(object):
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class ShrinkTick(unittest.TestCase):
    """`Recorder._shrink_tick` -- the detector feed itself (adopted from the
    adversarial mutation pass: every one of these killed a surviving
    mutant)."""

    def setUp(self):
        self._mono = record.time.monotonic
        self._onscreen = record.dev.window_onscreen
        self.clock = _Clock(100.0)
        record.time.monotonic = self.clock

    def tearDown(self):
        record.time.monotonic = self._mono
        record.dev.window_onscreen = self._onscreen

    def _fire(self, r, entries, dt=6.0):
        """Two ticks `dt` apart with `entries` as the filtered list."""
        r._shrink_tick(entries)
        self.clock.t += dt
        r._shrink_tick(entries)

    def test_mic_anchor_never_reaches_the_mailbox(self):
        record.dev.window_onscreen = lambda wid: False
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td, mic=0)
            # worker 0 (wid 100, the mic anchor) vanishes; worker 1 stays.
            self._fire(r, [{"id": 101}])
            self.assertFalse(r._shrink_requested.is_set())
            self.assertIsNone(r._shrink_pending)

    def test_anchor_still_counts_toward_the_global_vanish_guard(self):
        # Decision 1 reads "captured wids", not "shrinkable wids": the mic
        # anchor + one card vanishing TOGETHER is a Minimize-All/app event,
        # so the other card must be suppressed, not solo-exited.
        record.dev.window_onscreen = lambda wid: False
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td, n=3, mic=0)
            self._fire(r, [{"id": 102}])      # anchor 100 AND card 101 vanish
            self.assertFalse(r._shrink_requested.is_set())

    def test_absent_from_list_but_onscreen_counts_as_seen(self):
        # Filtered off the list (other display / tiny / alpha shim) but the
        # direct query says ONSCREEN: must never arm, even delivery-dead.
        record.dev.window_onscreen = lambda wid: True
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            self._fire(r, [{"id": 100}])       # wid 101 absent from the list
            self.assertFalse(r._shrink_requested.is_set())

    def test_dup_frames_are_not_arrivals(self):
        # Hidden window whose worker keeps APPENDING dup (freeze) frames:
        # appended advances, appended-dup is flat -> must still exit.
        record.dev.window_onscreen = lambda wid: False
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            w = r._sck_workers[1]
            w.stat = {"appended": 600, "dup": 0}
            r._shrink_tick([{"id": 100}])
            self.clock.t += 6.0
            w.stat = {"appended": 960, "dup": 360}   # +360, all dup-fill
            r._shrink_tick([{"id": 100}])
            self.assertTrue(r._shrink_requested.is_set())
            self.assertEqual(r._shrink_pending, {101: 100.0})

    def test_kill_switch_disarms_the_tick(self):
        record.dev.window_onscreen = lambda wid: False
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            r._shrink_enabled = False
            self._fire(r, [{"id": 100}])
            self.assertFalse(r._shrink_requested.is_set())

    def test_ever_paused_disarms_the_tick(self):
        record.dev.window_onscreen = lambda wid: False
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            r._ever_paused = True
            self._fire(r, [{"id": 100}])
            self.assertFalse(r._shrink_requested.is_set())

    def test_fired_exit_sets_the_event_and_the_mailbox(self):
        record.dev.window_onscreen = lambda wid: False
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            self._fire(r, [{"id": 100}])
            self.assertTrue(r._shrink_requested.is_set())
            self.assertEqual(r._shrink_pending, {101: 100.0})

    def test_hidden_mic_anchor_stashes_the_freeze_note_once(self):
        # Decision 6's hint: the anchor met the full exit bar (hidden +
        # delivery-dead past the debounce) and was spared -- one note per
        # hide episode, delivered by the run loop; the SEEN edge queues a
        # clear (so a reloaded bar never keeps a stale hint) and re-arms.
        record.dev.window_onscreen = lambda wid: False
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td, mic=0)
            # Anchor (wid 100) vanishes FIRST (solo); card 101 stays.
            self._fire(r, [{"id": 101}])
            self.assertEqual(r._anchor_note_pending, ("hidden", "App0"))
            self.assertFalse(r._shrink_requested.is_set())  # exit discarded
            # Still hidden: no re-note within the same episode.
            r._anchor_note_pending = None
            self.clock.t += 6.0
            r._shrink_tick([{"id": 101}])
            self.clock.t += 6.0
            r._shrink_tick([{"id": 101}])
            self.assertIsNone(r._anchor_note_pending)
            # Seen again: the CLEAR message queues, then a NEW hide episode
            # produces a fresh note.
            r._shrink_tick([{"id": 100}, {"id": 101}])
            self.assertEqual(r._anchor_note_pending, ("seen", None))
            self._fire(r, [{"id": 101}])
            self.assertEqual(r._anchor_note_pending, ("hidden", "App0"))

    def test_one_card_mic_fleet_still_gets_the_freeze_note(self):
        # The case where the hint matters MOST: a mic fleet shrunk to the
        # anchor alone -- the entire visible recording is a frozen card
        # while the mic keeps rolling. Exit detection needs >=2 cards; the
        # note must not (adversarially caught coverage hole).
        record.dev.window_onscreen = lambda wid: False
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td, mic=0)
            r._sck_workers.pop(1)              # shrunk to the anchor alone
            self._fire(r, [])
            self.assertEqual(r._anchor_note_pending, ("hidden", "App0"))
            self.assertFalse(r._shrink_requested.is_set())
            self.assertIsNone(r._shrink_pending)

    def test_the_tick_never_mutates_the_worker_list(self):
        # The poller thread OWNS detection only; `_sck_workers` belongs to
        # the run loop -- that single-writer rule is what makes
        # remove-before-SIGINT an ordering pin instead of a race guard.
        record.dev.window_onscreen = lambda wid: False
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            before = list(r._sck_workers)
            self._fire(r, [{"id": 100}])
            self.assertTrue(r._shrink_requested.is_set())
            self.assertEqual(r._sck_workers, before)


class DrainShrink(_MoovPatched):
    """`_drain_shrink` -- the mailbox swap the run loop performs, executed
    for real (extracted so it is pinnable; the source pins only ordered
    substrings)."""

    def test_drains_each_wid_with_its_own_seam(self):
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td, n=3)
            r._respawn_watchdog = lambda workers, kb: None
            r._shrink_pending = {101: 105.0, 102: 107.5}
            r._shrink_requested.set()
            r._drain_shrink(None, lambda st, **kw: None)
            self.assertFalse(r._shrink_requested.is_set())
            self.assertIsNone(r._shrink_pending)
            # Both retire (3 cards -> 1, never past the last-card guard),
            # each with ITS OWN seam -- never a shared/garbled one.
            seams = {d.worker.index: d.seam_t for d in r._departed}
            self.assertEqual(seams, {1: 105.0, 2: 107.5})

    def test_event_cleared_before_the_swap(self):
        src = inspect.getsource(record.Recorder._drain_shrink)
        self.assertLess(src.index("_shrink_requested.clear()"),
                        src.index("pending, self._shrink_pending"))


class _HangProc(object):
    """A proc that ignores SIGINT: wait() times out until kill()."""

    def __init__(self):
        self.killed = False
        self.signals = []
        self.pid = 4242
        self.returncode = None

    def poll(self):
        return -9 if self.killed else None

    def send_signal(self, sig):
        self.signals.append(sig)

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        import subprocess as sp
        if not self.killed:
            raise sp.TimeoutExpired("worker", timeout)
        return self.returncode


class TryShrinkDeadline(_MoovPatched):
    def test_harvest_deadline_falls_back_to_sigkill(self):
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            r._respawn_watchdog = lambda workers, kb: None
            hang = _HangProc()
            r._sck_workers[1].proc = hang
            ok = r._try_shrink(101, 105.0, None, lambda st, **kw: None)
            self.assertTrue(ok)
            self.assertTrue(hang.killed)         # SIGKILL fallback happened
            # moov salvage keeps the channel despite rc=-9.
            self.assertEqual(r._exit_marks, [1])


class GateFleetSalvage(_MoovPatched):
    """`_gate_fleet` -- the all-or-nothing finalize gate, and the moov salvage
    that keeps a SIGKILLed-but-playable take rather than discarding all N
    channels. The bar's inherited SIGINT block (cli._install_signal_quit leaks
    a blocked mask to every child; each worker now self-unblocks) lost EXACTLY
    this: both workers ignored the harvest SIGINT, were SIGKILLed at the 40s
    deadline (rc=-9), and every raw_i.mov still had a moov via Phase C's
    fragments -- yet the gate failed the whole take."""

    def _capture_stderr(self, fn):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            out = fn()
        return out, buf.getvalue()

    def test_clean_exit_passes(self):
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            self.assertEqual(r._gate_fleet(r._sck_workers), [])

    def test_sigint_stop_passes(self):
        import signal
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            for w in r._sck_workers:
                w.proc.returncode = -signal.SIGINT
            self.assertEqual(r._gate_fleet(r._sck_workers), [])

    def test_sigkill_but_moov_is_salvaged(self):
        import signal
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            for w in r._sck_workers:
                w.proc.returncode = -signal.SIGKILL
            failures, err = self._capture_stderr(
                lambda: r._gate_fleet(r._sck_workers))
            self.assertEqual(failures, [])           # take kept, not discarded
            self.assertIn("moov salvage", err)       # but the exit is announced

    def test_rc1_but_moov_is_salvaged(self):
        # A worker's own non-signal failure exit (rc=1) is salvaged too, as
        # long as the file is playable -- the same rule the shrink path uses.
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            r._sck_workers[0].proc.returncode = 1
            failures, err = self._capture_stderr(
                lambda: r._gate_fleet(r._sck_workers))
            self.assertEqual(failures, [])
            self.assertIn("moov salvage", err)

    def test_no_moov_still_fails(self):
        # Salvage is moov-gated: a SIGKILL that left no moov is a real loss and
        # must still fail the all-or-nothing take.
        import signal
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            r._sck_workers[1].proc.returncode = -signal.SIGKILL
            gone = r._sck_workers[1].raw_path
            record.sck.has_moov = lambda p, g=gone: p != g
            failures = r._gate_fleet(r._sck_workers)
            self.assertEqual(len(failures), 1)
            self.assertIn("no moov", failures[0][1])

    def test_missing_file_still_fails(self):
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            os.remove(r._sck_workers[0].raw_path)
            failures = r._gate_fleet(r._sck_workers)
            self.assertEqual(len(failures), 1)
            self.assertIn("no ", failures[0][1])


class FinalizeGaps(_MoovPatched):
    """Position-vs-index discipline in `_finalize_fleet_take` (adopted from
    the mutation pass: every fixture elsewhere had position == index)."""

    def _load(self, r):
        import json
        with open(r.meta_path) as f:
            return json.load(f)

    def test_exit_seams_key_position_not_channel_index(self):
        # An index GAP separates position from channel index: a joiner
        # allocated index 3 (index 2 burned by an aborted grow) departs.
        # Its POSITION is 2; keying by worker.index would point past the
        # channel list.
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            joiner = _worker(td, 3, t0=101.0, frames=480)
            r._join_marks = [3]
            r._next_channel_idx = 4
            r._departed = [record._DepartedChannel(joiner, 103.0)]
            r._exit_marks = [3]
            r._finalize_fleet_take(1440.0, 900.0, "quartz", 99.0)
            scenes = self._load(r)["capture_scenes"]
            self.assertEqual([len(s["channels"]) for s in scenes], [2, 3, 2])
            j = scenes[1]["channels"][2]
            self.assertEqual(j["file"], "raw_3.mov")
            self.assertEqual((j["frame_start"], j["frame_count"]), (0, 120))
            self.assertEqual(scenes[2]["t0_monotonic"], 103.0)

    def test_departed_original_sorts_before_a_surviving_joiner(self):
        # Original channel 1 departs AFTER a joiner (index 2) arrived:
        # survivors [0, 2] + departed [1] must plan in channel-index order.
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            joiner = _worker(td, 2, t0=101.0, frames=480)
            r._sck_workers.append(joiner)
            r._join_marks = [2]
            r._next_channel_idx = 3
            gone = r._sck_workers.pop(1)
            r._departed = [record._DepartedChannel(gone, 103.0)]
            r._exit_marks = [1]
            r._finalize_fleet_take(1440.0, 900.0, "quartz", 99.0)
            scenes = self._load(r)["capture_scenes"]
            self.assertEqual([len(s["channels"]) for s in scenes], [2, 3, 2])
            self.assertEqual([c["file"] for c in scenes[1]["channels"]],
                             ["raw_0.mov", "raw_1.mov", "raw_2.mov"])
            self.assertEqual([c["file"] for c in scenes[2]["channels"]],
                             ["raw_0.mov", "raw_2.mov"])


class FinalizeRouting(unittest.TestCase):
    """The two adversarially-found routing holes, pinned at the source
    level (the behavioral halves live in FinalizeFleetTake / TryShrink)."""

    def setUp(self):
        self.loop = inspect.getsource(record.Recorder._start_multi_native)

    def test_exit_routing_precedes_the_fleet_single_downconvert(self):
        # A 1-window take that grew and shrank back to one card still has a
        # departed channel only _finalize_fleet_take can express -- the
        # down-convert running first silently discarded its footage.
        self.assertLess(self.loop.index("_finalize_fleet_take"),
                        self.loop.index("_finalize_fleet_single"))

    def test_teardown_harvests_departed_workers_too(self):
        # A Ctrl+C that interrupts a shrink's own harvest wait must not
        # orphan the departing child.
        tail = self.loop.split("_stop_window_track()")[1]
        self.assertIn("d.worker for d in self._departed", tail)


# ---------------------------------------------------------------------------
# Auto-rejoin (M3.3)
# ---------------------------------------------------------------------------
class PlanRejoinReady(unittest.TestCase):
    """`sck.plan_rejoin_ready` -- the pure restore edge: back on the
    pickable list CONTINUOUSLY for the stability window, then ready."""

    S = 1.0

    def test_stable_presence_fires_once(self):
        w, ready = sck.plan_rejoin_ready({}, [7], {7}, 100.0, self.S)
        self.assertEqual(ready, [])
        w, ready = sck.plan_rejoin_ready(w, [7], {7}, 101.1, self.S)
        self.assertEqual(ready, [7])
        self.assertNotIn(7, w)                   # left the watch

    def test_absence_resets_the_stability_clock(self):
        w, _ = sck.plan_rejoin_ready({}, [7], {7}, 100.0, self.S)
        w, ready = sck.plan_rejoin_ready(w, [7], set(), 100.5, self.S)
        self.assertEqual(ready, [])
        w, ready = sck.plan_rejoin_ready(w, [7], {7}, 100.6, self.S)
        self.assertEqual(ready, [])              # clock restarted at 100.6
        w, ready = sck.plan_rejoin_ready(w, [7], {7}, 101.7, self.S)
        self.assertEqual(ready, [7])

    def test_return_after_absence_starts_from_zero(self):
        # 1.2s since the FIRST sighting but 0s since the return: a stale
        # carried-over `since` would fire here and defeat the stability
        # window entirely (a restore-animation flicker would rejoin).
        w, _ = sck.plan_rejoin_ready({}, [7], {7}, 100.0, self.S)
        w, _ = sck.plan_rejoin_ready(w, [7], set(), 100.5, self.S)
        w, ready = sck.plan_rejoin_ready(w, [7], {7}, 101.2, self.S)
        self.assertEqual(ready, [])

    def test_unwatched_wids_fall_out(self):
        w, _ = sck.plan_rejoin_ready({}, [7], {7}, 100.0, self.S)
        w, _ = sck.plan_rejoin_ready(w, [], {7}, 100.5, self.S)
        self.assertEqual(w, {})


class WatchRejoins(unittest.TestCase):
    """`Recorder._watch_rejoins` via `_shrink_tick` -- the poller feed."""

    def setUp(self):
        self._mono = record.time.monotonic
        self._onscreen = record.dev.window_onscreen
        self.clock = _Clock(100.0)
        record.time.monotonic = self.clock
        record.dev.window_onscreen = lambda wid: False

    def tearDown(self):
        record.time.monotonic = self._mono
        record.dev.window_onscreen = self._onscreen

    def _departed_rec(self, td):
        r = _fleet_rec(td)
        gone = r._sck_workers.pop(1)
        r._departed = [record._DepartedChannel(gone, 95.0)]
        return r

    def test_restored_departed_wid_fires_after_the_stability_window(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._departed_rec(td)
            r._shrink_tick([{"id": 100}, {"id": 101}])   # wid 101 is back
            self.clock.t += 1.2
            r._shrink_tick([{"id": 100}, {"id": 101}])
            self.assertTrue(r._rejoin_requested.is_set())
            self.assertEqual(r._rejoin_pending, {101})

    def test_rejoin_watch_runs_even_on_a_one_card_fleet(self):
        # The flagship case: a 2-card take shrunk to 1 -- exit detection
        # needs >=2 live cards, the rejoin watch must NOT.
        with tempfile.TemporaryDirectory() as td:
            r = self._departed_rec(td)
            self.assertEqual(len(r._sck_workers), 1)
            r._shrink_tick([{"id": 101}])
            self.clock.t += 1.2
            r._shrink_tick([{"id": 101}])
            self.assertTrue(r._rejoin_requested.is_set())

    def test_attempted_wids_are_not_watched(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._departed_rec(td)
            r._rejoin_attempted = {101}
            r._shrink_tick([{"id": 101}])
            self.clock.t += 1.2
            r._shrink_tick([{"id": 101}])
            self.assertFalse(r._rejoin_requested.is_set())

    def test_still_hidden_departed_wid_never_fires(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._departed_rec(td)
            r._shrink_tick([{"id": 100}])
            self.clock.t += 5.0
            r._shrink_tick([{"id": 100}])
            self.assertFalse(r._rejoin_requested.is_set())

    def test_recaptured_wid_is_not_watched(self):
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            gone = r._sck_workers.pop(1)
            r._departed = [record._DepartedChannel(gone, 95.0)]
            back = _worker(td, 2)
            back.window = _pick(1)            # wid 101 re-captured via chip
            back.entry = back.window
            r._sck_workers.append(back)
            r._shrink_tick([{"id": 100}, {"id": 101}])
            self.clock.t += 1.2
            r._shrink_tick([{"id": 100}, {"id": 101}])
            self.assertFalse(r._rejoin_requested.is_set())

    def test_at_cap_fleet_never_arms_the_watch(self):
        # At the cap there is nothing an auto-rejoin could do; arming would
        # end in a drain-side skip. The watch stays quiet until a shrink
        # frees a slot -- and the attempt stays UN-burned for that moment.
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td, n=4)
            gone = r._sck_workers.pop(1)      # wid 101 departs
            r._departed = [record._DepartedChannel(gone, 95.0)]
            grown = _worker(td, 4)            # a 4th live card: at _MAX_FLEET
            r._sck_workers.append(grown)
            self.assertEqual(len(r._sck_workers), record._MAX_FLEET)
            ents = [{"id": w} for w in (100, 101, 102, 103, 104)]
            r._shrink_tick(ents)
            self.clock.t += 1.2
            r._shrink_tick(ents)
            self.assertFalse(r._rejoin_requested.is_set())

    def test_stale_watch_state_is_cleared_when_nothing_is_watchable(self):
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            gone = r._sck_workers.pop(1)
            r._departed = [record._DepartedChannel(gone, 95.0)]
            r._shrink_tick([{"id": 100}, {"id": 101}])   # arms at t=100.0
            r._rejoin_attempted = {101}                   # now unwatchable
            self.clock.t += 0.5
            r._shrink_tick([{"id": 100}, {"id": 101}])   # must clear state
            r._rejoin_attempted = set()                   # watchable again
            self.clock.t += 0.7                           # t=101.2
            r._shrink_tick([{"id": 100}, {"id": 101}])
            # A fresh watch has 0s of stability; a stale carried-over
            # `since`=100.0 would fire (1.2s >= 1.0s) right here.
            self.assertFalse(r._rejoin_requested.is_set())

    def test_redeparture_earns_a_fresh_attempt(self):
        # _try_shrink clears the wid from _rejoin_attempted: one attempt PER
        # DEPARTURE, not per window.
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            r._respawn_watchdog = lambda workers, kb: None
            r._rejoin_attempted = {101}
            real_moov = record.sck.has_moov
            record.sck.has_moov = lambda path: True
            try:
                r._try_shrink(101, 105.0, None, lambda st, **kw: None)
            finally:
                record.sck.has_moov = real_moov
            self.assertNotIn(101, r._rejoin_attempted)


class TryRejoin(_MoovPatched):
    """`_try_rejoin` -- drain-side eligibility with an injectable resolver
    (the scope's named pin)."""

    def _rec(self, td):
        r = _fleet_rec(td)
        gone = r._sck_workers.pop(1)
        r._departed = [record._DepartedChannel(gone, 95.0, rank=1)]
        r._exit_marks = [1]
        return r

    def test_manual_pick_always_wins(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            r._grow_pending = _pick(9)
            out = r._try_rejoin(101, [], [], None, lambda st, **kw: None,
                                resolver=lambda w: _pick(1))
            self.assertEqual(out, "defer")
            self.assertNotIn(101, r._rejoin_attempted)   # attempt not burned

    def test_one_attempt_marked_even_when_the_resolver_fails(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            out = r._try_rejoin(101, [], [], None, lambda st, **kw: None,
                                resolver=lambda w: None)
            self.assertEqual(out, "failed")
            self.assertIn(101, r._rejoin_attempted)      # chip-only now

    def test_already_recaptured_wid_is_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            back = _worker(td, 2)
            back.window = _pick(1)                        # same wid 101
            back.entry = back.window
            r._sck_workers.append(back)
            out = r._try_rejoin(101, [], [], None, lambda st, **kw: None,
                                resolver=lambda w: _pick(1))
            self.assertEqual(out, "skipped")

    def test_at_cap_skip_does_not_burn_the_attempt(self):
        # Decision 10's attempt is "resolve -> grow"; a cap skip does
        # neither, so the window keeps its attempt for when a later shrink
        # frees a slot (adversarially caught: burning here left a
        # still-on-screen window chip-only forever).
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            for i in (2, 3, 4):
                r._sck_workers.append(_worker(td, i))
            self.assertEqual(len(r._sck_workers), record._MAX_FLEET)
            out = r._try_rejoin(101, [], [], None, lambda st, **kw: None,
                                resolver=lambda w: _pick(1))
            self.assertEqual(out, "skipped")
            self.assertNotIn(101, r._rejoin_attempted)

    def test_rank_comes_from_the_most_recent_departure(self):
        # depart (rank 1) -> chip re-add -> depart again (rank 2): two
        # departure records for one wid; the returnee takes the LAST slot.
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            gone = r._sck_workers.pop(1)
            again = _worker(td, 2)
            again.window = _pick(1)
            again.entry = again.window
            r._departed = [record._DepartedChannel(gone, 95.0, rank=1),
                           record._DepartedChannel(again, 105.0, rank=2)]
            r._next_channel_idx = 3

            def fake_grow(entry, so, se, kb, notify):
                w = _worker(td, r._alloc_channel_index(), t0=110.0)
                w.window = entry
                w.entry = entry
                r._sck_workers.append(w)
                r._display_rank.setdefault(w.index, w.index)
                return True

            r._try_grow = fake_grow
            out = r._try_rejoin(101, [], [], None, lambda st, **kw: None,
                                resolver=lambda w: _pick(1))
            self.assertEqual(out, "joined")
            self.assertEqual(r._display_rank[r._sck_workers[-1].index], 2)

    def test_failed_auto_attempt_uses_honest_copy(self):
        # A failed AUTO attempt must not read as a failed USER action: the
        # wrapped notify rewrites _try_grow's manual grow_error copy
        # (M3.4's copy pass; adversarially flagged in the M3.3 verify).
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            notes = []
            r._try_grow = (lambda entry, so, se, kb, notify:
                           (notify("recording",
                                   grow_error="couldn't start the new "
                                              "window — still recording"),
                            False)[1])
            out = r._try_rejoin(101, [], [], None,
                                lambda st, **kw: notes.append((st, kw)),
                                resolver=lambda w: _pick(1))
            self.assertEqual(out, "failed")
            self.assertEqual(len(notes), 1)
            msg = notes[0][1]["grow_error"]
            self.assertIn("automatically re-add", msg)
            self.assertIn("App1", msg)
            self.assertNotIn("couldn't start the new window", msg)

    def test_success_rides_try_grow_and_inherits_the_rank(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._rec(td)
            calls = []

            def fake_grow(entry, so, se, kb, notify):
                w = _worker(td, r._alloc_channel_index(), t0=110.0)
                w.window = entry
                w.entry = entry
                r._sck_workers.append(w)
                r._display_rank.setdefault(w.index, w.index)
                calls.append(entry)
                return True

            r._try_grow = fake_grow
            out = r._try_rejoin(101, [], [], None, lambda st, **kw: None,
                                resolver=lambda w: _pick(1))
            self.assertEqual(out, "joined")
            self.assertEqual(len(calls), 1)
            new = r._sck_workers[-1]
            self.assertEqual(new.index, 2)                # fresh index/file
            self.assertEqual(r._display_rank[new.index], 1)   # slot kept


class DrainRejoin(_MoovPatched):
    def test_deferred_wid_is_dropped_unattempted_for_the_watch(self):
        # Deliberately NOT re-queued: merging back would make the run loop
        # a second `_rejoin_pending` writer racing the poller (adversarially
        # caught). Un-attempted, the watch re-fires it after a fresh ~1s
        # stability window instead.
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td)
            gone = r._sck_workers.pop(1)
            r._departed = [record._DepartedChannel(gone, 95.0)]
            r._grow_pending = _pick(9)                    # manual pick pending
            r._rejoin_pending = {101}
            r._rejoin_requested.set()
            r._drain_rejoin([], [], None, lambda st, **kw: None)
            self.assertIsNone(r._rejoin_pending)
            self.assertFalse(r._rejoin_requested.is_set())
            self.assertNotIn(101, r._rejoin_attempted)    # watch re-arms it


class RejoinSlotOrder(_MoovPatched):
    """Decision 11 end-to-end: the returnee keeps its SLOT in the per-scene
    channel order (3 originals so slot-order and index-order differ)."""

    def _load(self, r):
        import json
        with open(r.meta_path) as f:
            return json.load(f)

    def test_returnee_occupies_the_departed_slot(self):
        with tempfile.TemporaryDirectory() as td:
            r = _fleet_rec(td, n=3)
            for w in r._sck_workers:
                w.stat = {"appended": 900, "dup": 0}      # run past the rejoin
            gone = r._sck_workers.pop(1)                  # ch1 departs @105
            r._departed = [record._DepartedChannel(gone, 105.0, rank=1)]
            r._exit_marks = [1]
            back = _worker(td, 3, t0=110.0, frames=300)   # rejoins @110
            r._sck_workers.append(back)
            r._join_marks = [3]
            r._next_channel_idx = 4
            r._display_rank[3] = 1                        # inherited slot
            r._finalize_fleet_take(1440.0, 900.0, "quartz", 99.0)
            scenes = self._load(r)["capture_scenes"]
            self.assertEqual([len(s["channels"]) for s in scenes], [3, 2, 3])
            # The final scene lays out raw_3 in the MIDDLE slot -- where the
            # departed raw_1 sat -- not appended at the right edge.
            self.assertEqual([c["file"] for c in scenes[2]["channels"]],
                             ["raw_0.mov", "raw_3.mov", "raw_2.mov"])
            # Earlier scenes keep index order (ranks == indexes there).
            self.assertEqual([c["file"] for c in scenes[0]["channels"]],
                             ["raw_0.mov", "raw_1.mov", "raw_2.mov"])


if __name__ == "__main__":
    unittest.main()
