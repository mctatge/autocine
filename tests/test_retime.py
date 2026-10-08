"""Unit tests for the auto speed-up engine (autocine/retime.py).

Everything here is a pure-Python decision core -- no macOS permissions,
no video decoding, no ffmpeg subprocess. The end-to-end render integration
lives in test_render_speedup.py.
"""

import unittest

import numpy as np

from autocine import retime


class TimeMapIdentityTests(unittest.TestCase):
    """The empty-spans TimeMap must be an EXACT identity: bit-exact with
    the pre-feature render path (this is the off-switch contract)."""

    def test_no_spans_reports_identity(self):
        tm = retime.TimeMap([], duration=10.0)
        self.assertTrue(tm.identity)

    def test_no_spans_output_duration_equals_input(self):
        tm = retime.TimeMap([], duration=42.5)
        self.assertEqual(tm.output_duration, 42.5)

    def test_no_spans_warp_is_identity(self):
        tm = retime.TimeMap([], duration=10.0)
        t = np.linspace(0.0, 10.0, 101)
        # array-equal (not just close): the identity path returns a copy
        # of the input, no float ops
        self.assertTrue(np.array_equal(tm.warp(t), t))

    def test_no_spans_emission_is_arange(self):
        tm = retime.TimeMap([], duration=10.0)
        fps = 60.0
        frame_times = np.arange(600) / fps
        emit, out_ord, n = tm.emission(frame_times)
        self.assertTrue(emit.all())
        self.assertTrue(np.array_equal(out_ord, np.arange(600)))
        self.assertEqual(n, 600)

    def test_no_spans_slowness_is_all_ones(self):
        tm = retime.TimeMap([], duration=10.0)
        g = tm.slowness(np.linspace(0.0, 10.0, 501))
        self.assertTrue(np.all(g == 1.0))

    def test_ineligible_spans_reduce_to_identity(self):
        # rate <= 1 or end <= start: dropped -> identity
        for spans in ([{"start": 5, "end": 10, "rate": 1.0}],
                      [{"start": 5, "end": 5, "rate": 6}],
                      [{"start": 10, "end": 5, "rate": 6}]):
            tm = retime.TimeMap(spans, duration=20.0)
            self.assertTrue(tm.identity, spans)


class WarpMathTests(unittest.TestCase):
    """Closed-form warp math -- monotonicity, boundary values, ramp shrink."""

    def test_warp_matches_boundary_closed_form(self):
        # single 10s span [10, 20] at rate 6, ramp 0.5
        tm = retime.TimeMap([{"start": 10.0, "end": 20.0, "rate": 6.0}],
                            ramp=0.5, duration=60.0)
        r, d = 6.0, 0.5
        # τ(a) = a
        self.assertAlmostEqual(tm.warp(10.0), 10.0, places=9)
        # τ(a+d) - τ(a) = d*(1+1/r)/2
        self.assertAlmostEqual(tm.warp(10.5) - tm.warp(10.0),
                               d * (1.0 + 1.0 / r) / 2.0, places=9)
        # plateau: τ(b-d) - τ(a+d) = (b-2d)/r  = 9/6 = 1.5
        self.assertAlmostEqual(tm.warp(19.5) - tm.warp(10.5),
                               (10.0 - 2 * d) / r, places=9)
        # exit ramp is symmetric to entry
        self.assertAlmostEqual(tm.warp(20.0) - tm.warp(19.5),
                               d * (1.0 + 1.0 / r) / 2.0, places=9)
        # whole-span delta = d*(1+1/r) + (b-a-2d)/r
        span_delta = d * (1.0 + 1.0 / r) + (10.0 - 2 * d) / r
        self.assertAlmostEqual(tm.warp(20.0) - tm.warp(10.0), span_delta,
                               places=9)
        # τ(60) = 60 - 10 + delta  = 52.08333...
        self.assertAlmostEqual(tm.warp(60.0), 50.0 + span_delta, places=9)
        # output_duration matches the last-boundary warp
        self.assertAlmostEqual(tm.output_duration, tm.warp(60.0), places=9)

    def test_warp_matches_numeric_integral(self):
        """The closed-form τ must equal the numeric integral of slowness."""
        tm = retime.TimeMap(
            [{"start": 10.0, "end": 20.0, "rate": 6.0},
             {"start": 30.0, "end": 40.0, "rate": 4.0}],
            ramp=0.5, duration=50.0)
        t = np.linspace(0.0, 50.0, 200001)
        g = tm.slowness(t)
        tau_num = np.concatenate([[0.0], np.cumsum(0.5 * (g[:-1] + g[1:])) *
                                  (t[1] - t[0])])
        tau_form = tm.warp(t)
        # trapezoid error is O(dx^2); with 200k samples this is well under
        # 1e-4 across a 50s window with two 10s ramped spans
        self.assertLess(float(np.max(np.abs(tau_form - tau_num))), 1e-3)

    def test_warp_strictly_monotone(self):
        tm = retime.TimeMap(
            [{"start": 5.0, "end": 15.0, "rate": 8.0}],
            ramp=0.5, duration=20.0)
        t = np.linspace(0.0, 20.0, 20001)
        w = tm.warp(t)
        self.assertTrue(np.all(np.diff(w) > 0.0))

    def test_ramp_shrinks_for_short_span(self):
        # b - a = 0.6 < 2*ramp: d clipped to 0.3, no plateau -- pure ramp
        # in and out. Warp still monotone, no negative slowness anywhere.
        tm = retime.TimeMap(
            [{"start": 10.0, "end": 10.6, "rate": 4.0}],
            ramp=0.5, duration=15.0)
        t = np.linspace(9.5, 11.0, 5001)
        w = tm.warp(t)
        self.assertTrue(np.all(np.diff(w) > 0.0))
        g = tm.slowness(t)
        # peak speed reached: min(g) == 1/rate exactly (at ramp meeting point)
        self.assertAlmostEqual(g.min(), 1.0 / 4.0, places=6)

    def test_overlapping_spans_merge_max_rate_wins(self):
        tm = retime.TimeMap(
            [{"start": 5.0, "end": 12.0, "rate": 3.0},
             {"start": 10.0, "end": 15.0, "rate": 6.0}],
            ramp=0.5, duration=20.0)
        # After merge: one span [5, 15] at rate 6
        self.assertEqual(len(tm.spans), 1)
        a, b, r = tm.spans[0]
        self.assertEqual((a, b, r), (5.0, 15.0, 6.0))

    def test_spans_clip_to_duration(self):
        tm = retime.TimeMap(
            [{"start": 8.0, "end": 30.0, "rate": 6.0}],
            ramp=0.5, duration=20.0)
        self.assertEqual(tm.spans, [(8.0, 20.0, 6.0)])
        # output_duration is bounded: definitely less than input duration
        self.assertLess(tm.output_duration, 20.0)


class EmissionTests(unittest.TestCase):
    """Frame emission over the source-frame grid."""

    def test_plateau_emits_one_in_r(self):
        tm = retime.TimeMap([{"start": 10.0, "end": 20.0, "rate": 6.0}],
                            ramp=0.5, duration=60.0)
        fps = 60.0
        frame_times = np.arange(int(60 * fps)) / fps
        emit, out_ord, n = tm.emission(frame_times)
        # deep inside plateau (skip ramp region)
        mask = (frame_times >= 12.0) & (frame_times < 18.0)
        ratio = emit[mask].sum() / mask.sum()
        self.assertAlmostEqual(ratio, 1.0 / 6.0, places=2)

    def test_source_frame_never_emits_more_than_one_output(self):
        """Mutation pin: g <= 1 => cum step <= 1 => out_ord can only advance
        by at most 1 per source frame. This is what makes it safe to
        `cap.read()` once per emit frame."""
        tm = retime.TimeMap([{"start": 5.0, "end": 15.0, "rate": 6.0}],
                            ramp=0.5, duration=20.0)
        frame_times = np.arange(1200) / 60.0
        emit, out_ord, n = tm.emission(frame_times)
        diffs = np.diff(out_ord)
        self.assertTrue(np.all(diffs <= 1))
        self.assertTrue(np.all(diffs >= 0))

    def test_emit_count_matches_n_out_total_exactly(self):
        """A/V-sync mutation pin: emit.sum() == n_out_total, always.

        Prior implementation computed n_out_total from the discrete
        cumsum of slowness while segments_for_audio used the closed-form
        warp; the two could disagree by ~1 frame due to rectangle-vs-
        analytic quadrature error, and that off-by-one leaked into every
        A/V-retimed render as ~16ms of audio hangover.
        """
        # sample a bunch of rates/spans/durations to nail down the invariant
        cases = [
            ([{"start": 10.0, "end": 20.0, "rate": 6.0}], 60.0),
            ([{"start": 5.0, "end": 15.0, "rate": 8.0}], 20.0),
            ([{"start": 2.0, "end": 8.0, "rate": 3.5},
              {"start": 12.0, "end": 18.0, "rate": 6.0}], 30.0),
            ([{"start": 0.0, "end": 10.0, "rate": 4.0}], 10.0),  # spans-clip
        ]
        for spans, dur in cases:
            tm = retime.TimeMap(spans, ramp=0.5, duration=dur)
            fps = 60.0
            frame_times = np.arange(int(round(dur * fps))) / fps
            emit, out_ord, n = tm.emission(frame_times)
            self.assertEqual(int(emit.sum()), n,
                             "emit-count vs n_out_total mismatch for "
                             "spans={}, dur={}: {} != {}".format(
                                 spans, dur, int(emit.sum()), n))

    def test_audio_video_frame_counts_match(self):
        """A/V-sync mutation pin: n_out_total * (1/fps) equals
        segments_for_audio's total output duration -- so ffmpeg's audio
        graph and the emitted video have identical length in frames."""
        tm = retime.TimeMap([{"start": 10.0, "end": 20.0, "rate": 6.0}],
                            ramp=0.5, duration=60.0)
        fps = 60.0
        frame_times = np.arange(int(60 * fps)) / fps
        _emit, _out_ord, n = tm.emission(frame_times)
        segs = tm.segments_for_audio(0.0, 60.0)
        audio_out = sum((b - a) / r for (a, b, r) in segs)
        self.assertAlmostEqual(n / fps, audio_out, places=6)


class ActivityDetectionTests(unittest.TestCase):
    """activity_times, drag_spans, idle_candidates."""

    def test_activity_includes_all_event_kinds(self):
        acts = retime.activity_times(
            clicks_t=[1.0, 5.0], moves_t=[], moves_x=[], moves_y=[],
            ups_t=[2.0], keys_t=[3.0], scrolls_t=[4.0])
        self.assertTrue(np.array_equal(acts, np.array([1.0, 2.0, 3.0, 4.0, 5.0])))

    def test_move_speed_gate_drops_drift(self):
        # 60 frames at 0.1px/frame = 6 px/s -- drift, excluded by 90 px/s gate
        t = np.arange(60) / 60.0
        x = 100.0 + 0.1 * np.arange(60)
        y = np.full(60, 100.0)
        acts = retime.activity_times([], t, x, y, [], [], [], move_speed=90.0)
        self.assertEqual(acts.size, 0)

    def test_move_speed_gate_admits_intent(self):
        # 500 px in 1 sample = huge speed; both admitted
        t = np.array([0.0, 0.02])
        x = np.array([100.0, 600.0])
        y = np.array([100.0, 100.0])
        acts = retime.activity_times([], t, x, y, [], [], [], move_speed=90.0)
        # the second sample (fast) is included; first has no dt-neighbor so
        # is not evaluated
        self.assertEqual(acts.size, 1)
        self.assertAlmostEqual(acts[0], 0.02)

    def test_drag_pairs_click_with_first_up(self):
        # click at 1, up at 3 (2s drag); click at 5, up at 7 (2s drag)
        spans = retime.drag_spans([1.0, 5.0], [3.0, 7.0])
        self.assertEqual(spans, [(1.0, 3.0), (5.0, 7.0)])

    def test_drag_short_click_up_not_a_drag(self):
        # click at 1, up at 1.1 (0.1s -- just a click)
        spans = retime.drag_spans([1.0], [1.1])
        self.assertEqual(spans, [])

    def test_drag_stuck_button_capped(self):
        # 40s "drag" clamped to drag_cap=30
        spans = retime.drag_spans([1.0], [41.0], cap=30.0)
        self.assertEqual(spans, [(1.0, 31.0)])

    def test_idle_candidates_empty_events_whole_clip(self):
        idle = retime.idle_candidates([], [], duration=30.0,
                                      pad=0.5, min_idle=3.0)
        self.assertEqual(idle, [(0.0, 30.0)])

    def test_idle_candidates_no_span_if_too_short(self):
        idle = retime.idle_candidates([], [], duration=2.0,
                                      pad=0.5, min_idle=3.0)
        self.assertEqual(idle, [])

    def test_idle_candidates_padding_and_min_idle(self):
        # activity at 5, 25; duration 30, pad 0.8, min_idle 3
        # active windows: [4.2, 5.8], [24.2, 25.8]
        # idle: [0, 4.2] (4.2s ≥ 3 → keep), [5.8, 24.2] (18.4s → keep),
        #       [25.8, 30] (4.2s → keep)
        idle = retime.idle_candidates([5.0, 25.0], [], duration=30.0,
                                      pad=0.8, min_idle=3.0)
        self.assertEqual(len(idle), 3)
        self.assertAlmostEqual(idle[0][0], 0.0)
        self.assertAlmostEqual(idle[0][1], 4.2)
        self.assertAlmostEqual(idle[1][0], 5.8)

    def test_idle_candidates_drags_absorbed(self):
        # drag from 5..15 with click at 5: drag span is active wholesale, so
        # [5-0.5, 15+0.5] = [4.5, 15.5] is one merged active block.
        # With min_idle=6, the two flanking gaps [0, 4.5] and [15.5, 17] are
        # both too short to survive -- but the mid-drag "gap" doesn't even
        # appear as a candidate (drag was absorbed), which is the invariant.
        idle = retime.idle_candidates([5.0], [(5.0, 15.0)], duration=17.0,
                                      pad=0.5, min_idle=6.0)
        for (a, b) in idle:
            self.assertFalse(a < 10.0 < b,
                             "mid-drag idle gap: drag not absorbed")

    def test_idle_candidates_drag_no_split_across_drag_span(self):
        idle = retime.idle_candidates([5.0], [(5.0, 15.0)], duration=30.0,
                                      pad=0.5, min_idle=3.0)
        # active window: [4.5, 15.5]; gaps [0,4.5] and [15.5,30] both >= 3
        self.assertEqual(len(idle), 2)
        self.assertAlmostEqual(idle[0][1], 4.5)
        self.assertAlmostEqual(idle[1][0], 15.5)


class PlanSpeedSpansTests(unittest.TestCase):
    def test_silence_gate_intersects(self):
        # idle would be [4.5, 20] alone; silence only [10, 20] -> intersection
        # yields [10, 20] (>= min_idle) after re-filter
        activity = np.array([4.0])
        spans = retime.plan_speed_spans(activity, [], 20.0, rate=6.0,
                                        params={"pad": 0.5, "min_idle": 3.0},
                                        silence=[(10.0, 20.0)])
        self.assertEqual(len(spans), 1)
        self.assertAlmostEqual(spans[0]["start"], 10.0)
        self.assertAlmostEqual(spans[0]["end"], 20.0)

    def test_silence_none_means_gate_not_applied_engine_speeds(self):
        # PURE PLANNER contract: silence=None means "gate not applied"
        # (this is the code-path the render.py caller uses ONLY when the
        # session has no audio stream at all -- no probe was run, no
        # narration to protect). The failure-mode conservative-fallback
        # for a FAILED probe on a session WITH audio is handled in the
        # render integration layer (_plan_timemap), not here. See
        # test_render_speedup.SilenceGateFailsClosed for that pin.
        activity = np.array([4.0])
        spans = retime.plan_speed_spans(activity, [], 20.0, rate=6.0,
                                        params={"pad": 0.5, "min_idle": 3.0},
                                        silence=None)
        self.assertEqual(len(spans), 2)

    def test_empty_silence_gates_out_everything(self):
        # silence=[] means "no silence found" -- intersect is [] -- nothing
        # gets sped. This is the intended safe default when a session has
        # a mic track and the probe genuinely found continuous audio.
        activity = np.array([4.0])
        spans = retime.plan_speed_spans(activity, [], 20.0, rate=6.0,
                                        params={"pad": 0.5, "min_idle": 3.0},
                                        silence=[])
        self.assertEqual(spans, [])

    def test_off_override_subtracts_from_auto(self):
        activity = np.array([])
        spans = retime.plan_speed_spans(activity, [], 30.0, rate=6.0,
            params={"pad": 0.5, "min_idle": 3.0},
            overrides=[{"mode": "off", "start": 10.0, "end": 15.0}])
        # auto: [0, 30] whole clip. After subtracting off [10, 15]:
        # [0, 10] and [15, 30] both >= 3
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(spans[0]["end"], 10.0)
        self.assertAlmostEqual(spans[1]["start"], 15.0)

    def test_force_override_adds_span_with_its_own_rate(self):
        # dense activity, no idle candidates. Force span is the only auto
        # output (no merge with an overlapping auto span to force max-rate).
        activity = np.array([float(i) * 0.5 for i in range(20)])  # 0..9.5
        spans = retime.plan_speed_spans(activity, [], 10.0, rate=6.0,
            params={"pad": 0.5, "min_idle": 3.0},
            overrides=[{"mode": "force", "start": 2.0, "end": 8.0,
                        "rate": 4.0}])
        self.assertEqual(len(spans), 1)
        self.assertAlmostEqual(spans[0]["start"], 2.0)
        self.assertAlmostEqual(spans[0]["end"], 8.0)
        self.assertEqual(spans[0]["rate"], 4.0)

    def test_force_and_auto_overlap_merges_max_rate_wins(self):
        # Force at rate 4 overlapping auto (uses render rate 6) -> merged
        # with rate 6 (max-rate-wins is deliberate; a manually forced span
        # can only ACCELERATE what auto would have done, never slow it).
        activity = np.array([])
        spans = retime.plan_speed_spans(activity, [], 20.0, rate=6.0,
            params={"pad": 0.5, "min_idle": 3.0},
            overrides=[{"mode": "force", "start": 5.0, "end": 15.0,
                        "rate": 4.0}])
        # auto: [0, 20] at 6; merged with force [5, 15] at 4 -> [0, 20]@6
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["rate"], 6.0)

    def test_max_spans_cap(self):
        # 10 non-overlapping force spans (well-spaced so nothing merges),
        # no auto activity. Cap at 5 keeps the 5 longest.
        activity = np.array([])
        overrides = [{"mode": "force", "start": float(i * 100),
                      "end": float(i * 100 + (i + 1)), "rate": 6.0}
                     for i in range(10)]
        spans = retime.plan_speed_spans(activity, [], 10000.0, rate=6.0,
                                        # off any auto span with a huge
                                        # min_idle so only forces qualify
                                        params={"max_spans": 5, "pad": 0,
                                                "min_idle": 100000.0},
                                        overrides=overrides)
        self.assertEqual(len(spans), 5)
        # kept the 5 longest (last 5, since each is longer than the prior)
        starts = sorted(s["start"] for s in spans)
        self.assertEqual(starts, [500.0, 600.0, 700.0, 800.0, 900.0])


class MotionGateTests(unittest.TestCase):
    """Pure decision cores for the visual motion gate. The integration
    (subprocess ffmpeg + cv2 decode) is covered in test_render_speedup."""

    def _still(self, h=100, w=100, val=128):
        return np.full((h, w), val, dtype=np.uint8)

    def _noisy(self, h=100, w=100, seed=0):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, size=(h, w), dtype=np.uint8)

    def test_all_still_frames_all_static(self):
        frames = [self._still() for _ in range(5)]
        mask = retime.motion_mask(frames, thresh_frac=0.02, diff_thresh=10,
                                  menubar_frac=0.03)
        self.assertEqual(mask.size, 4)
        self.assertTrue(mask.all())

    def test_frame_full_change_not_static(self):
        frames = [self._still(val=0), self._still(val=200), self._still(val=0)]
        mask = retime.motion_mask(frames)
        self.assertEqual(mask.size, 2)
        self.assertFalse(mask.any())

    def test_menubar_flicker_excluded(self):
        """A tiny bright-changing strip at the top must NOT trip the gate
        (that's the macOS menubar clock)."""
        a = self._still()
        b = self._still()
        # flicker in the top 2% (menubar_frac=0.03 covers this)
        b[:2, :] = 255
        mask = retime.motion_mask([a, b], thresh_frac=0.02, diff_thresh=10,
                                  menubar_frac=0.03)
        self.assertTrue(mask.all(), "menubar-only change tripped the gate")

    def test_small_caret_blink_stays_static(self):
        """A 4x36-px caret blink is ~0.14% of a 100x100 frame -- well below
        the 2% threshold. Idle stretches with just a caret must compress."""
        a = self._still()
        b = self._still()
        b[40:44, 30:66] = 255
        mask = retime.motion_mask([a, b], thresh_frac=0.02, diff_thresh=10,
                                  menubar_frac=0.03)
        self.assertTrue(mask.all())

    def test_static_sub_spans_all_static(self):
        times = [0.0, 0.25, 0.5, 0.75, 1.0]
        mask = np.ones(4, dtype=bool)
        subs = retime.static_sub_spans(times, mask, min_span=0.5, edge_pad=0)
        self.assertEqual(subs, [(0.0, 1.0)])

    def test_static_sub_spans_split_at_motion(self):
        # 6 probe times, 5 pairs: [static, static, MOTION, static, static]
        times = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
        mask = np.array([True, True, False, True, True])
        subs = retime.static_sub_spans(times, mask, min_span=0.4, edge_pad=0)
        # first static run: pairs 0-1 -> times[0..2] = [0, 0.5]
        # second static run: pairs 3-4 -> times[3..5] = [0.75, 1.25]
        self.assertEqual(len(subs), 2)
        self.assertAlmostEqual(subs[0][0], 0.0)
        self.assertAlmostEqual(subs[0][1], 0.5)
        self.assertAlmostEqual(subs[1][0], 0.75)
        self.assertAlmostEqual(subs[1][1], 1.25)

    def test_static_sub_spans_min_span_drops_shortest(self):
        times = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
        mask = np.array([True, False, True, True, True])
        subs = retime.static_sub_spans(times, mask, min_span=0.5, edge_pad=0)
        # first run: pair 0 -> [0, 0.25], span 0.25 < 0.5 -> dropped
        # second run: pairs 2-4 -> [0.5, 1.25], span 0.75 >= 0.5 -> kept
        self.assertEqual(len(subs), 1)
        self.assertAlmostEqual(subs[0][0], 0.5)
        self.assertAlmostEqual(subs[0][1], 1.25)

    def test_static_sub_spans_edge_pad_shrinks(self):
        times = [0.0, 1.0]
        mask = np.array([True])
        subs = retime.static_sub_spans(times, mask, min_span=0.5, edge_pad=0.1)
        # [0, 1] shrunk to [0.1, 0.9], span 0.8 >= 0.5 -> kept
        self.assertEqual(len(subs), 1)
        self.assertAlmostEqual(subs[0][0], 0.1)
        self.assertAlmostEqual(subs[0][1], 0.9)

    def test_static_sub_spans_no_static_returns_empty(self):
        times = [0.0, 0.25, 0.5]
        mask = np.array([False, False])
        subs = retime.static_sub_spans(times, mask, min_span=0.1, edge_pad=0)
        self.assertEqual(subs, [])

    def test_plan_speed_spans_motion_static_hook_drops(self):
        """The motion_static hook can drop an entire span (playing-video
        gate rejects it wholesale)."""
        activity = np.array([])
        spans = retime.plan_speed_spans(
            activity, [], duration=20.0, rate=6.0,
            params={"pad": 0.5, "min_idle": 3.0},
            motion_static=lambda s: [])
        self.assertEqual(spans, [])

    def test_plan_speed_spans_motion_static_splits(self):
        """Motion in the middle of a candidate splits it into two shorter
        sub-spans; both survive min_idle."""
        activity = np.array([])
        spans = retime.plan_speed_spans(
            activity, [], duration=20.0, rate=6.0,
            params={"pad": 0.5, "min_idle": 3.0},
            motion_static=lambda s: [
                {"start": 0.0, "end": 8.0},
                {"start": 12.0, "end": 20.0},
            ])
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0]["rate"], 6.0)
        self.assertEqual(spans[1]["rate"], 6.0)

    def test_plan_speed_spans_force_ranges_bypass_motion_gate(self):
        """Force ranges bypass the motion gate -- author's explicit
        choice, e.g. compressing a download-progress-bar animation."""
        activity = np.array([])
        spans = retime.plan_speed_spans(
            activity, [], duration=20.0, rate=6.0,
            params={"pad": 0.5, "min_idle": 3.0},
            motion_static=lambda s: [],   # gate vetoes every auto span
            overrides=[{"mode": "force", "start": 5.0, "end": 15.0}])
        # Auto = [], but force adds [5, 15]
        self.assertEqual(len(spans), 1)
        self.assertAlmostEqual(spans[0]["start"], 5.0)
        self.assertEqual(spans[0]["rate"], 6.0)


class AudioSegmentsTests(unittest.TestCase):
    def test_atempo_chain_decomposition(self):
        self.assertEqual(retime.atempo_chain(1.0), [])
        self.assertEqual(retime.atempo_chain(2.0), [2.0])
        self.assertEqual(retime.atempo_chain(1.5), [1.5])
        self.assertEqual(retime.atempo_chain(4.8), [2.0, 2.0, 1.2])
        # rate 8 = 2 * 2 * 2
        chain = retime.atempo_chain(8.0)
        self.assertTrue(all(0.5 < r <= 2.0 for r in chain))
        prod = 1.0
        for r in chain:
            prod *= r
        self.assertAlmostEqual(prod, 8.0, places=6)

    def test_segments_for_audio_gapless_cover(self):
        tm = retime.TimeMap([{"start": 10.0, "end": 20.0, "rate": 6.0}],
                            ramp=0.5, duration=60.0)
        segs = tm.segments_for_audio(0.0, 60.0)
        # three segments: [0,10]@1, [10,20]@4.8, [20,60]@1
        self.assertEqual(len(segs), 3)
        # gapless
        for i in range(len(segs) - 1):
            self.assertAlmostEqual(segs[i][1], segs[i + 1][0])
        # covers full window
        self.assertAlmostEqual(segs[0][0], 0.0)
        self.assertAlmostEqual(segs[-1][1], 60.0)
        # rate_eff of the middle segment matches (b-a)/(tau(b)-tau(a))
        _a, _b, r = segs[1]
        self.assertAlmostEqual(r, 10.0 / (tm.warp(20.0) - tm.warp(10.0)),
                               places=6)

    def test_segments_reconstruct_output_duration(self):
        """A/V-sync mutation pin: Σ(b-a)/rate_eff over all segments
        equals TimeMap.output_duration (within 1e-6)."""
        tm = retime.TimeMap(
            [{"start": 10.0, "end": 20.0, "rate": 6.0},
             {"start": 40.0, "end": 50.0, "rate": 6.0}],
            ramp=0.5, duration=60.0)
        segs = tm.segments_for_audio(0.0, 60.0)
        recon = sum((b - a) / r for (a, b, r) in segs)
        self.assertAlmostEqual(recon, tm.output_duration, places=6)

    def test_segments_identity_returns_single_segment(self):
        tm = retime.TimeMap([], duration=20.0)
        segs = tm.segments_for_audio(2.0, 12.0)
        self.assertEqual(len(segs), 1)
        a, b, r = segs[0]
        self.assertAlmostEqual(a, 0.0)
        self.assertAlmostEqual(b, 10.0)
        self.assertEqual(r, 1.0)


class SilenceDetectParserTests(unittest.TestCase):
    def test_parses_paired_start_end(self):
        text = (
            "[silencedetect @ 0x1] silence_start: 0.5\n"
            "[silencedetect @ 0x1] silence_end: 5.2 | silence_duration: 4.7\n"
        )
        self.assertEqual(retime.parse_silencedetect(text, 30.0),
                         [(0.5, 5.2)])

    def test_trailing_unclosed_silence_closes_at_duration(self):
        text = "[silencedetect] silence_start: 15.3\n"
        self.assertEqual(retime.parse_silencedetect(text, 30.0),
                         [(15.3, 30.0)])

    def test_garbage_lines_skipped(self):
        text = "random line\n[silencedetect] silence_start: 1.0\n"
        self.assertEqual(retime.parse_silencedetect(text, 10.0),
                         [(1.0, 10.0)])

    def test_empty_input_empty_output(self):
        self.assertEqual(retime.parse_silencedetect("", 10.0), [])
        self.assertEqual(retime.parse_silencedetect(None, 10.0), [])


class SpanAlgebraTests(unittest.TestCase):
    def test_intersect_disjoint(self):
        self.assertEqual(retime.intersect_spans(
            [(0, 5)], [(6, 10)]), [])

    def test_intersect_full_overlap(self):
        self.assertEqual(retime.intersect_spans(
            [(0, 10)], [(2, 5)]), [(2, 5)])

    def test_intersect_multiple(self):
        A = [(0, 3), (5, 8), (10, 15)]
        B = [(2, 6), (12, 20)]
        self.assertEqual(retime.intersect_spans(A, B),
                         [(2, 3), (5, 6), (12, 15)])

    def test_subtract_middle(self):
        self.assertEqual(retime.subtract_spans([(0, 10)], [(3, 6)]),
                         [(0, 3), (6, 10)])

    def test_subtract_nothing(self):
        self.assertEqual(retime.subtract_spans([(0, 10)], []),
                         [(0, 10)])

    def test_subtract_full(self):
        self.assertEqual(retime.subtract_spans([(2, 8)], [(0, 20)]), [])

    def test_union_merges_overlaps(self):
        self.assertEqual(retime.union_spans([(0, 5), (3, 8), (10, 12)]),
                         [(0, 8), (10, 12)])


class CutMapTests(unittest.TestCase):
    """Cuts (ripple delete) as a zero-tau piece kind in TimeMap.

    The off-switch half of the contract: cuts=None / cuts=[] must leave
    every TimeMap surface array-equal with today. The feature half: a cut
    removes exactly its span from output time, emission, and the audio
    segment list -- and composes with speed spans (cut wins on overlap).
    """

    def test_empty_cuts_is_identity(self):
        for cuts in (None, [], ()):
            tm = retime.TimeMap([], duration=10.0, cuts=cuts)
            self.assertTrue(tm.identity)
            t = np.linspace(0.0, 10.0, 101)
            self.assertTrue(np.array_equal(tm.warp(t), t))
            self.assertEqual(tm.output_duration, 10.0)
            self.assertEqual(tm.cut_spans, [])

    def test_empty_cuts_speedup_map_unchanged(self):
        spans = [{"start": 2.0, "end": 6.0, "rate": 4.0}]
        base = retime.TimeMap(spans, duration=10.0)
        with_empty = retime.TimeMap(spans, duration=10.0, cuts=[])
        t = np.linspace(0.0, 10.0, 401)
        self.assertTrue(np.array_equal(base.warp(t), with_empty.warp(t)))
        self.assertEqual(base.spans, with_empty.spans)

    def test_cuts_alone_not_identity(self):
        tm = retime.TimeMap([], duration=10.0, cuts=[(2.0, 3.0)])
        self.assertFalse(tm.identity)
        self.assertEqual(tm.cut_spans, [(2.0, 3.0)])

    def test_warp_flat_across_cut(self):
        tm = retime.TimeMap([], duration=10.0, cuts=[(2.0, 5.0)])
        # before: identity; inside: flat at the seam; after: shifted
        self.assertAlmostEqual(tm.warp(1.0), 1.0)
        self.assertAlmostEqual(tm.warp(2.0), 2.0)
        self.assertAlmostEqual(tm.warp(3.7), 2.0)
        self.assertAlmostEqual(tm.warp(5.0), 2.0)
        self.assertAlmostEqual(tm.warp(8.0), 5.0)
        self.assertAlmostEqual(tm.output_duration, 7.0)

    def test_output_duration_shrinks_by_unioned_cut_length(self):
        # overlapping cuts union to [1, 4] + [6, 7] = 4s removed
        tm = retime.TimeMap([], duration=10.0,
                            cuts=[{"start": 1.0, "end": 3.0},
                                  {"start": 2.0, "end": 4.0},
                                  (6.0, 7.0)])
        self.assertEqual(tm.cut_spans, [(1.0, 4.0), (6.0, 7.0)])
        self.assertAlmostEqual(tm.output_duration, 6.0)

    def test_cuts_clip_to_duration(self):
        tm = retime.TimeMap([], duration=10.0, cuts=[(8.0, 99.0), (-5.0, 1.0)])
        self.assertEqual(tm.cut_spans, [(0.0, 1.0), (8.0, 10.0)])
        self.assertAlmostEqual(tm.output_duration, 7.0)

    def test_emission_drops_exactly_covered_frames(self):
        fps = 30.0
        tm = retime.TimeMap([], duration=4.0, cuts=[(1.0, 2.0)])
        frame_times = np.arange(120) / fps
        emit, out_ord, n = tm.emission(frame_times)
        # frames 30..59 cover [1.0, 2.0): dropped; everything else emits
        self.assertTrue(emit[:30].all())
        self.assertFalse(emit[30:60].any())
        self.assertTrue(emit[60:].all())
        self.assertEqual(int(emit.sum()), 90)
        self.assertEqual(n, 90)
        # <= 1 emission per source frame, output ordinals dense
        kept = out_ord[emit]
        self.assertTrue(np.array_equal(kept, np.arange(90)))

    def test_head_and_tail_cuts_emit_correctly(self):
        fps = 30.0
        tm = retime.TimeMap([], duration=4.0, cuts=[(0.0, 1.0), (3.0, 4.0)])
        frame_times = np.arange(120) / fps
        emit, out_ord, n = tm.emission(frame_times)
        self.assertFalse(emit[:30].any())
        self.assertTrue(emit[30:90].all())
        self.assertFalse(emit[90:].any())
        self.assertEqual(n, 60)
        self.assertTrue(np.array_equal(out_ord[emit], np.arange(60)))

    def test_n_out_total_matches_analytic_duration(self):
        # NON-frame-aligned cut (as quantize_cut_spans would not produce,
        # but the map must stay self-consistent anyway)
        fps = 30.0
        tm = retime.TimeMap([], duration=4.0, cuts=[(0.51, 1.49)])
        frame_times = np.arange(120) / fps
        emit, _ord, n = tm.emission(frame_times)
        self.assertEqual(n, int(np.floor(fps * tm.output_duration + 1e-9)))
        self.assertEqual(int(emit.sum()), n)

    def test_segments_for_audio_omits_cut_ranges(self):
        tm = retime.TimeMap([], duration=60.0, cuts=[(10.0, 20.0)])
        segs = tm.segments_for_audio(0.0, 60.0)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0], (0.0, 10.0, 1.0))
        self.assertEqual(segs[1], (20.0, 60.0, 1.0))
        # kept lengths sum to the output duration (the A/V-sync pin)
        recon = sum((b - a) / r for (a, b, r) in segs)
        self.assertAlmostEqual(recon, tm.output_duration, places=6)

    def test_segments_for_audio_respects_trim_window(self):
        tm = retime.TimeMap([], duration=60.0, cuts=[(10.0, 20.0)])
        segs = tm.segments_for_audio(5.0, 30.0)
        # trimmed-local coordinates: [0,5] kept, [5,15] cut, [15,25] kept
        self.assertEqual(segs, [(0.0, 5.0, 1.0), (15.0, 25.0, 1.0)])

    def test_cut_wins_over_speedup(self):
        tm = retime.TimeMap([{"start": 10.0, "end": 20.0, "rate": 6.0}],
                            ramp=0.5, duration=60.0, cuts=[(12.0, 15.0)])
        self.assertEqual([(a, b) for (a, b, _r) in tm.spans],
                         [(10.0, 12.0), (15.0, 20.0)])
        self.assertEqual(tm.cut_spans, [(12.0, 15.0)])
        segs = tm.segments_for_audio(0.0, 60.0)
        # no audio segment overlaps the cut interior, every rate is finite
        for a, b, r in segs:
            self.assertTrue(np.isfinite(r))
            self.assertFalse(a < 15.0 - 1e-9 and b > 12.0 + 1e-9,
                             "audio segment {}..{} overlaps the cut".format(a, b))
        recon = sum((b - a) / r for (a, b, r) in segs)
        self.assertAlmostEqual(recon, tm.output_duration, places=5)

    def test_speedup_swallowed_by_cut_vanishes(self):
        tm = retime.TimeMap([{"start": 10.0, "end": 12.0, "rate": 6.0}],
                            duration=60.0, cuts=[(9.0, 13.0)])
        self.assertEqual(tm.spans, [])
        self.assertEqual(tm.cut_spans, [(9.0, 13.0)])
        self.assertAlmostEqual(tm.output_duration, 56.0)

    def test_keep_mask_half_open_seam(self):
        tm = retime.TimeMap([], duration=10.0, cuts=[(2.0, 5.0)])
        times = np.array([1.0, 2.0, 3.5, 4.999, 5.0, 6.0])
        keep = tm.keep_mask(times)
        # t == cut start DROPPED, t == cut end KEPT (earlier-segment seam)
        self.assertTrue(np.array_equal(
            keep, np.array([True, False, False, False, True, True])))

    def test_slowness_zero_inside_cut(self):
        tm = retime.TimeMap([], duration=10.0, cuts=[(2.0, 5.0)])
        g = tm.slowness(np.array([1.0, 2.0, 3.0, 5.0, 6.0]))
        self.assertTrue(np.array_equal(g, np.array([1.0, 0.0, 0.0, 1.0, 1.0])))

    def test_warp_spans_shrinks_straddling_range(self):
        tm = retime.TimeMap([], duration=10.0, cuts=[(2.0, 5.0)])
        spans = tm.warp_spans([{"start": 1.0, "end": 3.0, "id": "z1"},
                               {"start": 2.5, "end": 4.5, "id": "z2"},
                               {"start": 6.0, "end": 8.0, "id": "z3"}])
        # straddler: [1, 3] -> [1, 2] (end collapses to the seam)
        self.assertAlmostEqual(spans[0]["start"], 1.0)
        self.assertAlmostEqual(spans[0]["end"], 2.0)
        # wholly inside: collapses to zero length at the seam
        self.assertAlmostEqual(spans[1]["start"], 2.0)
        self.assertAlmostEqual(spans[1]["end"], 2.0)
        # after: rides the shift
        self.assertAlmostEqual(spans[2]["start"], 3.0)
        self.assertAlmostEqual(spans[2]["end"], 5.0)
        self.assertEqual(spans[1]["id"], "z2")

    def test_atempo_chain_rejects_non_finite(self):
        for bad in (float("inf"), float("-inf"), float("nan")):
            with self.assertRaises(ValueError):
                retime.atempo_chain(bad)


class QuantizeCutSpansTests(unittest.TestCase):
    def test_exact_frame_boundaries_unchanged(self):
        fps = 30.0
        out = retime.quantize_cut_spans([(30 / fps, 60 / fps)], fps)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0][0], 1.0)
        self.assertAlmostEqual(out[0][1], 2.0)

    def test_snaps_floor_start_ceil_end(self):
        fps = 30.0
        (a, b), = retime.quantize_cut_spans([(0.51, 0.52)], fps)
        self.assertAlmostEqual(a, 15 / fps)   # floor(15.3)
        self.assertAlmostEqual(b, 16 / fps)   # ceil(15.6)

    def test_merges_overlaps_and_snap_adjacency(self):
        fps = 30.0
        out = retime.quantize_cut_spans(
            [{"start": 1.0, "end": 2.01}, {"start": 2.02, "end": 3.0}], fps)
        # ceil(2.01*30)=61 and floor(2.02*30)=60 touch -> one merged range
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0][0], 1.0)
        self.assertAlmostEqual(out[0][1], 3.0)

    def test_clips_to_duration_and_drops_degenerate(self):
        fps = 30.0
        out = retime.quantize_cut_spans([(-1.0, 0.5), (9.9, 20.0), (5.0, 5.0)],
                                        fps, duration=10.0)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0][0], 0.0)
        self.assertAlmostEqual(out[0][1], 0.5)
        self.assertAlmostEqual(out[1][0], 9.9)
        self.assertAlmostEqual(out[1][1], 10.0)

    def test_matches_emission_frame_count(self):
        """The quantizer's promise: removed seconds * fps is EXACTLY the
        number of frames emission drops."""
        fps = 30.0
        cuts = retime.quantize_cut_spans([(0.51, 1.49), (2.2, 2.9)], fps,
                                         duration=4.0)
        removed = sum(b - a for (a, b) in cuts)
        tm = retime.TimeMap([], duration=4.0, cuts=cuts)
        frame_times = np.arange(120) / fps
        emit, _ord, n = tm.emission(frame_times)
        self.assertEqual(120 - int(emit.sum()),
                         int(round(removed * fps)))
        self.assertEqual(n, int(emit.sum()))


if __name__ == "__main__":
    unittest.main()
