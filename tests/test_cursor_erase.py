"""The cursor eraser (`edits.render.cursor_erase`, `autocine/eraser.py`).

Removes the macOS pointer that a `cursor_mode: "system"` take burns into the
pixels, by repainting each frame's pointer box with the pixels that were
genuinely visible there before the pointer arrived or after it left.

What the tests are actually pinning, in order of how much it would hurt to
get wrong:

  * **Off is byte-identical.** This stage sits in the one loop every export
    runs through, on a plain session, a window-captured one, and a session
    recorded with a synthetic cursor (where it must decline to run at all).
  * **On is EXACT where the recording supports it.** The synthetic sessions
    here are built pixel by pixel, so "the cursor is gone" is asserted as
    "the frame equals the background it was drawn on", not as a similarity
    score.
  * **It refuses to fabricate.** When the content under a parked pointer
    changes while it is hidden, the pixels never existed and no search finds
    them. The eraser has to notice and fall back -- and, the failure this
    caught on real footage, it must NOT confidently paint a stale scene over
    a live one.

`testsrc2` (the harness the other render tests use) is animated, so it is
useless for the middle group: there is no "the background" to compare
against. These build their own frames with numpy instead, which is also what
lets the decode be faked entirely for the unit-level tests.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from autocine import edits as ed
from autocine import eraser
from autocine import render

from tests.test_capture_window import _mk_session, _cw


# -- synthetic footage ---------------------------------------------------

W, H = 320, 200
FPS = 30
# The pointer, as a solid block with a bright rim: opaque, high contrast, and
# extending down-right of its hotspot exactly like the real arrow does.
CUR_W, CUR_H = 14, 20


def _background(seed=7):
    """A busy static background: 20px blocks of saturated colour.

    Deliberately edgy rather than smooth -- an eraser that quietly blurs
    would pass on a flat field, and the ring test has to survive real
    high-contrast borders without reading them as a scene change.
    """
    rng = np.random.RandomState(seed)
    blocks = rng.randint(0, 256, size=((H + 19) // 20, (W + 19) // 20, 3))
    return np.repeat(np.repeat(blocks, 20, axis=0), 20, axis=1
                     )[:H, :W].astype(np.uint8)


def _draw_cursor(frame, x, y):
    x, y = int(round(x)), int(round(y))
    x1, y1 = min(W, x + CUR_W), min(H, y + CUR_H)
    if x1 <= x or y1 <= y:
        return
    frame[y:y1, x:x1] = 20
    frame[y:min(H, y + 2), x:x1] = 240
    frame[y:y1, x:min(W, x + 2)] = 240


def _frames(positions, bg=None, mutate=None):
    """One frame per position; `mutate(frame, i)` edits the background first."""
    base = _background() if bg is None else bg
    out = []
    for i, (x, y) in enumerate(positions):
        fr = base.copy()
        if mutate is not None:
            mutate(fr, i)
        if x is not None:
            _draw_cursor(fr, x, y)
        out.append(fr)
    return out


class _FakeCapture(object):
    """cv2's `read()`/`grab()` over a list of arrays -- the seam that lets
    `eraser.plan` be driven with no video file and no permissions."""

    def __init__(self, frames, start=0):
        self.frames = frames
        self.i = int(start)

    def read(self):
        if self.i >= len(self.frames):
            return False, None
        fr = self.frames[self.i].copy()
        self.i += 1
        return True, fr

    def grab(self):
        if self.i >= len(self.frames):
            return False
        self.i += 1
        return True

    def release(self):
        pass


def _plan_and_apply(frames, positions, start=0, end=None, params=None):
    """Run the real three-pass pipeline over `frames`. Returns the erased
    frames plus the eraser (for its stats)."""
    n = len(frames)
    end = n if end is None else end
    times = np.arange(n) / float(FPS)
    ev_t = np.array([i / float(FPS) for i, p in enumerate(positions)
                     if p[0] is not None])
    ev_x = np.array([p[0] for p in positions if p[0] is not None], dtype=float)
    ev_y = np.array([p[1] for p in positions if p[0] is not None], dtype=float)
    bx, by, fx, fy, ring = eraser.box_padding(1.0, 1.0, params)
    boxes, hots, tracked = eraser.boxes_for_track(
        times, ev_t, ev_x, ev_y, W, H, bx, by, fx, fy, ring=ring)
    e = eraser.plan(_FakeCapture(frames), None, boxes, hots, tracked, FPS,
                    start_idx=start, end_idx=end, params=params)
    if e is None:
        return None, None
    out = []
    for i in range(start, end):
        fr = frames[i].copy()
        e.step(i)
        e.erase(fr, i)
        e.observe(fr, i)
        out.append(fr)
    return out, e


def _walk(x0, y0, dx, dy, n):
    return [(x0 + dx * i, y0 + dy * i) for i in range(n)]


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _events_for(positions, fps=FPS, scale=1.0):
    return [{"t": i / float(fps), "type": "move",
             "x": p[0] / scale, "y": p[1] / scale}
            for i, p in enumerate(positions) if p[0] is not None]


def _mk_pixel_session(root, frames, events, fps=FPS, logical_w=W,
                      logical_h=H, cursor_mode="system"):
    """A session whose raw.mov is EXACTLY these frames (lossless x264)."""
    os.makedirs(root, exist_ok=True)
    raw = os.path.join(root, "raw.mov")
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-video_size", "{}x{}".format(W, H), "-r", str(fps),
         "-i", "pipe:0", "-c:v", "libx264", "-qp", "0",
         "-pix_fmt", "yuv444p", raw], stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed writing the synthetic session")
    with open(os.path.join(root, "events.jsonl"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump({"raw": "raw.mov", "events": "events.jsonl", "fps": fps,
                   "logical_w": logical_w, "logical_h": logical_h,
                   "t0_monotonic": 0.0, "cursor_mode": cursor_mode,
                   "duration": len(frames) / float(fps)}, f)


# -- schema --------------------------------------------------------------


class CursorEraseSchema(unittest.TestCase):
    """It is a render OPTION (a look switch), not timeline content -- so it
    lives in the preset's render dict and rides every doc rebuild for free.
    That is the difference from `crop`, which needed hand-wiring into four
    separate allowlists."""

    def test_default_is_off(self):
        self.assertFalse(ed.default_edits()["render"]["cursor_erase"])
        self.assertFalse(ed.normalize_edits({})["render"]["cursor_erase"])

    def test_it_round_trips(self):
        doc = ed.normalize_edits({"render": {"cursor_erase": True}})
        self.assertTrue(doc["render"]["cursor_erase"])

    def test_junk_is_coerced_not_raised(self):
        for bad in ("yes", 3, [], {}, None):
            ed.normalize_edits({"render": {"cursor_erase": bad}})

    def test_merge_patches_it(self):
        m = ed.merge_edits(ed.default_edits(),
                           {"render": {"cursor_erase": True}}, duration=10.0)
        self.assertTrue(m["render"]["cursor_erase"])
        self.assertTrue(m["presets"][0]["render"]["cursor_erase"])

    def test_preset_moves_carry_it(self):
        m = ed.merge_edits(ed.default_edits(),
                           {"render": {"cursor_erase": True}}, duration=10.0)
        pid = m["presets"][0]["id"]
        self.assertTrue(
            ed.set_active_preset(m, pid, duration=10.0)["render"]["cursor_erase"])
        self.assertTrue(
            ed.duplicate_preset(m, duration=10.0)["render"]["cursor_erase"])


# -- geometry ------------------------------------------------------------


class RectMinus(unittest.TestCase):
    """`_rect_minus` is what keeps the per-frame cost proportional to the box
    instead of to the screen, so it has to be exactly right about coverage."""

    def _area(self, rects):
        return sum((r[2] - r[0]) * (r[3] - r[1]) for r in rects)

    def test_disjoint_returns_the_whole_rect(self):
        got = eraser._rect_minus((0, 0, 10, 10), (50, 50, 60, 60))
        self.assertEqual(got, [(0, 0, 10, 10)])

    def test_fully_covered_returns_nothing(self):
        self.assertEqual(eraser._rect_minus((2, 2, 8, 8), (0, 0, 10, 10)), [])

    def test_partial_overlap_conserves_area(self):
        c, p = (0, 0, 10, 10), (5, 5, 20, 20)
        self.assertEqual(self._area(eraser._rect_minus(c, p)), 100 - 25)

    def test_the_pieces_are_disjoint_and_exact(self):
        """Overlapping pieces would double-count a pixel into two runs."""
        c, p = (0, 0, 20, 20), (6, 8, 14, 12)
        mask = np.zeros((20, 20), dtype=int)
        for (x0, y0, x1, y1) in eraser._rect_minus(c, p):
            mask[y0:y1, x0:x1] += 1
        self.assertEqual(mask.max(), 1)
        self.assertEqual(int(mask.sum()), 400 - 8 * 4)
        self.assertEqual(int(mask[8:12, 6:14].sum()), 0)

    def test_an_empty_subject_is_empty(self):
        self.assertEqual(eraser._rect_minus((5, 5, 5, 9), (0, 0, 1, 1)), [])


class CursorTrackAndBoxes(unittest.TestCase):

    def _boxes(self, ev_t, ev_x, ev_y, n=6, back=4, fwd=6, ring=0.0):
        return eraser.boxes_for_track(
            np.arange(n) / float(FPS), np.asarray(ev_t, dtype=float),
            np.asarray(ev_x, dtype=float), np.asarray(ev_y, dtype=float),
            W, H, back, back, fwd, fwd, ring=ring)

    def test_the_track_is_a_step_function(self):
        """`move` lines are only written when the pointer MOVES, so the
        position between two of them is the earlier one held -- the pointer
        is still on screen while it is still."""
        t = np.arange(6) / float(FPS)
        x, y = eraser.cursor_track(t, [0.0, 4 / float(FPS)], [10, 90],
                                   [20, 80])
        self.assertEqual(list(x), [10, 10, 10, 10, 90, 90])
        self.assertEqual(list(y), [20, 20, 20, 20, 80, 80])

    def test_before_the_first_sample_the_first_position_is_held(self):
        t = np.arange(3) / float(FPS)
        x, _ = eraser.cursor_track(t, [10.0], [55], [55])
        self.assertEqual(list(x), [55, 55, 55])

    def test_an_empty_track_yields_no_boxes(self):
        b, h, tr = self._boxes([], [], [])
        self.assertEqual(b.shape, (0, 4))
        self.assertIsNone(eraser.region_of(b))

    def test_the_box_opens_in_every_direction_from_the_hotspot(self):
        """The arrow's hotspot is its TIP so it lies down-right, but an
        I-beam's is its CENTRE. A box that only opened forward would cut
        every centre-hotspot cursor in half."""
        b, _, _ = self._boxes([0.0], [100], [100], n=1, back=4, fwd=6)
        self.assertEqual(list(b[0]), [96, 96, 107, 107])

    def test_the_box_widens_over_events_inside_one_frame(self):
        """A 60fps frame is exposed while the pointer keeps moving, so the
        arrow really can be between two logged points. Pinning the box to the
        held position alone leaves a smear of arrow behind."""
        dt = 1.0 / FPS
        b, _, _ = self._boxes([0.0, dt * 0.3, dt * 0.6], [100, 140, 180],
                              [100, 100, 100], n=2)
        self.assertEqual(b[0][0], 96)          # back margin off the earliest
        self.assertEqual(b[0][2], 187)         # forward margin off the latest

    def test_the_hot_rect_is_tight_and_the_tracked_rect_is_wide(self):
        b, hot, tr = self._boxes([0.0], [100], [100], n=1, back=4, fwd=6,
                                 ring=5)
        self.assertTrue(hot[0][0] > b[0][0] and hot[0][2] < b[0][2])
        self.assertEqual(list(tr[0]), [91, 91, 112, 112])

    def test_a_box_off_the_captured_area_is_empty_not_degenerate(self):
        b, _, _ = self._boxes([0.0], [-500], [-500], n=1)
        self.assertEqual(list(b[0]), [0, 0, 0, 0])
        self.assertIsNone(eraser.region_of(b))

    def test_boxes_clamp_into_the_frame(self):
        b, _, _ = self._boxes([0.0], [W - 1], [H - 1], n=1, back=4, fwd=40)
        self.assertEqual((b[0][2], b[0][3]), (W, H))


# -- the algorithm, on frames we own ------------------------------------


class ErasesExactly(unittest.TestCase):
    """On a static background the recovery is not approximate: the pixels
    under the pointer were genuinely visible in other frames, so the repaired
    frame must equal the background bit for bit."""

    def test_a_moving_pointer_is_removed_exactly(self):
        pos = _walk(40, 40, 6, 3, 30)
        frames = _frames(pos)
        out, e = _plan_and_apply(frames, pos)
        bg = _background()
        # Frame 0's pixels were never seen uncovered (nothing precedes it),
        # so it is the one frame with no clean sample on either side yet.
        for i in range(2, len(out)):
            self.assertTrue((out[i] == bg).all(),
                            "frame {} still differs from the background at {} "
                            "pixels".format(i, int((out[i] != bg).any(axis=2).sum())))
        self.assertEqual(e.stats()["px_fallback_frac"], 0.0)

    def test_a_parked_pointer_is_removed_from_the_frames_it_sat_still_on(self):
        """The whole point of the two-sided search: while the pointer is
        parked there is no clean sample anywhere nearby in time, only one
        before it arrived and one after it leaves."""
        pos = _walk(40, 40, 8, 4, 6) + [(80, 60)] * 40 + _walk(88, 64, 8, 4, 6)
        frames = _frames(pos)
        out, e = _plan_and_apply(frames, pos)
        bg = _background()
        for i in range(10, 44):
            self.assertTrue((out[i] == bg).all(), "parked frame %d" % i)

    def test_the_forward_sample_alone_is_enough_at_the_end_of_a_clip(self):
        """A run that never closes has no `post`. It still has `pre`, and a
        pointer that arrives and stays put to the end must still come off."""
        pos = _walk(40, 40, 8, 4, 6) + [(80, 60)] * 10
        out, _ = _plan_and_apply(_frames(pos), pos)
        bg = _background()
        self.assertTrue((out[-1] == bg).all())

    def test_a_centre_hotspot_shape_comes_off_too(self):
        """The box has to open backwards or a crosshair/I-beam loses its top
        half -- pinned here by drawing the block CENTRED on the logged point
        instead of down-right of it."""
        pos = _walk(60, 60, 5, 5, 24)
        bg = _background()
        frames = []
        for (x, y) in pos:
            fr = bg.copy()
            _draw_cursor(fr, x - CUR_W // 2, y - CUR_H // 2)
            frames.append(fr)
        out, _ = _plan_and_apply(frames, pos)
        for i in range(2, len(out)):
            self.assertTrue((out[i] == bg).all(), "frame %d" % i)

    def test_a_trimmed_render_still_has_its_history(self):
        """The render starts at `start_idx`, but the clean pixels for a run
        that opened before it are further back. The prepass seeds them."""
        pos = _walk(40, 40, 6, 3, 10) + [(100, 70)] * 30
        frames = _frames(pos)
        out, _ = _plan_and_apply(frames, pos, start=25, end=40)
        bg = _background()
        self.assertTrue((out[0] == bg).all())


class RefusesToFabricate(unittest.TestCase):
    """The honest failure case: content that changed while it was hidden."""

    @staticmethod
    def _mutation(i):
        """Repaint a band under the parked pointer from frame 20 onward."""
        def _m(fr, idx):
            if idx >= 20:
                fr[50:110, 60:180] = 128
        return _m

    def test_it_falls_back_instead_of_pasting_a_stale_scene(self):
        pos = _walk(40, 40, 6, 3, 4) + [(70, 70)] * 40 + _walk(76, 74, 8, 6, 8)
        frames = _frames(pos, mutate=self._mutation(0))
        out, e = _plan_and_apply(frames, pos)
        self.assertGreater(e.stats()["px_fallback_frac"], 0.0,
                           "content that changed under a parked pointer must "
                           "register as unrecoverable")

    def test_it_does_not_paint_the_old_scene_over_the_new_one(self):
        """The failure this test exists for, seen on real footage: a pointer
        parked for 17 minutes bracketed a document that scrolled away and
        back, both samples agreed, and the eraser confidently painted the
        wrong paragraph over the right one. Agreement across a long gap is
        not proof, and the verification ring is what says so.

        Here the band under the pointer is grey from frame 20 to 45 and the
        original background on either side, so `pre` and `post` agree on
        something the middle frames never showed.
        """
        pos = _walk(30, 30, 6, 4, 4) + [(60, 60)] * 50 + _walk(66, 64, 8, 6, 8)

        def _m(fr, idx):
            if 20 <= idx < 45:
                fr[40:120, 40:200] = 128

        frames = _frames(pos, mutate=_m)
        out, e = _plan_and_apply(frames, pos)
        bg = _background()
        for i in (28, 34, 40):
            band = out[i][40:120, 40:200]
            # It may inpaint, and it may leave pixels alone. What it may NOT
            # do is put the ORIGINAL background back over a band the frame
            # showed as flat grey.
            restored = (np.abs(band.astype(int)
                               - bg[40:120, 40:200].astype(int)).max(axis=2)
                        < 8).mean()
            self.assertLess(restored, 0.25,
                            "frame {} was repainted with the pre/post scene "
                            "({:.0%} of the changed band matches the OLD "
                            "background)".format(i, restored))

    def test_it_leaves_a_frame_ALONE_rather_than_damaging_it(self):
        """The defect this test exists for, found by rendering a real 4-minute
        take end to end and looking at the output.

        On a frame over a live spreadsheet the eraser correctly concluded it
        could recover nothing -- the ring veto fired and NOT ONE pixel of the
        box was trusted -- and then inpainted 4668 px of it anyway, blurring
        three cell values into illegibility. It knew, and damaged the frame
        regardless. Across that take, 14.6% of frames came out with a
        footprint 3-6x the size of a real cursor.

        The cause is that "the frame differs from the plate here" means two
        different things, and only one of them is the cursor. Where the plate
        cannot be trusted the difference is just the plate being wrong, so the
        footprint may not grow through those pixels -- and when none of them
        can be trusted, the honest output is the frame UNTOUCHED. A pointer
        left on screen is visible and obvious; deleted spreadsheet cells are
        silent corruption.
        """
        # The pointer parks, and the entire neighbourhood is repainted while
        # it sits there -- so no sample, near or far, describes this frame.
        pos = _walk(30, 30, 6, 4, 4) + [(120, 100)] * 40

        def _m(fr, idx):
            if 12 <= idx < 34:
                fr[:] = _background(seed=99)

        frames = _frames(pos, mutate=_m)
        out, e = _plan_and_apply(frames, pos)
        mid = 22
        self.assertTrue((out[mid] == frames[mid]).all(),
                        "a frame the eraser cannot recover must come out "
                        "byte-identical, not blurred")
        self.assertGreater(e.stats()["frames_declined"], 0)
        self.assertIn("LEFT ALONE", e.report())

    def test_declining_is_not_the_default_answer(self):
        """The guard above must not be so eager that it stops erasing
        ordinary footage -- on a static background nothing is ever declined."""
        pos = _walk(40, 40, 6, 3, 30)
        _, e = _plan_and_apply(_frames(pos), pos)
        self.assertEqual(e.stats()["frames_declined"], 0)

    def test_the_report_names_the_fallback(self):
        pos = _walk(30, 30, 6, 4, 4) + [(60, 60)] * 40

        def _m(fr, idx):
            if idx >= 15:
                fr[40:120, 40:200] = 128

        _, e = _plan_and_apply(_frames(pos, mutate=_m), pos)
        self.assertIn("cursor erase", e.report())
        self.assertIn("fallback", e.report())


class PlanEdges(unittest.TestCase):

    def test_no_coverage_plans_nothing(self):
        pos = [(None, None)] * 8
        frames = _frames([(None, None)] * 8)
        out, e = _plan_and_apply(frames, pos)
        self.assertIsNone(e)

    def test_a_bounded_walk_sizes_its_plates_to_the_walk(self):
        """The editor's still walks a window, not the file. Sizing the plates
        to every box in the take would allocate the whole screen for one
        frame."""
        pos = _walk(10, 10, 8, 6, 30)
        frames = _frames(pos)
        times = np.arange(len(frames)) / float(FPS)
        ev_t = np.arange(len(pos)) / float(FPS)
        bx, by, fx, fy, ring = eraser.box_padding(1.0, 1.0, None)
        boxes, hots, tracked = eraser.boxes_for_track(
            times, ev_t, np.array([p[0] for p in pos], dtype=float),
            np.array([p[1] for p in pos], dtype=float), W, H,
            bx, by, fx, fy, ring=ring)
        wide = eraser.plan(_FakeCapture(frames), None, boxes, hots, tracked,
                           FPS, start_idx=0, end_idx=len(frames))
        narrow = eraser.plan(_FakeCapture(frames, start=18), None, boxes,
                             hots, tracked, FPS, start_idx=20, end_idx=21,
                             walk_start=18, walk_limit=24)
        def _area(r):
            return (r[2] - r[0]) * (r[3] - r[1])
        self.assertLess(_area(narrow.region), _area(wide.region) / 2.0)


# -- end to end ----------------------------------------------------------


class CursorEraseEndToEnd(unittest.TestCase):
    """render() / preview_frame() on a real (if synthetic) session."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.pos = _walk(40, 40, 5, 3, 45)
        cls.sd = os.path.join(cls.tmp, "sess")
        _mk_pixel_session(cls.sd, _frames(cls.pos), _events_for(cls.pos))
        # Same footage, recorded with the cursor hidden: nothing is burned in.
        cls.syn = os.path.join(cls.tmp, "syn")
        _mk_pixel_session(cls.syn, _frames(cls.pos), _events_for(cls.pos),
                          cursor_mode="synthetic")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _render(self, name, **kw):
        out = os.path.join(self.tmp, name)
        kw.setdefault("motion_blur", False)
        kw.setdefault("facecam", False)
        render.render(self.sd, out_path=out, **kw)
        return out

    def test_off_is_byte_identical(self):
        """The one invariant this whole feature is gated on: it sits in the
        loop every export runs through."""
        a = self._render("off_a.mp4")
        b = self._render("off_b.mp4", cursor_erase=False)
        self.assertEqual(_md5(a), _md5(b))

    def test_a_synthetic_cursor_session_declines_and_stays_identical(self):
        """`--cursor synthetic` switched ffmpeg's `-capture_cursor` off, so
        there is nothing in the pixels and the extra decode would buy
        nothing. Turning the option ON must therefore change no byte."""
        a = os.path.join(self.tmp, "syn_off.mp4")
        b = os.path.join(self.tmp, "syn_on.mp4")
        render.render(self.syn, out_path=a, motion_blur=False, facecam=False)
        render.render(self.syn, out_path=b, motion_blur=False, facecam=False,
                      cursor_erase=True)
        self.assertEqual(_md5(a), _md5(b))

    def test_on_changes_the_export(self):
        self.assertNotEqual(_md5(self._render("e_off.mp4")),
                            _md5(self._render("e_on.mp4", cursor_erase=True)))

    def test_the_exported_pixels_no_longer_hold_a_pointer(self):
        """Decoded back out of the mp4 and compared against the background it
        was drawn on. Not bit-exact -- the export re-encodes -- so the
        measure is how much of the pointer's own footprint still differs."""
        out = self._render("gone.mp4", cursor_erase=True, style="clean",
                           click_fx=False)
        bg = _background().astype(int)
        got = _decode(out)
        worst = 0.0
        for i in (20, 30, 40):
            x, y = self.pos[i]
            box = (slice(int(y), int(y) + CUR_H), slice(int(x), int(x) + CUR_W))
            d = np.abs(got[i][box].astype(int) - bg[box]).max(axis=2)
            worst = max(worst, float((d > 40).mean()))
        self.assertLess(worst, 0.05,
                        "the pointer's footprint still differs from the "
                        "background in {:.0%} of its pixels".format(worst))
        # ...and the same measure on the un-erased export, so the assertion
        # above is known to be measuring something.
        ref = _decode(self._render("still_there.mp4", click_fx=False))
        x, y = self.pos[30]
        box = (slice(int(y), int(y) + CUR_H), slice(int(x), int(x) + CUR_W))
        d = np.abs(ref[30][box].astype(int) - bg[box]).max(axis=2)
        self.assertGreater(float((d > 40).mean()), 0.5)

    def test_preview_frame_erases_too(self):
        """Paused preview is a server render of the same frame; if it did not
        erase, the editor would show a cursor the export does not have."""
        plain = render.preview_frame(self.sd, 30 / float(FPS),
                                     motion_blur=False, facecam=False)
        erased = render.preview_frame(self.sd, 30 / float(FPS),
                                      motion_blur=False, facecam=False,
                                      cursor_erase=True)
        self.assertEqual(plain.shape, erased.shape)
        self.assertGreater(np.abs(plain.astype(int)
                                  - erased.astype(int)).max(), 40)

    def test_it_composes_with_trim_and_speedup(self):
        """The speed-up plan drops frames, and a dropped frame is where the
        clean sample for the NEXT run start lives -- so the eraser has to be
        fed those frames rather than letting them be grabbed past."""
        out = self._render("retimed.mp4", cursor_erase=True, trim_start=0.2,
                           speedup=True, speedup_rate=3.0,
                           speedup_silence_gate=False,
                           speedup_motion_gate=False)
        self.assertTrue(os.path.getsize(out) > 0)


def _render_args(argv=()):
    """An argparse namespace off the REAL render flag definitions, so a flag
    renamed in cli.py fails here rather than silently passing."""
    import argparse
    from autocine import cli
    p = argparse.ArgumentParser()
    cli._add_render_opts(p)
    return p.parse_args(list(argv))


def _decode(path):
    import cv2
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


class CursorEraseWithCaptureWindow(unittest.TestCase):
    """A window-captured session: the eraser must run in the CROPPED source
    space, which is the space the event coordinates are mapped into. Off by
    one crop origin and every box lands somewhere else entirely."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.sd = os.path.join(cls.tmp, "cw")
        _mk_session(cls.sd, [{"t": 0.5, "type": "down", "x": 60, "y": 40}],
                    duration=2, fps=30, width=640, height=360,
                    logical_w=320, logical_h=180,
                    capture_window=_cw((20, 10, 100, 80)))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_off_is_byte_identical_here_too(self):
        a = os.path.join(self.tmp, "a.mp4")
        b = os.path.join(self.tmp, "b.mp4")
        render.render(self.sd, out_path=a, motion_blur=False, facecam=False)
        render.render(self.sd, out_path=b, motion_blur=False, facecam=False,
                      cursor_erase=False)
        self.assertEqual(_md5(a), _md5(b))

    def test_on_runs_without_raising_in_window_space(self):
        out = os.path.join(self.tmp, "on.mp4")
        render.render(self.sd, out_path=out, motion_blur=False, facecam=False,
                      cursor_erase=True)
        self.assertTrue(os.path.getsize(out) > 0)

    def test_it_composes_with_the_editor_crop(self):
        out = os.path.join(self.tmp, "both.mp4")
        render.render(self.sd, out_path=out, motion_blur=False, facecam=False,
                      cursor_erase=True,
                      crop_rect={"x": 10, "y": 20, "w": 120, "h": 100})
        self.assertTrue(os.path.getsize(out) > 0)


class EraseUnlocksTheSyntheticCursor(unittest.TestCase):
    """`cursor_erase` + `cursor_fx` -- the two opposites compose.

    The synthetic cursor has always enforced one rule, "never two cursors on
    screen", and it used to implement that rule as "the take must have been
    recorded with --cursor synthetic". The eraser satisfies the SAME rule a
    second way: it lifts the recorded pointer out of the source frame,
    upstream of the warp, so what `CursorFX` paints into has no pointer in it
    either. Setting both is therefore a retrofit -- a synthetic presentation
    cursor on footage that was not recorded for it.

    What is pinned here is that the relaxation is exactly that wide and no
    wider: `cursor_fx` ALONE on a system take still draws nothing, byte for
    byte, because that pointer really is still in the picture.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.pos = _walk(40, 40, 5, 3, 45)
        cls.sd = os.path.join(cls.tmp, "sess")
        _mk_pixel_session(cls.sd, _frames(cls.pos), _events_for(cls.pos))
        cls.syn = os.path.join(cls.tmp, "syn")
        _mk_pixel_session(cls.syn, _frames(cls.pos), _events_for(cls.pos),
                          cursor_mode="synthetic")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    IDX = 30

    def _still(self, session, **kw):
        kw.setdefault("motion_blur", False)
        kw.setdefault("facecam", False)
        kw.setdefault("click_fx", False)
        return render.preview_frame(session, self.IDX / float(FPS), **kw)

    def test_the_gate_is_a_truth_table(self):
        """One helper decides this in all four render paths, so its table is
        worth stating outright."""
        sysm, synm = {"cursor_mode": "system"}, {"cursor_mode": "synthetic"}
        draws = render._cursor_fx_draws
        self.assertFalse(draws(False, sysm, True), "effect off draws nothing")
        self.assertFalse(draws(False, synm, False))
        self.assertFalse(draws(True, sysm, False),
                         "the recorded pointer is still in those pixels")
        self.assertTrue(draws(True, sysm, True), "erased: the frame is free")
        self.assertTrue(draws(True, synm, False))
        self.assertTrue(draws(True, synm, True),
                        "the eraser is the no-op here, not the cursor")
        # No key at all means an older session, and record.py's default has
        # always been "system" -- the same reading `_erase_applies` uses.
        self.assertFalse(draws(True, {}, False))
        self.assertTrue(draws(True, {}, True))

    def test_the_fx_alone_still_draws_nothing(self):
        plain = self._still(self.sd)
        fx = self._still(self.sd, cursor_fx=True)
        self.assertTrue(np.array_equal(plain, fx),
                        "cursor_fx on a system take must stay a no-op "
                        "until the recorded pointer is gone")

    def test_erasing_lets_it_in(self):
        erased = self._still(self.sd, cursor_erase=True)
        both = self._still(self.sd, cursor_erase=True, cursor_fx=True)
        self.assertFalse(np.array_equal(erased, both),
                         "with the pointer erased, cursor_fx must draw")

    def test_what_it_draws_is_glued_to_the_recorded_position(self):
        """Not just "the frames differ" -- everything that changed has to sit
        in the pointer's own neighbourhood, mapped through the same camera
        transform the eraser worked in."""
        erased = self._still(self.sd, cursor_erase=True)
        both = self._still(self.sd, cursor_erase=True, cursor_fx=True)
        diff = np.abs(both.astype(int) - erased.astype(int)).max(axis=2)
        sx = both.shape[1] / float(W)
        sy = both.shape[0] / float(H)
        x, y = self.pos[self.IDX]
        # Generous margins on purpose -- this is a containment test, not a
        # pixel-alignment one. The glyph hangs down-right of its hotspot and
        # carries a drop shadow, and the DRAWN point trails the recorded one
        # by a few px because the position is EMA-smoothed (that lag is the
        # feature). Measured here: ~7px behind on a 5px/frame walk.
        x0 = int((x - 14) * sx); x1 = int((x + 14) * sx)
        y0 = int((y - 10) * sy); y1 = int((y + 20) * sy)
        outside = diff.copy()
        outside[y0:y1, x0:x1] = 0
        self.assertEqual(int(outside.max()), 0,
                         "the drawn cursor leaked outside its own box")
        self.assertGreater(int(diff[y0:y1, x0:x1].max()), 30,
                           "nothing was drawn inside it either")

    def test_a_synthetic_take_is_unchanged_by_the_relaxation(self):
        """`cursor_erase` is the no-op on a take with nothing burned in, so
        it must not become a way to change what the cursor looks like."""
        alone = self._still(self.syn, cursor_fx=True)
        with_erase = self._still(self.syn, cursor_fx=True, cursor_erase=True)
        self.assertTrue(np.array_equal(alone, with_erase))

    def test_the_export_says_when_it_skips(self):
        """A silent no-op was fine when nothing could be done about it. Now
        that --cursor-erase is the fix, the render has to name it."""
        import contextlib
        import io as _io
        buf = _io.StringIO()
        out = os.path.join(self.tmp, "skipped.mp4")
        with contextlib.redirect_stdout(buf):
            render.render(self.sd, out_path=out, motion_blur=False,
                          facecam=False, cursor_fx=True)
        self.assertIn("cursor-erase", buf.getvalue())
        # ...and it really was a no-op while it said so.
        plain = os.path.join(self.tmp, "skipped_ref.mp4")
        render.render(self.sd, out_path=plain, motion_blur=False,
                      facecam=False)
        self.assertEqual(_md5(out), _md5(plain))

    def test_the_export_draws_it_too(self):
        """preview_frame and render() are separate paths; the paused still
        agreeing with the export is the whole point of the gate being one
        helper."""
        a = os.path.join(self.tmp, "retro_off.mp4")
        b = os.path.join(self.tmp, "retro_on.mp4")
        for path, fx in ((a, False), (b, True)):
            render.render(self.sd, out_path=path, motion_blur=False,
                          facecam=False, click_fx=False, cursor_erase=True,
                          cursor_fx=fx)
        self.assertNotEqual(_md5(a), _md5(b))


class RenderOptionReachesEverySink(unittest.TestCase):
    """`docs/architecture.md`: an option resolved in one place and passed on by
    hand in four is exactly how `window_layout` shipped rendering two
    different videos from one edits.json."""

    def test_the_resolver_carries_it(self):
        from autocine import studio_app
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        kw = studio_app.StudioState(tmp)._render_kwargs_from_edits(
            ed.normalize_edits({"render": {"cursor_erase": True}}))
        self.assertTrue(kw["cursor_erase"])

    def test_the_live_options_patch_carries_it(self):
        """`_patch_from_options` is a FIFTH allowlist, between the editor's
        unsaved option dict and the render. Missing here, the paused still
        keeps showing the old answer until an autosave lands -- which is what
        a real click on the switch found."""
        from autocine import studio_app
        patch = studio_app._patch_from_options({"cursor_erase": True})
        self.assertTrue(patch["render"]["cursor_erase"])

    def test_render_and_preview_both_accept_it(self):
        import inspect
        for fn in (render.render, render.preview_frame):
            self.assertIn("cursor_erase",
                          inspect.signature(fn).parameters, fn.__name__)

    def test_the_cli_honours_a_saved_option(self):
        """A toggle turned on in the editor has to survive `studio.py
        render`, or one edits.json makes two different videos."""
        from autocine import cli
        args = _render_args()
        self.assertIsNone(args.cursor_erase)
        self.assertTrue(cli._render_kwargs(args, {"cursor_erase": True})
                        ["cursor_erase"])
        self.assertFalse(cli._render_kwargs(args, {})["cursor_erase"])

    def test_the_flag_wins_over_the_saved_option(self):
        from autocine import cli
        args = _render_args(["--cursor-erase"])
        self.assertTrue(cli._render_kwargs(args, {"cursor_erase": False})
                        ["cursor_erase"])

    def test_the_DRAWN_cursor_is_read_from_the_saved_edits_too(self):
        """On a system take the eraser and the synthetic cursor are a pair.
        Half a pair surviving `studio.py render` is the worst outcome of the
        three: the recorded pointer is lifted out and nothing replaces it, so
        the export has no cursor at all."""
        from autocine import cli
        args = _render_args()
        self.assertIsNone(args.cursor_fx)
        kw = cli._render_kwargs(args, {"cursor_erase": True, "cursor_fx": True,
                                       "cursor_size": 1.6})
        self.assertTrue(kw["cursor_fx"])
        self.assertEqual(kw["cursor_params"], {"scale": 1.6})
        plain = cli._render_kwargs(args, {})
        self.assertFalse(plain["cursor_fx"])
        self.assertIsNone(plain["cursor_params"],
                          "no flag and no saved size must leave "
                          "effects.CURSOR_DEFAULTS untouched")

    def test_the_cursor_flags_still_win_over_the_saved_ones(self):
        from autocine import cli
        args = _render_args(["--cursor-fx", "--cursor-size", "2.5"])
        kw = cli._render_kwargs(args, {"cursor_fx": False, "cursor_size": 1.0})
        self.assertTrue(kw["cursor_fx"])
        self.assertEqual(kw["cursor_params"], {"scale": 2.5})


if __name__ == "__main__":
    unittest.main()
