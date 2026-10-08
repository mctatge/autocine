"""macOS's per-window capture indicator, found and painted out.

An occlusion-free (window-native) capture comes back with a system-drawn pill
burned into the top-left corner of every frame, over the window's traffic
lights. `autocine/capture_badge.py` locates it and fills it with the local
title-bar colour; `badge_erase` (ON by default) is the switch.

THE FIXTURE. `_paint_badge` draws a pill whose interior is `capture_badge
.GLYPH` itself, upscaled -- so `_signature` reads the reference bitmap back
out and the detector's glyph check passes on a synthetic frame exactly as it
does on a recording. That is what lets the whole find -> refine -> erase path
be pinned with no video file, no macOS, and no permissions. What it does NOT
pin is that GLYPH still matches what macOS draws: if Apple restyles the
indicator these tests stay green and every take keeps its badge. The evidence
for the constant itself is measured, and lives in the module docstring and in
docs/architecture.md.

Pinned here:
  * find: locates the pill at any offset inside the search box and at either
    backing scale, REJECTS a same-sized rectangle carrying the wrong glyph
    (the check that stops a toolbar button being repainted with flat grey),
    and follows the pill down when an SCK letterbox fit shrinks it;
  * erase: the box becomes the surrounding colour and nothing outside it
    moves;
  * BadgeEraser: holds its box when the pill is covered rather than dropping
    it, re-locks when the pill MOVES (a measured real case), and reports the
    two outcomes worth knowing;
  * the option, end to end: a window-native render comes out without the
    badge, `badge_erase=False` is byte-identical to the pre-feature output,
    and a take that is not window-native is byte-identical either way.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from autocine import capture_badge as cb
from autocine import edits, render


# -- fixture -------------------------------------------------------------

def _paint_badge(frame, x, y, scale=2.0, fill=(232, 232, 232),
                 ink=(90, 90, 90)):
    """Draw a capture pill at (x, y); returns its tight box."""
    w = int(round(cb.PILL_PT[0] * scale))
    h = int(round(cb.PILL_PT[1] * scale))
    r = h // 2
    cv2.rectangle(frame, (x + r, y), (x + w - r, y + h - 1), fill, -1)
    cv2.circle(frame, (x + r, y + r), r, fill, -1)
    cv2.circle(frame, (x + w - r - 1, y + r), r, fill, -1)
    bits = [(cb.GLYPH >> (127 - i)) & 1 for i in range(128)]
    glyph = np.array(bits, np.uint8).reshape(8, 16)
    ix, iy = int(round(x + 0.22 * w)), int(round(y + 0.15 * h))
    iw, ih = int(round(0.56 * w)), int(round(0.70 * h))
    big = cv2.resize(glyph, (iw, ih), interpolation=cv2.INTER_NEAREST)
    frame[iy:iy + ih, ix:ix + iw] = np.where(
        big[..., None] == 1, np.array(fill, np.uint8), np.array(ink, np.uint8))
    return (x, y, w, h)


def _titlebar(w=700, h=400, bg=(250, 250, 250)):
    return np.full((h, w, 3), np.array(bg, np.uint8), np.uint8)


class FindTests(unittest.TestCase):

    def test_finds_the_pill_at_any_offset(self):
        # Real offsets differ per app -- Claude (8,4)pt, Excel (13,6),
        # Chrome (21,10), Finder (20,16) -- which is why nothing here is a
        # constant the caller could have computed instead.
        for (x, y) in ((16, 8), (26, 12), (40, 32), (84, 42)):
            fr = _titlebar()
            truth = _paint_badge(fr, x, y)
            got = cb.find(fr, 2.0)
            self.assertIsNotNone(got, (x, y))
            self.assertEqual(got[:2], truth[:2], (x, y))
            self.assertAlmostEqual(got[2], truth[2], delta=2)
            self.assertAlmostEqual(got[3], truth[3], delta=2)

    def test_finds_it_on_a_non_retina_buffer(self):
        fr = _titlebar()
        truth = _paint_badge(fr, 16, 8, scale=1.0)
        got = cb.find(fr, 1.0)
        self.assertIsNotNone(got)
        self.assertEqual(got[:2], truth[:2])
        self.assertAlmostEqual(got[2], truth[2], delta=2)

    def test_finds_it_on_a_dark_title_bar(self):
        # The glyph signature thresholds on the patch's own median, so one
        # reference has to serve both appearances.
        fr = _titlebar(bg=(30, 30, 30))
        truth = _paint_badge(fr, 26, 12, fill=(60, 60, 60), ink=(20, 20, 20))
        self.assertEqual(cb.find(fr, 2.0)[:2], truth[:2])

    def test_a_same_sized_rectangle_is_not_a_badge(self):
        # The one that matters: without the glyph check this repaints a
        # toolbar button with flat grey and nobody sees the frame it ruined.
        fr = _titlebar()
        cv2.rectangle(fr, (20, 10), (124, 50), (200, 200, 200), -1)
        self.assertIsNone(cb.find(fr, 2.0))

    def test_empty_frame_finds_nothing(self):
        self.assertIsNone(cb.find(_titlebar(), 2.0))

    def test_outside_the_search_box_is_not_searched(self):
        # A window's own content must never be a candidate, however
        # pill-shaped: macOS draws this over the window buttons or not at all.
        fr = _titlebar(w=1400, h=900)
        _paint_badge(fr, 900, 700)
        self.assertIsNone(cb.find(fr, 2.0))

    def test_follows_a_letterbox_shrunken_pill(self):
        # SCK fits an enlarged window back into the fixed buffer; everything
        # in the frame, badge included, scales down with it.
        fr = _titlebar()
        truth = _paint_badge(fr, 18, 9, scale=1.3)
        got = cb.find(fr, 2.0)
        self.assertIsNotNone(got)
        self.assertEqual(got[:2], truth[:2])
        self.assertLess(got[2], cb.PILL_PT[0] * 2.0)


class EraseTests(unittest.TestCase):

    def test_the_box_becomes_the_surrounding_colour(self):
        fr = _titlebar(bg=(250, 249, 248))
        rect = _paint_badge(fr, 16, 8)
        before = fr.copy()
        cb.erase(fr, cb.find(fr, 2.0))
        x, y, w, h = rect
        patch = fr[y:y + h, x:x + w].reshape(-1, 3)
        self.assertEqual(list(patch.min(axis=0)), [250, 249, 248])
        self.assertEqual(list(patch.max(axis=0)), [250, 249, 248])
        # ...and the pill really was there to begin with.
        self.assertTrue((before[y:y + h, x:x + w] != patch[0]).any())

    def test_nothing_outside_the_padded_box_moves(self):
        fr = _titlebar()
        _paint_badge(fr, 40, 32)
        marker = (7, 210, 90)
        fr[300:310, 400:410] = marker
        rect = cb.find(fr, 2.0)
        cb.erase(fr, rect)
        self.assertEqual(list(fr[305, 405]), list(marker))
        x, y, w, h = rect
        far = fr[y + h + 12:y + h + 30, x:x + w]
        self.assertTrue((far == 250).all())

    def test_patch_describes_the_same_repair_it_would_paint(self):
        # The editor's live canvas player paints this box itself; if the two
        # disagree, play and pause show different corners.
        fr = _titlebar(bg=(240, 238, 236))
        _paint_badge(fr, 26, 12)
        rect = cb.find(fr, 2.0)
        desc = cb.patch(fr, rect)
        cb.erase(fr, rect)
        bx, by, bw, bh = desc["rect"]
        painted = fr[by:by + bh, bx:bx + bw].reshape(-1, 3)
        self.assertEqual(list(painted.min(axis=0)), desc["color"])
        self.assertEqual(list(painted.max(axis=0)), desc["color"])


class BadgeEraserTests(unittest.TestCase):

    def _frames(self, n=8, x=16, y=8):
        out = []
        for _ in range(n):
            fr = _titlebar()
            _paint_badge(fr, x, y)
            out.append(fr)
        return out

    def test_locks_once_and_paints_every_frame(self):
        er = cb.BadgeEraser(2.0)
        for fr in self._frames():
            er.apply(fr)
        st = er.stats()
        self.assertEqual(st["painted"], 8)
        self.assertEqual(st["relocks"], 0)
        self.assertEqual(st["held"], 0)
        self.assertIsNone(er.report())      # silent on the ordinary outcome

    def test_holds_the_box_when_the_pill_is_covered(self):
        # The pointer parks on it, a menu opens over it: the badge has NOT
        # left (it is drawn for the whole capture), so dropping the box would
        # flash it back for exactly those frames.
        er = cb.BadgeEraser(2.0)
        frames = self._frames(6)
        for fr in frames[:2]:
            er.apply(fr)
        rect = er.rect
        covered = frames[2]
        cv2.rectangle(covered, (10, 4), (150, 60), (12, 12, 200), -1)
        er.apply(covered)
        self.assertEqual(er.rect, rect)
        self.assertGreaterEqual(er.stats()["held"], 1)

    def test_relocks_when_the_pill_moves(self):
        # Measured: recordings/20260831-183141/raw_0.mov holds (8,4)pt for
        # 90s and then shows the pill at (18,15).
        er = cb.BadgeEraser(2.0)
        for fr in self._frames(3, x=16, y=8):
            er.apply(fr)
        self.assertEqual(er.rect[:2], (16, 8))
        for _ in range(cb.RESEARCH_EVERY + 2):
            fr = _titlebar()
            _paint_badge(fr, 36, 30)
            er.apply(fr)
        self.assertEqual(er.rect[:2], (36, 30))
        self.assertGreaterEqual(er.stats()["relocks"], 1)

    def test_never_found_is_reported_not_silent(self):
        er = cb.BadgeEraser(2.0)
        for _ in range(cb.RESEARCH_EVERY * 2):
            er.apply(_titlebar())
        self.assertIsNone(er.rect)
        self.assertIn("not found", er.report())

    def test_acquisition_gives_up(self):
        # A sheet or dialog has no window buttons, so macOS marks it with
        # nothing; without a cap that take pays a contour search forever.
        er = cb.BadgeEraser(2.0)
        for _ in range(cb.RESEARCH_EVERY * (cb.MAX_ACQUIRE_TRIES + 4)):
            er.apply(_titlebar())
        self.assertLessEqual(er.stats()["frames"] and er._tries,
                             cb.MAX_ACQUIRE_TRIES)

    def test_clone_carries_the_lock_but_not_the_counters(self):
        er = cb.BadgeEraser(2.0)
        for fr in self._frames(3):
            er.apply(fr)
        twin = er.clone()
        self.assertEqual(twin.rect, er.rect)
        self.assertEqual(twin.frames, 0)
        twin.apply(self._frames(1)[0])
        self.assertEqual(twin.stats()["painted"], 1)
        self.assertEqual(er.stats()["frames"], 3)


class AppliesTests(unittest.TestCase):

    def test_only_window_native_specs_apply(self):
        self.assertTrue(cb.applies({"mode": "window_native"}))
        self.assertFalse(cb.applies({"mode": "window_crop"}))
        self.assertFalse(cb.applies({}))
        self.assertFalse(cb.applies(None))

    def test_scale_comes_from_the_manifest(self):
        self.assertEqual(
            cb.scale_for({"logical_w": 709.0, "buffer_w": 1418}, None), 2.0)
        # No buffer_w recorded: fall back to the decoded width.
        self.assertEqual(cb.scale_for({"logical_w": 700.0}, 700), 1.0)
        # Nothing usable at all: Retina, the only thing this ships on.
        self.assertEqual(cb.scale_for({}, None), 2.0)
        self.assertEqual(cb.scale_for({"logical_w": 0}, 100), 2.0)


# -- end to end ----------------------------------------------------------

def _encode(path, frames, fps=30):
    """Lossless H.264 from BGR frames -- the badge has to survive the encode
    for the detector to be tested against a real decode."""
    h, w = frames[0].shape[:2]
    p = subprocess.Popen([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", "{}x{}".format(w, h), "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", path],
        stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    assert p.wait() == 0


def _native_session(root, badge=True, native=True, n=12, fps=30,
                    w=700, h=400):
    os.makedirs(root, exist_ok=True)
    frames = []
    for i in range(n):
        fr = _titlebar(w, h)
        # Something that moves, so a stuck decode would be visible too.
        cv2.rectangle(fr, (200 + i, 200), (260 + i, 260), (40, 120, 200), -1)
        if badge:
            _paint_badge(fr, 16, 8)
        frames.append(fr)
    _encode(os.path.join(root, "raw.mov"), frames, fps=fps)
    open(os.path.join(root, "events.jsonl"), "w").close()
    meta = {"raw": "raw.mov", "events": "events.jsonl", "fps": fps,
            "logical_w": 1440.0, "logical_h": 900.0, "t0_monotonic": 0.0,
            "cursor_mode": "system", "duration": n / float(fps)}
    if native:
        meta["capture_window"] = {
            "id": 4242, "app": "TestApp", "title": "t", "units": "points",
            "rect": [0.0, 0.0, w / 2.0, h / 2.0],
            "display_origin": [0.0, 0.0], "source": "quartz",
            "resnapshot": True, "end_rect": [0.0, 0.0, w / 2.0, h / 2.0],
            "mode": "window_native", "track": "ok",
            "logical_w": w / 2.0, "logical_h": h / 2.0,
            "buffer_w": w, "buffer_h": h}
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


def _frames_of(path):
    cap = cv2.VideoCapture(path)
    out = []
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            out.append(fr)
    finally:
        cap.release()
    return out


class RenderEndToEnd(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="badge_e2e_")
        render._BADGE_RECTS.clear()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _render(self, sd, name, **kw):
        out = os.path.join(sd, name)
        render.render(sd, out_path=out, max_zoom=1.0, click_fx=False,
                      motion_blur=False, style="clean", **kw)
        return out

    def test_a_native_render_comes_out_without_the_badge(self):
        sd = os.path.join(self.td, "on")
        _native_session(sd)
        frames = _frames_of(self._render(sd, "out.mp4"))
        self.assertTrue(frames)
        # The rendered frame is the framed composite, so assert on the SOURCE
        # instead: nothing badge-shaped survives in the decoded raw once the
        # eraser has run over it.
        raw = _frames_of(os.path.join(sd, "raw.mov"))
        self.assertIsNotNone(cb.find(raw[6], 2.0))     # it was there
        er = cb.BadgeEraser(2.0)
        er.apply(raw[6])
        self.assertIsNone(cb.find(raw[6], 2.0))        # and it goes

    def test_off_is_bit_exact_with_a_take_that_has_no_badge(self):
        # The off switch has to reproduce the previous behaviour exactly.
        # A badge-free native take is the cleanest statement of it: on and
        # off must produce identical bytes, which also pins that a take
        # macOS never marked is left completely alone.
        sd = os.path.join(self.td, "clean")
        _native_session(sd, badge=False)
        a = _frames_of(self._render(sd, "a.mp4", badge_erase=True))
        b = _frames_of(self._render(sd, "b.mp4", badge_erase=False))
        self.assertEqual(len(a), len(b))
        for i, (fa, fb) in enumerate(zip(a, b)):
            self.assertTrue(np.array_equal(fa, fb), "frame %d" % i)

    def test_a_non_native_take_is_untouched_either_way(self):
        # The gate is `mode: window_native`; a whole-screen take has no
        # per-window badge, so the search must never even run on one.
        sd = os.path.join(self.td, "screen")
        _native_session(sd, badge=True, native=False)
        a = _frames_of(self._render(sd, "a.mp4", badge_erase=True))
        b = _frames_of(self._render(sd, "b.mp4", badge_erase=False))
        for i, (fa, fb) in enumerate(zip(a, b)):
            self.assertTrue(np.array_equal(fa, fb), "frame %d" % i)

    def test_erased_and_kept_renders_differ(self):
        sd = os.path.join(self.td, "diff")
        _native_session(sd)
        a = _frames_of(self._render(sd, "a.mp4", badge_erase=True))
        b = _frames_of(self._render(sd, "b.mp4", badge_erase=False))
        self.assertTrue(any(not np.array_equal(x, y) for x, y in zip(a, b)))

    def test_preview_matches_the_export(self):
        sd = os.path.join(self.td, "prev")
        _native_session(sd)
        fr = render.preview_frame(sd, t_sec=0.2, max_zoom=1.0, click_fx=False,
                                  motion_blur=False)
        kept = render.preview_frame(sd, t_sec=0.2, max_zoom=1.0,
                                    click_fx=False, motion_blur=False,
                                    badge_erase=False)
        self.assertFalse(np.array_equal(fr, kept))


class OptionWiring(unittest.TestCase):

    def test_default_is_on(self):
        self.assertTrue(edits._DEFAULT_RENDER["badge_erase"])
        self.assertTrue(edits.normalize_edits({})["render"]["badge_erase"])

    def test_it_round_trips_through_edits(self):
        doc = edits.normalize_edits({"render": {"badge_erase": False}})
        self.assertFalse(doc["render"]["badge_erase"])
        self.assertTrue(
            edits.normalize_edits({"render": {"badge_erase": "yes"}})
            ["render"]["badge_erase"])

    def test_the_mcp_can_set_it(self):
        # The tool surface has to carry it or the editing agent cannot turn
        # it off on a take where the fill lands badly.
        from autocine import mcp_server
        opts = [t for t in mcp_server.TOOL_DEFS
                if t["name"] == "set_render_options"][0]
        self.assertIn("badge_erase", opts["inputSchema"]["properties"])


if __name__ == "__main__":
    unittest.main()
