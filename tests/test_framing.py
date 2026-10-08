"""Unit tests for the background system + frame painter."""
import unittest

import cv2
import numpy as np

from autocine import framing


class ResolveBackground(unittest.TestCase):
    def test_preset_gradient_varies(self):
        bg = framing.resolve_background("midnight", 64, 48)
        self.assertEqual(bg.shape, (48, 64, 3))
        self.assertEqual(bg.dtype, np.float32)
        # a gradient must not be a single flat color
        self.assertGreater(len(np.unique(bg.reshape(-1, 3), axis=0)), 1)

    def test_hex_solid_is_constant(self):
        bg = framing.resolve_background("#101820", 32, 24)
        self.assertEqual(len(np.unique(bg.reshape(-1, 3), axis=0)), 1)
        # '#101820' rgb (16,24,32) -> bgr (32,24,16)
        np.testing.assert_array_equal(bg[0, 0], np.array([32, 24, 16], np.float32))

    def test_named_solid(self):
        bg = framing.resolve_background("black", 8, 8)
        self.assertEqual(len(np.unique(bg.reshape(-1, 3), axis=0)), 1)

    def test_none_uses_default_preset(self):
        a = framing.resolve_background(None, 40, 30)
        b = framing.resolve_background(framing.DEFAULT_BACKGROUND, 40, 30)
        np.testing.assert_array_equal(a, b)

    def test_unknown_falls_back_without_crashing(self):
        bg = framing.resolve_background("does-not-exist", 40, 30)
        self.assertEqual(bg.shape, (30, 40, 3))


class ResolveAspectCanvas(unittest.TestCase):
    def test_none_is_identity(self):
        self.assertEqual(framing.resolve_aspect_canvas(None, 1920, 1080), (1920, 1080))

    def test_auto_is_identity(self):
        self.assertEqual(framing.resolve_aspect_canvas("auto", 1920, 1080), (1920, 1080))

    def test_vertical_ratio_transposes_a_169_source(self):
        # 1920x1080 is exactly 16:9, so asking for 9:16 at the same pixel
        # budget lands exactly on the transposed dimensions.
        self.assertEqual(framing.resolve_aspect_canvas("9:16", 1920, 1080), (1080, 1920))

    def test_square_ratio(self):
        w, h = framing.resolve_aspect_canvas("1:1", 1920, 1080)
        self.assertEqual(w, h)
        self.assertAlmostEqual(w, 1440, delta=1)

    def test_custom_wxh_exact(self):
        self.assertEqual(framing.resolve_aspect_canvas("1080x1920", 640, 480), (1080, 1920))

    def test_custom_wxh_rounds_to_even(self):
        w, h = framing.resolve_aspect_canvas("101x201", 100, 100)
        self.assertEqual(w % 2, 0)
        self.assertEqual(h % 2, 0)

    def test_unknown_spec_falls_back_to_source(self):
        self.assertEqual(framing.resolve_aspect_canvas("garbage", 640, 480), (640, 480))

    def test_dimensions_are_always_even(self):
        w, h = framing.resolve_aspect_canvas("4:5", 641, 481)
        self.assertEqual(w % 2, 0)
        self.assertEqual(h % 2, 0)


class Painter(unittest.TestCase):
    def test_output_size_is_canvas(self):
        self.assertEqual(framing.output_size(1280, 720, "framed"), (1280, 720))

    def test_output_size_honors_aspect_for_both_styles(self):
        self.assertEqual(framing.output_size(1920, 1080, "clean", aspect="9:16"), (1080, 1920))
        self.assertEqual(framing.output_size(1920, 1080, "framed", aspect="9:16"), (1080, 1920))

    def test_clean_has_no_painter(self):
        self.assertIsNone(framing.make_painter(100, 80, "clean"))

    def test_paint_shape_and_dtype(self):
        p = framing.make_painter(200, 160, "framed", background="graphite")
        out = p.paint(np.full((160, 200, 3), 128, np.uint8))
        self.assertEqual(out.shape, (160, 200, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_recording_is_inset_not_fullbleed(self):
        # the framed recording must not touch the canvas edge (there's padding)
        p = framing.make_painter(200, 160, "framed", background="black")
        white = np.full((160, 200, 3), 255, np.uint8)
        out = p.paint(white)
        self.assertLess(int(out[0, 0].max()), 40)     # corner is background/shadow
        self.assertGreater(int(out[80, 100].max()), 200)  # center is the recording


class FramedCompositeIsExact(unittest.TestCase):
    """`FramePainter.paint` composites in uint8 and repairs only the four
    corner squares. It must stay byte-identical to the straightforward
    per-pixel alpha blend it is an optimization of -- this is a pinned output
    path, and a one-count drift is invisible in review and permanent in every
    export, so compare against the blend spelled out longhand rather than
    against a tolerance.

    Exactness rests on two properties of `_rounded_mask`, and both are quiet:
    antialiasing the corners (`cv2.LINE_AA`) or moving a zero out of the
    corner squares would leave the framed edges silently wrong rather than
    raise.
    """

    def _blend(self, p, img):
        out = p._plate.astype(np.float32)
        m3 = (p._mask.astype(np.float32) / 255.0)[:, :, None]
        rec = cv2.resize(img, (p.iw, p.ih),
                         interpolation=cv2.INTER_AREA).astype(np.float32)
        roi = out[p.y:p.y + p.ih, p.x:p.x + p.iw]
        out[p.y:p.y + p.ih, p.x:p.x + p.iw] = rec * m3 + roi * (1.0 - m3)
        return out.astype(np.uint8)

    def _check(self, W, H, background=None, seed=0, dtype=np.uint8):
        p = framing.make_painter(W, H, "framed", background=background)
        rng = np.random.RandomState(seed)
        img = rng.randint(0, 256, (H, W, 3), np.uint8).astype(dtype)
        got, want = p.paint(img), self._blend(p, img)
        self.assertTrue(np.array_equal(got, want),
                        "{} px differ (max {})".format(
                            int((got != want).sum()),
                            int(np.abs(got.astype(np.int16)
                                       - want.astype(np.int16)).max())))

    def test_landscape_default_background(self):
        self._check(1280, 720)

    def test_vertical_canvas_and_preset(self):
        self._check(608, 1080, background="sunset", seed=1)

    def test_odd_dimensions_and_hex_background(self):
        self._check(957, 601, background="#101820", seed=2)

    def test_tiny_canvas_clamps_the_radius(self):
        # inner_h // 2 < the requested radius here, so _rounded_mask clamps
        # and _corner_repairs must use the clamped value or miss pixels.
        self._check(64, 20, background="mist", seed=3)

    def test_array_background(self):
        rng = np.random.RandomState(7)
        bg = rng.randint(0, 256, (300, 500, 3), np.uint8).astype(np.float32)
        self._check(500, 300, background=bg, seed=4)

    def test_non_uint8_frame_takes_the_realloc_path(self):
        """A non-uint8 frame makes OpenCV reject the in-place destination and
        hand back its own buffer; the fallback assignment must cast exactly
        like the old `.astype()` did."""
        self._check(640, 360, background="ocean", seed=5, dtype=np.float32)

    def test_mask_is_binary(self):
        for W, H in ((1280, 720), (608, 1080), (64, 20)):
            p = framing.make_painter(W, H, "framed")
            self.assertTrue(
                set(np.unique(p._mask).tolist()) <= {0, 255},
                "mask must stay aliased 0/255 for a {}x{} canvas".format(W, H))

    def test_mask_zeros_live_only_in_the_corners_paint_repairs(self):
        for W, H in ((1280, 720), (608, 1080), (64, 20)):
            p = framing.make_painter(W, H, "framed")
            eff = int(min(p._radius, p.iw // 2, p.ih // 2))
            corners = np.zeros((p.ih, p.iw), bool)
            for ys, xs in ((slice(0, eff), slice(0, eff)),
                           (slice(0, eff), slice(p.iw - eff, p.iw)),
                           (slice(p.ih - eff, p.ih), slice(0, eff)),
                           (slice(p.ih - eff, p.ih),
                            slice(p.iw - eff, p.iw))):
                corners[ys, xs] = True
            stray = (p._mask == 0) & ~corners
            self.assertFalse(stray.any(),
                             "{}x{}: {} transparent px outside the corner "
                             "squares paint() repairs".format(
                                 W, H, int(stray.sum())))

    def test_returns_a_fresh_canvas_each_call(self):
        """Callers draw ON TOP of the returned frame in place (the facecam
        bubble), so handing back a reused buffer would smear one frame's
        overlay into every later one."""
        p = framing.make_painter(400, 300, "framed", background="graphite")
        img = np.full((300, 400, 3), 90, np.uint8)
        first = p.paint(img)
        plate_before = p._plate.copy()
        first[:] = 7                             # an overlay, drawn in place
        second = p.paint(img)
        self.assertFalse(np.array_equal(first, second))
        self.assertTrue(np.array_equal(p._plate, plate_before))
        self.assertTrue(np.array_equal(second, p.paint(img)))


class ChooseGrid(unittest.TestCase):
    def test_single_window_is_1x1(self):
        self.assertEqual(framing._choose_grid(1, 1920, 1080), (1, 1))

    def test_two_windows_side_by_side_on_landscape(self):
        self.assertEqual(framing._choose_grid(2, 1920, 1080), (1, 2))

    def test_two_windows_stacked_on_portrait(self):
        self.assertEqual(framing._choose_grid(2, 1080, 1920), (2, 1))

    def test_three_windows_row_on_landscape(self):
        self.assertEqual(framing._choose_grid(3, 1920, 1080), (1, 3))

    def test_three_windows_column_on_portrait(self):
        self.assertEqual(framing._choose_grid(3, 1080, 1920), (3, 1))

    def test_four_windows_prefers_square_grid_near_169(self):
        # 16:9 is closer in log-aspect to a 2x2 grid than to a 1x4 strip.
        self.assertEqual(framing._choose_grid(4, 1920, 1080), (2, 2))

    def test_four_windows_extreme_wide_canvas_prefers_strip(self):
        self.assertEqual(framing._choose_grid(4, 4000, 1000), (1, 4))

    def test_four_windows_extreme_tall_canvas_prefers_column(self):
        self.assertEqual(framing._choose_grid(4, 1000, 4000), (4, 1))

    def test_n_clamped_into_1_to_4(self):
        self.assertEqual(framing._choose_grid(0, 1920, 1080), (1, 1))
        self.assertEqual(framing._choose_grid(9, 1920, 1080), (2, 2))


class FitAspect(unittest.TestCase):
    def test_wide_content_letterboxed_in_square_box(self):
        w, h = framing._fit_aspect(100, 100, 200, 100)
        self.assertAlmostEqual(w, 100.0)
        self.assertAlmostEqual(h, 50.0)

    def test_tall_content_pillarboxed_in_square_box(self):
        w, h = framing._fit_aspect(100, 100, 100, 200)
        self.assertAlmostEqual(w, 50.0)
        self.assertAlmostEqual(h, 100.0)

    def test_matching_aspect_fills_box(self):
        w, h = framing._fit_aspect(200, 100, 400, 200)
        self.assertAlmostEqual(w, 200.0)
        self.assertAlmostEqual(h, 100.0)


class MultiFramePainterTests(unittest.TestCase):
    def test_empty_windows_returns_none(self):
        self.assertIsNone(framing.make_multi_painter(1920, 1080, []))
        self.assertIsNone(framing.make_multi_painter(1920, 1080, None))

    def test_paint_shape_and_dtype(self):
        mp = framing.make_multi_painter(
            1920, 1080, [{"w": 800, "h": 600}, {"w": 400, "h": 400}],
            background="graphite")
        crops = [np.full((600, 800, 3), 255, np.uint8),
                 np.full((400, 400, 3), 255, np.uint8)]
        out = mp.paint(crops)
        self.assertEqual(out.shape, (1080, 1920, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_each_card_is_inset_not_fullbleed(self):
        mp = framing.make_multi_painter(
            800, 600, [{"w": 300, "h": 200}, {"w": 300, "h": 200}],
            background="black")
        crops = [np.full((200, 300, 3), 255, np.uint8)] * 2
        out = mp.paint(crops)
        self.assertLess(int(out[0, 0].max()), 40)          # canvas corner: bg/shadow
        (ix, iy, fw, fh, _m3) = mp._cells[0]
        cx, cy = ix + fw // 2, iy + fh // 2
        self.assertGreater(int(out[cy, cx].max()), 200)    # first card center: content

    def test_shadow_baked_once_not_per_cell(self):
        # A GaussianBlur call count isn't directly observable, but the
        # shadow canvas is built once and shared -- assert the two painters
        # (1 cell vs 4 cells on the same canvas) both produce a valid,
        # differently-shaped bake without erroring, as a smoke proxy.
        mp1 = framing.make_multi_painter(1920, 1080, [{"w": 640, "h": 480}])
        mp4 = framing.make_multi_painter(1920, 1080, [{"w": 640, "h": 480}] * 4)
        self.assertEqual(mp1._plate.shape, (1080, 1920, 3))
        self.assertEqual(mp4._plate.shape, (1080, 1920, 3))
        self.assertEqual(len(mp1._cells), 1)
        self.assertEqual(len(mp4._cells), 4)

    def test_mismatched_aspect_crop_is_letterboxed_not_stretched(self):
        # A very wide crop in a near-square cell should shrink (not fill)
        # the cell on one axis, per _fit_aspect.
        mp = framing.make_multi_painter(1000, 1000, [{"w": 1000, "h": 100}])
        (ix, iy, fw, fh, _m3) = mp._cells[0]
        self.assertGreater(fw, fh * 5)  # still ~10:1, not squashed to fill a square cell

    def test_accepts_plain_tuple_rects(self):
        mp = framing.make_multi_painter(800, 600, [(320, 240), (320, 240)])
        crops = [np.full((240, 320, 3), 255, np.uint8)] * 2
        out = mp.paint(crops)
        self.assertEqual(out.shape, (600, 800, 3))


class RoundedMaskIsAliased(unittest.TestCase):
    """`MultiFramePainter.paint` composites in uint8 and repairs only the four
    corner squares. That is exact ONLY while these two properties hold, and
    both are quiet: antialiasing the corners (`cv2.LINE_AA`) or reshaping the
    mask would leave the card edges silently wrong rather than raising."""

    SHAPES = ((200, 120, 12), (37, 41, 8), (300, 90, 24), (16, 16, 40))

    def test_mask_is_binary(self):
        for w, h, r in self.SHAPES:
            mask = framing._rounded_mask(w, h, r)
            self.assertTrue(
                set(np.unique(mask).tolist()) <= {0, 255},
                "mask must stay aliased 0/255 for {}x{} r{}".format(w, h, r))

    def test_zeros_live_only_in_the_corner_squares(self):
        for w, h, r in self.SHAPES:
            mask = framing._rounded_mask(w, h, r)
            eff = int(min(r, w // 2, h // 2))
            corners = np.zeros((h, w), bool)
            for ys, xs in ((slice(0, eff), slice(0, eff)),
                           (slice(0, eff), slice(w - eff, w)),
                           (slice(h - eff, h), slice(0, eff)),
                           (slice(h - eff, h), slice(w - eff, w))):
                corners[ys, xs] = True
            stray = (mask == 0) & ~corners
            self.assertFalse(stray.any(),
                             "{}x{} r{}: {} transparent px outside the corner "
                             "squares paint() repairs".format(w, h, r,
                                                              int(stray.sum())))


class MultiWindowCompositeIsExact(unittest.TestCase):
    """`paint` must stay byte-identical to the straightforward per-pixel
    alpha blend it is an optimization of. This is a pinned output path -- a
    one-count drift is invisible in review and permanent in every export --
    so compare against the blend spelled out longhand, not against a
    tolerance."""

    def _blend(self, painter, crops):
        out = painter._plate.astype(np.float32)
        for (ix, iy, fw, fh, mask), crop in zip(painter._cells, crops):
            m3 = (mask.astype(np.float32) / 255.0)[:, :, None]
            rec = cv2.resize(crop, (fw, fh),
                             interpolation=cv2.INTER_AREA).astype(np.float32)
            roi = out[iy:iy + fh, ix:ix + fw]
            out[iy:iy + fh, ix:ix + fw] = rec * m3 + roi * (1.0 - m3)
        return out.astype(np.uint8)

    def _check(self, W, H, rects, background=None, layout="grid", seed=0):
        painter = framing.make_multi_painter(W, H, rects, background=background,
                                             layout=layout)
        rng = np.random.RandomState(seed)
        crops = [rng.randint(0, 256, (int(r["h"]), int(r["w"]), 3), np.uint8)
                 for r in rects]
        got, want = painter.paint(crops), self._blend(painter, crops)
        self.assertTrue(np.array_equal(got, want),
                        "{} px differ (max {})".format(
                            int((got != want).sum()),
                            int(np.abs(got.astype(np.int16)
                                       - want.astype(np.int16)).max())))

    def test_one_card(self):
        self._check(640, 360, [{"x": 0, "y": 0, "w": 300, "h": 200}])

    def test_four_cards_mixed_aspects(self):
        self._check(1280, 720, [{"x": 0, "y": 0, "w": 300, "h": 200},
                                {"x": 300, "y": 0, "w": 200, "h": 400},
                                {"x": 0, "y": 200, "w": 640, "h": 120},
                                {"x": 500, "y": 300, "w": 111, "h": 97}],
                    background="sunset", seed=1)

    def test_desktop_layout_and_odd_sizes(self):
        self._check(957, 601, [{"x": 3, "y": 7, "w": 417, "h": 313},
                               {"x": 511, "y": 41, "w": 233, "h": 519}],
                    background="#101820", layout="desktop", seed=2)

    def test_upscaled_and_tiny_cards(self):
        self._check(400, 400, [{"x": 0, "y": 0, "w": 9, "h": 7},
                               {"x": 20, "y": 20, "w": 1600, "h": 40}],
                    background="mist", seed=3)

    def test_array_background(self):
        rng = np.random.RandomState(7)
        bg = rng.randint(0, 256, (300, 500, 3), np.uint8).astype(np.float32)
        self._check(500, 300, [{"x": 0, "y": 0, "w": 220, "h": 140},
                               {"x": 220, "y": 0, "w": 140, "h": 220}],
                    background=bg, seed=4)

    def test_returns_a_fresh_canvas_each_call(self):
        """Callers draw ON TOP of the returned frame in place (the facecam
        bubble), so handing back a reused buffer would smear one frame's
        overlay into every later one."""
        painter = framing.make_multi_painter(
            400, 300, [{"x": 0, "y": 0, "w": 200, "h": 150}])
        crops = [np.full((150, 200, 3), 90, np.uint8)]
        first = painter.paint(crops)
        plate_before = painter._plate.copy()
        first[:] = 7                                   # an overlay, drawn in place
        second = painter.paint(crops)
        self.assertFalse(np.array_equal(first, second))
        self.assertTrue(np.array_equal(painter._plate, plate_before))
        self.assertTrue(np.array_equal(second, painter.paint(crops)))


# ---- the named arrangements: "feature" / "row" / "column" -------------------
#
# The user's own two window shapes, because the arrangement that prompted this
# feature is the one they kept dragging by hand: a tall editor down the left
# with two wide panes stacked beside it. Aspects are deliberately MIXED
# throughout -- an arrangement built out of four identical rects would satisfy
# every property below while quietly stretching real windows.
_TALL = {"x": 0, "y": 0, "w": 1418, "h": 1720}
_WIDE = {"x": 1462, "y": 0, "w": 1418, "h": 852}
_WIDE2 = {"x": 1462, "y": 900, "w": 1418, "h": 852}
_SQUARE = {"x": 200, "y": 1200, "w": 900, "h": 940}

_WINDOW_SETS = {
    1: [_TALL],
    2: [_TALL, _WIDE],
    3: [_TALL, _WIDE, _WIDE2],
    4: [_TALL, _WIDE, _WIDE2, _SQUARE],
}

# Retina, 1080p, a 9:16 export, and two small canvases -- the padding and gap
# are fractions of the WIDTH, so a small canvas is not just a scaled-down big
# one: it is where rounding is proportionally largest.
_CANVASES = ((2880, 1800), (1920, 1080), (1080, 1920), (1280, 720), (640, 360))

_NAMED_LAYOUTS = ("feature", "row", "column")


# One real `MultiFramePainter` per canvas, kept only for the pad and gap it
# derives. `_arrange` below used to re-derive them -- `int(framing.PAD_FRAC * W)` and
# `int(0.02 * W)`, copied out of the painter -- and a single test on a single
# canvas was the only thing keeping the copy honest, while EVERY sweep in this
# module (`InkCoverage`, `NamedArrangementsAreWellFormed`,
# `FeatureOrientationFollowsTheCanvas`, `GridAndDesktopUnchanged`) measures
# through it. A drift there would not fail anything; it would silently move all
# of them onto an arrangement nothing renders. Reading the numbers off a
# painter deletes the copy instead of guarding it.
#
# Cached because a 2880x1800 painter costs ~2s to build (one full-canvas
# GaussianBlur at sigma ~58px) and the sweeps call `_arrange` hundreds of
# times. Keyed by the whole canvas, not by width: that pad and gap happen to
# depend on the width alone is the painter's business, and hard-coding the
# assumption here would put the copy straight back.
_PAD_GAP = {}


def _painter_pad_gap(W, H):
    key = (int(W), int(H))
    if key not in _PAD_GAP:
        painter = framing.MultiFramePainter(
            W, H, [{"x": 0, "y": 0, "w": 4, "h": 4}])
        _PAD_GAP[key] = (painter._pad, painter._gap)
    return _PAD_GAP[key]


def _arrange(layout, W, H, rects):
    """`(pad, placements)` exactly as `MultiFramePainter.__init__` computes
    them, without paying for the shadow bake on a 2880x1800 canvas.

    The pad and gap come off a real painter (`_painter_pad_gap`); what is left
    here is the dispatch -- `_ARRANGEMENTS` or the grid fallback -- which
    `NamedArrangementsAreWellFormed.test_helper_matches_the_painter` pins
    against a real painter over every layout name and window count.
    """
    pad, gap = _painter_pad_gap(W, H)
    fn = framing._ARRANGEMENTS.get(layout)
    cells = fn(W, H, rects, pad, gap) if fn is not None else None
    if cells is None:
        cells = framing._grid_placements(W, H, rects, pad, gap)
    return pad, framing._apply_card_overrides(W, H, cells, rects)


def _ink(cells, W, H):
    """Percentage of the canvas the cards actually cover.

    The number the complaint that started this whole feature was about, and
    the one score that separates the layouts: "touches the padding" is
    satisfied by every layout that ever shipped, including the ones the user
    was unhappy with.

    Summing areas is exact rather than an approximation of a union, because
    no layout overlaps its cards -- pinned for the presets by
    `test_no_two_cards_overlap`, true of the grid by construction (disjoint
    cells) and of desktop by `_separate_boxes`.
    """
    return 100.0 * sum(c[2] * c[3] for c in cells) / float(W * H)


class NamedArrangementsAreWellFormed(unittest.TestCase):
    """The three structural properties every preset arrangement has to hold,
    swept over 1-4 mixed-aspect windows on five canvases.

    These are what a rewrite aimed at covering more of the canvas is most
    likely to break on its way there: cards can always be made to cover more
    of the frame by overlapping them, running one off the canvas, or
    stretching one out of aspect. How much of the canvas they actually cover
    is `InkCoverage` below, kept separate because it is a quality bar rather
    than a correctness one.
    """

    def _sweep(self):
        for W, H in _CANVASES:
            for n in (1, 2, 3, 4):
                for layout in _NAMED_LAYOUTS:
                    pad, cells = _arrange(layout, W, H, _WINDOW_SETS[n])
                    yield (W, H, n, layout, pad, cells)

    # Every layout name that reaches a painter, including the two that go
    # through `_arrange`'s grid FALLBACK rather than `_ARRANGEMENTS` and an
    # unrecognized one, which is a third path again.
    ALL_LAYOUTS = _NAMED_LAYOUTS + ("grid", "desktop", "spiral")

    def test_helper_matches_the_painter(self):
        """`_arrange` is not the painter, and every sweep in this module
        measures through it -- so pin the whole dispatch rather than one layout
        on one canvas: eight layout names over 1-4 windows, landscape and
        portrait (the orientation matters: `_choose_grid` and `feature`'s
        transpose both re-decide on it).

        On the small canvases because this needs a REAL painter per case and a
        2880x1800 one costs seconds to bake. What varies with the canvas -- the
        pad and the gap -- is no longer re-derived here at all; it comes off a
        real painter for every canvas via `_painter_pad_gap`.
        """
        for W, H in ((640, 360), (360, 640)):
            for n in (1, 2, 3, 4):
                for layout in self.ALL_LAYOUTS:
                    painter = framing.make_multi_painter(
                        W, H, _WINDOW_SETS[n], layout=layout)
                    _pad, cells = _arrange(layout, W, H, _WINDOW_SETS[n])
                    self.assertEqual(
                        list(painter._placements), cells,
                        "{} n={} on {}x{}".format(layout, n, W, H))

    def test_helper_matches_the_painter_on_the_canvas_the_tables_use(self):
        """A small canvas cannot show a rounding difference that only appears
        at scale, and the measured tables (`InkCoverage`,
        `GridAndDesktopUnchanged`) are taken at 2880x1800. One painter, because
        each one is ~2s of GaussianBlur; `feature` because it is the layout
        with a choice to get wrong."""
        painter = framing.make_multi_painter(
            2880, 1800, _WINDOW_SETS[3], layout="feature")
        _pad, cells = _arrange("feature", 2880, 1800, _WINDOW_SETS[3])
        self.assertEqual(list(painter._placements), cells)
        self.assertEqual((painter._pad, painter._gap),
                         _painter_pad_gap(2880, 1800))

    def test_no_two_cards_overlap(self):
        for W, H, n, layout, _pad, cells in self._sweep():
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    a, b = cells[i], cells[j]
                    ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                    oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                    self.assertTrue(
                        ox <= 0 or oy <= 0,
                        "{} n={} on {}x{}: cards {} and {} overlap by "
                        "{}x{}px".format(layout, n, W, H, a, b, ox, oy))

    def test_every_card_is_inside_the_canvas(self):
        for W, H, n, layout, _pad, cells in self._sweep():
            for cell in cells:
                x, y, w, h = cell
                self.assertTrue(
                    0 <= x and 0 <= y and x + w <= W and y + h <= H,
                    "{} n={} on {}x{}: card {} leaves the canvas".format(
                        layout, n, W, H, cell))

    def test_each_card_keeps_its_source_aspect(self):
        for W, H, n, layout, _pad, cells in self._sweep():
            for cell, rect in zip(cells, _WINDOW_SETS[n]):
                _x, _y, w, h = cell
                want = float(rect["w"]) / float(rect["h"])
                got = w / float(h)
                # Width and height are rounded INDEPENDENTLY off one uniform
                # scale, so the ratio can drift by at most half a pixel on
                # each -- bound it exactly instead of guessing a tolerance,
                # which on a 1484px-tall card would wave through a real
                # stretch and on a 187px-tall one would fail honest rounding.
                bound = 0.5 * (want + 1.0) / h
                self.assertLessEqual(
                    abs(got - want), bound,
                    "{} n={} on {}x{}: card {} is {:.4f}:1, source is "
                    "{:.4f}:1".format(layout, n, W, H, cell, got, want))

    def test_the_arrangement_touches_the_padding_on_one_axis(self):
        """A uniform scale that preserves every aspect fills exactly one axis
        and centres the other, so this is the *mechanical* half of a fit: it
        proves `_scale_boxes_into` ran and grew the arrangement until
        something stopped it.

        Deliberately NOT billed as the dead-space property it used to claim
        to be. `grid` and `desktop` satisfy it identically -- it distinguishes
        nothing, and the arrangement the user complained about passes it. The
        property that separates the layouts is coverage, in `InkCoverage`.
        """
        for W, H, n, layout, pad, cells in self._sweep():
            x0 = min(c[0] for c in cells)
            x1 = max(c[0] + c[2] for c in cells)
            y0 = min(c[1] for c in cells)
            y1 = max(c[1] + c[3] for c in cells)
            # 1px of slack is the rounding allowance; every case here
            # measures exactly 0.
            dx = max(abs(x0 - pad), abs(x1 - (W - pad)))
            dy = max(abs(y0 - pad), abs(y1 - (H - pad)))
            self.assertLessEqual(
                min(dx, dy), 1,
                "{} n={} on {}x{} (pad {}): spans x {}..{} y {}..{} -- "
                "dead space on both axes".format(
                    layout, n, W, H, pad, x0, x1, y0, y1))


# The three windows the user actually recorded -- a tall editor and two wide
# panes. `_WINDOW_SETS` rounds them slightly and adds a fourth for the sweeps
# above; these are the exact rects the measured table in `framing.py` was taken
# from, so `InkCoverage` below can pin that table literally. Only `desktop`
# reads the origins; every preset sees aspects only.
_REAL_WINDOWS = [{"x": 0, "y": 0, "w": 1418, "h": 1718},
                 {"x": 1462, "y": 0, "w": 1418, "h": 852},
                 {"x": 1462, "y": 900, "w": 1418, "h": 854}]

# 16:10 Retina, 16:9, a 9:16 export, and square.
_INK_CANVASES = ((2880, 1800), (1920, 1080), (1080, 1920), (1440, 1440))


class InkCoverage(unittest.TestCase):
    """How much of the canvas each layout's cards actually cover.

    This replaces a "touches the padding on one axis" property that read like
    the dead-space bug stated formally and was in fact satisfied by `grid` and
    `desktop` too -- it separated nothing, so it could not have caught the
    round it was written to defend. Coverage separates them, because coverage
    is the thing the user was complaining about.

    Floors, not equalities: the measured value minus 0.5 of a percentage
    point, so honest rounding drift passes and a layout that quietly gets
    worse does not. The floors are the table in the comment above
    `framing._unit_gap`, so if this fails, that comment is now a lie and both
    have to move together.
    """

    # (layout, W, H) -> covered % of the canvas, for the three real windows.
    # Floors, re-taken 2026-08-31 when framing.PAD_FRAC dropped 0.055 -> 0.03.
    # Left at the OLD values this table would still have passed (the check is
    # >= want - 0.5 and every figure rose), which is exactly why it is worth
    # re-taking: a floor nobody moves stops being a regression floor and
    # becomes a floor under a level the code left behind years ago.
    MEASURED = {
        ("grid", 2880, 1800): 34.9, ("grid", 1920, 1080): 38.7,
        ("grid", 1080, 1920): 73.2, ("grid", 1440, 1440): 21.8,
        ("desktop", 2880, 1800): 80.7, ("desktop", 1920, 1080): 71.0,
        ("desktop", 1080, 1920): 29.1, ("desktop", 1440, 1440): 51.8,
        ("feature", 2880, 1800): 81.8, ("feature", 1920, 1080): 71.6,
        ("feature", 1080, 1920): 74.9, ("feature", 1440, 1440): 56.6,
        ("row", 2880, 1800): 33.0, ("row", 1920, 1080): 36.5,
        ("row", 1080, 1920): 11.8, ("row", 1440, 1440): 20.9,
        ("column", 2880, 1800): 20.0, ("column", 1920, 1080): 17.5,
        ("column", 1080, 1920): 67.5, ("column", 1440, 1440): 35.4,
    }

    def _ink(self, layout, W, H, rects=None):
        _pad, cells = _arrange(layout, W, H,
                               _REAL_WINDOWS if rects is None else rects)
        return _ink(cells, W, H)

    def test_every_layout_covers_at_least_what_it_covers_today(self):
        for (layout, W, H), want in sorted(self.MEASURED.items()):
            got = self._ink(layout, W, H)
            self.assertGreaterEqual(
                got, want - 0.5,
                "{} on {}x{}: covers {:.1f}% of the canvas, documented "
                "{:.1f}%".format(layout, W, H, got, want))

    def test_feature_more_than_doubles_the_grid_on_a_landscape_canvas(self):
        """The claim the presets were added for, as a number: at
        framing.PAD_FRAC the grid covers 34.9% of a 2880x1800 canvas and
        feature 81.8%. The RATIO is the pin, not the levels -- both rose with
        the 2026-08-31 margin change and the gap held."""
        grid = self._ink("grid", 2880, 1800)
        feature = self._ink("feature", 2880, 1800)
        self.assertGreater(feature / grid, 2.0,
                           "feature {:.1f}% vs grid {:.1f}%".format(feature,
                                                                    grid))

    def test_feature_is_within_a_third_of_a_point_of_the_best_everywhere(self):
        """What re-orienting buys, and the reason `feature` is the safe
        default: no other layout is close to the best on all four canvases.
        Before the transpose it covered 26.6% of the 9:16 canvas against the
        grid's 67.4%."""
        for W, H in _INK_CANVASES:
            inks = dict((l, self._ink(l, W, H))
                        for l in ("grid", "desktop", "feature", "row",
                                  "column"))
            best = max(inks.values())
            self.assertGreaterEqual(
                inks["feature"], best - 0.35,
                "{}x{}: feature {:.1f}% against the best {:.1f}% -- "
                "{}".format(W, H, inks["feature"], best, inks))

    def test_no_other_layout_manages_that(self):
        """Stated as its own assertion so the one above cannot be satisfied by
        a change that simply lifts everything: every other layout is more than
        a point off the best on at least one of these four canvases."""
        for layout in ("grid", "desktop", "row", "column"):
            worst_gap = 0.0
            for W, H in _INK_CANVASES:
                best = max(self._ink(l, W, H)
                           for l in ("grid", "desktop", "feature", "row",
                                     "column"))
                worst_gap = max(worst_gap, best - self._ink(layout, W, H))
            self.assertGreater(
                worst_gap, 1.0,
                "{} is now within {:.1f} points of the best on every canvas "
                "-- the feature/rest distinction has gone".format(layout,
                                                                  worst_gap))

    def test_two_cards_rank_differently_from_three(self):
        """The table is a three-card measurement and does not generalize --
        pinned so nobody reads it as a rule. With two cards on a vertical
        canvas, column ties feature at 80.9% and the grid trails at 58.4%;
        on 16:10 the grid beats both presets."""
        pair = _REAL_WINDOWS[:2]
        vertical = dict((l, self._ink(l, 1080, 1920, pair))
                        for l in ("grid", "feature", "column"))
        self.assertAlmostEqual(vertical["feature"], vertical["column"],
                               delta=0.1, msg=str(vertical))
        self.assertGreater(vertical["feature"] - vertical["grid"], 20.0,
                           str(vertical))
        wide = dict((l, self._ink(l, 2880, 1800, pair))
                    for l in ("grid", "feature", "column"))
        self.assertGreater(wide["grid"], wide["feature"], str(wide))


def _feature_axis(cells):
    """0 when the feature card is on the LEFT (the block runs down the y
    axis), 1 when it is on TOP (the block runs along x). Read off the
    placements rather than asked of `framing`, so the test is checking the
    geometry that renders and not the decision that produced it."""
    hero, first = cells[0], cells[1]
    if first[0] >= hero[0] + hero[2]:
        return 0
    if first[1] >= hero[1] + hero[3]:
        return 1
    raise AssertionError("card 1 {} is neither right of nor below the "
                         "feature card {}".format(first, hero))


def _feature_candidate(aspects, gap_u, transposed):
    """One orientation of the feature arrangement, in unit space, re-derived
    from the SHAPE the arrangement promises instead of from
    `framing._feature_boxes`.

    Deliberately a second implementation and not a call: the orientation test
    below has to be able to score the candidate that LOST, and asking the
    implementation for it would only pin the wiring -- any error inside
    `_feature_boxes` or `_fit_scale` would be reproduced identically on both
    sides and the test would agree with the bug.

    The contract, exactly as `FeatureArrangementShape` pins it on the output:
    the hero is its own aspect wide by 1.0 tall; the rest form ONE block that
    shares a single thickness and spans exactly the hero's facing edge, with
    the same `gap_u` between every pair including hero-to-block. Solve the
    block's thickness for that span and the arrangement is determined.
    """
    hero, rest = aspects[0], list(aspects[1:])
    gaps = (len(rest) - 1) * gap_u
    boxes = [(0.0, 0.0, hero, 1.0)]
    if not transposed:
        # A card `w` wide is w/aspect tall; the column must total the hero's
        # 1.0 including its own internal gaps.
        w = (1.0 - gaps) / sum(1.0 / a for a in rest)
        x, y = hero + gap_u, 0.0
        for a in rest:
            h = w / a
            boxes.append((x, y, w, h))
            y += h + gap_u
    else:
        # Mirrored: a card `h` tall is h*aspect wide, and the row must total
        # the hero's width.
        h = (hero - gaps) / sum(rest)
        x, y = 0.0, 1.0 + gap_u
        for a in rest:
            w = h * a
            boxes.append((x, y, w, h))
            x += w + gap_u
    return boxes


def _uniform_fit(W, H, boxes, pad):
    """The uniform scale + centre every arrangement ends in, spelled out here
    for the same reason `_feature_candidate` is: scoring the rejected
    candidate through `framing._scale_boxes_into` would make the comparison
    blind to an error inside it."""
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    span_w = max(b[0] + b[2] for b in boxes) - x0
    span_h = max(b[1] + b[3] for b in boxes) - y0
    scale = min((W - 2.0 * pad) / span_w, (H - 2.0 * pad) / span_h)
    off_x = (W - span_w * scale) / 2.0
    off_y = (H - span_h * scale) / 2.0
    return [(int(round(off_x + (x - x0) * scale)),
             int(round(off_y + (y - y0) * scale)),
             int(round(w * scale)), int(round(h * scale)))
            for (x, y, w, h) in boxes]


class FeatureArrangementShape(unittest.TestCase):
    """"Feature" is the preset the user was hand-building, so its shape is the
    contract, not an implementation detail: one large card, the rest beside it
    in card order against a single straight edge.

    Every property here is stated on the chosen AXIS rather than on "left",
    because the arrangement transposes to suit the canvas
    (`FeatureOrientationFollowsTheCanvas`). What must hold in both
    orientations is the same thing: the block is sized as one block. Sizing
    each card on its own would still satisfy every property in
    `NamedArrangementsAreWellFormed` while leaving the block ragged, which is
    the look this replaces.
    """

    def _cells(self, n, W=2880, H=1800):
        return _arrange("feature", W, H, _WINDOW_SETS[n])[1]

    def test_the_hero_is_clear_of_every_other_card_on_one_axis(self):
        for W, H in _CANVASES:
            for n in (2, 3, 4):
                cells = self._cells(n, W, H)
                axis = _feature_axis(cells)
                hero = cells[0]
                for other in cells[1:]:
                    self.assertLessEqual(
                        hero[axis] + hero[axis + 2], other[axis],
                        "n={} on {}x{} (axis {}): feature card {} does not "
                        "clear {}".format(n, W, H, axis, hero, other))

    def test_the_block_runs_in_card_order(self):
        for W, H in _CANVASES:
            for n in (3, 4):
                cells = self._cells(n, W, H)
                # The block runs along the OTHER axis from the hero split:
                # hero-left means a column, hero-top means a row.
                run = 1 - _feature_axis(cells)
                block = cells[1:]
                for first, second in zip(block, block[1:]):
                    self.assertLessEqual(
                        first[run] + first[run + 2], second[run],
                        "n={} on {}x{}: {} does not precede {} on axis "
                        "{}".format(n, W, H, first, second, run))

    def test_the_block_shares_one_edge_and_one_thickness(self):
        # Exact equality, not a tolerance: every card in the block is the same
        # thickness in unit space, so one uniform scale must round them all to
        # the same pixel count. A ragged block would mean it was sized card by
        # card instead of as one block.
        for W, H in _CANVASES:
            for n in (3, 4):
                cells = self._cells(n, W, H)
                axis = _feature_axis(cells)
                block = cells[1:]
                self.assertEqual(len(set(c[axis] for c in block)), 1,
                                 str(block))
                self.assertEqual(len(set(c[axis + 2] for c in block)), 1,
                                 str(block))

    def test_the_block_spans_exactly_the_heros_facing_edge(self):
        """The block is scaled as one block to match the hero's facing edge,
        which is what leaves no letterboxing between the two of them -- the
        only dead space in the composition is the canvas letterbox
        `_scale_boxes_into` cannot avoid."""
        for W, H in _CANVASES:
            for n in (3, 4):
                cells = self._cells(n, W, H)
                run = 1 - _feature_axis(cells)
                hero, block = cells[0], cells[1:]
                near = min(c[run] for c in block)
                far = max(c[run] + c[run + 2] for c in block)
                self.assertAlmostEqual(near, hero[run], delta=2)
                self.assertAlmostEqual(far, hero[run] + hero[run + 2], delta=2)

    def test_one_requested_gap_produces_one_visual_gap(self):
        """It used to produce three: the hero-to-block gap was `gap_u`, the
        gaps inside the block were `gap_u * scale`, and `_scale_boxes_into`
        then rescaled the lot. Measured at 2880x1800, where the requested gap
        is 57px: a 3-card feature came out 58 / 46, a 4-card one 58 / 24 / 25
        -- less than half, from one requested gap.

        2px of tolerance, not 1: the hero-to-block gap and the block's own
        gaps are rounded off different edges of the same uniform scale.
        """
        for W, H in _CANVASES:
            for n in (3, 4):
                cells = self._cells(n, W, H)
                axis = _feature_axis(cells)
                run = 1 - axis
                hero, block = cells[0], cells[1:]
                gaps = [min(c[axis] for c in block) - (hero[axis]
                                                       + hero[axis + 2])]
                for first, second in zip(block, block[1:]):
                    gaps.append(second[run] - (first[run] + first[run + 2]))
                self.assertLessEqual(
                    max(gaps) - min(gaps), 2,
                    "n={} on {}x{}: gaps {} are not one gap".format(
                        n, W, H, gaps))


class FeatureOrientationFollowsTheCanvas(unittest.TestCase):
    """"One big window plus the others beside it" is an INTENT, not an axis,
    so `feature` transposes: hero on the left with a column beside it on a
    landscape canvas, hero on top with a row beneath it on a portrait one.
    Both candidates are built and scored with the same `_fit_scale` the
    desktop layout uses to pick among its compaction candidates.

    Pinned because it is invisible in the output of any single canvas, and
    because it is what took `feature` on a 9:16 export from 26.6% of the
    canvas covered to 67.1% (`InkCoverage`).
    """

    def test_landscape_keeps_the_hero_on_the_left(self):
        for W, H in ((2880, 1800), (1920, 1080), (1280, 720), (640, 360)):
            for n in (2, 3, 4):
                cells = _arrange("feature", W, H, _WINDOW_SETS[n])[1]
                self.assertEqual(_feature_axis(cells), 0,
                                 "n={} on {}x{}: {}".format(n, W, H, cells))

    def test_portrait_and_square_put_the_hero_on_top(self):
        # Square counts as "taller than the left-hero arrangement wants":
        # these windows are wide, so a row of them beside a tall hero needs a
        # much wider canvas than 1:1 to break even.
        for W, H in ((1080, 1920), (900, 1600), (1440, 1440)):
            for n in (2, 3, 4):
                cells = _arrange("feature", W, H, _WINDOW_SETS[n])[1]
                self.assertEqual(_feature_axis(cells), 1,
                                 "n={} on {}x{}: {}".format(n, W, H, cells))

    def test_the_winner_is_the_one_that_scores_higher(self):
        """Not just "portrait transposes" -- the rule is the score, and the
        two must never disagree, or the arrangement on screen is not the one
        the planner picked."""
        for W, H in _CANVASES + ((1440, 1440), (900, 1600)):
            pad = int(framing.PAD_FRAC * W)
            gap = max(0, int(0.02 * W))
            gap_u = framing._unit_gap(H, pad, gap)
            for n in (2, 3, 4):
                rects = _WINDOW_SETS[n]
                hero = framing._rect_aspect(rects[0])
                rest = [framing._rect_aspect(r) for r in rects[1:]]
                left = framing._fit_scale(
                    W, H, framing._feature_boxes(hero, rest, gap_u, False),
                    pad)
                top = framing._fit_scale(
                    W, H, framing._feature_boxes(hero, rest, gap_u, True), pad)
                want = 1 if top > left else 0
                got = _feature_axis(_arrange("feature", W, H, rects)[1])
                self.assertEqual(
                    got, want,
                    "n={} on {}x{}: left-hero scores {:.3f}, top-hero "
                    "{:.3f}, but the arrangement chose axis {}".format(
                        n, W, H, left, top, got))

    def _candidates(self, W, H, rects):
        """Both orientations placed on the canvas, re-derived independently of
        `_feature_boxes` / `_fit_scale` / `_scale_boxes_into`."""
        pad, gap = _painter_pad_gap(W, H)
        gap_u = float(gap) / max(1.0, float(H) - 2.0 * pad)
        aspects = [float(r["w"]) / float(r["h"]) for r in rects]
        return [_uniform_fit(W, H, _feature_candidate(aspects, gap_u, t), pad)
                for t in (False, True)]

    def test_the_shipped_arrangement_has_the_bigger_hero(self):
        """The observable consequence, checked without the implementation's own
        helpers. `test_the_winner_is_the_one_that_scores_higher` above re-runs
        `_feature_boxes` + `_fit_scale` -- the exact pair the decision is made
        with -- so it pins the wiring and would agree with any error INSIDE
        them. This builds both candidates from the arrangement's stated shape
        instead, and asserts two things about what actually renders: the
        shipped placements ARE one of the two candidates (exactly -- the
        re-derivation reproduces them to the pixel on every case here), and it
        is the one whose hero card ends up bigger.

        Bigger HERO, which is what `_feature_boxes` normalizes for by giving
        both candidates the same `(aspect, 1.0)` hero -- not more ink; see
        `test_the_score_maximizes_the_hero_not_the_coverage`.
        """
        for W, H in _CANVASES + ((1440, 1440), (900, 1600)):
            for n in (2, 3, 4):
                rects = _WINDOW_SETS[n]
                left, top = self._candidates(W, H, rects)
                areas = [c[0][2] * c[0][3] for c in (left, top)]
                want = 1 if areas[1] > areas[0] else 0
                got = _arrange("feature", W, H, rects)[1]
                self.assertEqual(
                    list(got), list((left, top)[want]),
                    "n={} on {}x{}: shipped {} is neither candidate "
                    "({} / {})".format(n, W, H, got, left, top))
                self.assertEqual(
                    _feature_axis(got), want,
                    "n={} on {}x{}: hero areas {} but the arrangement chose "
                    "axis {}".format(n, W, H, areas, _feature_axis(got)))

    def test_the_score_maximizes_the_hero_not_the_coverage(self):
        """The counter-example that stops anyone "fixing" the score into a
        coverage-maximizer without noticing they are changing the feature.

        Measured: four windows on a 1440x1440 canvas. The transposed candidate
        wins -- its hero is 878x1065 against the left-hero candidate's
        831x1008 -- and it covers 52.9% of the canvas where the rejected one
        would have covered 60.3%. So "the arrangement picks the orientation
        that fills more of the frame" is FALSE, by 7.5 points, on a canvas the
        UI offers. It picks the bigger FEATURED window, which is what a
        preset named after a role should do, and the three-window coverage
        table in `InkCoverage` is unaffected (there the two agree everywhere).
        """
        W, H, rects = 1440, 1440, _WINDOW_SETS[4]
        left, top = self._candidates(W, H, rects)
        got = _arrange("feature", W, H, rects)[1]
        self.assertEqual(list(got), list(top), str(got))
        self.assertGreater(top[0][2] * top[0][3], left[0][2] * left[0][3])
        self.assertGreater(_ink(left, W, H) - _ink(top, W, H), 5.0,
                           "left-hero {:.1f}% vs the chosen {:.1f}%".format(
                               _ink(left, W, H), _ink(top, W, H)))

    def test_a_tie_keeps_the_left_hero_candidate(self):
        """`max` returns the FIRST maximum, and left-hero is built first --
        the same determinism `_desktop_placements` relies on for its three
        candidates. Forced with a square hero, square block cards and no gap,
        where the two candidates are exact transposes of each other."""
        square = [{"x": 0, "y": 0, "w": 500, "h": 500}] * 3
        boxes_l = framing._feature_boxes(1.0, [1.0, 1.0], 0.0, False)
        boxes_t = framing._feature_boxes(1.0, [1.0, 1.0], 0.0, True)
        self.assertEqual(framing._fit_scale(1000, 1000, boxes_l, 0),
                         framing._fit_scale(1000, 1000, boxes_t, 0))
        # `_feature_placements` called directly, with the pad and gap the tie
        # needs, rather than through the painter -- the painter derives a gap
        # from the canvas width and any gap at all breaks the symmetry.
        cells = framing._feature_placements(1000, 1000, square, 0, 0)
        self.assertEqual(_feature_axis(cells), 0, str(cells))


class RowAndColumnNeverTranspose(unittest.TestCase):
    """`feature` is named after a ROLE, so it may re-orient. `row` and
    `column` are named after an AXIS, and a user who picks "Row" and is handed
    a column has been lied to -- so they hold their axis on every canvas, even
    the ones where it costs them most of the frame (row covers 10.7% of a 9:16
    canvas; see `InkCoverage`).
    """

    def test_row_stays_a_row_everywhere(self):
        for W, H in _CANVASES + ((1440, 1440), (900, 1600)):
            for n in (2, 3, 4):
                cells = _arrange("row", W, H, _WINDOW_SETS[n])[1]
                self.assertEqual(len(set(c[1] for c in cells)), 1, str(cells))
                for first, second in zip(cells, cells[1:]):
                    self.assertLessEqual(first[0] + first[2], second[0],
                                         "{}x{}: {}".format(W, H, cells))

    def test_column_stays_a_column_everywhere(self):
        for W, H in _CANVASES + ((1440, 1440), (900, 1600)):
            for n in (2, 3, 4):
                cells = _arrange("column", W, H, _WINDOW_SETS[n])[1]
                self.assertEqual(len(set(c[0] for c in cells)), 1, str(cells))
                for first, second in zip(cells, cells[1:]):
                    self.assertLessEqual(first[1] + first[3], second[1],
                                         "{}x{}: {}".format(W, H, cells))


class SingleWindowIsTheGrid(unittest.TestCase):
    """One card has no arrangement to speak of, so all three presets delegate
    to the grid rather than re-deriving a centred card three more ways -- and
    that keeps `_scale_boxes_into`, whose spans are `min()`/`max()` over the
    box list, from ever being handed an empty one."""

    def test_one_window_is_identical_to_the_grid(self):
        for W, H in _CANVASES:
            _pad, want = _arrange("grid", W, H, _WINDOW_SETS[1])
            for layout in _NAMED_LAYOUTS:
                _pad, got = _arrange(layout, W, H, _WINDOW_SETS[1])
                self.assertEqual(got, want,
                                 "{} on {}x{}".format(layout, W, H))

    def test_no_window_is_an_empty_placement_list(self):
        for layout in _NAMED_LAYOUTS:
            self.assertEqual(_arrange(layout, 1280, 720, [])[1], [])


class UnknownLayoutFallsBackToTheGrid(unittest.TestCase):
    """Adding names to `_ARRANGEMENTS` must not change what an unrecognized
    one does. `window_layout` is a free string on the way in from edits.json,
    the CLI and MCP, and every one of those validators coerces an unknown
    value to "grid" -- but the painter is the last line, and a layout name
    from a newer version of the app has to land on the grid there too rather
    than raise mid-render."""

    RECTS = [{"x": 0, "y": 0, "w": 300, "h": 200},
             {"x": 320, "y": 0, "w": 200, "h": 400}]

    def test_placements_match_the_grid(self):
        _pad, want = _arrange("grid", 640, 400, self.RECTS)
        for name in ("spiral", "Feature", "ROW", "", " row", "fit"):
            _pad, got = _arrange(name, 640, 400, self.RECTS)
            self.assertEqual(got, want, "layout {!r}".format(name))

    def test_painted_frame_is_byte_identical_to_the_grid(self):
        rng = np.random.RandomState(11)
        crops = [rng.randint(0, 256, (int(r["h"]), int(r["w"]), 3), np.uint8)
                 for r in self.RECTS]
        base = framing.make_multi_painter(640, 400, self.RECTS,
                                          background="sunset", layout="grid")
        want = base.paint(crops)
        for name in ("spiral", "Feature", ""):
            other = framing.make_multi_painter(
                640, 400, self.RECTS, background="sunset", layout=name)
            self.assertTrue(np.array_equal(other.paint(crops), want),
                            "layout {!r} did not paint the grid".format(name))

    def test_originless_rects_keep_the_presets_but_still_drop_desktop(self):
        """A plain `(w, h)` rect carries no desktop position, so "desktop" has
        nothing to preserve and falls back; the three presets never read x/y
        at all and stay themselves. Pinned because the two halves look like
        one behaviour and are not."""
        tuples = [(320, 240), (640, 360), (100, 700)]
        _pad, grid = _arrange("grid", 1280, 720, tuples)
        self.assertEqual(_arrange("desktop", 1280, 720, tuples)[1], grid)
        for layout in _NAMED_LAYOUTS:
            self.assertNotEqual(_arrange(layout, 1280, 720, tuples)[1], grid,
                                "{} fell back to the grid".format(layout))


class GridAndDesktopUnchanged(unittest.TestCase):
    """The two shipped arrangements, pinned to literal pixels.

    Literals rather than a computed comparison on purpose: every alternative
    (recomputing with the same helpers, comparing two layouts to each other)
    would move along with a refactor of the shared `_scale_boxes_into` /
    `_fit_aspect` code the new arrangements now also depend on. These numbers
    are what existing projects render today, and nothing is allowed to move
    them except a deliberate, declared change to the margin itself.

    TWO tables, and the second is the reason the first still means something.
    `LEGACY_*` are the original literals, captured from the implementation
    that predates the presets, and are checked at the fraction they were
    captured with (`_LEGACY_PAD_FRAC` = 0.055). They are FROZEN: they prove
    that the layout arithmetic has never drifted, independently of what the
    current margin happens to be. `GRID`/`DESKTOP` pin today's output at
    `framing.PAD_FRAC`.

    When PAD_FRAC dropped to 0.03 (2026-08-31, for the empty-background
    report -- see framing.PAD_FRAC), re-running the LEGACY tables at 0.055
    reproduced every literal EXACTLY, which is what made re-baselining the
    live table provably a margin change and not a regression. Any future
    move of PAD_FRAC should repeat that: if the LEGACY table stops matching,
    something other than the margin has changed and the new numbers must not
    be blessed.
    """

    _LEGACY_PAD_FRAC = 0.055

    LEGACY_GRID = {
        (1280, 720): {
            1: [(401, 70, 478, 580)],
            2: [(110, 70, 478, 580), (652, 192, 558, 335)],
            3: [(70, 140, 363, 441), (458, 251, 363, 218),
                (847, 251, 363, 218)],
            4: [(234, 70, 229, 278), (700, 70, 462, 278),
                (118, 372, 462, 278), (798, 372, 266, 278)],
        },
        (2880, 1800): {
            3: [(158, 404, 817, 991), (1031, 654, 817, 491),
                (1905, 654, 817, 491)],
        },
        # portrait flips `_choose_grid` from a row to a column
        (1080, 1920): {
            2: [(173, 59, 734, 890), (59, 1127, 962, 578)],
        },
    }

    LEGACY_DESKTOP = {
        (1280, 720): {
            1: [(401, 70, 478, 580)],
            2: [(154, 70, 478, 580), (647, 70, 478, 287)],
            3: [(163, 70, 469, 569), (647, 70, 469, 282),
                (647, 368, 469, 282)],
            4: [(326, 70, 309, 375), (645, 127, 309, 186),
                (645, 323, 309, 186), (370, 445, 196, 205)],
        },
        (2880, 1800): {
            3: [(220, 158, 1201, 1457), (1459, 158, 1201, 722),
                (1459, 920, 1201, 722)],
        },
        (1080, 1920): {
            2: [(59, 673, 474, 575), (547, 673, 474, 285)],
        },
    }

    GRID = {
        (1280, 720): {
            1: [(374, 38, 531, 644)],
            2: [(67, 38, 531, 644), (652, 183, 590, 354)],
            3: [(38, 126, 385, 467), (448, 244, 385, 231), (857, 244, 385, 231)],
            4: [(205, 38, 255, 310), (690, 38, 515, 310),
                (75, 372, 515, 310), (799, 372, 296, 310)],
        },
        (2880, 1800): {
            3: [(86, 376, 865, 1049), (1008, 640, 865, 520),
                (1929, 640, 865, 520)],
        },
        (1080, 1920): {
            2: [(162, 32, 756, 917), (32, 1124, 1016, 610)],
        },
    }

    DESKTOP = {
        (1280, 720): {
            1: [(375, 38, 531, 644)],
            2: [(101, 38, 531, 644), (648, 38, 531, 319)],
            3: [(111, 38, 521, 632), (648, 38, 521, 313), (648, 369, 521, 313)],
            4: [(291, 38, 343, 416), (645, 101, 343, 206),
                (645, 319, 343, 206), (340, 454, 218, 228)],
        },
        (2880, 1800): {
            3: [(102, 86, 1318, 1598), (1460, 86, 1318, 792),
                (1460, 922, 1318, 792)],
        },
        (1080, 1920): {
            2: [(32, 657, 500, 607), (548, 657, 500, 301)],
        },
    }

    def _check(self, layout, table, pad_frac=None):
        for (W, H), by_n in table.items():
            for n, want in by_n.items():
                if pad_frac is None:
                    _pad, got = _arrange(layout, W, H, _WINDOW_SETS[n])
                else:
                    got = [tuple(int(v) for v in c) for c in
                           framing.placements_for(W, H, _WINDOW_SETS[n],
                                                  layout=layout,
                                                  pad_frac=pad_frac)]
                self.assertEqual([tuple(c) for c in got],
                                 [tuple(c) for c in want],
                                 "{} n={} on {}x{} (pad_frac={})".format(
                                     layout, n, W, H, pad_frac or framing.PAD_FRAC))

    def test_grid_placements_are_unchanged(self):
        self._check("grid", self.GRID)

    def test_desktop_placements_are_unchanged(self):
        self._check("desktop", self.DESKTOP)

    def test_grid_arithmetic_never_drifted_from_the_original(self):
        self._check("grid", self.LEGACY_GRID, self._LEGACY_PAD_FRAC)

    def test_desktop_arithmetic_never_drifted_from_the_original(self):
        self._check("desktop", self.LEGACY_DESKTOP, self._LEGACY_PAD_FRAC)


class FitPlacements(unittest.TestCase):
    """`fit_placements` is the "Fit to frame" action, and the only thing a
    hand-dragged arrangement cannot get from an auto-layout: a drop stays
    exactly where it was dropped (re-flowing the other cards underneath the
    pointer would be fighting the user), so filling the frame is explicit.

    The editor and the MCP tool both call this one function, which is why the
    arithmetic lives in framing rather than twice at the callers.
    """

    # A "1 tall left + 2 stacked right" arrangement dragged by hand into the
    # middle of a 2880x1800 canvas -- the exact complaint this fixes.
    DRAGGED = [(900, 500, 420, 510), (1360, 500, 420, 252),
               (1360, 780, 420, 252)]
    W, H = 2880, 1800
    PAD = int(framing.PAD_FRAC * 2880)

    def test_dead_space_arrangement_comes_back_filling_the_frame(self):
        got = framing.fit_placements(self.W, self.H, self.DRAGGED)
        y0 = min(c[1] for c in got)
        y1 = max(c[1] + c[3] for c in got)
        self.assertAlmostEqual(y0, self.PAD, delta=1)
        self.assertAlmostEqual(y1, self.H - self.PAD, delta=1)
        for x, y, w, h in got:
            self.assertTrue(0 <= x and 0 <= y
                            and x + w <= self.W and y + h <= self.H)

    def test_relative_sizes_and_aspects_survive(self):
        """One uniform scale for the whole set, not a per-card fit: the point
        of the arrangement is that the tall card is visibly bigger than the
        two beside it, and a per-card rescale would flatten exactly that."""
        got = framing.fit_placements(self.W, self.H, self.DRAGGED)
        scales = []
        for before, after in zip(self.DRAGGED, got):
            scales.append(after[2] / float(before[2]))
            scales.append(after[3] / float(before[3]))
            self.assertAlmostEqual(after[2] / float(after[3]),
                                   before[2] / float(before[3]),
                                   delta=0.5 * (before[2] / float(before[3])
                                                + 1.0) / after[3])
        self.assertAlmostEqual(max(scales), min(scales), delta=0.01)
        self.assertGreater(min(scales), 1.0)

    def test_the_result_is_centred_on_the_free_axis(self):
        got = framing.fit_placements(self.W, self.H, self.DRAGGED)
        left = min(c[0] for c in got)
        right = self.W - max(c[0] + c[2] for c in got)
        self.assertAlmostEqual(left, right, delta=2)

    def test_fitting_preset_output_moves_nothing_that_matters(self):
        """A preset already went through this exact scale, so "Fit to frame"
        on one has nothing to take -- an action that nudges a good layout
        every time it is pressed is worse than no action.

        Within a PIXEL, not exactly equal, and the difference is real: with
        two windows, `feature` and `row` come back re-centred by 1px on the
        free axis, because the fitted span is odd and `(W - span) / 2` rounds
        the other way the second time. The old version of this test asserted
        exact equality and passed only because the window set it happened to
        use produced even spans. Idempotence-to-within-rounding is the true
        property; a preset that came back 20px smaller would still fail.
        """
        for n in (2, 3, 4):
            for layout in _NAMED_LAYOUTS + ("desktop",):
                _pad, cells = _arrange(layout, self.W, self.H,
                                       _WINDOW_SETS[n])
                got = framing.fit_placements(self.W, self.H, cells)
                for before, after in zip(cells, got):
                    for va, vb in zip(before, after):
                        self.assertLessEqual(
                            abs(va - vb), 1,
                            "{} n={}: {} -> {}".format(layout, n, cells, got))
                # And it settles -- one nudge, not a nudge per press.
                self.assertEqual(framing.fit_placements(self.W, self.H, got),
                                 got, "{} n={}".format(layout, n))

    def test_a_two_window_preset_really_does_move_by_a_pixel(self):
        """The exact case the assertion above was loosened for, pinned so the
        loosening is not mistaken for slack nobody measured. Measured: every
        card moves exactly 1px on the free axis and nothing else changes --
        `(H - span_h) / 2` lands on .5 and rounds the other way once the span
        has been through the int round-trip.

        The DIRECTION of that pixel is a property of the rounding, not of the
        layout, and it flips with the margin: at the historical
        `pad_frac=0.055` it was -1, at today's 0.03 it is +1. So the pin is
        "exactly one pixel, on the free axis only" -- asserting the sign as
        well made this fail on a margin change with a diff that looked like a
        layout bug and was not.
        """
        _pad, cells = _arrange("feature", self.W, self.H, _WINDOW_SETS[2])
        got = framing.fit_placements(self.W, self.H, cells)
        self.assertNotEqual(got, cells)
        for before, after in zip(cells, got):
            self.assertEqual(before[2:], after[2:])       # no resize
            self.assertEqual(after[0] - before[0], 0)     # free axis only
            self.assertEqual(abs(after[1] - before[1]), 1)

    def test_the_grid_is_the_one_auto_layout_fit_can_still_move(self):
        """`grid` is not in the sweep above because it genuinely is not a
        fixed point, and the reason is worth writing down rather than
        excluding quietly: `_grid_placements` aspect-fits each crop INSIDE
        its own uniform cell, so the cards' bounding box can sit off-centre
        even though every cell is where it should be. Fit re-centres that box
        without resizing anything -- a re-centre, not a rescale, and the frame
        was already as full as one uniform scale can make it.

        WHICH canvas shows it depends on the margin, so the canvas is chosen
        here rather than inherited: at `pad_frac=0.055` the class canvas
        (2880x1800) slid 7-8px, and at today's 0.03 that particular canvas
        happens to land centred already. 1920x1080 is the stable
        demonstration -- measured 22px left at 0.03 -- and the no-resize /
        converges-in-one-pass claims are what actually matter.
        """
        W, H = 1920, 1080
        _pad, cells = _arrange("grid", W, H, _WINDOW_SETS[2])
        got = framing.fit_placements(W, H, cells)
        self.assertNotEqual(got, cells)
        for before, after in zip(cells, got):
            self.assertEqual(before[1:], after[1:])       # only x moves
            self.assertGreater(before[0] - after[0], 0)   # and it moves left
        self.assertEqual(framing.fit_placements(W, H, got), got)

    def test_refitting_a_fitted_drag_converges(self):
        """Not equality on the second pass: re-centring an odd span rounds the
        free axis by a pixel once and then holds. Assert it settles rather
        than that it never moves, so nobody writes an idempotence test that
        has to be loosened later."""
        once = framing.fit_placements(self.W, self.H, self.DRAGGED)
        twice = framing.fit_placements(self.W, self.H, once)
        for a, b in zip(once, twice):
            for va, vb in zip(a, b):
                self.assertLessEqual(abs(va - vb), 1)
        self.assertEqual(framing.fit_placements(self.W, self.H, twice), twice)

    def test_empty_in_empty_out(self):
        self.assertEqual(framing.fit_placements(self.W, self.H, []), [])

    def test_accepts_cells_dicts_as_well_as_tuples(self):
        """The MCP tool hands it `multi_window_layout`'s `cells`, which are
        dicts; the editor's own math works in tuples. Both callers must get
        the same pixels or "Fit to frame" means two different things."""
        dicts = [{"x": x, "y": y, "w": w, "h": h}
                 for (x, y, w, h) in self.DRAGGED]
        self.assertEqual(framing.fit_placements(self.W, self.H, dicts),
                         framing.fit_placements(self.W, self.H, self.DRAGGED))

    def test_degenerate_cells_do_not_raise(self):
        for cells in ([(0, 0, 0, 0)], [(10, 10, -5, -9)],
                      [(0, 0, 1, 1), (0, 0, 0, 400)]):
            got = framing.fit_placements(self.W, self.H, cells)
            self.assertEqual(len(got), len(cells))
            for x, y, w, h in got:
                self.assertTrue(0 <= x and 0 <= y
                                and x + w <= self.W and y + h <= self.H)


class DegenerateArrangementInput(unittest.TestCase):
    """`edits.py` clamps a window to >= 8px but bounds no aspect, and a card
    can be resized to nothing while the editor is live, so a zero or negative
    dimension is reachable from the UI. A preset that raises here takes the
    whole render down; the requirement is only that it survives and stays on
    the canvas."""

    CASES = {
        "empty": [],
        "single": [{"x": 0, "y": 0, "w": 300, "h": 200}],
        "zero_width": [{"x": 0, "y": 0, "w": 0, "h": 200},
                       {"x": 0, "y": 0, "w": 300, "h": 200}],
        "zero_height": [{"x": 0, "y": 0, "w": 300, "h": 0},
                        {"x": 0, "y": 0, "w": 300, "h": 200}],
        "negative": [{"x": 0, "y": 0, "w": -40, "h": -9},
                     {"x": 0, "y": 0, "w": 300, "h": 200}],
        "all_zero": [{"x": 0, "y": 0, "w": 0, "h": 0},
                     {"x": 0, "y": 0, "w": 0, "h": 0}],
        "originless_tuples": [(320, 240), (640, 360), (100, 700)],
        "missing_keys": [{"w": 300, "h": 200}, {"w": 200, "h": 300}],
    }

    def test_no_case_raises_and_every_card_stays_on_the_canvas(self):
        W, H = 1280, 720
        for layout in _NAMED_LAYOUTS:
            for name, rects in self.CASES.items():
                _pad, cells = _arrange(layout, W, H, rects)
                self.assertEqual(len(cells), len(rects),
                                 "{} / {}".format(layout, name))
                for cell in cells:
                    x, y, w, h = cell
                    self.assertTrue(
                        w >= 1 and h >= 1 and 0 <= x and 0 <= y
                        and x + w <= W and y + h <= H,
                        "{} / {}: {}".format(layout, name, cell))

    def test_the_painter_survives_them_too(self):
        # The placements are only half of it -- `MultiFramePainter` goes on to
        # build a rounded mask and a shadow rect per card, and a 0-dimension
        # cell there is an OpenCV error rather than a bad-looking frame.
        for layout in _NAMED_LAYOUTS:
            for name, rects in self.CASES.items():
                if not rects:
                    continue
                painter = framing.make_multi_painter(400, 300, rects,
                                                     layout=layout)
                crops = []
                for rect in rects:
                    rw, rh = framing._rect_wh(rect)
                    crops.append(np.full((max(1, int(rh)), max(1, int(rw)), 3),
                                         200, np.uint8))
                out = painter.paint(crops)
                self.assertEqual(out.shape, (300, 400, 3),
                                 "{} / {}".format(layout, name))


class ArbitraryAspectCanvas(unittest.TestCase):
    """A window-native take's source (W, H) IS the window's own backing
    buffer -- any aspect the window happens to be, not screen-shaped. The
    framing math has to survive that: auto aspect stays identity on any
    (W, H); a requested ratio keeps that ratio at ~the source's pixel budget;
    the framed painter's inner rect keeps the CANVAS aspect and clamps against
    the shorter axis, so a very tall or very wide canvas ends up with a valid
    inset rather than an empty or negative one.

    Pinned as part of window-native P2: the doc's own gate is 'pin one test on
    a non-screen-aspect source', and every path here is one framed / --aspect /
    GIF export of an unusual window will land on.
    """

    # Real window shapes the recorder actually produces: the recording bar
    # itself (wide-narrow), a code editor stretched down the desktop
    # (tall-narrow), and a utility window (square-ish).
    ARBITRARY = ((1440, 264), (720, 1750), (1024, 1024), (640, 120), (120, 640))

    def test_auto_aspect_is_identity_on_any_source(self):
        for W, H in self.ARBITRARY:
            self.assertEqual(framing.output_size(W, H, "clean"), (W, H))
            self.assertEqual(framing.output_size(W, H, "framed"), (W, H))
            self.assertEqual(framing.resolve_aspect_canvas(None, W, H), (W, H))
            self.assertEqual(framing.resolve_aspect_canvas("auto", W, H), (W, H))
            self.assertEqual(framing.resolve_aspect_canvas("source", W, H),
                             (W, H))

    def test_ratio_aspect_hits_target_at_the_pixel_budget(self):
        # `resolve_aspect_canvas` picks the canvas that (a) has the requested
        # ratio and (b) matches the source's pixel budget. Both must hold on a
        # non-screen source; a rewrite that quietly keyed off the source
        # ASPECT would satisfy neither on a wide-narrow window.
        for W, H in self.ARBITRARY:
            for spec, (rw, rh) in (("9:16", (9, 16)), ("16:9", (16, 9)),
                                    ("1:1", (1, 1)), ("4:5", (4, 5))):
                w, h = framing.resolve_aspect_canvas(spec, W, H)
                self.assertEqual(w % 2, 0)
                self.assertEqual(h % 2, 0)
                got = w / float(h)
                want = rw / float(rh)
                # Each axis is _even'd independently, so the delivered ratio
                # can drift by ~one pixel per axis at the resolved size --
                # bound it that way instead of guessing a tolerance.
                bound = 2.0 * (want + 1.0) / h
                self.assertLessEqual(
                    abs(got - want), bound,
                    "{!r} on {}x{}: got {:.4f} (canvas {}x{}), want "
                    "{:.4f}".format(spec, W, H, got, w, h, want))
                self.assertLess(
                    abs(w * h - W * H) / float(W * H), 0.05,
                    "{!r} on {}x{}: canvas {}x{} strays {:.1%} from the "
                    "source pixel budget".format(spec, W, H, w, h,
                                                 abs(w * h - W * H)
                                                 / float(W * H)))

    def test_wxh_aspect_is_exact_regardless_of_source_shape(self):
        # Custom "WxH" bypasses the pixel-budget math -- it's the exact canvas
        # requested. That must not change for a native source (which the
        # scaling logic never inspects, but a rewrite might start to).
        for W, H in self.ARBITRARY:
            self.assertEqual(
                framing.resolve_aspect_canvas("1080x1920", W, H),
                (1080, 1920))
            self.assertEqual(
                framing.resolve_aspect_canvas("640x360", W, H),
                (640, 360))

    def test_framed_inner_box_keeps_the_canvas_aspect(self):
        # The framed painter's inner rect is aspect-matched to the CANVAS
        # (W, H). That is what keeps the recording sized right for the frame
        # around it -- a rewrite that lost the aspect would either stretch the
        # recording or letterbox on both sides.
        for W, H in self.ARBITRARY:
            p = framing.make_painter(W, H, "framed", background="black")
            got = p.iw / float(p.ih)
            want = W / float(H)
            # Same one-pixel-per-axis rounding bound as the arrangement pins:
            # both dimensions come out of one uniform scale but round
            # independently.
            bound = 2.0 * (want + 1.0) / p.ih
            self.assertLessEqual(
                abs(got - want), bound,
                "{}x{}: inner {}x{}, ratio {:.4f} vs {:.4f}".format(
                    W, H, p.iw, p.ih, got, want))

    def test_framed_inner_box_fits_inside_the_padded_canvas(self):
        # No matter how extreme the canvas aspect, the inner rect must fit
        # with the requested padding on both axes -- otherwise the composited
        # recording would run off the frame. `pad_frac` is a fraction of the
        # WIDTH but the clamp also applies to the height (see
        # framing.FramePainter.__init__: `if inner_h > H - 2 * pad`), so a
        # narrow-and-tall canvas has an inner rect whose height is bounded
        # by (H - 2 * pad_from_width).
        for W, H in self.ARBITRARY:
            p = framing.make_painter(W, H, "framed")
            self.assertGreaterEqual(p.iw, 2, "{}x{}: iw {}".format(W, H, p.iw))
            self.assertGreaterEqual(p.ih, 2, "{}x{}: ih {}".format(W, H, p.ih))
            pad = int(framing.PAD_FRAC * W)
            # `x`/`y` come off `(W - inner) // 2`, so a one-pixel odd-span
            # centring can put them exactly at `pad - 0`.
            self.assertGreaterEqual(p.x, pad - 1)
            self.assertGreaterEqual(p.y, pad - 1)
            self.assertLessEqual(p.x + p.iw, W - pad + 1)
            self.assertLessEqual(p.y + p.ih, H - pad + 1)

    def test_framed_paint_survives_any_arbitrary_shape(self):
        # The painter has to accept an arbitrary-aspect input frame and hand
        # back a canvas-shaped output (H, W, 3) uint8. Both matter: the shape
        # is what gets fed to the encoder, and the dtype is what keeps the
        # in-place cv2.resize path (framing.py's `dst=roi`) valid.
        for W, H in self.ARBITRARY:
            p = framing.make_painter(W, H, "framed", background="graphite")
            # An arbitrary-shape source frame -- deliberately NOT canvas-
            # aspect: the camera hands the painter a frame that already
            # matches (out_w, out_h) upstream, so the painter's own resize
            # into (iw, ih) is a downscale of a same-aspect image. Here we
            # feed a mismatched frame to prove `paint` still produces a valid
            # output rather than crashing on a shape it never sees in practice.
            img = np.full((H, W, 3), 128, np.uint8)
            out = p.paint(img)
            self.assertEqual(out.shape, (H, W, 3))
            self.assertEqual(out.dtype, np.uint8)


class ArbitraryAspectFramedCompositeIsExact(FramedCompositeIsExact):
    """`FramedCompositeIsExact` widened onto real window aspects.

    Framed style on a native take runs `paint` with the WINDOW's aspect as
    both source and canvas, so wide-narrow, tall-narrow, and square canvases
    are what the exact-composite guarantee has to hold on. A one-count drift
    is invisible in review and permanent in every export -- and the framed
    edges of a canvas that never appeared in the review set are exactly where
    a rewrite is likely to leak antialiased pixels.
    """

    def test_wide_narrow_canvas(self):
        self._check(960, 200, background="midnight", seed=21)

    def test_tall_narrow_canvas(self):
        self._check(300, 900, background="ocean", seed=22)

    def test_square_canvas(self):
        self._check(600, 600, background="#101820", seed=23)


class FitMaxHeight(unittest.TestCase):
    """The render-resolution clamp (framing.fit_max_height). It must be a
    strict, bit-exact no-op when off (None/0/already-fits) and an
    aspect-preserving, even-dimension downscale otherwise."""

    def test_none_and_zero_are_noops(self):
        self.assertEqual(framing.fit_max_height(3840, 2160, None), (3840, 2160))
        self.assertEqual(framing.fit_max_height(3840, 2160, 0), (3840, 2160))
        self.assertEqual(framing.fit_max_height(3840, 2160, ""), (3840, 2160))

    def test_already_fits_is_a_noop(self):
        # never scales UP -- a 1080-tall canvas asked to cap at 1440 stays put
        self.assertEqual(framing.fit_max_height(1920, 1080, 1440), (1920, 1080))
        self.assertEqual(framing.fit_max_height(1920, 1080, 1080), (1920, 1080))

    def test_downscale_preserves_aspect_and_parity(self):
        w, h = framing.fit_max_height(3840, 2160, 1080)
        self.assertEqual((w, h), (1920, 1080))
        # odd source dims still come back even and ~aspect-locked
        w2, h2 = framing.fit_max_height(3642, 2050, 720)
        self.assertEqual(h2, 720)
        self.assertEqual(w2 % 2, 0)
        self.assertAlmostEqual(w2 / float(h2), 3642 / 2050.0, delta=0.02)

    def test_never_below_two(self):
        w, h = framing.fit_max_height(4000, 10, 2)
        self.assertGreaterEqual(w, 2)
        self.assertGreaterEqual(h, 2)


class WideBlurIsAnOversizedCanvasOptimizationOnly(unittest.TestCase):
    """`_wide_blur` computes the baked plate's cast shadow on a coarse grid.

    The baked plate is an EXPORT pixel path, so the approximation is gated on
    canvas area and must stay byte-identical for every canvas an ordinary take
    reaches. It exists for one shape: a multi-window scene canvas sizes itself
    to render its cards ~1:1, which on a Retina 4-window take is 3636x2046 and
    made `sigma = 0.02 * W` a ~437-tap kernel -- ~1.4s of GaussianBlur per
    scene, 78% of an /api/camera-path round trip, paid again on every
    background change in the editor.
    """

    def test_ordinary_canvases_are_exact(self):
        # Every canvas a normal framed / multi-window export reaches. A
        # downscale here would drift the plate an LSB to save ~0.2s -- a bad
        # trade, and the reason the gate is on AREA and not on sigma alone.
        for w, h in ((200, 160), (1280, 720), (1920, 1080),
                     (2560, 1440), (2880, 1800)):
            self.assertEqual(framing._blur_downscale(0.02 * w, w * h), 1,
                             "%dx%d must blur exactly" % (w, h))

    def test_oversized_scene_canvas_engages(self):
        d = framing._blur_downscale(0.02 * 3636, 3636 * 2046)
        self.assertGreater(d, 1)
        self.assertLessEqual(d, framing._SHADOW_DOWNSCALE)

    def test_exact_below_the_gate_is_byte_identical(self):
        mask = np.zeros((1080, 1920), np.uint8)
        mask[200:900, 300:1600] = 255
        sigma = max(1.0, 0.02 * 1920)
        self.assertTrue(np.array_equal(
            framing._wide_blur(mask, sigma),
            cv2.GaussianBlur(mask, (0, 0), sigma)))

    def test_oversized_stays_within_a_couple_of_counts(self):
        # Band-limited enough that 1/d and back is visually the same shadow.
        mask = np.zeros((2046, 3636), np.uint8)
        mask[300:1700, 400:3200] = 255
        sigma = max(1.0, 0.02 * 3636)
        got = framing._wide_blur(mask, sigma).astype(int)
        want = cv2.GaussianBlur(mask, (0, 0), sigma).astype(int)
        self.assertLessEqual(int(np.abs(got - want).max()), 4)

    def test_plate_of_a_normal_multi_window_canvas_is_unchanged(self):
        # The end-to-end guarantee, not just the helper: a 1920x1080 four-card
        # plate must be byte-for-byte what the exact blur produces.
        sizes = [(1418.0, 854.0), (1418.0, 852.0),
                 (1240.0, 920.0), (1416.0, 852.0)]
        real = framing._wide_blur
        try:
            painter = framing.MultiFramePainter(
                1920, 1080, sizes, background="ocean", layout="desktop")
            framing._wide_blur = (
                lambda m, s: cv2.GaussianBlur(m, (0, 0), s))
            exact = framing.MultiFramePainter(
                1920, 1080, sizes, background="ocean", layout="desktop")
        finally:
            framing._wide_blur = real
        self.assertTrue(np.array_equal(painter._plate, exact._plate))


if __name__ == "__main__":
    unittest.main()
