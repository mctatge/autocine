"""Record-time multi-window capture, end to end, without permissions.

Four separable things get pinned here:

  * `framing._desktop_placements` -- the layout that keeps a desktop
    arrangement instead of gridding it, and the grid path staying BIT-EXACT
    through the refactor that made room for it;
  * `record.Recorder`'s `capture_windows` meta block, including the rule that
    a pick of ONE collapses back into today's single-window crop;
  * the `edits.windows` schema additions (`window_id`, `layout`), which must
    be invisible when unset or every window entry ever saved changes shape;
  * `render._build_grid_tracks` binding by id when a card carries one.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from autocine import edits, framing, record as rec, render as ren  # noqa: E402


def _entry(wid, x, y, w, h, app="App", title="Title"):
    return {"id": wid, "app": app, "title": title, "label": app + " — " + title,
            "x": float(x), "y": float(y), "w": float(w), "h": float(h),
            "display_id": 1, "display_origin": [0.0, 0.0], "main_display": True}


def _placement_overlaps(cells):
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            a, b = cells[i], cells[j]
            ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
            oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
            if ox > 0 and oy > 0:
                return True
    return False


class DesktopPlacement(unittest.TestCase):
    """The layout the user actually asked for: their desktop, tidied."""

    W, H, PAD, GAP = 1280, 720, 70, 25

    def _place(self, rects):
        return framing._desktop_placements(self.W, self.H, rects,
                                           self.PAD, self.GAP)

    def test_overlapping_windows_come_out_separated(self):
        cells = self._place([{"x": 100, "y": 100, "w": 800, "h": 600},
                             {"x": 700, "y": 300, "w": 800, "h": 600}])
        self.assertFalse(_placement_overlaps(cells))

    def test_left_stays_left_and_top_stays_top(self):
        cells = self._place([{"x": 0, "y": 0, "w": 700, "h": 900},
                             {"x": 760, "y": 0, "w": 600, "h": 400},
                             {"x": 760, "y": 430, "w": 600, "h": 450}])
        self.assertLess(cells[0][0], cells[1][0])       # 1 left of 2
        self.assertLess(cells[1][1], cells[2][1])       # 2 above 3

    def test_each_cards_aspect_survives(self):
        rects = [{"x": 0, "y": 0, "w": 800, "h": 200},
                 {"x": 900, "y": 0, "w": 300, "h": 700}]
        for rect, cell in zip(rects, self._place(rects)):
            src = rect["w"] / float(rect["h"])
            got = cell[2] / float(cell[3])
            self.assertAlmostEqual(src, got, delta=src * 0.03)

    def test_relative_size_survives(self):
        """A big window must stay visibly bigger than a small one -- that's
        the difference between 'like my desktop' and 'a grid'."""
        cells = self._place([{"x": 0, "y": 0, "w": 900, "h": 700},
                             {"x": 1000, "y": 0, "w": 300, "h": 240}])
        self.assertGreater(cells[0][2] * cells[0][3],
                           4 * cells[1][2] * cells[1][3])

    def test_dead_space_is_squeezed_out(self):
        """Two small windows at opposite corners of a big desktop should end
        up filling the canvas, not sitting as two dots in the corners."""
        cells = self._place([{"x": 0, "y": 0, "w": 300, "h": 200},
                             {"x": 1100, "y": 700, "w": 300, "h": 200}])
        span = max(c[0] + c[2] for c in cells) - min(c[0] for c in cells)
        self.assertGreater(span, (self.W - 2 * self.PAD) * 0.9)

    def test_everything_lands_inside_the_canvas(self):
        cells = self._place([{"x": -200, "y": -100, "w": 900, "h": 700},
                             {"x": 1300, "y": 800, "w": 400, "h": 300}])
        for x, y, w, h in cells:
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, self.W)
            self.assertLessEqual(y + h, self.H)

    def test_single_window_is_centred_not_gridded(self):
        cells = self._place([{"x": 300, "y": 200, "w": 800, "h": 600}])
        self.assertEqual(len(cells), 1)
        x, y, w, h = cells[0]
        self.assertAlmostEqual(x + w / 2.0, self.W / 2.0, delta=2)
        self.assertAlmostEqual(y + h / 2.0, self.H / 2.0, delta=2)

    def test_identical_rects_are_deterministic(self):
        rects = [{"x": 50, "y": 50, "w": 400, "h": 300},
                 {"x": 50, "y": 50, "w": 400, "h": 300}]
        self.assertEqual(self._place(rects), self._place(rects))
        self.assertFalse(_placement_overlaps(self._place(rects)))

    def test_one_window_containing_another_still_separates(self):
        cells = self._place([{"x": 0, "y": 0, "w": 1440, "h": 900},
                             {"x": 300, "y": 200, "w": 400, "h": 300}])
        self.assertFalse(_placement_overlaps(cells))

    def test_originless_rects_fall_back_to_the_grid(self):
        """`make_multi_painter` also takes plain (w, h) tuples. Those carry no
        desktop arrangement, so treating the missing origin as (0, 0) would
        stack every card in the corner -- return None and let the caller grid
        them instead."""
        self.assertIsNone(framing._desktop_placements(
            self.W, self.H, [(320, 240), (320, 240)], self.PAD, self.GAP))


class GridPathUnchanged(unittest.TestCase):
    """The grid was extracted, not rewritten. Same integers out."""

    def test_extracted_grid_matches_the_painter(self):
        rects = [{"x": 0, "y": 0, "w": 800, "h": 600},
                 {"x": 0, "y": 0, "w": 400, "h": 400},
                 {"x": 0, "y": 0, "w": 640, "h": 480}]
        painter = framing.make_multi_painter(1920, 1080, rects)
        pad = int(framing.PAD_FRAC * 1920)
        gap = max(0, int(0.02 * 1920))
        placements = framing._grid_placements(1920, 1080, rects, pad, gap)
        got = [(c["x"], c["y"], c["w"], c["h"]) for c in painter.cells]
        self.assertEqual(got, placements)

    def test_default_layout_is_the_grid(self):
        rects = [{"x": 0, "y": 0, "w": 800, "h": 600},
                 {"x": 900, "y": 0, "w": 400, "h": 400}]
        default = framing.make_multi_painter(1280, 720, rects)
        explicit = framing.make_multi_painter(1280, 720, rects, layout="grid")
        self.assertEqual(default.cells, explicit.cells)

    def test_desktop_layout_actually_differs(self):
        rects = [{"x": 0, "y": 0, "w": 700, "h": 800},
                 {"x": 900, "y": 0, "w": 400, "h": 300}]
        grid = framing.make_multi_painter(1280, 720, rects, layout="grid")
        desk = framing.make_multi_painter(1280, 720, rects, layout="desktop")
        self.assertNotEqual(grid.cells, desk.cells)

    def test_unknown_layout_name_falls_back_to_the_grid(self):
        rects = [{"x": 0, "y": 0, "w": 800, "h": 600},
                 {"x": 900, "y": 0, "w": 400, "h": 400}]
        grid = framing.make_multi_painter(1280, 720, rects, layout="grid")
        odd = framing.make_multi_painter(1280, 720, rects, layout="nonsense")
        self.assertEqual(grid.cells, odd.cells)


class ManualCardPlacement(unittest.TestCase):
    """A dragged card wins over whatever the auto-layout wanted."""

    def test_layout_override_places_the_card(self):
        rects = [{"x": 0, "y": 0, "w": 800, "h": 600,
                  "layout": {"x": 0.5, "y": 0.25, "w": 0.4, "h": 0.5}},
                 {"x": 900, "y": 0, "w": 400, "h": 400}]
        cells = framing.make_multi_painter(1000, 800, rects,
                                           layout="desktop").cells
        self.assertEqual((cells[0]["x"], cells[0]["y"]), (500, 200))
        self.assertEqual((cells[0]["w"], cells[0]["h"]), (400, 400))

    def test_override_applies_in_grid_mode_too(self):
        rects = [{"x": 0, "y": 0, "w": 800, "h": 600,
                  "layout": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}},
                 {"x": 900, "y": 0, "w": 400, "h": 400}]
        cells = framing.make_multi_painter(1000, 800, rects,
                                           layout="grid").cells
        self.assertEqual((cells[0]["x"], cells[0]["y"]), (100, 80))

    def test_override_is_clamped_into_the_canvas(self):
        rects = [{"x": 0, "y": 0, "w": 800, "h": 600,
                  "layout": {"x": 0.9, "y": 0.9, "w": 0.5, "h": 0.5}}]
        cells = framing.make_multi_painter(1000, 800, rects).cells
        self.assertLessEqual(cells[0]["x"] + cells[0]["w"], 1000)
        self.assertLessEqual(cells[0]["y"] + cells[0]["h"], 800)

    def test_malformed_override_is_ignored_not_fatal(self):
        rects = [{"x": 0, "y": 0, "w": 800, "h": 600,
                  "layout": {"x": 0.1}},
                 {"x": 900, "y": 0, "w": 400, "h": 400}]
        plain = framing.make_multi_painter(1000, 800, [
            {"x": 0, "y": 0, "w": 800, "h": 600},
            {"x": 900, "y": 0, "w": 400, "h": 400}])
        self.assertEqual(framing.make_multi_painter(1000, 800, rects).cells,
                         plain.cells)


class WindowEntrySchema(unittest.TestCase):
    """New keys must be INVISIBLE when unset, or every saved window entry
    changes shape and every off-switch claim in the docs stops being true."""

    def test_plain_entry_still_normalizes_to_five_keys(self):
        got = edits.normalize_edits({"windows": [{"x": 1, "y": 2, "w": 3, "h": 4}]})
        self.assertEqual(set(got["windows"][0]), {"id", "x", "y", "w", "h"})

    def test_window_id_round_trips(self):
        got = edits.normalize_edits(
            {"windows": [{"x": 1, "y": 2, "w": 3, "h": 4, "window_id": 9856}]})
        self.assertEqual(got["windows"][0]["window_id"], 9856)

    def test_window_id_of_true_is_rejected(self):
        """`True` is an int in Python; a window id that came from a JSON
        `true` is a bug wearing a valid value."""
        got = edits.normalize_edits(
            {"windows": [{"x": 1, "y": 2, "w": 3, "h": 4, "window_id": True}]})
        self.assertNotIn("window_id", got["windows"][0])

    def test_garbage_window_id_is_dropped(self):
        got = edits.normalize_edits(
            {"windows": [{"x": 1, "y": 2, "w": 3, "h": 4, "window_id": "nope"}]})
        self.assertNotIn("window_id", got["windows"][0])

    def test_layout_round_trips_and_clamps(self):
        got = edits.normalize_edits({"windows": [
            {"x": 1, "y": 2, "w": 3, "h": 4,
             "layout": {"x": -0.5, "y": 0.25, "w": 2.0, "h": 0.5}}]})
        lay = got["windows"][0]["layout"]
        self.assertEqual(lay["x"], 0.0)
        self.assertEqual(lay["y"], 0.25)
        self.assertEqual(lay["w"], 1.0)
        self.assertEqual(lay["h"], 0.5)

    def test_partial_layout_is_dropped(self):
        got = edits.normalize_edits(
            {"windows": [{"x": 1, "y": 2, "w": 3, "h": 4, "layout": {"x": 0.1}}]})
        self.assertNotIn("layout", got["windows"][0])

    def test_window_layout_defaults_to_grid(self):
        self.assertEqual(edits.default_edits()["render"]["window_layout"], "grid")

    def test_unknown_window_layout_falls_back(self):
        got = edits.normalize_edits({"render": {"window_layout": "spiral"}})
        self.assertEqual(got["render"]["window_layout"], "grid")


class MaterializingFocusRanges(unittest.TestCase):
    """`edits.focus` is what makes every arc of the composition camera an
    object the user owns -- retimable, retargetable, deletable. Same
    materialize-once contract the auto zooms and the capture-window pick use,
    and the same reason: re-running would resurrect what they deleted."""

    RANGES = [{"start": 1.0, "end": 4.0, "card": 0, "level": "focus"},
              {"start": 4.0, "end": 8.0, "card": 0, "level": "full"}]

    def test_defaults_are_off_and_empty(self):
        d = edits.default_edits()
        self.assertFalse(d["render"]["window_focus"])
        self.assertEqual(d["focus"], [])
        self.assertFalse(d["focus_initialized"])

    def test_materializes_the_plan_once(self):
        doc, changed = edits.initialize_focus_ranges(edits.default_edits(),
                                                     self.RANGES)
        self.assertTrue(changed)
        norm = edits.normalize_edits(doc, duration=20.0)
        self.assertEqual([f["level"] for f in norm["focus"]],
                         ["focus", "full"])
        self.assertEqual([f["card"] for f in norm["focus"]], [0, 0])
        again, changed = edits.initialize_focus_ranges(doc, self.RANGES)
        self.assertFalse(changed)
        self.assertEqual(again["focus"], doc["focus"])

    def test_never_clobbers_arcs_the_user_already_edited(self):
        doc = edits.default_edits()
        doc["focus"] = [{"id": "focus-mine", "start": 2.0, "end": 3.0,
                         "card": 1, "level": "full"}]
        out, changed = edits.initialize_focus_ranges(doc, self.RANGES)
        self.assertTrue(changed)                    # the flag flipped
        self.assertEqual(out["focus"], doc["focus"])    # the arcs did not

    def test_materializes_even_with_the_toggle_off(self):
        """Inert, not absent: the plan is what the editor shows so the user
        can decide, and flipping the switch on must not need a reload."""
        doc, _ = edits.initialize_focus_ranges(edits.default_edits(),
                                               self.RANGES)
        self.assertFalse(doc["render"]["window_focus"])
        self.assertEqual(len(doc["focus"]), 2)

    def test_level_survives_normalization_symbolically(self):
        """`level` must stay "focus"/"full" rather than being coerced to a
        number -- both stages are derived from the card's own cell, so a
        symbol keeps meaning the right thing after a resize or an aspect
        change."""
        norm = edits.normalize_edits({"focus": self.RANGES}, duration=20.0)
        self.assertEqual(norm["focus"][1]["level"], "full")
        # A hand-set numeric level is a 0..1 emphasis AMOUNT (what
        # camera._normalize_focus_manual reads it as), so it is clamped to
        # [0, 1] here to match -- not floored at 1.0. Flooring silently turned
        # every numeric level into the strongest setting, inverting intent:
        # `level: 0.25` ("a quieter grow") rendered as "full". A quieter value
        # now survives as itself, an out-of-range one clamps, junk falls back.
        odd = edits.normalize_edits(
            {"focus": [{"start": 0, "end": 1, "card": 0, "level": 0.25},
                       {"start": 2, "end": 3, "card": 0, "level": 1.8},
                       {"start": 4, "end": 5, "card": 0, "level": "wat"}]},
            duration=20.0)
        self.assertEqual(odd["focus"][0]["level"], 0.25)  # survives, not 1.0
        self.assertEqual(odd["focus"][1]["level"], 1.0)   # 1.8 clamps to 1.0
        self.assertEqual(odd["focus"][2]["level"], "focus")

    def test_focus_is_timeline_content_not_preset_look(self):
        """Like zooms/suppressed/markers: a patch replaces the whole array,
        and switching presets must not take the arcs with it."""
        base = edits.normalize_edits({"focus": self.RANGES}, duration=20.0)
        merged = edits.merge_edits(base, {"render": {"style": "framed"}},
                                   duration=20.0)
        self.assertEqual(len(merged["focus"]), 2)
        cleared = edits.merge_edits(base, {"focus": []}, duration=20.0)
        self.assertEqual(cleared["focus"], [])


class SeedingFromAPick(unittest.TestCase):
    def test_seeds_windows_and_switches_to_desktop(self):
        specs = [{"x": 0, "y": 0, "w": 100, "h": 100, "window_id": 7},
                 {"x": 200, "y": 0, "w": 100, "h": 100, "window_id": 8}]
        doc, changed = edits.initialize_capture_windows(edits.default_edits(),
                                                        specs)
        self.assertTrue(changed)
        norm = edits.normalize_edits(doc)
        self.assertEqual([w["window_id"] for w in norm["windows"]], [7, 8])
        self.assertEqual(norm["render"]["window_layout"], "desktop")

    def test_is_idempotent(self):
        specs = [{"x": 0, "y": 0, "w": 100, "h": 100, "window_id": 7}]
        doc, _ = edits.initialize_capture_windows(edits.default_edits(), specs)
        again, changed = edits.initialize_capture_windows(doc, specs)
        self.assertFalse(changed)
        self.assertEqual(again["windows"], doc["windows"])

    def test_never_clobbers_windows_the_user_already_has(self):
        doc = edits.default_edits()
        doc["windows"] = [{"id": "window-mine", "x": 5, "y": 5, "w": 9, "h": 9}]
        out, changed = edits.initialize_capture_windows(
            doc, [{"x": 0, "y": 0, "w": 100, "h": 100, "window_id": 7}])
        self.assertTrue(changed)                       # the flag flipped
        self.assertEqual(out["windows"], doc["windows"])   # the cards did not

    def test_no_specs_leaves_the_grid_default(self):
        doc, changed = edits.initialize_capture_windows(edits.default_edits(), [])
        self.assertTrue(changed)
        self.assertEqual(doc["windows"], [])
        self.assertEqual(edits.normalize_edits(doc)["render"]["window_layout"],
                         "grid")

    def test_multi_native_defaults_to_desktop_with_no_specs(self):
        # An occlusion-free session has no specs and no `windows` array (its
        # cards ARE the recorded channels), but it should still open looking
        # like the screen -- not the tiny-thumbnail grid.
        doc, changed = edits.initialize_capture_windows(
            edits.default_edits(), [], multi_native=True)
        self.assertTrue(changed)
        self.assertEqual(doc["windows"], [])
        self.assertEqual(edits.normalize_edits(doc)["render"]["window_layout"],
                         "desktop")

    def test_multi_native_never_clobbers_a_user_layout(self):
        # Re-open (flag already set) is a no-op: the desktop flip only happens
        # on the first initialization, so a user who switched to grid keeps it.
        doc = edits.default_edits()
        doc["render"]["window_layout"] = "grid"
        doc["capture_windows_initialized"] = True
        out, changed = edits.initialize_capture_windows(
            doc, [], multi_native=True)
        self.assertFalse(changed)
        self.assertEqual(edits.normalize_edits(out)["render"]["window_layout"],
                         "grid")


class SeededCameraDefault(unittest.TestCase):
    """A session recorded with a multi-window PICK opens with the composition
    camera on -- the worked-in card grows, then the frame pushes in -- and
    with the inside-the-card camera off.

    Both are plain render options, so this is about the session's STARTING
    value only: `default_edits()` is untouched, and hand-drawn cards (which
    never come through the seeder) still start with neither camera. That
    split is the same one `window_layout` already makes."""

    SPECS = [{"x": 0, "y": 0, "w": 100, "h": 100, "window_id": 7},
             {"x": 200, "y": 0, "w": 100, "h": 100, "window_id": 8}]

    def _render(self, doc):
        return edits.normalize_edits(doc)["render"]

    def test_a_pick_opens_with_the_outer_frame_camera(self):
        doc, _ = edits.initialize_capture_windows(edits.default_edits(),
                                                  self.SPECS)
        r = self._render(doc)
        self.assertTrue(r["window_focus"])
        self.assertFalse(r["window_zoom"],
                         "the inside-the-card camera is a different look, "
                         "not the default one")

    def test_an_occlusion_free_take_opens_the_same_way(self):
        doc, _ = edits.initialize_capture_windows(edits.default_edits(), [],
                                                  multi_native=True)
        r = self._render(doc)
        self.assertTrue(r["window_focus"])
        self.assertFalse(r["window_zoom"])

    def test_the_global_default_is_untouched(self):
        r = edits.default_edits()["render"]
        self.assertFalse(r["window_focus"])
        self.assertFalse(r["window_zoom"])

    def test_hand_drawn_cards_start_with_neither_camera(self):
        # No specs and not native: nothing was picked off the screen, so the
        # steady side-by-side layout is still what you get.
        doc, _ = edits.initialize_capture_windows(edits.default_edits(), [])
        r = self._render(doc)
        self.assertFalse(r["window_focus"])
        self.assertFalse(r["window_zoom"])

    def test_reopening_never_re_enables_a_camera_the_user_turned_off(self):
        doc, _ = edits.initialize_capture_windows(edits.default_edits(),
                                                  self.SPECS)
        doc["render"]["window_focus"] = False
        again, changed = edits.initialize_capture_windows(doc, self.SPECS)
        self.assertFalse(changed)
        self.assertFalse(self._render(again)["window_focus"])


class RecorderMetaBlock(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="mw_meta_")

    def _rec(self, **kw):
        return rec.Recorder(self.dir, 2, **kw)

    def test_ordinary_take_has_neither_block(self):
        r = self._rec()
        self.assertIsNone(r._capture_window_meta())
        self.assertIsNone(r._capture_windows_meta())

    def test_a_pick_of_one_collapses_to_the_crop_path(self):
        """One window is better served by the crop, which keeps the auto-zoom
        camera alive -- so a one-element list must not become a composition."""
        r = self._rec(capture_windows=[_entry(11, 0, 0, 400, 300)])
        self.assertEqual(r.capture_windows, [])
        self.assertIsNotNone(r.capture_window)
        self.assertIsNone(r._capture_windows_meta())
        self.assertIsNotNone(r._capture_window_meta())

    def test_two_windows_write_the_multi_block_only(self):
        r = self._rec(capture_windows=[_entry(11, 0, 0, 400, 300),
                                       _entry(22, 500, 0, 400, 300)])
        self.assertIsNone(r._capture_window_meta())
        block = r._capture_windows_meta()
        self.assertEqual(block["units"], "points")
        self.assertEqual([w["id"] for w in block["windows"]], [11, 22])
        self.assertEqual(block["windows"][0]["rect"], [0.0, 0.0, 400.0, 300.0])

    def test_pick_order_is_card_order(self):
        r = self._rec(capture_windows=[_entry(33, 500, 0, 400, 300),
                                       _entry(11, 0, 0, 400, 300)])
        block = r._capture_windows_meta()
        self.assertEqual([w["id"] for w in block["windows"]], [33, 11])

    def test_track_is_per_window(self):
        r = self._rec(capture_windows=[_entry(11, 0, 0, 400, 300),
                                       _entry(22, 500, 0, 400, 300)])
        r._win_last_rect[11] = [0, 0, 400, 300]      # only this one was seen
        block = r._capture_windows_meta()
        self.assertEqual(block["windows"][0]["track"], "ok")
        self.assertEqual(block["windows"][1]["track"], "failed")

    def test_non_dict_entries_are_dropped(self):
        r = self._rec(capture_windows=[_entry(11, 0, 0, 400, 300), None,
                                       _entry(22, 500, 0, 400, 300)])
        block = r._capture_windows_meta()
        self.assertEqual([w["id"] for w in block["windows"]], [11, 22])

    def test_arrange_state_rides_the_block(self):
        r = self._rec(capture_windows=[_entry(11, 0, 0, 400, 300),
                                       _entry(22, 500, 0, 400, 300)])
        self.assertEqual(r._capture_windows_meta()["arrange"], "none")
        r._arrange_state = "moved"
        self.assertEqual(r._capture_windows_meta()["arrange"], "moved")

    def test_meta_dict_carries_the_block(self):
        r = self._rec(capture_windows=[_entry(11, 0, 0, 400, 300),
                                       _entry(22, 500, 0, 400, 300)])
        meta = r._meta_dict(1440.0, 900.0, "quartz", 0.0, None)
        self.assertIn("capture_windows", meta)
        self.assertNotIn("capture_window", meta)

    def test_ordinary_meta_gains_no_new_key(self):
        meta = self._rec()._meta_dict(1440.0, 900.0, "quartz", 0.0, None)
        self.assertNotIn("capture_windows", meta)
        self.assertNotIn("capture_window", meta)


class GeometrySampleZ(unittest.TestCase):
    """The z rank is the only thing that can later say a window was covered.
    It is also a new key on a line format with a byte-exactness claim, so it
    has to be omitted when unknown."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="mw_z_")
        self.rec = rec.Recorder(self.dir, 2)
        self.rec._ev_file = open(os.path.join(self.dir, "events.jsonl"), "w")

    def tearDown(self):
        if self.rec._ev_file:
            self.rec._ev_file.close()

    def _lines(self):
        self.rec._ev_file.flush()
        with open(os.path.join(self.dir, "events.jsonl")) as f:
            return [json.loads(x) for x in f if x.strip()]

    def test_z_is_written_when_known(self):
        self.rec._write_window_event(7, [0, 0, 10, 10], z=3)
        self.assertEqual(self._lines()[0]["z"], 3)

    def test_z_is_absent_when_unknown(self):
        self.rec._write_window_event(7, [0, 0, 10, 10])
        self.assertNotIn("z", self._lines()[0])

    def test_z_never_carries_identity(self):
        self.rec._write_window_event(7, [0, 0, 10, 10], z=0)
        line = self._lines()[0]
        for banned in ("app", "title", "owner", "pid", "label", "name"):
            self.assertNotIn(banned, line)


class BindByWindowId(unittest.TestCase):
    """A card that knows its window must not be re-guessed by overlap."""

    def _ev(self, frames=30):
        # Two windows far apart; the card's rect matches NEITHER well, so the
        # IoU path would refuse to bind and leave it static.
        t, rect, wid = [], [], []
        for i in range(frames):
            t.append(i * 0.05); rect.append([0, 0, 400, 300]); wid.append(11)
            t.append(i * 0.05); rect.append([800, 500, 400, 300]); wid.append(22)
        return {"windows_t": np.asarray(t, dtype=float),
                "windows_rect": np.asarray(rect, dtype=float),
                "windows_id": np.asarray(wid, dtype=int)}

    def _build(self, spec):
        return ren._build_grid_tracks(
            self._ev(), [spec], None, None, 1440, 900, 1.0, 1.0,
            np.linspace(0.0, 1.0, 30), lambda x: x, 60.0)

    def test_a_pinned_card_binds_even_with_poor_overlap(self):
        got = self._build({"x": 100, "y": 100, "w": 900, "h": 700,
                           "window_id": 22})
        self.assertIsNotNone(got[0])

    def test_the_same_card_without_a_pin_stays_static(self):
        got = self._build({"x": 100, "y": 100, "w": 900, "h": 700})
        self.assertIsNone(got[0])

    def test_an_unknown_pin_falls_back_to_overlap(self):
        got = self._build({"x": 0, "y": 0, "w": 400, "h": 300,
                           "window_id": 999})
        self.assertIsNotNone(got[0])       # matched window 11 by IoU

    def test_pinning_beats_a_better_overlap_elsewhere(self):
        """The rect sits exactly on window 11, but says it is window 22.
        Identity wins: the picker had the real id and the IoU search is only
        ever a reconstruction."""
        pinned = self._build({"x": 0, "y": 0, "w": 400, "h": 300,
                              "window_id": 22})[0]
        plain = self._build({"x": 0, "y": 0, "w": 400, "h": 300})[0]
        self.assertIsNotNone(pinned)
        self.assertIsNotNone(plain)
        self.assertFalse(np.allclose(pinned.rect_at(0), plain.rect_at(0)))


class OcclusionReport(unittest.TestCase):
    def _ev(self, rows):
        return {"windows_t": np.asarray([r[0] for r in rows], dtype=float),
                "windows_id": np.asarray([r[1] for r in rows], dtype=int),
                "windows_rect": np.asarray([r[2] for r in rows], dtype=float),
                "windows_z": np.asarray([r[3] for r in rows], dtype=int)}

    def test_a_covered_window_accumulates_time(self):
        ev = self._ev([
            (0.0, 11, [0, 0, 400, 300], 1),
            (0.0, 22, [100, 100, 400, 300], 0),   # on top, overlapping
            (2.0, 11, [0, 0, 400, 300], 1),
            (2.0, 22, [100, 100, 400, 300], 0),
        ])
        got = ren.occlusion_report(ev, [11, 22])
        self.assertAlmostEqual(got.get(11, 0.0), 2.0, places=2)
        self.assertNotIn(22, got)             # nothing was on top of 22

    def test_separated_windows_report_nothing(self):
        ev = self._ev([
            (0.0, 11, [0, 0, 400, 300], 1),
            (0.0, 22, [800, 0, 400, 300], 0),
            (2.0, 11, [0, 0, 400, 300], 1),
            (2.0, 22, [800, 0, 400, 300], 0),
        ])
        self.assertEqual(ren.occlusion_report(ev, [11, 22]), {})

    def test_a_session_with_no_z_track_reports_nothing(self):
        ev = self._ev([
            (0.0, 11, [0, 0, 400, 300], -1),
            (0.0, 22, [100, 100, 400, 300], -1),
            (2.0, 11, [0, 0, 400, 300], -1),
        ])
        self.assertEqual(ren.occlusion_report(ev, [11, 22]), {})

    def test_empty_track_is_not_an_error(self):
        self.assertEqual(ren.occlusion_report({}, [1]), {})


if __name__ == "__main__":
    unittest.main()


class MultiNativeCanvasSizing(unittest.TestCase):
    """The composite canvas is sized from the window buffers, not hardcoded.

    It used to be a flat 1920x1080, which on a Retina Mac downscaled every
    card to ~0.55x -- roughly a third of the captured pixels survived, and the
    text came out mushy with nothing in the output hinting why. The
    whole-screen path never had this problem: `framing.output_size` passes the
    REAL source dims through, so "auto" renders at native resolution.
    """

    # Three ~600pt windows captured at 2x, the shape that exposed this.
    RECTS = [{"x": 0, "y": 0, "w": 620, "h": 460},
             {"x": 640, "y": 0, "w": 620, "h": 460},
             {"x": 0, "y": 480, "w": 620, "h": 460}]
    DIMS = [(1240, 920), (1240, 920), (1240, 920)]

    def _scales(self, w, h, layout="grid"):
        from autocine import framing
        cells = framing.placements_for(w, h, self.RECTS, layout=layout)
        return [min(fw / float(bw), fh / float(bh))
                for (_x, _y, fw, fh), (bw, bh) in zip(cells, self.DIMS)]

    def test_auto_canvas_reaches_native_scale_or_the_cap(self):
        from autocine import render
        w, h = render._multi_native_canvas(self.RECTS, self.DIMS, "clean",
                                           None, "grid")
        self.assertGreater(w, render._MULTI_NATIVE_DEFAULT_W)
        capped = (max(w, h) >= render._MULTI_NATIVE_MAX_DIM
                  or w * h >= render._MULTI_NATIVE_MAX_PIXELS * 0.98)
        for s in self._scales(w, h):
            if capped:
                # Three 2x Retina windows in a grid genuinely need more than
                # 4K to be native. The cap binds first, and a capped take is
                # merely as soft as it used to be -- never softer.
                self.assertGreater(s, 0.85, "capped, but worse than expected")
            else:
                self.assertGreaterEqual(s, 0.98, "card still downscaled")

    def test_it_is_a_large_improvement_on_the_old_default(self):
        from autocine import render
        w, h = render._multi_native_canvas(self.RECTS, self.DIMS, "clean",
                                           None, "grid")
        before = min(self._scales(1920, 1080))
        after = min(self._scales(w, h))
        self.assertGreater(after, before * 1.5)

    def test_the_old_default_was_actually_lossy(self):
        # Guards the guard: if this ever stops being true the fix is moot.
        for s in self._scales(1920, 1080):
            self.assertLess(s, 0.7)

    def test_an_exact_aspect_wins(self):
        # "WxH" is the caller naming the canvas; never second-guess it.
        from autocine import render
        self.assertEqual(
            render._multi_native_canvas(self.RECTS, self.DIMS, "clean",
                                        "1600x900", "grid"),
            (1600, 900))

    def test_the_cap_bounds_the_encode(self):
        from autocine import render
        huge = [(8000, 6000)] * 3
        w, h = render._multi_native_canvas(self.RECTS, huge, "clean", None,
                                           "grid")
        self.assertLessEqual(max(w, h), render._MULTI_NATIVE_MAX_DIM)
        self.assertLessEqual(w * h, render._MULTI_NATIVE_MAX_PIXELS * 1.02)

    def test_a_vertical_export_may_grow_TALL(self):
        # The cap is a longest-edge + pixel budget, not a width/height pair:
        # capping height at 1080p's would leave 9:16 permanently soft.
        from autocine import render
        w, h = render._multi_native_canvas(self.RECTS, self.DIMS, "clean",
                                           "9:16", "grid")
        self.assertGreater(h, w)
        self.assertGreater(h, 2160)

    def test_placements_for_matches_the_painter(self):
        # The sizer measures with placements_for and the render then builds a
        # painter; if they ever disagree the canvas is sized for a layout
        # nothing draws.
        from autocine import framing
        for layout in ("grid", "desktop", "feature", "row", "column"):
            cheap = framing.placements_for(1920, 1080, self.RECTS, layout=layout)
            painter = framing.make_multi_painter(1920, 1080, self.RECTS,
                                                 layout=layout)
            real = [(c["x"], c["y"], c["w"], c["h"]) for c in painter.cells]
            self.assertEqual([tuple(int(v) for v in p) for p in cheap], real,
                             layout)


def _make_multi_native_session(root, name="mn", n=2, secs=8.0, fps=30,
                               bw=240, bh=160, clicks=None):
    """A synthetic occlusion-free take: N `raw_i.mov` buffers + a
    `capture_channels` manifest + events, enough for the camera/focus emitters
    to plan. Windows are laid out side by side (points), scale 2x -> buffer.
    `clicks` is a list of (t_media, x_pt, y_pt); they land in window 0 by
    default so card 0 is the one that zooms/focuses.
    """
    import subprocess
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    lw, lh = bw // 2, bh // 2                    # logical points, scale 2
    channels = []
    for i in range(n):
        raw = os.path.join(d, "raw_{}.mov".format(i))
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", "testsrc2=size={}x{}:rate={}:duration={}".format(
                 bw, bh, fps, secs),
             "-pix_fmt", "yuv420p", raw],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        channels.append({
            "role": "screen_window", "file": "raw_{}.mov".format(i),
            "mode": "window_native", "id": 1000 + i,
            "rect": [float(i * (lw + 20)), 0.0, float(lw), float(lh)],
            "logical_w": float(lw), "logical_h": float(lh),
            "buffer_w": bw, "buffer_h": bh,
            "t0_monotonic": 100.0 + i * 0.01,     # tiny per-channel drift
        })
    if clicks is None:
        clicks = [(2.0, 30, 40), (2.4, 32, 42), (2.8, 30, 41), (3.2, 33, 40)]
    events = []
    for (t, x, y) in clicks:
        events.append({"t": 100.0 + t, "type": "down", "x": x, "y": y})
    with open(os.path.join(d, "events.jsonl"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    meta = {"fps": fps, "t0_monotonic": 100.0, "events": "events.jsonl",
            "width": bw, "height": bh, "capture_channels": channels}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    return d


class MultiNativePreviewCameras(unittest.TestCase):
    """The editor's LIVE preview of per-card zoom + window-focus for an
    occlusion-free take. The emitters must (a) go bit-inert when their toggle
    is off and (b) reproduce EXACTLY what the export composites -- the browser
    re-derives its crop/cell from these numbers, so a drift is a visible
    preview-vs-export split. Mirrors PerCardAutoZoom / WindowFocus for the
    display-crop path.
    """

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="mn_prev_")
        cls.dir = _make_multi_native_session(cls.root)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.root, ignore_errors=True)

    def _inputs(self):
        sp = ren._native_camera_inputs(self.dir)
        self.assertIsNotNone(sp)
        return sp

    def test_card_paths_off_switch_is_none(self):
        self.assertIsNone(ren.multi_native_card_paths(
            self.dir, window_zoom=False, max_zoom=2.2))

    def test_card_paths_engage_and_match_the_builder(self):
        cp = ren.multi_native_card_paths(
            self.dir, window_zoom=True, max_zoom=2.2, params={}, stride=2)
        self.assertIsNotNone(cp)
        self.assertEqual(cp["fps"], 30.0)
        self.assertEqual(cp["stride"], 2)
        zoomed = [i for i, c in enumerate(cp["cards"]) if c is not None]
        self.assertTrue(zoomed, "no card zoomed -- fixture clicks too sparse")
        # The emitter must sample EXACTLY what the export's builder produces.
        sp = self._inputs()
        exp = ren._build_native_card_cameras(
            sp["channels"], sp["native_tracks"], sp["clicks_t"],
            sp["clicks_x"], sp["clicks_y"], sp["frame_times"], max_zoom=2.2,
            params={}, suppressed_ranges=None,
            plan_duration=sp["n_out"] / sp["src_fps"], src_fps=sp["src_fps"],
            enabled=True)
        for i, path in enumerate(exp):
            got = cp["cards"][i]
            if path is None:
                self.assertIsNone(got)
                continue
            self.assertEqual(
                got["z"], [round(float(v), 4) for v in path[::2, 2]])
            self.assertEqual(
                got["cx"], [round(float(v), 2) for v in path[::2, 0]])
            self.assertEqual(
                got["cy"], [round(float(v), 2) for v in path[::2, 1]])

    def test_focus_cells_off_switch_is_none(self):
        painter = ren.multi_native_layout(self.dir, window_layout="grid")["painter"]
        self.assertIsNone(ren.multi_native_focus_cells(
            self.dir, painter, window_focus=False, max_zoom=2.2))

    def test_focus_cells_match_the_focus_layout(self):
        layout = ren.multi_native_layout(self.dir, window_layout="grid")
        painter = layout["painter"]
        fc = ren.multi_native_focus_cells(
            self.dir, painter, window_focus=True, max_zoom=2.2, params={},
            stride=2)
        self.assertIsNotNone(fc)
        # Rebuild the export's _FocusLayout and compare frame-for-frame.
        sp = self._inputs()
        pct = ren._native_card_click_times(
            sp["native_tracks"], sp["clicks_t"], sp["clicks_x"], sp["clicks_y"])
        wff = [{"w": (t.raw_w if t else 1), "h": (t.raw_h if t else 1)}
               for t in sp["native_tracks"]]
        emph = ren._build_native_focus_emphasis(
            wff, pct, sp["frame_times"], 2.2, {}, None,
            sp["n_out"] / sp["src_fps"], sp["src_fps"], manual=None)
        self.assertIsNotNone(emph)
        from autocine import camera
        fl = ren._FocusLayout(
            painter, emph, lean=camera.build_params(2.2, {}).focus_lean)
        for k, j in enumerate(range(0, len(emph), 2)):
            cells, order = fl.cells_at(j)
            cam = fl.camera_at(j)
            self.assertEqual(
                fc["frames"][k]["c"],
                [[c["x"], c["y"], c["w"], c["h"], c["radius"]] for c in cells])
            self.assertEqual(fc["frames"][k]["o"], list(order))
            exp_z = ([round(float(cam[0]), 1), round(float(cam[1]), 1),
                      round(float(cam[2]), 4)] if cam else None)
            self.assertEqual(fc["frames"][k]["z"], exp_z)

    def test_focus_is_auto_planned_never_materialized(self):
        # Multi-native focus must stay auto-planned: every card shares the same
        # click TIMES (clamp-not-drop), so a materialized per-card span plan
        # would arbitrate differently than the auto-plan. `_load_edits_with_auto`
        # leaves it stale so render/export/preview all auto-plan the same. The
        # earlier bug materialized `focus: []`, which SUPPRESSED focus entirely.
        from autocine import studio_app
        state = studio_app.StudioState(self.root)
        info = state._describe(self.dir, include_click_times=True)
        self.assertTrue(info.get("multi_native"))
        doc = state._load_edits_with_auto(self.dir, info)
        # Not materialized (stays stale) -> render sees focus_ranges=None.
        self.assertFalse(doc.get("focus_initialized"))
        self.assertEqual(list(doc.get("focus") or []), [])
        self.assertTrue(edits.focus_plan_is_stale(doc))
        # And with the toggle on, the live preview still animates focus by
        # auto-planning -- the whole point of leaving it un-materialized.
        doc["render"]["window_focus"] = True
        doc["render"]["window_zoom"] = True
        edits.save_edits(self.dir, doc, duration=info["duration"])
        payload = state.camera_path(os.path.basename(self.dir), {"stride": 2})
        self.assertIsNotNone(payload.get("focus_cells"))
        self.assertIn("bg_jpeg_base64", payload)
