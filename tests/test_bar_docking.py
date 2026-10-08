"""Parking the pill on the window the picker just elected.

`_NativeBarApi.dock_to_windows` and the hand-off to `bar_fit`. Everything
Cocoa is stubbed at the `bar_native` seam, so this runs with no pywebview, no
NSWindow and no permissions -- the geometry itself is pinned in
`test_bar_native.DockGeometry`.

The interesting part is the RACE, not the arithmetic: `set_frame` is async, so
a `bar_fit` arriving right after a dock would read a pre-dock frame and drag
the pill back. The dock is therefore left PENDING as well as applied, and
whichever call lands last agrees with the other.
"""
import os
import unittest

from autocine import bar_native
from autocine import devices as dev
from autocine import studio_app


SCREEN = (0, 0, 1440, 900)          # Cocoa (x, y, w, h) of the main screen
FULLSCREEN_WINDOW = {"id": 13502, "x": 0, "y": 0, "w": 1440, "h": 900}
SIDE_WINDOW = {"id": 11192, "x": 800, "y": 100, "w": 640, "h": 400}


class _Harness(unittest.TestCase):
    """An api whose window "moves" when set_frame is called, so a later
    frame_of read sees where the previous call actually left it."""

    def setUp(self):
        self.frame = (100, 700, 400, 136)   # Cocoa; the pill's starting frame
        self.moves = []
        self.saved = []
        self.rects = {w["id"]: w for w in (FULLSCREEN_WINDOW, SIDE_WINDOW)}

        def _set_frame(win, frame):
            self.frame = frame
            self.moves.append(frame)
            return True

        self._patch(bar_native, "set_frame", _set_frame)
        self._patch(bar_native, "frame_of", lambda win: self.frame)
        self._patch(bar_native, "screen_frames", lambda: [SCREEN])
        self._patch(bar_native, "_origin_screen_frame", lambda: SCREEN)
        self._patch(bar_native, "save_bar_position",
                    lambda x, y, path=None: self.saved.append((x, y)) or True)
        self._patch(dev, "window_rect_points",
                    lambda wid, **kw: self.rects.get(int(wid)))

        self.api = studio_app._NativeBarApi()
        self.api.window = object()

    def _patch(self, mod, name, value):
        orig = getattr(mod, name)
        setattr(mod, name, value)
        self.addCleanup(lambda: setattr(mod, name, orig))

    def visible_bottom(self, frame):
        """Layout-coord y of the PILL's bottom edge inside a window frame."""
        _sx, sy, _sw, sh = SCREEN
        return sy + sh - (frame[1] + bar_native.SHADOW_PAD)

    def centre_x(self, frame):
        return frame[0] + frame[2] / 2.0


class DockToWindows(_Harness):
    def test_a_single_pick_lands_bottom_centre_on_that_window(self):
        self.assertEqual(self.api.dock_to_windows([13502]), {"ok": True})
        moved = self.moves[-1]
        self.assertEqual(self.centre_x(moved), 720)            # 1440 / 2
        self.assertEqual(self.visible_bottom(moved),
                         900 - bar_native.DOCK_MARGIN)

    def test_a_multi_pick_docks_to_the_union(self):
        self.api.dock_to_windows([13502, 11192])
        moved = self.moves[-1]
        # union of a full-screen window and one inside it is the full screen
        self.assertEqual(self.centre_x(moved), 720)

    def test_docks_to_the_one_window_when_it_is_not_full_screen(self):
        self.api.dock_to_windows([11192])
        moved = self.moves[-1]
        self.assertEqual(self.centre_x(moved), 800 + 640 / 2.0)
        self.assertEqual(self.visible_bottom(moved),
                         100 + 400 - bar_native.DOCK_MARGIN)

    def test_a_stale_id_leaves_the_pill_alone(self):
        got = self.api.dock_to_windows([999999])
        self.assertEqual(got["ok"], False)
        self.assertEqual(self.moves, [])

    def test_clearing_the_pick_leaves_the_pill_alone(self):
        """Back to Full screen sends an empty list. Undoing the dock would be
        worse than leaving it: the user can see where the pill is."""
        self.assertEqual(self.api.dock_to_windows([])["ok"], False)
        self.assertEqual(self.moves, [])

    def test_the_dragged_position_is_never_overwritten(self):
        """bar-pos.json holds where the user DRAGGED the pill. A pick docks for
        this session only -- a later drag saves, exactly as it always did."""
        self.api.dock_to_windows([13502])
        self.assertEqual(self.saved, [])
        self.api.bar_moved()
        self.assertEqual(len(self.saved), 1)

    def test_an_unreadable_frame_still_arms_the_pending_dock(self):
        """No frame to size against now, but the fit that follows has one."""
        self._patch(bar_native, "frame_of", lambda win: None)
        self.assertEqual(self.api.dock_to_windows([13502]), {"ok": True})
        self.assertEqual(self.moves, [])
        self.assertIsNotNone(self.api._dock_anchor)


class DockThenFit(_Harness):
    """The hand-off: a dock is applied now AND consumed by the next fit."""

    def test_the_fit_after_a_dock_re_docks_at_the_new_size(self):
        self.api.dock_to_windows([13502])
        self.api.bar_fit(300, 72)
        fitted = self.moves[-1]
        self.assertEqual(fitted[2:], (300 + 2 * bar_native.SHADOW_PAD,
                                      72 + 2 * bar_native.SHADOW_PAD))
        # resized, but still docked: both anchors survive the size change
        self.assertEqual(self.centre_x(fitted), 720)
        self.assertEqual(self.visible_bottom(fitted),
                         900 - bar_native.DOCK_MARGIN)

    def test_the_anchor_is_consumed_and_the_anti_jump_rule_returns(self):
        """Only the FIRST fit after a pick docks. After that `fit_frame`'s
        centre-x + top-edge pinning owns the position again, or every face
        change would yank the pill back to the window."""
        self.api.dock_to_windows([13502])
        self.api.bar_fit(300, 72)
        docked = self.moves[-1]
        self.assertIsNone(self.api._dock_anchor)

        self.api.bar_fit(200, 72)
        after = self.moves[-1]
        self.assertEqual(self.centre_x(after), self.centre_x(docked))
        self.assertEqual(after[1] + after[3], docked[1] + docked[3])  # top edge

    def test_a_fit_with_nothing_pending_is_unchanged(self):
        """The off switch: no pick, no dock, and bar_fit behaves exactly as it
        did before docking existed."""
        self.api.bar_fit(300, 72)
        w, h, _rects = bar_native.content_layout(300, 72)
        expected = bar_native.clamp_to_screens(
            bar_native.fit_frame((100, 700, 400, 136), w, h), [SCREEN])
        self.assertEqual(self.moves, [expected])

    def test_a_racing_fit_cannot_drag_the_pill_back(self):
        """The whole reason the anchor is pending: simulate the fit reading the
        pre-dock frame (set_frame hasn't been serviced by the main loop yet)."""
        self.api.dock_to_windows([13502])
        self.frame = (100, 700, 400, 136)          # stale read
        self.api.bar_fit(300, 72)
        self.assertEqual(self.centre_x(self.moves[-1]), 720)


class PickerResultDocks(_Harness):
    """`picker_result` is the only caller in overlay mode, and the <select>
    fallback reaches `dock_to_windows` from bar.js instead."""

    def test_confirming_the_picker_docks(self):
        self._patch(studio_app._NativeBarApi, "_eval_bar",
                    lambda self, script: None)
        self._patch(studio_app._NativeBarApi, "_close_picker",
                    lambda self: False)
        self.api.picker_result([13502], False)
        self.assertEqual(self.centre_x(self.moves[-1]), 720)

    def test_cancelling_the_picker_does_not_move_the_pill(self):
        self._patch(studio_app._NativeBarApi, "_eval_bar",
                    lambda self, script: None)
        self._patch(studio_app._NativeBarApi, "_close_picker",
                    lambda self: False)
        self.api.picker_cancelled()
        self.assertEqual(self.moves, [])


class BarJsWiring(unittest.TestCase):
    """The <select> fallback path lives in bar.js, so it gets a source pin --
    the overlay picker docks from Python and would hide a regression here."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "studio_web", "bar.js")
        with open(path, encoding="utf-8") as f:
            cls.js = f.read()

    def test_the_select_change_handler_docks(self):
        i = self.js.index('ui.window.addEventListener("change"')
        self.assertIn('bridgeCall("dock_to_windows"', self.js[i:i + 600])

    def test_the_bridge_name_matches_the_python_method(self):
        self.assertTrue(hasattr(studio_app._NativeBarApi, "dock_to_windows"))


if __name__ == "__main__":
    unittest.main()
