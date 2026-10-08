"""bar_native geometry + position persistence.

Pure math and file IO only — no pywebview, no Cocoa, no permissions. The
Cocoa glue in bar_native is deliberately soft-failing so importing this module
on any machine is safe.
"""
import builtins
import json
import os
import shutil
import tempfile
import threading
import time
import types
import unittest

from autocine import bar_native as bn


class ContentLayout(unittest.TestCase):
    def test_pill_only_is_padded_on_every_side(self):
        w, h, rects = bn.content_layout(702, 72, pad=40)
        self.assertEqual((w, h), (702 + 80, 72 + 80))
        self.assertEqual(len(rects), 1)
        self.assertEqual(rects[0], (40.0, 40.0, 702.0, 72.0))

    def test_pill_is_centred_when_the_hint_is_wider(self):
        w, h, rects = bn.content_layout(265, 72, hint_w=600, hint_h=30, pad=40)
        self.assertEqual(w, 600 + 80)
        # pill row + gap + hint row, all inside the padding
        self.assertEqual(h, 72 + 30 + bn.BODY_GAP + 80)
        pill, hint = rects
        self.assertAlmostEqual(pill[0] + pill[2] / 2.0, w / 2.0)
        self.assertAlmostEqual(hint[0] + hint[2] / 2.0, w / 2.0)
        self.assertEqual(hint[1], 40 + 72 + bn.BODY_GAP)

    def test_no_hint_row_when_the_hint_is_hidden(self):
        _w, h, rects = bn.content_layout(300, 72, hint_w=0, hint_h=0, pad=10)
        self.assertEqual(len(rects), 1)
        self.assertEqual(h, 72 + 20)     # no stray gap


class FitFrame(unittest.TestCase):
    """Cocoa coords: (x, y, w, h) with y the BOTTOM edge."""

    def test_keeps_centre_x_and_top_edge(self):
        frame = (100, 200, 880, 136)          # top edge = 336, centre = 540
        x, y, w, h = bn.fit_frame(frame, 782, 152)
        self.assertEqual((w, h), (782, 152))
        self.assertEqual(x + w / 2.0, 540)    # centre-x pinned
        self.assertEqual(y + h, 336)          # top edge pinned

    def test_growing_and_shrinking_are_inverses(self):
        frame = (100, 200, 880, 136)
        grown = bn.fit_frame(frame, 400, 300)
        back = bn.fit_frame(grown, 880, 136)
        self.assertEqual(back, (100, 200, 880, 136))

    def test_degenerate_sizes_are_clamped_not_crashed(self):
        self.assertEqual(bn.fit_frame((0, 0, 10, 10), 0, -5)[2:], (1, 1))


class ClampFrame(unittest.TestCase):
    screen = (0, 0, 1440, 900)

    def test_leaves_an_on_screen_frame_alone(self):
        f = (100, 100, 400, 200)
        self.assertEqual(bn.clamp_frame(f, self.screen), f)

    def test_pulls_an_off_screen_frame_back(self):
        self.assertEqual(bn.clamp_frame((1300, -50, 400, 200), self.screen),
                         (1040, 0, 400, 200))

    def test_oversized_frame_pins_to_the_low_edge(self):
        self.assertEqual(bn.clamp_frame((-30, -30, 2000, 1200), self.screen),
                         (0, 0, 2000, 1200))

    def test_margin_is_honoured(self):
        self.assertEqual(bn.clamp_frame((0, 0, 100, 100), self.screen, margin=12),
                         (12, 12, 100, 100))


class ClampToScreens(unittest.TestCase):
    """The multi-display fix: a frame is clamped onto the display it sits on,
    not onto one hard-coded screen. Cocoa coords throughout — the SAMSUNG here
    mirrors this machine's real arrangement (primary at origin, second display
    up and to the left)."""

    PRIMARY = (0, 0, 1440, 900)
    SECOND = (-912, 900, 1366, 768)

    def test_no_screens_leaves_the_frame_untouched(self):
        """AppKit unreadable -> `screen_frames()` is [] -> leave the pill where
        the user dragged it rather than move it somewhere unjustified."""
        f = (5000, 5000, 400, 136)
        self.assertEqual(bn.clamp_to_screens(f, []), f)

    def test_a_frame_on_the_primary_clamps_against_the_primary(self):
        f = (100, 100, 400, 136)
        self.assertEqual(bn.clamp_to_screens(f, [self.PRIMARY, self.SECOND]),
                         bn.clamp_frame(f, self.PRIMARY))

    def test_a_frame_on_the_second_display_stays_there(self):
        """The bug: the old single-screen clamp was against the primary, so a
        pill dragged fully onto the second display was hauled back on the next
        fit. It must clamp against the display it overlaps most instead."""
        f = (-800, 1000, 400, 136)                  # squarely on the SAMSUNG
        got = bn.clamp_to_screens(f, [self.PRIMARY, self.SECOND])
        self.assertEqual(got, bn.clamp_frame(f, self.SECOND))
        self.assertEqual(got, f)                    # already inside -> untouched

    def test_screen_order_does_not_decide_the_target(self):
        f = (-800, 1000, 400, 136)
        self.assertEqual(bn.clamp_to_screens(f, [self.PRIMARY, self.SECOND]),
                         bn.clamp_to_screens(f, [self.SECOND, self.PRIMARY]))

    def test_a_frame_overlapping_neither_snaps_to_the_nearest_centre(self):
        """Mid-drag over the dead gap in a staggered layout: no overlap with
        any display, so pick the one whose centre is nearest and clamp onto
        it. Just left of the SAMSUNG, far from the primary."""
        f = (-3000, 1200, 400, 136)
        self.assertEqual(bn.clamp_to_screens(f, [self.PRIMARY, self.SECOND]),
                         bn.clamp_frame(f, self.SECOND))

    def test_partial_overlap_picks_the_larger_share(self):
        """Straddling the boundary, but mostly on the second display -> clamp
        against the second."""
        f = (-500, 1000, 400, 136)                  # x -500..-100, all y on SECOND
        self.assertEqual(bn.clamp_to_screens(f, [self.PRIMARY, self.SECOND]),
                         bn.clamp_frame(f, self.SECOND))

    def test_degenerate_screens_are_ignored(self):
        """A zero-area screen never wins; it is filtered before the choice."""
        f = (100, 100, 400, 136)
        self.assertEqual(
            bn.clamp_to_screens(f, [(0, 0, 0, 0), self.PRIMARY]),
            bn.clamp_frame(f, self.PRIMARY))


class OriginScreen(unittest.TestCase):
    """`origin_screen()` feeds `create_window(screen=...)`, which is the whole
    drag fix: it pins pywebview's coordinate anchor to the primary display so a
    globally-reported drag maps correctly. If it returns None the backend falls
    back to `mainScreen()` and the pill can't be dragged across displays."""

    def _with_fake_webview(self, screens_attr):
        """Run origin_screen() with `webview.screens` set to `screens_attr`."""
        fake = types.ModuleType("webview")
        fake.screens = screens_attr
        real_import = builtins.__import__

        def _fake_import(name, *a, **kw):
            if name == "webview":
                return fake
            return real_import(name, *a, **kw)

        builtins.__import__ = _fake_import
        try:
            return bn.origin_screen()
        finally:
            builtins.__import__ = real_import

    def test_reads_the_screens_property_without_calling_it(self):
        """REGRESSION: `webview.screens` is a proxy_tools module-property,
        ACCESSED not called. `webview.screens()` raises "'list' object is not
        callable", which the broad except swallowed to None -- silently
        reverting to the mainScreen() anchor and un-fixing the drag. A plain
        list stands in for the resolved proxy: it indexes but is not callable,
        so a test that expects the first entry only passes on attribute access."""
        primary = types.SimpleNamespace(x=0, y=0, width=1440, height=900)
        second = types.SimpleNamespace(x=-912, y=900, width=1366, height=768)
        got = self._with_fake_webview([primary, second])
        self.assertIs(got, primary)

    def test_no_screens_is_a_quiet_none(self):
        self.assertIsNone(self._with_fake_webview([]))

    def test_missing_pywebview_is_a_quiet_none(self):
        real_import = builtins.__import__

        def _fake_import(name, *a, **kw):
            if name == "webview":
                raise ImportError("no pywebview")
            return real_import(name, *a, **kw)

        builtins.__import__ = _fake_import
        try:
            self.assertIsNone(bn.origin_screen())
        finally:
            builtins.__import__ = real_import


class HitRects(unittest.TestCase):
    def test_layout_rect_flips_into_cocoa_coords(self):
        frame = (100, 200, 800, 150)          # top edge = 350
        # a 700x72 rect 40px below the window's top-left
        got = bn.hit_rects_to_screen(frame, [(50, 40, 700, 72)])
        self.assertEqual(got, [(150, 350 - 40 - 72, 700, 72)])

    def test_round_trip_through_content_layout_stays_inside_the_frame(self):
        w, h, rects = bn.content_layout(702, 72)
        frame = (0, 0, w, h)
        for rx, ry, rw, rh in bn.hit_rects_to_screen(frame, rects):
            self.assertGreaterEqual(rx, 0)
            self.assertGreaterEqual(ry, 0)
            self.assertLessEqual(rx + rw, w)
            self.assertLessEqual(ry + rh, h)

    def test_malformed_entries_are_skipped(self):
        got = bn.hit_rects_to_screen((0, 0, 10, 10),
                                     [None, (1, 2), "nope", (0, 0, 0, 5),
                                      (0, 0, 4, 4)])
        self.assertEqual(len(got), 1)

    def test_point_in_rects(self):
        rects = [(0, 0, 10, 10), (100, 100, 5, 5)]
        self.assertTrue(bn.point_in_rects(5, 5, rects))
        self.assertTrue(bn.point_in_rects(102, 102, rects))
        self.assertFalse(bn.point_in_rects(50, 50, rects))
        self.assertFalse(bn.point_in_rects(0, 0, []))


class ClickThroughDecisions(unittest.TestCase):
    """The pointer-position -> ignoresMouseEvents logic, without a live
    NSWindow (mouse reads, the frame read, and the AppKit write are all
    injectable seams).

    The bar for every case here is: an unclickable pill is a worse bug than
    dead space, so anything unexpected must resolve to "interactive".
    """

    # window at Cocoa (0, 0, 400, 200); pill 200x72 inset 100/40 in layout
    # coords -> Cocoa (100, 200-40-72, 200, 72) = (100, 88, 200, 72)
    FRAME = (0.0, 0.0, 400.0, 200.0)
    PILL = [(100.0, 40.0, 200.0, 72.0)]
    INSIDE = (150.0, 120.0)
    OUTSIDE = (10.0, 190.0)

    def _rig(self, rects, x=0.0, y=0.0, buttons=0, armed=True,
             apply_ok=True, frame=FRAME):
        self.applied = []
        self.mouse = [x, y, buttons]
        ct = bn.ClickThrough(
            None,
            mouse_state=lambda: tuple(self.mouse),
            apply_ignore=lambda ig: (self.applied.append(ig), apply_ok)[1],
            frame_reader=lambda: frame,
        )
        ct._armed = armed
        ct.set_rects(rects)
        return ct

    def test_pointer_over_the_pill_keeps_the_window_interactive(self):
        ct = self._rig(self.PILL, *self.INSIDE)
        self.assertFalse(ct.should_ignore(*self.INSIDE))
        self.assertEqual(self.applied, [False])

    def test_pointer_over_transparent_slack_is_click_through(self):
        ct = self._rig(self.PILL, *self.OUTSIDE)
        self.assertTrue(ct.should_ignore(*self.OUTSIDE))
        self.assertEqual(self.applied, [True])

    def test_rects_track_a_window_that_moved(self):
        """Regression: rects are layout-relative and resolved against the LIVE
        frame. Freezing them in screen coords at report time meant an async
        set_frame left them describing a stale position — the pointer was then
        never 'inside' and the pill latched unclickable."""
        moved = (500.0, 300.0, 400.0, 200.0)
        ct = self._rig(self.PILL, armed=True, frame=moved)
        self.assertTrue(ct.should_ignore(*self.INSIDE))          # old spot
        self.assertFalse(ct.should_ignore(650.0, 420.0))         # new spot

    def test_not_armed_is_always_interactive(self):
        """Regression: set_rects used to be able to turn click-through ON
        before (or without) the watcher running, with nothing left to turn it
        back off."""
        ct = self._rig(self.PILL, *self.OUTSIDE, armed=False)
        self.assertEqual(self.applied, [False])
        ct._update()
        self.assertEqual(self.applied, [False])

    def test_no_rects_yet_stays_interactive(self):
        ct = self._rig([], *self.OUTSIDE)
        self.assertFalse(ct.should_ignore(*self.OUTSIDE))
        self.assertEqual(self.applied, [False])

    def test_unreadable_frame_stays_interactive(self):
        ct = self._rig(self.PILL, *self.OUTSIDE, frame=None)
        self.assertFalse(ct.should_ignore(*self.OUTSIDE))

    def test_hysteresis_arms_just_outside_the_edge(self):
        ct = self._rig(self.PILL, armed=True)
        # pill spans Cocoa x 100..300; a couple of points clear is still "in"
        self.assertFalse(ct.should_ignore(97.0, 120.0))
        self.assertTrue(ct.should_ignore(80.0, 120.0))

    def test_state_is_only_pushed_on_change(self):
        ct = self._rig(self.PILL, *self.OUTSIDE)
        self.assertEqual(self.applied, [True])
        ct._update()
        ct._update()
        self.assertEqual(self.applied, [True])          # still just the one
        self.mouse[0], self.mouse[1] = self.INSIDE
        ct._update()
        self.assertEqual(self.applied, [True, False])

    def test_never_flips_mid_drag(self):
        """Dropping the flag while a button is down would kill the drag."""
        ct = self._rig(self.PILL, *self.INSIDE)
        self.assertEqual(self.applied, [False])
        self.mouse[0], self.mouse[1], self.mouse[2] = 10, 190, 1   # dragged out
        ct._update()
        self.assertEqual(self.applied, [False])         # unchanged
        self.mouse[2] = 0                                # released
        ct._update()
        self.assertEqual(self.applied, [False, True])

    def test_a_failed_apply_is_retried(self):
        ct = self._rig(self.PILL, *self.OUTSIDE, apply_ok=False)
        ct._update()
        self.assertEqual(self.applied, [True, True])   # not latched on failure

    def test_no_mouse_reading_is_survivable(self):
        """No AppKit (or a failed read) -> no-op, never an exception."""
        applied = []
        ct = bn.ClickThrough(None, mouse_state=lambda: None,
                             apply_ignore=lambda ig: (applied.append(ig), True)[1])
        ct.set_rects(self.PILL)
        ct._update()
        self.assertEqual(applied, [])

    def test_start_refuses_to_arm_without_pyobjc(self):
        ct = bn.ClickThrough(None, mouse_state=lambda: None,
                             apply_ignore=lambda ig: True)
        self.assertFalse(ct.start())
        self.assertFalse(ct._armed)

    def test_kill_switch_disables_arming(self):
        os.environ["AUTOCINE_NO_CLICKTHROUGH"] = "1"
        try:
            ct = bn.ClickThrough(None, mouse_state=lambda: (0, 0, 0),
                                 apply_ignore=lambda ig: True)
            self.assertFalse(ct.start())
            self.assertFalse(ct._armed)
        finally:
            del os.environ["AUTOCINE_NO_CLICKTHROUGH"]

    def test_stop_restores_interactivity(self):
        ct = self._rig(self.PILL, *self.OUTSIDE)
        self.assertEqual(self.applied, [True])
        ct.stop()
        self.assertEqual(self.applied, [True, False])
        self.assertFalse(ct._armed)

    def test_stop_is_quiet_when_already_interactive(self):
        ct = self._rig(self.PILL, *self.INSIDE)
        ct.stop()
        self.assertEqual(self.applied, [False])   # no redundant write

    def test_a_dying_watcher_restores_interactivity(self):
        """The watcher is the only thing that can turn click-through back off,
        so if it exits the flag has to come off with it."""
        ct = self._rig(self.PILL, *self.OUTSIDE)
        self.assertEqual(self.applied, [True])
        ct._stop.set()          # make the loop exit immediately
        ct._loop()
        self.assertEqual(self.applied, [True, False])
        self.assertFalse(ct._armed)

    def test_poller_converges_on_a_real_window_frame(self):
        """End-to-end through start(): the thread runs and un-latches."""
        applied = []
        box = {"frame": (1000.0, 1000.0, 400.0, 200.0)}   # pill far from (0,0)
        ct = bn.ClickThrough(
            None,
            mouse_state=lambda: (0.0, 0.0, 0),
            apply_ignore=lambda ig: (applied.append(ig), True)[1],
            frame_reader=lambda: box["frame"])
        ct.POLL_SEC = 0.005
        self.assertTrue(ct.start())
        try:
            ct.set_rects(self.PILL)
            deadline = time.time() + 2.0
            while applied != [True] and time.time() < deadline:
                time.sleep(0.01)
            self.assertEqual(applied, [True])
            box["frame"] = (-100.0, -100.0, 400.0, 200.0)  # pill now under (0,0)
            while applied != [True, False] and time.time() < deadline:
                time.sleep(0.01)
            self.assertEqual(applied, [True, False])
        finally:
            ct.stop()


class ExcludeFromCapture(unittest.TestCase):
    """The sharingType write, against a stand-in NSWindow.

    This pins that we make the CALL, with the right value, and that a window
    which won't take it fails soft. It proves nothing about whether an
    avfoundation recording then leaves the window out — that needs real pixels.
    """

    class _Win(object):
        def __init__(self):
            self.shared = []

        def setSharingType_(self, value):
            self.shared.append(value)

    def test_sharing_none_is_requested(self):
        win = self._Win()
        self.assertTrue(bn._apply_sharing_none(win))
        self.assertEqual(win.shared, [0])          # NSWindowSharingNone
        self.assertEqual(bn.NSWINDOW_SHARING_NONE, 0)

    def test_a_window_that_refuses_the_selector_fails_soft(self):
        class _Old(object):
            def setSharingType_(self, value):
                raise AttributeError("unrecognized selector")

        self.assertFalse(bn._apply_sharing_none(_Old()))

    def test_no_nswindow_is_a_quiet_no_op(self):
        """No pywebview/pyobjc behind the window -> False, never an exception,
        and the caller keeps today's behavior (window shows up in the take)."""
        self.assertFalse(bn.exclude_from_capture(None))


class JoinAllSpaces(unittest.TestCase):
    """The collectionBehavior write, against a stand-in NSWindow.

    Pins the VALUE, because the two bits are what make the pill usable over a
    full-screen app and dropping either one is a silent regression: without
    CanJoinAllSpaces it stays on the Space it was born on, without
    FullScreenAuxiliary macOS defers it until the app leaves full screen. Says
    nothing about what a real Space switch does — that needs the real pill.
    """

    class _Win(object):
        def __init__(self):
            self.behaviors = []

        def setCollectionBehavior_(self, value):
            self.behaviors.append(value)

    def test_both_bits_are_requested(self):
        win = self._Win()
        self.assertTrue(bn._apply_all_spaces(win))
        self.assertEqual(win.behaviors, [1 | 256])
        self.assertEqual(bn.ALL_SPACES_BEHAVIOR, 1 | 256)

    def test_behavior_is_replaced_not_merged(self):
        """NSWindowCollectionBehaviorManaged (1<<2) is mutually exclusive with
        CanJoinAllSpaces, so the write must never OR into what's there."""
        self.assertFalse(bn.ALL_SPACES_BEHAVIOR & (1 << 2))

    def test_a_window_that_refuses_the_selector_fails_soft(self):
        class _Old(object):
            def setCollectionBehavior_(self, value):
                raise AttributeError("unrecognized selector")

        self.assertFalse(bn._apply_all_spaces(_Old()))

    def test_no_nswindow_is_a_quiet_no_op(self):
        self.assertFalse(bn.join_all_spaces(None))

    def test_kill_switch_skips_the_call(self):
        """AUTOCINE_NO_ALL_SPACES=1 restores the pre-feature behavior, which
        matters on avfoundation: there the pill can't be excluded from a take,
        so a Space it can't reach is a take it can't spoil."""
        old = os.environ.get("AUTOCINE_NO_ALL_SPACES")
        os.environ["AUTOCINE_NO_ALL_SPACES"] = "1"
        try:
            self.assertFalse(bn.join_all_spaces(object()))
        finally:
            if old is None:
                os.environ.pop("AUTOCINE_NO_ALL_SPACES", None)
            else:
                os.environ["AUTOCINE_NO_ALL_SPACES"] = old


class CaptureExclusion(unittest.TestCase):
    def test_no_nswindow_is_a_quiet_no_op(self):
        self.assertFalse(bn.exclude_from_capture(None))

    def test_kill_switch_skips_the_call(self):
        """AUTOCINE_NO_CAPTURE_EXCLUDE=1 leaves sharingType alone so the bar
        can be screenshotted (website / README / bug report). It must return
        before touching the window at all -- a stand-in object here would
        otherwise raise inside _ns_window."""
        old = os.environ.get("AUTOCINE_NO_CAPTURE_EXCLUDE")
        os.environ["AUTOCINE_NO_CAPTURE_EXCLUDE"] = "1"
        try:
            self.assertFalse(bn.exclude_from_capture(object()))
        finally:
            if old is None:
                os.environ.pop("AUTOCINE_NO_CAPTURE_EXCLUDE", None)
            else:
                os.environ["AUTOCINE_NO_CAPTURE_EXCLUDE"] = old


class PanelWindows(unittest.TestCase):
    """The pywebview class swap, without pywebview.

    Only the swap's WIRING is testable offline -- that it targets
    `BrowserView.WindowHost`, that the mask constant is right, and that it
    fails soft and honours its kill switch. Whether a panel then follows the
    user across Spaces is a window-server question; `tools/` probes measured
    it (NSPanel + mask 128 follows, everything else is pinned) and no unit
    test can.
    """

    def test_the_nonactivating_mask_is_the_appkit_value(self):
        self.assertEqual(bn.NSWINDOW_STYLE_MASK_NONACTIVATING_PANEL, 128)

    def test_the_swap_targets_pywebviews_window_class(self):
        """It patches `BrowserView.WindowHost`, the class attribute pywebview
        looks up at window-creation time. A rename upstream must fail here
        rather than silently leaving plain NSWindows behind."""
        cocoa = types.ModuleType("cocoa")

        class _BrowserView(object):
            class WindowHost(object):
                pass

        cocoa.BrowserView = _BrowserView
        before = _BrowserView.WindowHost
        sentinel = object()
        real_make = bn._make_panel_host
        bn._make_panel_host = lambda: sentinel
        real_import = __import__

        def _fake_import(name, *a, **kw):
            if name == "webview.platforms":
                mod = types.ModuleType("webview.platforms")
                mod.cocoa = cocoa
                return mod
            return real_import(name, *a, **kw)

        builtins.__import__ = _fake_import
        try:
            self.assertTrue(bn.use_panel_windows())
        finally:
            builtins.__import__ = real_import
            bn._make_panel_host = real_make
        self.assertIs(_BrowserView.WindowHost, sentinel)
        self.assertIsNot(_BrowserView.WindowHost, before)

    def test_no_pywebview_fails_soft(self):
        """Without pywebview the caller keeps plain windows -- i.e. today's
        behavior minus the Spaces fix, never an exception at bar start-up."""
        real_import = builtins.__import__

        def _missing_webview(name, *args, **kwargs):
            if name == "webview.platforms":
                raise ImportError("simulated missing pywebview")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = _missing_webview
        try:
            self.assertFalse(bn.use_panel_windows())
        finally:
            builtins.__import__ = real_import

    def test_kill_switch_skips_the_swap(self):
        old = os.environ.get("AUTOCINE_NO_PANEL_WINDOWS")
        os.environ["AUTOCINE_NO_PANEL_WINDOWS"] = "1"
        calls = []
        real_make = bn._make_panel_host
        bn._make_panel_host = lambda: calls.append(1)
        try:
            self.assertFalse(bn.use_panel_windows())
            self.assertEqual(calls, [])
        finally:
            bn._make_panel_host = real_make
            if old is None:
                os.environ.pop("AUTOCINE_NO_PANEL_WINDOWS", None)
            else:
                os.environ["AUTOCINE_NO_PANEL_WINDOWS"] = old


class UnionRect(unittest.TestCase):
    def test_one_rect_is_itself(self):
        self.assertEqual(bn.union_rect([(10, 20, 30, 40)]), (10, 20, 30, 40))

    def test_spans_every_rect(self):
        got = bn.union_rect([(0, 0, 100, 100), (200, 50, 100, 200)])
        self.assertEqual(got, (0, 0, 300, 250))

    def test_nothing_usable_is_none(self):
        self.assertIsNone(bn.union_rect([]))
        self.assertIsNone(bn.union_rect(None))

    def test_malformed_and_degenerate_entries_are_skipped(self):
        got = bn.union_rect([("x", 0, 10, 10), (0, 0, 0, 50), (5, 5, 10, 10)])
        self.assertEqual(got, (5, 5, 10, 10))


class DockGeometry(unittest.TestCase):
    """Layout coords in, Cocoa coords out. Screen is injected so this runs
    with no Cocoa at all."""

    SCREEN = (0, 0, 1440, 900)          # Cocoa (x, y, w, h) of the main screen

    def test_anchor_is_bottom_centre_of_the_target(self):
        cx, by = bn.dock_anchor((0, 0, 1440, 900), margin=32)
        self.assertEqual((cx, by), (720.0, 868.0))

    def test_cocoa_frame_is_the_inverse_of_to_layout_xy(self):
        frame = bn.to_cocoa_frame(520, 796, 400, 100, screen=self.SCREEN)
        self.assertEqual(frame, (520, 4, 400, 100))
        sx, sy, _sw, sh = self.SCREEN
        x, y, _w, h = frame
        self.assertEqual((int(x - sx), int(sy + sh - (y + h))), (520, 796))

    def test_visible_bottom_lands_on_the_anchor(self):
        """`dock_frame` positions the WINDOW, but the anchor names the PILL's
        bottom edge — the difference is the shadow slack below the content."""
        anchor = bn.dock_anchor((0, 0, 1440, 900))
        frame = bn.dock_frame(anchor, (400, 100), screen=self.SCREEN)
        _sx, sy, _sw, sh = self.SCREEN
        visible_bottom_layout = sy + sh - (frame[1] + bn.SHADOW_PAD)
        self.assertEqual(visible_bottom_layout, anchor[1])

    def test_pill_is_centred_on_the_target(self):
        anchor = bn.dock_anchor((200, 100, 800, 600))
        frame = bn.dock_frame(anchor, (400, 100), screen=self.SCREEN)
        self.assertEqual(frame[0] + frame[2] / 2.0, 600)   # 200 + 800/2

    def test_the_window_box_stays_inside_a_fullscreen_target(self):
        """DOCK_MARGIN > SHADOW_PAD, so a dock onto a full-screen window needs
        no clamping — otherwise every such dock would be pulled up 4pt."""
        self.assertGreater(bn.DOCK_MARGIN, bn.SHADOW_PAD)
        anchor = bn.dock_anchor((0, 0, 1440, 900))
        frame = bn.dock_frame(anchor, (400, 100), screen=self.SCREEN)
        self.assertEqual(bn.clamp_frame(frame, self.SCREEN), frame)

    def test_an_unreadable_screen_is_a_quiet_none(self):
        """No Cocoa (or a screen that won't read) must not raise across the JS
        bridge — the caller treats None as "leave the pill where it is"."""
        real = bn.screen_frames
        bn.screen_frames = lambda: []
        try:
            self.assertIsNone(bn._origin_screen_frame())
            self.assertIsNone(bn.to_cocoa_frame(0, 0, 10, 10))
            self.assertIsNone(bn.dock_frame((100.0, 200.0), (400, 100)))
        finally:
            bn.screen_frames = real


class GlassGeometry(unittest.TestCase):
    """Page rects -> the frames + mask radii of the native glass. Pure."""

    def test_capsule_radius_is_half_the_shorter_side(self):
        """`border-radius: 999px` is clamped by CSS to half the shorter side,
        so the pill (76 tall) and a one- or two-line hint are all capsules."""
        self.assertEqual(bn.capsule_radius(682, 76), 38.0)
        self.assertEqual(bn.capsule_radius(600, 45), 22.5)
        self.assertEqual(bn.capsule_radius(20, 76), 10.0)
        self.assertEqual(bn.capsule_radius(0, 0), 0.0)

    def test_flipped_host_keeps_layout_y(self):
        """WKWebView is flipped (measured), so a layout rect maps straight
        across -- only the inset moves it."""
        got = bn.glass_frames([(28, 28, 682, 76)], host_h=132, flipped=True)
        self.assertEqual(got, [(29.0, 29.0, 680.0, 74.0, 37.0)])

    def test_unflipped_host_flips_y_against_its_height(self):
        got = bn.glass_frames([(28, 28, 682, 76)], host_h=132, flipped=False)
        # 132 - (28 + 1) - 74 = 29: symmetric here because pad is equal
        self.assertEqual(got, [(29.0, 29.0, 680.0, 74.0, 37.0)])
        got = bn.glass_frames([(28, 28, 682, 76)], host_h=200, flipped=False)
        self.assertEqual(got[0][1], 200 - 29 - 74)

    def test_glass_sits_inside_the_border(self):
        """Inset by the CSS border width on every side, radius shrunk with it
        so the curve stays concentric with the border's."""
        (x, y, w, h, r), = bn.glass_frames([(10, 20, 100, 40)], 0, True)
        self.assertEqual((x, y, w, h), (11.0, 21.0, 98.0, 38.0))
        self.assertEqual(r, 19.0)
        self.assertEqual(bn.GLASS_INSET, 1)

    def test_every_glass_frame_lies_inside_its_click_rect(self):
        """Through content_layout, as bar_fit does: pill + hint, both inside
        the window and inside the regions that catch clicks."""
        w, h, rects = bn.content_layout(682, 76, hint_w=600, hint_h=29)
        frames = bn.glass_frames(rects, h, flipped=True)
        self.assertEqual(len(frames), 2)
        for (gx, gy, gw, gh, _r), (rx, ry, rw, rh) in zip(frames, rects):
            self.assertGreater(gx, rx)
            self.assertGreater(gy, ry)
            self.assertLess(gx + gw, rx + rw)
            self.assertLess(gy + gh, ry + rh)
            self.assertLessEqual(gx + gw, w)
            self.assertLessEqual(gy + gh, h)
        # both centred on the window, like the page centres them
        for gx, _gy, gw, _gh, _r in frames:
            self.assertAlmostEqual(gx + gw / 2.0, w / 2.0)

    def test_the_pill_stays_put_across_face_changes(self):
        """The faces swing the width ~682 -> ~245 and the hint comes and goes.
        With fit_frame pinning centre-x and the top edge, the glass must land
        on the same screen spot every time -- same centre, same top."""
        frame = (300.0, 500.0, 738.0, 132.0)             # Cocoa, idle pill
        spots = []
        # even width deltas: fit_frame rounds the window's x to a whole point,
        # which moves the page and the glass together by <= 0.5pt otherwise
        for pill_w, hint_w, hint_h in ((682, 0, 0), (246, 0, 0),
                                       (246, 600, 29), (682, 600, 45)):
            w, h, rects = bn.content_layout(pill_w, 76, hint_w, hint_h)
            frame = bn.fit_frame(frame, w, h)
            gx, gy, gw, gh, r = bn.glass_frames(rects, h, flipped=True)[0]
            fx, fy, fw, fh = frame
            centre_x = fx + gx + gw / 2.0
            top = fy + fh - gy                          # Cocoa y of glass top
            spots.append((centre_x, top, r))
        self.assertEqual(len(set(spots)), 1, spots)

    def test_malformed_and_swallowed_rects_are_skipped(self):
        got = bn.glass_frames([None, "x", (1, 2), (0, 0, 2, 2), (0, 0, 10, 4)],
                              0, True)
        self.assertEqual(got, [(1.0, 1.0, 8.0, 2.0, 1.0)])
        self.assertEqual(bn.glass_frames(None, 0, True), [])

    def test_autoresizing_pins_centre_and_top(self):
        """Both x margins flexible (centred) and the BOTTOM margin flexible
        (top-anchored) -- bottom is max-y when flipped, min-y when not."""
        self.assertEqual(bn.glass_autoresizing_mask(True), 1 | 4 | 32)
        self.assertEqual(bn.glass_autoresizing_mask(False), 1 | 4 | 8)

    def test_the_hardcoded_appkit_values(self):
        self.assertEqual((bn.NSVIEW_MIN_X_MARGIN, bn.NSVIEW_MAX_X_MARGIN,
                          bn.NSVIEW_MIN_Y_MARGIN, bn.NSVIEW_MAX_Y_MARGIN),
                         (1, 4, 8, 32))
        self.assertEqual(bn.NSVISUAL_EFFECT_MATERIAL_HUD_WINDOW, 13)
        self.assertEqual(bn.NSVISUAL_EFFECT_BLENDING_BEHIND_WINDOW, 0)
        self.assertEqual(bn.NSVISUAL_EFFECT_STATE_ACTIVE, 1)
        self.assertEqual(bn.NSWINDOW_BELOW, -1)


class _FakeView(object):
    """Stand-in NSVisualEffectView: records what the glass does to it."""

    def __init__(self, flipped):
        self.flipped = flipped
        self.frame = None
        self.hidden = True
        self.mask = None
        self.mask_sets = 0
        self.removed = False

    def setFrame_(self, rect):
        (x, y), (w, h) = rect
        self.frame = (x, y, w, h)

    def setHidden_(self, hidden):
        self.hidden = bool(hidden)

    def setMaskImage_(self, image):
        self.mask = image
        self.mask_sets += 1

    def removeFromSuperview(self):
        self.removed = True


class _FakeHost(object):
    """Stand-in WKWebView."""

    def __init__(self, flipped=True, height=132.0):
        self.flipped = flipped
        self.height = height
        self.added = []

    def isFlipped(self):
        return self.flipped

    def bounds(self):
        size = types.SimpleNamespace(width=738.0, height=self.height)
        return types.SimpleNamespace(size=size)

    def addSubview_positioned_relativeTo_(self, view, place, other):
        self.added.append((view, place, other))


class GlassLifecycle(unittest.TestCase):
    """`Glass` with every Cocoa seam stubbed and the main-thread queue run
    synchronously. Pins create-once / re-frame / hide-extras and the
    soft-fail paths; what the blur LOOKS like needs the real pill."""

    PILL = [(28.0, 28.0, 682.0, 76.0)]
    PILL_HINT = [(69.0, 28.0, 600.0, 76.0), (28.0, 112.0, 682.0, 29.0)]

    def _glass(self, host=None, make_view=None, **kw):
        self.host = host if host is not None else _FakeHost()
        self.views = []
        self.masks = []

        def _view(flipped):
            v = _FakeView(flipped)
            self.views.append(v)
            return v

        def _mask(radius):
            self.masks.append(radius)
            return ("mask", radius)

        return bn.Glass(None, host_reader=kw.get("reader", lambda: self.host),
                        call_on_main=kw.get("call", lambda fn: (fn(), True)[1]),
                        make_view=make_view or _view, make_mask=_mask)

    def setUp(self):
        self._old_env = os.environ.pop("AUTOCINE_NO_VIBRANCY", None)

    def tearDown(self):
        os.environ.pop("AUTOCINE_NO_VIBRANCY", None)
        if self._old_env is not None:
            os.environ["AUTOCINE_NO_VIBRANCY"] = self._old_env

    def test_install_puts_hidden_views_below_the_web_content(self):
        g = self._glass()
        self.assertTrue(g.install())
        self.assertTrue(g.installed)
        self.assertEqual(len(self.views), bn.Glass.MAX_VIEWS)
        for view, place, other in self.host.added:
            self.assertEqual((place, other), (bn.NSWINDOW_BELOW, None))
            self.assertTrue(view.hidden)            # nothing shows until framed
            self.assertTrue(view.flipped)           # told the host's flip

    def test_update_frames_the_pill_and_hides_the_spare(self):
        g = self._glass()
        g.install()
        self.assertTrue(g.update(self.PILL))
        pill, spare = self.views
        self.assertEqual(pill.frame, (29.0, 29.0, 680.0, 74.0))
        self.assertFalse(pill.hidden)
        self.assertEqual(pill.mask, ("mask", 37.0))
        self.assertTrue(spare.hidden)

    def test_views_are_reused_as_the_hint_comes_and_goes(self):
        g = self._glass()
        g.install()
        g.update(self.PILL)
        g.update(self.PILL_HINT)
        pill, hint = self.views
        self.assertFalse(hint.hidden)
        self.assertEqual(hint.frame, (29.0, 113.0, 680.0, 27.0))
        self.assertEqual(hint.mask, ("mask", 13.5))
        g.update(self.PILL)
        self.assertTrue(hint.hidden)
        self.assertFalse(pill.hidden)
        self.assertEqual(len(self.views), bn.Glass.MAX_VIEWS)   # never rebuilt
        self.assertEqual(len(self.host.added), bn.Glass.MAX_VIEWS)

    def test_masks_are_built_once_per_radius_and_set_only_on_change(self):
        g = self._glass()
        g.install()
        for _ in range(3):
            g.update(self.PILL)                 # face changes keep the height
        g.update([(28.0, 28.0, 245.0, 76.0)])
        self.assertEqual(self.masks, [37.0])
        self.assertEqual(self.views[0].mask_sets, 1)

    def test_update_installs_lazily_when_shown_never_ran(self):
        g = self._glass()
        self.assertFalse(g.installed)
        g.update(self.PILL)
        self.assertTrue(g.installed)
        self.assertFalse(self.views[0].hidden)

    def test_an_unflipped_host_is_flipped_against_its_height(self):
        g = self._glass(host=_FakeHost(flipped=False, height=200.0))
        g.update(self.PILL)
        self.assertEqual(self.views[0].frame, (29.0, 200 - 29 - 74, 680.0, 74.0))
        self.assertFalse(self.views[0].flipped)

    def test_no_web_view_yet_is_not_a_failure(self):
        """`shown` can race the web view; a later update must still install."""
        box = {"host": None}
        g = self._glass(reader=lambda: box["host"])
        g.install()
        self.assertFalse(g.installed)
        box["host"] = _FakeHost()
        self.host = box["host"]
        g.update(self.PILL)
        self.assertTrue(g.installed)

    def test_a_view_that_will_not_build_fails_soft_and_stays_off(self):
        made = []

        def _flaky(flipped):
            if made:
                raise RuntimeError("no NSVisualEffectView")
            v = _FakeView(flipped)
            made.append(v)
            return v

        g = self._glass(make_view=_flaky)
        self.assertFalse(g._install_now())
        self.assertFalse(g.installed)
        self.assertTrue(made[0].removed)        # half-built install undone
        self.assertFalse(g.update(self.PILL))   # and never retried
        self.assertFalse(g.install())

    def test_a_failed_update_hides_the_glass_for_good(self):
        g = self._glass()
        g.install()
        g.update(self.PILL)

        def _boom(rect):
            raise RuntimeError("setFrame failed")

        self.views[0].setFrame_ = _boom
        g.update(self.PILL_HINT)
        self.assertFalse(g.installed)
        self.assertTrue(all(v.hidden for v in self.views))
        self.assertFalse(g.update(self.PILL))

    def test_a_teardown_tells_the_page_to_drop_the_translucent_pill(self):
        """A bar_fit reply only ever ADDS the class, so the page is told
        directly -- off the main thread, since evaluate_js blocks on a reply
        the main thread would have to deliver."""
        import threading
        told = threading.Event()
        calls = []
        window = types.SimpleNamespace(
            evaluate_js=lambda js: (calls.append(js), told.set()))
        g = bn.Glass(window, host_reader=lambda: _FakeHost(),
                     call_on_main=lambda fn: (fn(), True)[1],
                     make_view=_FakeView, make_mask=lambda r: r)
        g.install()
        g._teardown()
        self.assertTrue(told.wait(2.0))
        self.assertEqual(calls, [bn.GLASS_LOST_JS])

    def test_the_first_visible_frame_tells_the_page_once(self):
        """The page is told directly the first time a view actually shows
        (the lazy path's fit replied before the views existed), and only
        once; a frame with no rects shows nothing and says nothing."""
        import threading
        told = threading.Event()
        calls = []
        window = types.SimpleNamespace(
            evaluate_js=lambda js: (calls.append(js), told.set()))
        g = bn.Glass(window, host_reader=lambda: _FakeHost(),
                     call_on_main=lambda fn: (fn(), True)[1],
                     make_view=_FakeView, make_mask=lambda r: r)
        g.update([])
        time.sleep(0.05)
        self.assertEqual(calls, [])
        g.update(self.PILL)
        self.assertTrue(told.wait(2.0))
        g.update(self.PILL_HINT)
        g.update(self.PILL)
        time.sleep(0.05)
        self.assertEqual(calls, [bn.GLASS_ON_JS])

    def test_on_and_lost_scripts_agree_on_the_page_flag(self):
        """A late "glass on" (or a late bar_fit reply) must not undo a loss:
        both scripts go through window.autocineGlassLost, and bar.js's
        applyGlass reads the same flag."""
        self.assertIn("autocineGlassLost=true", bn.GLASS_LOST_JS)
        self.assertIn("classList.remove('glass')", bn.GLASS_LOST_JS)
        self.assertTrue(bn.GLASS_ON_JS.startswith("window.autocineGlassLost||"))
        self.assertIn("classList.add('glass')", bn.GLASS_ON_JS)
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, os.pardir, "studio_web", "bar.js"),
                  encoding="utf-8") as f:
            src = f.read()
        body = src.split("function applyGlass")[1].split("\n  }")[0]
        self.assertIn("window.autocineGlassLost", body)
        self.assertNotIn("toggle", body)       # a reply never removes it

    def test_no_main_thread_queue_means_not_installed(self):
        g = self._glass(call=lambda fn: False)
        self.assertFalse(g.install())
        self.assertFalse(g.update(self.PILL))
        self.assertFalse(g.installed)

    def test_kill_switch_touches_nothing(self):
        os.environ["AUTOCINE_NO_VIBRANCY"] = "1"
        queued = []
        g = self._glass(call=lambda fn: (queued.append(fn), True)[1])
        self.assertFalse(g.install())
        self.assertFalse(g.update(self.PILL))
        self.assertEqual(queued, [])
        self.assertFalse(g.installed)

    def test_no_pywebview_is_a_quiet_no_op(self):
        """The real seams with nothing behind them: no web view to find."""
        g = bn.Glass(None, call_on_main=lambda fn: (fn(), True)[1])
        self.assertTrue(g.install())            # queued...
        self.assertFalse(g.installed)           # ...and found nothing


class BarFitReportsGlass(unittest.TestCase):
    """`_NativeBarApi.bar_fit`'s "glass" flag: only when the views exist,
    and the re-frame is queued AFTER set_frame so it tracks the resize."""

    def setUp(self):
        from autocine import studio_app
        self.order = []
        self._patch(bn, "frame_of", lambda win: (100, 700, 400, 136))
        self._patch(bn, "screen_frames", lambda: [(0, 0, 1440, 900)])
        self._patch(bn, "set_frame",
                    lambda win, frame: self.order.append("set_frame") or True)
        self.api = studio_app._NativeBarApi()
        self.api.window = object()

    def _patch(self, mod, name, value):
        orig = getattr(mod, name)
        setattr(mod, name, value)
        self.addCleanup(lambda: setattr(mod, name, orig))

    def _stub_glass(self, installed, queued=True):
        test = self

        class _Stub(object):
            def __init__(self):
                self.installed = installed
                self.rects = None

            def update(self, rects):
                test.order.append("glass")
                self.rects = rects
                return queued

        self.api.glass = _Stub()
        return self.api.glass

    def test_no_glass_object_reports_nothing(self):
        got = self.api.bar_fit(682, 76)
        self.assertTrue(got["ok"])
        self.assertNotIn("glass", got)

    def test_installed_glass_is_reported_and_follows_set_frame(self):
        stub = self._stub_glass(installed=True)
        got = self.api.bar_fit(682, 76, 600, 29)
        self.assertIs(got.get("glass"), True)
        self.assertEqual(self.order, ["set_frame", "glass"])
        self.assertEqual(stub.rects, bn.content_layout(682, 76, 600, 29)[2])

    def test_not_installed_is_not_reported(self):
        self._stub_glass(installed=False)
        self.assertNotIn("glass", self.api.bar_fit(682, 76))

    def test_an_update_that_could_not_be_queued_is_not_reported(self):
        self._stub_glass(installed=True, queued=False)
        self.assertNotIn("glass", self.api.bar_fit(682, 76))


class BarFitIsSerialized(unittest.TestCase):
    """Two fits in one JS tick run on two pywebview bridge threads. The main
    thread's queue is FIFO, so what matters is the ORDER each fit's work is
    queued in: the window resize, click-through rects and glass re-frame of
    one fit must reach the queue as a unit, and the unit that lands last must
    be the one the page sent last. The other tests run the queue
    synchronously and cannot see this."""

    WIDE = (1003, 76, 0, 0)          # idle face
    NARROW = (277, 76, 660, 44)      # recording face + two-line hint

    def setUp(self):
        from autocine import studio_app
        self.queue = []                      # the main thread's FIFO, in order
        self.qlock = threading.Lock()
        self.inside = threading.Event()
        self._patch(bn, "frame_of", lambda win: (100, 700, 1059, 132))
        self._patch(bn, "screen_frames", lambda: [(0, 0, 1440, 900)])
        self._patch(bn, "set_frame", lambda win, frame: self._q(
            "frame", (frame[2], frame[3])))
        test = self

        class _SlowClickThrough(object):
            """Dawdles between set_frame and the glass re-frame for the WIDE
            fit only -- the GIL-releasing ObjC reads ClickThrough._update does
            there are what let the other thread overtake in the real pill."""
            def set_rects(self, rects):
                test._q("rects", len(rects))
                if rects and rects[0][2] == test.WIDE[0]:
                    test.inside.set()
                    time.sleep(0.15)

        class _QueuedGlass(object):
            installed = True

            def update(self, rects):
                w, h, _ = bn.content_layout(*test._dims_of(rects))
                test._q("glass", (w, h))
                return True

        self.api = studio_app._NativeBarApi()
        self.api.window = object()
        self.api.click_through = _SlowClickThrough()
        self.api.glass = _QueuedGlass()

    def _patch(self, mod, name, value):
        orig = getattr(mod, name)
        setattr(mod, name, value)
        self.addCleanup(lambda: setattr(mod, name, orig))

    def _q(self, kind, what):
        with self.qlock:
            self.queue.append((kind, what))

    def _dims_of(self, rects):
        pill = rects[0]
        hint = rects[1] if len(rects) > 1 else (0, 0, 0, 0)
        return (pill[2], pill[3], hint[2], hint[3])

    def _last(self, kind):
        return [w for k, w in self.queue if k == kind][-1]

    def _race(self, first_args, second_args):
        """`first` enters bar_fit and stalls mid-fit; `second` is sent while
        it is stalled."""
        results = {}
        a = threading.Thread(target=lambda: results.__setitem__(
            "first", self.api.bar_fit(*first_args)))
        a.start()
        self.assertTrue(self.inside.wait(2.0))
        b = threading.Thread(target=lambda: results.__setitem__(
            "second", self.api.bar_fit(*second_args)))
        b.start()
        a.join(3.0)
        b.join(3.0)
        return results

    def test_each_fits_resize_and_glass_are_queued_together(self):
        self._race(self.WIDE, self.NARROW)
        kinds = [k for k, _ in self.queue]
        self.assertEqual(kinds, ["frame", "rects", "glass"] * 2)
        self.assertEqual(self._last("frame"), self._last("glass"))
        self.assertEqual(self._last("frame"),
                         bn.content_layout(*self.NARROW)[:2])

    def test_a_fit_that_lost_the_race_to_a_newer_one_is_dropped(self):
        """The page sent NARROW last (seq 2) but its thread got in first; the
        older WIDE fit (seq 1) arriving after it must not win."""
        self.api.bar_fit(*(self.NARROW + ("pg", 2)))
        got = self.api.bar_fit(*(self.WIDE + ("pg", 1)))
        self.assertEqual(got, {"ok": False, "stale": True})
        self.assertEqual(self._last("frame"), self._last("glass"))
        self.assertEqual(self._last("frame"),
                         bn.content_layout(*self.NARROW)[:2])

    def test_a_reloaded_page_starts_a_fresh_count(self):
        self.api.bar_fit(*(self.NARROW + ("old-page", 40)))
        got = self.api.bar_fit(*(self.WIDE + ("new-page", 1)))
        self.assertTrue(got["ok"])
        self.assertEqual(self._last("frame"), bn.content_layout(*self.WIDE)[:2])

    def test_a_fit_without_a_sequence_is_never_dropped(self):
        """The old four-argument call (and every other caller) is unchanged."""
        self.api.bar_fit(*(self.NARROW + ("pg", 9)))
        self.assertTrue(self.api.bar_fit(*self.WIDE)["ok"])
        self.assertEqual(self._last("frame"), bn.content_layout(*self.WIDE)[:2])


class LazyGlassReachesThePage(unittest.TestCase):
    """`shown` never installed: the fit that installs only QUEUES it, so its
    reply cannot say "glass". The page must still get the class once the
    queue drains -- modelled with a deferred queue, as the real main thread
    is."""

    def setUp(self):
        from autocine import studio_app
        self._old_env = os.environ.pop("AUTOCINE_NO_VIBRANCY", None)
        self.addCleanup(self._restore_env)
        for name, value in (("frame_of", lambda win: (100, 700, 738, 132)),
                            ("screen_frames", lambda: [(0, 0, 1440, 900)]),
                            ("set_frame", lambda win, frame: True)):
            orig = getattr(bn, name)
            setattr(bn, name, value)
            self.addCleanup(lambda n=name, o=orig: setattr(bn, n, o))
        self.deferred = []
        self.told = threading.Event()
        self.calls = []
        window = types.SimpleNamespace(
            evaluate_js=lambda js: (self.calls.append(js), self.told.set()))
        self.glass = bn.Glass(
            window, host_reader=lambda: _FakeHost(),
            call_on_main=lambda fn: (self.deferred.append(fn), True)[1],
            make_view=_FakeView, make_mask=lambda r: r)
        self.api = studio_app._NativeBarApi()
        self.api.window = window
        self.api.glass = self.glass

    def _restore_env(self):
        os.environ.pop("AUTOCINE_NO_VIBRANCY", None)
        if self._old_env is not None:
            os.environ["AUTOCINE_NO_VIBRANCY"] = self._old_env

    def test_the_installing_fit_replies_no_glass_then_the_page_is_told(self):
        got = self.api.bar_fit(682, 76)
        self.assertTrue(got["ok"])
        self.assertNotIn("glass", got)          # the install is only queued
        self.assertFalse(self.glass.installed)
        while self.deferred:                    # the main thread catches up
            self.deferred.pop(0)()
        self.assertTrue(self.glass.installed)
        self.assertFalse(self.glass._views[0].hidden)
        self.assertTrue(self.told.wait(2.0))
        self.assertEqual(self.calls, [bn.GLASS_ON_JS])
        # and from here on the replies carry it too
        self.assertIs(self.api.bar_fit(682, 76).get("glass"), True)


class Positions(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.path = os.path.join(self.td, "bar-pos.json")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_missing_file_reads_as_none(self):
        self.assertIsNone(bn.load_bar_position(self.path))
        self.assertIsNone(bn.load_face_position(self.path))

    def test_round_trip(self):
        self.assertTrue(bn.save_bar_position(120, 340, self.path))
        self.assertEqual(bn.load_bar_position(self.path), (120, 340))

    def test_bar_and_face_positions_are_independent(self):
        bn.save_bar_position(1, 2, self.path)
        bn.save_face_position(3, 4, self.path)
        self.assertEqual(bn.load_bar_position(self.path), (1, 2))
        self.assertEqual(bn.load_face_position(self.path), (3, 4))

    def test_corrupt_file_reads_as_none_and_is_recoverable(self):
        with open(self.path, "w") as f:
            f.write("{not json")
        self.assertIsNone(bn.load_bar_position(self.path))
        self.assertTrue(bn.save_bar_position(10, 20, self.path))
        self.assertEqual(bn.load_bar_position(self.path), (10, 20))

    def test_non_numeric_entries_are_rejected(self):
        with open(self.path, "w") as f:
            json.dump({"x": "left", "y": 5}, f)
        self.assertIsNone(bn.load_bar_position(self.path))
        self.assertFalse(bn.save_bar_position(None, 5, self.path))

    def test_unwritable_path_fails_soft(self):
        bad = os.path.join(self.td, "no-such-dir", "bar-pos.json")
        self.assertFalse(bn.save_bar_position(1, 2, bad))


class NotepadGeometry(unittest.TestCase):
    """The notes overlay stores SIZE next to position — it's resizable."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.path = os.path.join(self.td, "bar-pos.json")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_missing_reads_as_none(self):
        self.assertIsNone(bn.load_notepad_geometry(self.path))

    def test_round_trip(self):
        self.assertTrue(bn.save_notepad_geometry(20, 40, 360, 420, self.path))
        self.assertEqual(bn.load_notepad_geometry(self.path), (20, 40, 360, 420))

    def test_shares_the_file_with_the_pill_and_bubble(self):
        bn.save_bar_position(1, 2, self.path)
        bn.save_face_position(3, 4, self.path)
        bn.save_notepad_geometry(5, 6, 300, 200, self.path)
        self.assertEqual(bn.load_bar_position(self.path), (1, 2))
        self.assertEqual(bn.load_face_position(self.path), (3, 4))
        self.assertEqual(bn.load_notepad_geometry(self.path), (5, 6, 300, 200))

    def test_a_partial_record_reads_as_none(self):
        # width/height present but position missing -> not usable
        with open(self.path, "w") as f:
            json.dump({"notepad_w": 300, "notepad_h": 200}, f)
        self.assertIsNone(bn.load_notepad_geometry(self.path))

    def test_non_positive_size_is_rejected_on_write_and_read(self):
        self.assertFalse(bn.save_notepad_geometry(0, 0, 0, 200, self.path))
        with open(self.path, "w") as f:
            json.dump({"notepad_x": 1, "notepad_y": 2,
                       "notepad_w": -5, "notepad_h": 200}, f)
        self.assertIsNone(bn.load_notepad_geometry(self.path))

    def test_non_numeric_is_rejected(self):
        self.assertFalse(bn.save_notepad_geometry("x", 2, 3, 4, self.path))


class NotepadText(unittest.TestCase):
    """Notes content — a plain UTF-8 file, all failures read as empty."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.path = os.path.join(self.td, "notepad.txt")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_missing_reads_as_empty_string(self):
        self.assertEqual(bn.load_notepad_text(self.path), "")

    def test_round_trip_preserves_newlines_and_unicode(self):
        text = "Line one\nLine two — é, 你好\n\ttabbed"
        self.assertTrue(bn.save_notepad_text(text, self.path))
        self.assertEqual(bn.load_notepad_text(self.path), text)

    def test_none_is_stored_as_empty_not_a_crash(self):
        self.assertTrue(bn.save_notepad_text(None, self.path))
        self.assertEqual(bn.load_notepad_text(self.path), "")

    def test_overwrite_replaces(self):
        bn.save_notepad_text("first", self.path)
        bn.save_notepad_text("second", self.path)
        self.assertEqual(bn.load_notepad_text(self.path), "second")

    def test_unwritable_path_fails_soft(self):
        bad = os.path.join(self.td, "no-such-dir", "notepad.txt")
        self.assertFalse(bn.save_notepad_text("hi", bad))


if __name__ == "__main__":
    unittest.main()
