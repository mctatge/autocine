"""Window-enumeration filtering — offline, no Quartz needed.

Every case here is hand-built from a real CGWindowListCopyWindowInfo dump on a
two-display Mac (main 1440x900 at (0,0), a second 1366x768 at (-912,-768)), so
the junk shapes are the actual junk shapes macOS hands us.
"""

import sys
import unittest

from autocine import devices


MAIN = {"id": 1, "x": 0.0, "y": 0.0, "w": 1440.0, "h": 900.0, "main": True}
SECOND = {"id": 2, "x": -912.0, "y": -768.0, "w": 1366.0, "h": 768.0,
          "main": False}


def win(number=100, layer=0, owner="Google Chrome", name="New Tab", alpha=1.0,
        onscreen=True, pid=4242, x=100.0, y=100.0, w=800.0, h=600.0,
        bounds=True):
    info = {
        "kCGWindowNumber": number,
        "kCGWindowLayer": layer,
        "kCGWindowOwnerName": owner,
        "kCGWindowAlpha": alpha,
        "kCGWindowOwnerPID": pid,
    }
    if name is not None:
        info["kCGWindowName"] = name
    if onscreen:
        info["kCGWindowIsOnscreen"] = True
    if bounds:
        info["kCGWindowBounds"] = {"X": x, "Y": y, "Width": w, "Height": h}
    return info


def ids(entries):
    return [e["id"] for e in entries]


class LayerGate(unittest.TestCase):
    def test_non_zero_layers_dropped(self):
        # 25 menu-bar extras, 24 menubar, 23 Notification Centre, 20 Dock
        # icons, 5 side rails, 999/103 Chrome, 2005 OSDUIHelper, and the
        # cursor's huge negative layer.
        for layer in (25, 24, 23, 20, 5, 103, 999, 2005, -2147483624):
            got = devices._filter_windows([win(layer=layer)], [MAIN])
            self.assertEqual(got, [], "layer %d survived" % layer)

    def test_layer_zero_kept(self):
        got = devices._filter_windows([win(layer=0)], [MAIN])
        self.assertEqual(ids(got), [100])

    def test_missing_layer_treated_as_zero(self):
        info = win()
        del info["kCGWindowLayer"]
        self.assertEqual(ids(devices._filter_windows([info], [MAIN])), [100])

    def test_garbage_layer_dropped(self):
        got = devices._filter_windows([win(layer="frontmost")], [MAIN])
        self.assertEqual(got, [])


class AlphaGate(unittest.TestCase):
    def test_fully_transparent_dropped(self):
        got = devices._filter_windows([win(alpha=0.0)], [MAIN])
        self.assertEqual(got, [])

    def test_near_transparent_dropped(self):
        got = devices._filter_windows([win(alpha=0.05)], [MAIN])
        self.assertEqual(got, [])

    def test_faint_but_visible_kept(self):
        got = devices._filter_windows([win(alpha=0.5)], [MAIN])
        self.assertEqual(ids(got), [100])

    def test_missing_alpha_treated_as_opaque(self):
        info = win()
        del info["kCGWindowAlpha"]
        self.assertEqual(ids(devices._filter_windows([info], [MAIN])), [100])


class OnscreenGate(unittest.TestCase):
    def test_offscreen_key_absent_dropped(self):
        # macOS omits kCGWindowIsOnscreen entirely for minimized windows.
        got = devices._filter_windows([win(onscreen=False)], [MAIN])
        self.assertEqual(got, [])


class SizeGates(unittest.TestCase):
    def test_menubar_strip_dropped_on_height(self):
        # ~30 of these per dump: full-width, 24pt tall.
        got = devices._filter_windows(
            [win(x=0.0, y=0.0, w=1440.0, h=24.0)], [MAIN])
        self.assertEqual(got, [])

    def test_side_rail_dropped_on_width(self):
        # The 35x806 Grammarly rail clears the height gate easily.
        got = devices._filter_windows(
            [win(x=422.0, y=40.0, w=35.0, h=806.0)], [MAIN])
        self.assertEqual(got, [])

    def test_helper_shims_dropped_on_both(self):
        for w, h in ((0.0, 0.0), (1.0, 1.0), (54.0, 54.0)):
            got = devices._filter_windows([win(w=w, h=h)], [MAIN])
            self.assertEqual(got, [], "%gx%g survived" % (w, h))

    def test_exactly_at_the_gate_is_kept(self):
        got = devices._filter_windows(
            [win(w=devices.WINDOW_MIN_W_PT, h=devices.WINDOW_MIN_H_PT)], [MAIN])
        self.assertEqual(ids(got), [100])

    def test_gates_are_overridable(self):
        got = devices._filter_windows(
            [win(w=54.0, h=54.0)], [MAIN], min_w=10.0, min_h=10.0)
        self.assertEqual(ids(got), [100])

    def test_non_finite_bounds_dropped(self):
        got = devices._filter_windows(
            [win(x=float("nan"), y=0.0, w=800.0, h=600.0)], [MAIN])
        self.assertEqual(got, [])

    def test_missing_bounds_dropped(self):
        got = devices._filter_windows([win(bounds=False)], [MAIN])
        self.assertEqual(got, [])


class DisplayIntersection(unittest.TestCase):
    def test_off_display_window_dropped(self):
        # The transparent shims sit at y=-44 with a menubar-ish height; even at
        # full alpha they are entirely above the main display.
        got = devices._filter_windows(
            [win(x=0.0, y=-44.0, w=1440.0, h=24.0)], [MAIN])
        self.assertEqual(got, [])

    def test_partly_offscreen_window_kept_and_intersected(self):
        # Dragged half off the left edge and past the bottom.
        got = devices._filter_windows(
            [win(x=-200.0, y=500.0, w=800.0, h=600.0)], [MAIN])
        self.assertEqual(len(got), 1)
        e = got[0]
        self.assertEqual((e["x"], e["y"], e["w"], e["h"]),
                         (0.0, 500.0, 600.0, 400.0))

    def test_intersected_rect_can_fail_the_size_gate(self):
        # Only a 40pt-wide sliver of it is actually on the display.
        got = devices._filter_windows(
            [win(x=-760.0, y=100.0, w=800.0, h=600.0)], [MAIN])
        self.assertEqual(got, [])

    def test_unclipped_reports_the_windows_true_bounds(self):
        """`clip_to_display=False` is what occlusion-free capture needs: SCK
        records the window's own surface, so the part hanging off the screen
        IS in the file and a clipped rect describes less than the recording
        contains. Reproduces the reported case exactly -- a Word window at
        x=723 on a 1440-wide display, clipped to 717 against a buffer that
        held 1382 points of window (docs/architecture.md)."""
        args = dict(x=723.0, y=57.0, w=1382.0, h=835.0)
        clipped = devices._filter_windows([win(**args)], [MAIN])[0]
        self.assertEqual((clipped["x"], clipped["w"]), (723.0, 717.0))
        full = devices._filter_windows([win(**args)], [MAIN],
                                       clip_to_display=False)[0]
        self.assertEqual((full["x"], full["y"], full["w"], full["h"]),
                         (723.0, 57.0, 1382.0, 835.0))

    def test_unclipped_keeps_a_negative_origin(self):
        # Off the LEFT edge: the origin itself is outside the display, and
        # clipping moves it. Unclipped has to report where the window really
        # starts or the buffer's top-left maps to the wrong point.
        args = dict(x=-200.0, y=500.0, w=800.0, h=600.0)
        clipped = devices._filter_windows([win(**args)], [MAIN])[0]
        self.assertEqual((clipped["x"], clipped["w"]), (0.0, 600.0))
        full = devices._filter_windows([win(**args)], [MAIN],
                                       clip_to_display=False)[0]
        self.assertEqual((full["x"], full["w"]), (-200.0, 800.0))

    def test_unclipped_still_gates_on_the_visible_part(self):
        """The gates are NOT relaxed with the clip. A window with a 40pt
        sliver on screen is refused either way -- otherwise `clip_to_display`
        would quietly turn a size gate into an offer, and the picker would
        list windows nobody can see."""
        args = dict(x=-760.0, y=100.0, w=800.0, h=600.0)
        self.assertEqual(devices._filter_windows([win(**args)], [MAIN]), [])
        self.assertEqual(
            devices._filter_windows([win(**args)], [MAIN],
                                    clip_to_display=False), [])
        # Same for display ownership: entirely off the main display, gone.
        off = dict(x=0.0, y=-44.0, w=1440.0, h=24.0)
        self.assertEqual(
            devices._filter_windows([win(**off)], [MAIN],
                                    clip_to_display=False), [])

    def test_clipping_is_the_default_everywhere(self):
        """The display-crop pick must be untouched: an avfoundation crop can
        only cover pixels the capture contains, so the visible part stays the
        honest answer unless a caller asks otherwise."""
        import inspect
        for fn in (devices._filter_windows, devices.list_windows,
                   devices.window_rect_points):
            self.assertIs(
                inspect.signature(fn).parameters["clip_to_display"].default,
                True, "{} clips by default".format(fn.__name__))
        args = dict(x=723.0, y=57.0, w=1382.0, h=835.0)
        self.assertEqual(
            devices._filter_windows([win(**args)], [MAIN]),
            devices._filter_windows([win(**args)], [MAIN],
                                    clip_to_display=True))

    def test_a_fully_onscreen_window_is_identical_either_way(self):
        # The overwhelmingly common case: nothing to clip, so the switch is a
        # no-op and no take that never had the bug can change.
        args = dict(x=100.0, y=100.0, w=800.0, h=600.0)
        self.assertEqual(
            devices._filter_windows([win(**args)], [MAIN]),
            devices._filter_windows([win(**args)], [MAIN],
                                    clip_to_display=False))

    def test_no_displays_drops_everything(self):
        # Without Quartz there is no display geometry to crop against, so the
        # fail-safe answer is "no window capture".
        self.assertEqual(devices._filter_windows([win()], []), [])

    def test_owning_display_is_the_biggest_overlap(self):
        # Straddles the two displays but is mostly on the secondary one.
        got = devices._filter_windows(
            [win(x=-300.0, y=-300.0, w=500.0, h=500.0)], [MAIN, SECOND],
            main_only=False)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["display_id"], 2)
        self.assertEqual(got[0]["display_origin"], [-912.0, -768.0])
        self.assertFalse(got[0]["main_display"])
        # 500x300 on the secondary (150000) beats 200x200 on the main (40000).
        self.assertEqual((got[0]["x"], got[0]["y"], got[0]["w"], got[0]["h"]),
                         (-300.0, -300.0, 500.0, 300.0))


class MainOnly(unittest.TestCase):
    def test_secondary_display_dropped_by_default(self):
        # Real dump: the Eternum window lives at (-912,-768) 1366x768.
        got = devices._filter_windows(
            [win(owner="Eternum", name="Eternum",
                 x=-912.0, y=-768.0, w=1366.0, h=768.0)], [MAIN, SECOND])
        self.assertEqual(got, [])

    def test_secondary_display_kept_when_allowed(self):
        got = devices._filter_windows(
            [win(owner="Eternum", name="Eternum",
                 x=-912.0, y=-768.0, w=1366.0, h=768.0)], [MAIN, SECOND],
            main_only=False)
        self.assertEqual(ids(got), [100])
        self.assertFalse(got[0]["main_display"])
        self.assertEqual(got[0]["display_id"], 2)

    def test_main_display_metadata(self):
        got = devices._filter_windows([win()], [MAIN, SECOND])
        self.assertEqual(got[0]["display_id"], 1)
        self.assertEqual(got[0]["display_origin"], [0.0, 0.0])
        self.assertTrue(got[0]["main_display"])


class OwnChrome(unittest.TestCase):
    def test_exclude_pids_drops_our_windows(self):
        got = devices._filter_windows(
            [win(number=1, pid=999), win(number=2, pid=1234)], [MAIN],
            exclude_pids=(999,))
        self.assertEqual(ids(got), [2])

    def test_title_fallback_drops_the_bar_from_an_unknown_pid(self):
        # studio_app spawns `studio.py bar` detached, so its pid is not ours.
        entries = [win(number=1, owner="Python", name="AutoCine — New Recording"),
                   win(number=2, owner="Python", name="AutoCine — Facecam"),
                   win(number=3, owner="Python", name="Preview")]
        got = devices._filter_windows(entries, [MAIN])
        self.assertEqual(ids(got), [3])

    def test_owner_blacklist(self):
        for owner in ("Window Server", "Dock", "Wallpaper"):
            got = devices._filter_windows([win(owner=owner)], [MAIN])
            self.assertEqual(got, [], "%s survived" % owner)

    def test_blacklist_is_minimal(self):
        # Deliberately short: layer + size do the real work.
        self.assertEqual(len(devices.WINDOW_OWNER_BLACKLIST), 3)


class Naming(unittest.TestCase):
    def test_title_falls_back_to_owner_name(self):
        # kCGWindowName is '' without Screen Recording permission.
        for name in ("", None):
            got = devices._filter_windows(
                [win(owner="System Settings", name=name)], [MAIN])
            self.assertEqual(got[0]["app"], "System Settings")
            self.assertEqual(got[0]["title"], "System Settings")
            self.assertEqual(got[0]["label"], "System Settings")

    def test_label_joins_app_and_title(self):
        got = devices._filter_windows(
            [win(owner="Google Chrome", name="New Tab")], [MAIN])
        self.assertEqual(got[0]["label"], "Google Chrome — New Tab")

    def test_label_collapses_when_title_equals_app(self):
        got = devices._filter_windows(
            [win(owner="Eternum", name="Eternum")], [MAIN])
        self.assertEqual(got[0]["label"], "Eternum")

    def test_long_title_elided(self):
        long_title = "autocine — Python studio.py app — " + "x" * 80
        got = devices._filter_windows(
            [win(owner="Terminal", name=long_title)], [MAIN])
        label = got[0]["label"]
        self.assertTrue(label.startswith("Terminal — autocine"))
        self.assertTrue(label.endswith("…"))
        self.assertLessEqual(len(label), len("Terminal — ") + devices.WINDOW_TITLE_MAX)
        # The untruncated title is still available for tooltips etc.
        self.assertEqual(got[0]["title"], long_title)


class Ordering(unittest.TestCase):
    def test_front_to_back_order_preserved(self):
        # CGWindowListCopyWindowInfo returns z-order; a picker wants it as-is.
        entries = [win(number=9878), win(number=9853, owner="Terminal"),
                   win(number=10907, owner="System Settings")]
        got = devices._filter_windows(entries, [MAIN])
        self.assertEqual(ids(got), [9878, 9853, 10907])

    def test_junk_removal_does_not_reorder(self):
        entries = [win(number=1), win(number=11084, layer=25),
                   win(number=2), win(number=26, layer=24), win(number=3)]
        got = devices._filter_windows(entries, [MAIN])
        self.assertEqual(ids(got), [1, 2, 3])


class EntryShape(unittest.TestCase):
    def test_contract_keys(self):
        got = devices._filter_windows([win()], [MAIN])[0]
        self.assertEqual(sorted(got.keys()), sorted([
            "id", "app", "title", "label", "x", "y", "w", "h",
            "display_id", "display_origin", "main_display"]))
        for key in ("x", "y", "w", "h"):
            self.assertIsInstance(got[key], float)
        self.assertIsInstance(got["id"], int)


class _Raiser(object):
    """Stands in for a Quartz module that can't be used."""

    def __getattr__(self, name):
        raise ImportError("no Quartz here")


class SoftFail(unittest.TestCase):
    def setUp(self):
        self._saved = sys.modules.get("Quartz")
        sys.modules["Quartz"] = _Raiser()

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("Quartz", None)
        else:
            sys.modules["Quartz"] = self._saved

    def test_list_windows_returns_empty(self):
        self.assertEqual(devices.list_windows(), [])

    def test_displays_points_returns_empty(self):
        self.assertEqual(devices.displays_points(), [])

    def test_window_rect_points_returns_none(self):
        self.assertIsNone(devices.window_rect_points(9853))


if __name__ == "__main__":
    unittest.main()
