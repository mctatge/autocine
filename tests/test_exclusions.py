"""Capture exclusion: getting our own chrome's window ids to a content filter.

Stage 1 of the ScreenCaptureKit capture path. All offline -- synthetic Quartz
dicts, a stand-in NSWindow, and a real server object. No Quartz, no SCK, no
permissions, no recording.

Why this exists at all: `NSWindowSharingNone` was measured 2026-07-30 to be
IGNORED by ffmpeg's avfoundation screen input (the window still lands in the
pixels), while SCK exclusion by window id was measured to work. So the id is
the load-bearing handle now, and everything that produces one is pinned here.
"""
import unittest

from autocine import bar_native
from autocine import devices
from autocine import studio_app


def _win(pid, wid, layer=0, alpha=1.0, w=400.0, h=300.0, name=None):
    """One CGWindowListCopyWindowInfo-shaped entry."""
    info = {
        "kCGWindowOwnerPID": pid,
        "kCGWindowNumber": wid,
        "kCGWindowLayer": layer,
        "kCGWindowAlpha": alpha,
        "kCGWindowBounds": {"X": 0.0, "Y": 0.0, "Width": w, "Height": h},
        "kCGWindowIsOnscreen": True,
    }
    if name is not None:
        info["kCGWindowName"] = name
    return info


class IdsForPids(unittest.TestCase):
    """`devices._ids_for_pids` -- the pure half of the pid sweep."""

    def test_keeps_only_the_pids_asked_for(self):
        infos = [_win(10, 100), _win(11, 101), _win(10, 102)]
        self.assertEqual(devices._ids_for_pids(infos, [10]), [100, 102])

    def test_keeps_the_windows_the_picker_deliberately_drops(self):
        # This is the whole reason this isn't _filter_windows. The pill is
        # on_top (layer 25) and mostly transparent slack, and it is EXACTLY
        # the window we must not miss -- a window we fail to enumerate is a
        # window burned into the user's recording.
        infos = [_win(10, 100, layer=25, alpha=0.02, w=12.0, h=8.0)]
        self.assertEqual(devices._ids_for_pids(infos, [10]), [100])
        # ...and the picker really would have dropped it, so this test is
        # pinning a genuine difference rather than restating the same rule.
        self.assertEqual(devices._filter_windows(infos, [], main_only=False), [])

    def test_preserves_z_order_and_dedupes(self):
        infos = [_win(10, 300), _win(10, 100), _win(10, 300)]
        self.assertEqual(devices._ids_for_pids(infos, [10]), [300, 100])

    def test_survives_junk_entries(self):
        infos = [None, {}, _win(10, 100), {"kCGWindowOwnerPID": "x"},
                 {"kCGWindowOwnerPID": 10}]     # no window number
        self.assertEqual(devices._ids_for_pids(infos, [10]), [100])

    def test_unparseable_pids_are_ignored_not_fatal(self):
        infos = [_win(10, 100)]
        self.assertEqual(devices._ids_for_pids(infos, [None, "x", 10]), [100])

    def test_reads_no_window_titles(self):
        # PRIVACY: an id is an opaque handle; a title is the user's content.
        # A dict that EXPLODES if the title is touched proves we never do.
        class Exploding(dict):
            def get(self, key, default=None):
                if key == "kCGWindowName":
                    raise AssertionError("read a window title")
                return dict.get(self, key, default)

        info = Exploding(_win(10, 100, name="Private Thing"))
        self.assertEqual(devices._ids_for_pids([info], [10]), [100])

    def test_no_quartz_means_empty_not_a_crash(self):
        # Soft-fail like every other Quartz call here: [] means "exclude
        # nothing", which is today's behavior.
        orig = devices._copy_window_info
        devices._copy_window_info = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("no Quartz"))
        try:
            self.assertEqual(devices.window_ids_for_pids([1]), [])
        finally:
            devices._copy_window_info = orig


class WindowNumber(unittest.TestCase):
    """`bar_native.window_number` -- the bar's own NSWindow -> id."""

    class _FakeNS(object):
        def __init__(self, num):
            self._num = num

        def windowNumber(self):
            if self._num is None:
                raise RuntimeError("dead window")
            return self._num

    def _with_ns(self, ns):
        orig = bar_native._ns_window
        bar_native._ns_window = lambda w: ns
        self.addCleanup(lambda: setattr(bar_native, "_ns_window", orig))

    def test_reads_the_window_number(self):
        self._with_ns(self._FakeNS(22893))
        self.assertEqual(bar_native.window_number(object()), 22893)

    def test_no_nswindow_is_none_not_a_crash(self):
        self._with_ns(None)
        self.assertIsNone(bar_native.window_number(object()))

    def test_a_dead_window_is_none_not_a_crash(self):
        self._with_ns(self._FakeNS(None))
        self.assertIsNone(bar_native.window_number(object()))


class BarApiExclusions(unittest.TestCase):
    """`_NativeBarApi.capture_exclusions` -- all four of our windows."""

    def _api(self, numbers):
        api = studio_app._NativeBarApi()
        seq = list(numbers)
        orig = bar_native.window_number
        bar_native.window_number = lambda win: seq[win]
        self.addCleanup(lambda: setattr(bar_native, "window_number", orig))
        return api

    def test_reports_pill_bubble_picker_and_notepad(self):
        # One test pins the full ordered report, so dropping any window from
        # the loop fails here (not only in the per-window test_notepad case).
        api = self._api([11, 22, 33, 44])
        api.window, api.face_window = 0, 1
        api.picker_window, api.notepad_window = 2, 3
        self.assertEqual(api.capture_exclusions(),
                         {"ids": [11, 22, 33, 44]})

    def test_windows_that_are_not_open_are_simply_absent(self):
        api = self._api([11])
        api.window = 0            # bubble and picker stay None
        self.assertEqual(api.capture_exclusions(), {"ids": [11]})

    def test_one_unreadable_window_does_not_lose_the_others(self):
        # Excluding two of three still beats excluding none.
        api = self._api([11, None, 33])
        api.window, api.face_window, api.picker_window = 0, 1, 2
        self.assertEqual(api.capture_exclusions(), {"ids": [11, 33]})

    def test_no_windows_at_all_is_an_empty_report(self):
        api = self._api([])
        self.assertEqual(api.capture_exclusions(), {"ids": []})


class StateExcludeIds(unittest.TestCase):
    """`StudioState` -- reported ids UNION a live pid sweep."""

    def _state(self, tmp=None):
        import tempfile
        return studio_app.StudioState(tmp or tempfile.mkdtemp())

    def _sweep(self, ids):
        orig = studio_app.dev.window_ids_for_pids
        studio_app.dev.window_ids_for_pids = lambda pids: list(ids)
        self.addCleanup(
            lambda: setattr(studio_app.dev, "window_ids_for_pids", orig))

    def test_union_of_both_sources(self):
        st = self._state()
        st.set_bar_window_ids([1, 2])
        self._sweep([2, 3])
        self.assertEqual(st.capture_exclude_ids(), [1, 2, 3])

    def test_the_sweep_alone_still_reports(self):
        # The bar may not have pushed yet; the sweep is the backstop.
        st = self._state()
        self._sweep([7])
        self.assertEqual(st.capture_exclude_ids(), [7])

    def test_reported_ids_are_cleaned(self):
        st = self._state()
        self._sweep([])
        self.assertEqual(st.set_bar_window_ids([5, "6", None, 5, 0, -1, "x"]),
                         [5, 6])
        self.assertEqual(st.capture_exclude_ids(), [5, 6])

    def test_none_means_cleared_not_crashed(self):
        st = self._state()
        self._sweep([])
        st.set_bar_window_ids([1])
        self.assertEqual(st.set_bar_window_ids(None), [])
        self.assertEqual(st.capture_exclude_ids(), [])


class ExclusionEndpoints(unittest.TestCase):
    """POST/GET /api/bar/windows against a real server.

    Pins the transport, NOT a live data flow: no production code sends the
    POST, so `_bar_window_ids` is empty in every real take and
    `capture_exclude_ids` degrades to the pid sweep. These tests passing says
    the endpoint would work if something called it — nothing does.

    (An earlier version of this docstring called POST "the only way the ids
    can travel in the `studio.py app` topology". That is the one topology the
    pid sweep provably covers: the server spawns that bar, so it knows the
    pid, and both recorded exclusion passes came from there.)
    """

    def setUp(self):
        import tempfile, threading
        self.td = tempfile.mkdtemp()
        self.server, self.base = studio_app.build_server(
            "127.0.0.1", 0, recordings_root=self.td)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        # Pin the live sweep so the test doesn't depend on what is on the
        # developer's screen while it runs.
        orig = studio_app.dev.window_ids_for_pids
        studio_app.dev.window_ids_for_pids = lambda pids: []
        self.addCleanup(
            lambda: setattr(studio_app.dev, "window_ids_for_pids", orig))

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def _post(self, ids):
        import json as _json
        import urllib.request
        req = urllib.request.Request(
            self.base + "/api/bar/windows",
            data=_json.dumps({"ids": ids}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                studio_app._TOKEN_HEADER: self.server.security_token,
            }, method="POST")
        return _json.loads(urllib.request.urlopen(req).read().decode())

    def _get(self):
        import json as _json
        import urllib.request
        req = urllib.request.Request(
            self.base + "/api/bar/windows",
            headers={studio_app._TOKEN_HEADER: self.server.security_token})
        return _json.loads(urllib.request.urlopen(req).read().decode())

    def test_round_trip(self):
        self.assertEqual(self._post([42, 43])["ids"], [42, 43])
        self.assertEqual(self._get()["ids"], [42, 43])

    def test_get_before_any_post_is_empty_not_an_error(self):
        self.assertEqual(self._get()["ids"], [])

    def test_a_later_post_replaces_the_earlier_one(self):
        # Windows are created and destroyed on demand, so the newest report
        # is the truth -- accumulating would keep excluding dead ids.
        self._post([1, 2])
        self._post([3])
        self.assertEqual(self._get()["ids"], [3])


if __name__ == "__main__":
    unittest.main()
