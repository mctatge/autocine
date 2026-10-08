"""The invisible notes overlay — its native bridge and capture exclusion.

All offline: a stand-in NSWindow via injectable bar_native seams, a fake
`webview` module so `notepad_open` never opens a real window, and the real
server object. No pywebview, no Cocoa, no permissions, no recording.

The load-bearing claim under test is that the notes overlay is kept out of a
take by the SAME machinery as the pill and the facecam bubble: it is reported
by `capture_exclusions`, and — because it's a window the bar process owns — it
rides the pid sweep in `capture_exclude_ids` (pinned in test_exclusions). Its
invisibility is therefore inherited, not a second mechanism to keep in step.
"""
import sys
import types
import unittest

from autocine import bar_native
from autocine import studio_app


# --------------------------------------------------------------------------
# tiny fakes
# --------------------------------------------------------------------------
class _Signal(list):
    """A pywebview-ish event slot supporting `events.shown += handler`."""
    def __iadd__(self, fn):
        self.append(fn)
        return self


class _FakeWin(object):
    def __init__(self):
        self.events = types.SimpleNamespace(shown=_Signal())


class _FakeWebview(object):
    def __init__(self):
        self.created = []

    def create_window(self, title, **kw):
        self.created.append((title, kw))
        return _FakeWin()


class _Seams(object):
    """Swap bar_native functions for the duration of a test; auto-restore."""
    def __init__(self, case, **fns):
        for name, fn in fns.items():
            orig = getattr(bar_native, name)
            setattr(bar_native, name, fn)
            case.addCleanup(lambda n=name, o=orig: setattr(bar_native, n, o))


# --------------------------------------------------------------------------
# _NativeNotepadApi — the page's bridge
# --------------------------------------------------------------------------
class NotepadApi(unittest.TestCase):
    def _api(self):
        return studio_app._NativeNotepadApi(bar_api=object())

    def test_load_returns_the_saved_text(self):
        _Seams(self, load_notepad_text=lambda path=None: "my script")
        self.assertEqual(self._api().notepad_load(), {"text": "my script"})

    def test_save_persists_text_and_snapshots_geometry(self):
        saved = {}
        _Seams(
            self,
            save_notepad_text=lambda t, path=None: saved.__setitem__("text", t) or True,
            frame_of=lambda w: (100, 200, 360, 420),
            to_layout_xy=lambda f: (30, 40),
            save_notepad_geometry=lambda x, y, w, h, path=None:
                saved.__setitem__("geom", (x, y, w, h)) or True,
        )
        api = self._api()
        api.window = object()
        self.assertEqual(api.notepad_save("hello"), {"ok": True})
        # a native edge-resize fires no drag-end, so every text save also
        # snapshots the window box — that's how a resize gets persisted
        self.assertEqual(saved["text"], "hello")
        self.assertEqual(saved["geom"], (30, 40, 360, 420))

    def test_moved_persists_position_and_size(self):
        saved = {}
        _Seams(
            self,
            frame_of=lambda w: (100, 200, 360, 420),
            to_layout_xy=lambda f: (12, 34),
            save_notepad_geometry=lambda x, y, w, h, path=None:
                saved.__setitem__("g", (x, y, w, h)) or True,
        )
        api = self._api()
        api.window = object()
        self.assertEqual(api.notepad_moved(), {"ok": True})
        self.assertEqual(saved["g"], (12, 34, 360, 420))

    def test_resize_keeps_the_top_left_corner_fixed(self):
        """Cocoa y is the BOTTOM edge. Growing must pin the TOP-LEFT so the
        panel opens down-and-right under the grip, not up-and-away."""
        calls = {}
        _Seams(
            self,
            frame_of=lambda w: (100, 200, 360, 420),   # top edge = 620
            screen_frames=lambda: [],                  # skip clamping
            set_frame=lambda w, f: calls.__setitem__("f", f) or True,
        )
        api = self._api()
        api.window = object()
        self.assertEqual(api.notepad_resize(500, 600), {"ok": True})
        x, y, w, h = calls["f"]
        self.assertEqual((x, w, h), (100, 500, 600))   # left + new size
        self.assertEqual(y + h, 620)                   # top edge preserved

    def test_resize_on_an_unreadable_window_is_a_soft_no_op(self):
        _Seams(self, frame_of=lambda w: None)
        api = self._api()
        api.window = object()
        self.assertEqual(api.notepad_resize(300, 300), {"ok": False})

    def test_resize_rejects_junk_dimensions(self):
        _Seams(self, frame_of=lambda w: (0, 0, 10, 10))
        api = self._api()
        api.window = object()
        self.assertEqual(api.notepad_resize("wide", 300), {"ok": False})

    def test_persist_geometry_is_quiet_when_the_frame_is_unreadable(self):
        touched = {"saved": False}
        _Seams(
            self,
            frame_of=lambda w: None,
            save_notepad_geometry=lambda *a, **k: touched.__setitem__("saved", True),
        )
        api = self._api()
        api.window = object()
        api._persist_geometry()                        # must not raise
        self.assertFalse(touched["saved"])

    def test_close_delegates_to_the_bar(self):
        class _Bar(object):
            def notepad_close(self):
                return {"ok": True, "closed": True}
        api = studio_app._NativeNotepadApi(bar_api=_Bar())
        self.assertEqual(api.notepad_close(), {"ok": True, "closed": True})


# --------------------------------------------------------------------------
# _NativeBarApi — open / close / exclusion / teardown
# --------------------------------------------------------------------------
class BarApiNotepad(unittest.TestCase):
    def _stub_webview(self):
        fake = _FakeWebview()
        self.addCleanup(lambda: sys.modules.pop("webview", None))
        sys.modules["webview"] = fake
        return fake

    def _quiet_exclude(self):
        _Seams(self, exclude_from_capture=lambda w: True)

    def test_open_creates_a_resizable_on_top_window_and_stores_it(self):
        fake = self._stub_webview()
        self._quiet_exclude()
        _Seams(self, load_notepad_geometry=lambda path=None: None)
        api = studio_app._NativeBarApi()
        self.assertEqual(api.notepad_open(), {"ok": True})
        self.assertIsNotNone(api.notepad_window)
        self.assertEqual(len(fake.created), 1)
        title, kw = fake.created[0]
        self.assertEqual(title, "AutoCine — Notes")
        # the two properties that separate it from the facecam bubble
        self.assertTrue(kw["resizable"])
        self.assertTrue(kw["on_top"])
        self.assertTrue(kw["frameless"])
        self.assertTrue(kw["transparent"])
        # default box when nothing was saved
        self.assertEqual((kw["width"], kw["height"]), (360, 420))
        self.assertIsNone(kw["x"])
        self.assertIsNone(kw["y"])
        # the capture-exclusion attempt is wired to `shown`
        self.assertEqual(len(api.notepad_window.events.shown), 1)

    def test_open_reuses_the_saved_geometry(self):
        fake = self._stub_webview()
        self._quiet_exclude()
        _Seams(self, load_notepad_geometry=lambda path=None: (11, 22, 300, 250))
        api = studio_app._NativeBarApi()
        api.notepad_open()
        _title, kw = fake.created[0]
        self.assertEqual((kw["x"], kw["y"], kw["width"], kw["height"]),
                         (11, 22, 300, 250))

    def test_open_is_idempotent(self):
        fake = self._stub_webview()
        self._quiet_exclude()
        _Seams(self, load_notepad_geometry=lambda path=None: None)
        api = studio_app._NativeBarApi()
        api.notepad_open()
        again = api.notepad_open()
        self.assertEqual(again, {"ok": True, "reason": "already open"})
        self.assertEqual(len(fake.created), 1)          # no second window

    def test_open_without_pywebview_fails_soft(self):
        # No `webview` in sys.modules and none importable in the test env.
        self.addCleanup(lambda: sys.modules.pop("webview", None))
        sys.modules["webview"] = None                   # force ImportError
        api = studio_app._NativeBarApi()
        r = api.notepad_open()
        self.assertFalse(r["ok"])
        self.assertIn("pywebview", r["reason"])

    def test_close_destroys_the_window_persists_geometry_and_tells_the_bar(self):
        destroyed = []
        _Seams(self, exclude_from_capture=lambda w: True)
        orig_destroy = studio_app._destroy_async
        studio_app._destroy_async = lambda w: destroyed.append(w)
        self.addCleanup(lambda: setattr(studio_app, "_destroy_async", orig_destroy))

        api = studio_app._NativeBarApi()
        echoed = []
        api._eval_bar = lambda script: echoed.append(script)

        win = object()
        persisted = []

        class _NoteApi(object):
            def _persist_geometry(self):
                persisted.append(True)
        api.notepad_window = win
        api._notepad_api = _NoteApi()

        r = api.notepad_close()
        self.assertEqual(r, {"ok": True, "closed": True})
        self.assertIn(win, destroyed)
        self.assertEqual(persisted, [True])             # final geometry saved
        self.assertIsNone(api.notepad_window)
        self.assertIsNone(api._notepad_api)
        self.assertTrue(echoed and "__barNotesClosed" in echoed[0])

    def test_close_when_nothing_is_open_is_harmless(self):
        api = studio_app._NativeBarApi()
        api._eval_bar = lambda script: None
        orig_destroy = studio_app._destroy_async
        studio_app._destroy_async = lambda w: None
        self.addCleanup(lambda: setattr(studio_app, "_destroy_async", orig_destroy))
        self.assertEqual(api.notepad_close(), {"ok": True, "closed": False})

    def test_capture_exclusions_reports_the_notepad_window(self):
        """The completeness path: the notes overlay is one of the windows the
        bar reports for exclusion, alongside the pill/bubble/picker."""
        seq = {"pill": 11, "note": 44}
        _Seams(self, window_number=lambda win: seq[win])
        api = studio_app._NativeBarApi()
        api.window = "pill"
        api.notepad_window = "note"
        self.assertEqual(api.capture_exclusions(), {"ids": [11, 44]})

    def test_teardown_destroys_the_notepad_window(self):
        destroyed = []
        orig_destroy = studio_app._destroy_async
        studio_app._destroy_async = lambda w: destroyed.append(w)
        self.addCleanup(lambda: setattr(studio_app, "_destroy_async", orig_destroy))
        api = studio_app._NativeBarApi()
        note = object()
        api.notepad_window = note
        api.close_window()
        self.assertIn(note, destroyed)
        self.assertIsNone(api.notepad_window)


if __name__ == "__main__":
    unittest.main()
