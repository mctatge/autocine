"""Persisted app preferences, and the backend precedence they take part in.

All offline: settings go to a temp path, and `resolve_backend` is pure.
"""
import json
import os
import tempfile
import unittest

from autocine import sck
from autocine import settings


class SettingsFile(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "settings.json")

    def test_missing_file_is_empty_not_an_error(self):
        self.assertEqual(settings.load(self.path), {})
        self.assertIsNone(settings.get("capture_backend", from_path=self.path))

    def test_round_trip(self):
        settings.save({"capture_backend": "sck"}, self.path)
        self.assertEqual(settings.load(self.path), {"capture_backend": "sck"})

    def test_save_merges_rather_than_replaces(self):
        settings.save({"capture_backend": "sck"}, self.path)
        settings.save({}, self.path)
        self.assertEqual(settings.get("capture_backend", from_path=self.path),
                         "sck")

    def test_unknown_keys_are_dropped_on_write(self):
        settings.save({"capture_backend": "sck", "evil": "yes"}, self.path)
        with open(self.path) as f:
            self.assertNotIn("evil", json.load(f))

    def test_unknown_keys_are_dropped_on_read_too(self):
        # The file is a message from a possibly-different version of this
        # app; it must not be able to smuggle state in.
        with open(self.path, "w") as f:
            json.dump({"capture_backend": "sck", "future_key": 1}, f)
        self.assertEqual(settings.load(self.path), {"capture_backend": "sck"})

    def test_corrupt_file_reads_as_empty(self):
        # A preferences file must never be able to stop someone recording.
        for junk in ("{not json", "[]", '"a string"', ""):
            with open(self.path, "w") as f:
                f.write(junk)
            self.assertEqual(settings.load(self.path), {}, junk)

    def test_an_unwritable_path_does_not_raise(self):
        settings.save({"capture_backend": "sck"}, "/nope/nowhere/x.json")


class BackendPrecedence(unittest.TestCase):
    """explicit > env > saved > default."""

    def test_saved_is_used_when_nothing_else_says(self):
        self.assertEqual(sck.resolve_backend(None, {}, "sck"), "sck")

    def test_env_beats_saved(self):
        # A one-off env override must win for that launch WITHOUT quietly
        # rewriting the user's saved choice.
        self.assertEqual(
            sck.resolve_backend(None, {"AUTOCINE_CAPTURE_BACKEND":
                                       "avfoundation"}, "sck"),
            "avfoundation")

    def test_explicit_beats_everything(self):
        self.assertEqual(
            sck.resolve_backend("avfoundation",
                                {"AUTOCINE_CAPTURE_BACKEND": "sck"},
                                "sck"),
            "avfoundation")

    def test_a_corrupt_saved_value_falls_back_to_the_default(self):
        self.assertEqual(sck.resolve_backend(None, {}, "nonsense"),
                         "avfoundation")

    def test_no_saved_value_is_still_the_default(self):
        self.assertEqual(sck.resolve_backend(None, {}, None), "avfoundation")


class SettingsEndpoints(unittest.TestCase):
    def setUp(self):
        import threading
        from autocine import studio_app
        self.studio_app = studio_app
        self.td = tempfile.mkdtemp()
        # Point the module at a temp file so a test never touches the real one.
        self._orig = settings._PATH
        settings._PATH = os.path.join(self.td, "settings.json")
        self.addCleanup(lambda: setattr(settings, "_PATH", self._orig))
        self.server, self.base = studio_app.build_server(
            "127.0.0.1", 0, recordings_root=self.td)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def _get(self):
        import json as _j, urllib.request
        req = urllib.request.Request(
            self.base + "/api/settings",
            headers={self.studio_app._TOKEN_HEADER:
                     self.server.security_token})
        return _j.loads(urllib.request.urlopen(req).read().decode())

    def _post(self, body):
        import json as _j, urllib.request
        req = urllib.request.Request(
            self.base + "/api/settings", data=_j.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                self.studio_app._TOKEN_HEADER: self.server.security_token,
            }, method="POST")
        return _j.loads(urllib.request.urlopen(req).read().decode())

    def test_default_state(self):
        s = self._get()
        self.assertIsNone(s["capture_backend"])
        self.assertEqual(s["capture_backend_effective"], "avfoundation")

    def test_saving_changes_what_a_take_would_use(self):
        self._post({"capture_backend": "sck"})
        s = self._get()
        self.assertEqual(s["capture_backend"], "sck")
        self.assertEqual(s["capture_backend_effective"], "sck")

    def test_it_survives_a_new_server(self):
        # The point of the whole feature: a relaunch must not silently revert
        # to the engine that cannot hide the bar.
        self._post({"capture_backend": "sck"})
        self.assertEqual(settings.get("capture_backend"), "sck")

    def test_a_bad_backend_is_rejected_loudly(self):
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post({"capture_backend": "skc"})
        self.assertEqual(cm.exception.code, 400)

    def test_auto_add_starts_unchosen_so_the_bar_can_default_it_on(self):
        # Tri-state: None means "never chosen", which the bar reads as its
        # default-ON. Collapsing it to False here would make a fresh install
        # indistinguishable from someone who deliberately turned it off.
        self.assertIsNone(self._get()["auto_add_windows"])

    def test_auto_add_round_trips_both_ways(self):
        self._post({"auto_add_windows": False})
        self.assertIs(self._get()["auto_add_windows"], False)
        self.assertIs(settings.get("auto_add_windows"), False)
        self._post({"auto_add_windows": True})
        self.assertIs(self._get()["auto_add_windows"], True)

    def test_auto_add_survives_a_relaunch(self):
        # The whole point: it used to live in bar.js memory, so every bar
        # relaunch reverted it and the next take silently recorded none of
        # the windows the user brought up (reported 2026-08-31).
        self._post({"auto_add_windows": False})
        self.assertIs(settings.get("auto_add_windows"), False)

    def test_saving_one_preference_leaves_the_other_alone(self):
        self._post({"auto_add_windows": False})
        self._post({"capture_backend": "sck"})
        s = self._get()
        self.assertIs(s["auto_add_windows"], False)
        self.assertEqual(s["capture_backend"], "sck")

    def test_the_env_override_is_reported_as_locked(self):
        # If the environment wins, an editable picker would be lying.
        os.environ["AUTOCINE_CAPTURE_BACKEND"] = "sck"
        self.addCleanup(os.environ.pop, "AUTOCINE_CAPTURE_BACKEND", None)
        s = self._get()
        self.assertTrue(s["capture_backend_locked"])
        self.assertEqual(s["capture_backend_effective"], "sck")


if __name__ == "__main__":
    unittest.main()
