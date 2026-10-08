"""Optimistic-concurrency tests for edits.json: the rev counter, atomic
writes, and the save_session_edits base_rev compare-and-swap that stops one
client (web editor) from silently clobbering another's (MCP) edits.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest

from autocine import edits, studio_app


FFMPEG = shutil.which("ffmpeg")


def _make_session(root, name="20240101-000000", w=320, h=200, dur=1.0, fps=30):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    raw = os.path.join(d, "raw.mov")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=size={}x{}:rate={}:duration={}".format(w, h, fps, dur),
         "-pix_fmt", "yuv420p", raw],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with open(os.path.join(d, "events.jsonl"), "w") as f:
        f.write(json.dumps({"t": 0.5, "type": "down", "x": 100, "y": 80}) + "\n")
    meta = {"fps": fps, "logical_w": w, "logical_h": h, "t0_monotonic": 0.0,
            "raw": "raw.mov", "events": "events.jsonl",
            "cursor_mode": "system"}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    return d


class RevCounter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_defaults_and_normalization(self):
        self.assertEqual(edits.default_edits()["rev"], 0)
        self.assertEqual(edits.normalize_edits({})["rev"], 0)
        self.assertEqual(edits.normalize_edits({"rev": 7})["rev"], 7)
        self.assertEqual(edits.normalize_edits({"rev": -3})["rev"], 0)
        self.assertEqual(edits.normalize_edits({"rev": "junk"})["rev"], 0)

    def test_save_increments_and_load_roundtrips(self):
        saved1 = edits.save_edits(self.tmp, {"render": {"zoom": 3.0}})
        self.assertEqual(saved1["rev"], 1)
        loaded = edits.load_edits(self.tmp)
        self.assertEqual(loaded["rev"], 1)
        saved2 = edits.save_edits(self.tmp, loaded)
        self.assertEqual(saved2["rev"], 2)

    def test_merge_ignores_patch_rev(self):
        base = edits.save_edits(self.tmp, {})           # rev 1
        merged = edits.merge_edits(base, {"rev": 99, "render": {"zoom": 2.5}})
        self.assertEqual(merged["rev"], 1)

    def test_reset_bumps_past_current(self):
        edits.save_edits(self.tmp, {})                   # rev 1
        edits.save_edits(self.tmp, edits.load_edits(self.tmp))  # rev 2
        after = edits.reset_edits(self.tmp)
        self.assertEqual(after["rev"], 3)

    def test_write_is_atomic_no_tmp_left_behind(self):
        edits.save_edits(self.tmp, {})
        self.assertTrue(os.path.isfile(edits.edits_path(self.tmp)))
        self.assertFalse(os.path.exists(edits.edits_path(self.tmp) + ".tmp"))


@unittest.skipUnless(FFMPEG, "ffmpeg not available")
class SaveConflict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.name = "20240101-000000"
        _make_session(cls.root, cls.name)
        cls.state = studio_app.StudioState(cls.root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_stale_base_rev_conflicts_and_carries_current_doc(self):
        start = self.state.get_edits(self.name)["rev"]
        first = self.state.save_session_edits(
            self.name, {"render": {"zoom": 3.0}})
        self.assertEqual(first["rev"], start + 1)

        # a second client saves on top (like the MCP server would)
        second = self.state.save_session_edits(
            self.name, {"zooms": [{"start": 0.1, "end": 0.6, "level": 2.0}]},
            base_rev=first["rev"])
        self.assertEqual(second["rev"], start + 2)
        self.assertEqual(len(second["zooms"]), 1)

        # the first client retries with its stale rev -> refused, not clobbered
        with self.assertRaises(studio_app.EditsConflict) as ctx:
            self.state.save_session_edits(
                self.name, {"zooms": []}, base_rev=first["rev"])
        self.assertEqual(ctx.exception.current["rev"], start + 2)
        self.assertEqual(len(ctx.exception.current["zooms"]), 1)

        # on disk, the MCP zoom survived
        on_disk = edits.load_edits(os.path.join(self.root, self.name))
        self.assertEqual(len(on_disk["zooms"]), 1)

    def test_no_base_rev_means_last_write_wins_compat(self):
        before = self.state.get_edits(self.name)
        saved = self.state.save_session_edits(self.name, {"render": {"fade": 0.5}})
        self.assertEqual(saved["rev"], before["rev"] + 1)


if __name__ == "__main__":
    unittest.main()
