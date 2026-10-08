"""Unit tests for the file-watching helper of `studio app --reload`.

The full supervisor loop spawns real subprocesses, so we only cover the
pure decision core here (latest_mtime): pattern filtering, dotdir/pycache
skipping, missing dirs, empty trees. The `run` loop is a straightforward
subprocess.Popen + poll pattern that composes those; testing it end-to-end
would need a fake child that we can bump-then-observe, which is more brittle
than useful.
"""

import os
import shutil
import tempfile
import time
import unittest

from autocine import livereload


class LatestMtime(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _touch(self, rel, when=None):
        path = os.path.join(self.td, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("")
        if when is not None:
            os.utime(path, (when, when))
        return path

    def test_empty_dir_returns_zero(self):
        self.assertEqual(livereload.latest_mtime([self.td]), 0.0)

    def test_missing_dir_is_silently_zero(self):
        self.assertEqual(
            livereload.latest_mtime([os.path.join(self.td, "nope")]), 0.0)

    def test_picks_highest_mtime_across_files(self):
        self._touch("a.py", when=100.0)
        self._touch("sub/b.py", when=500.0)
        self._touch("sub/c.py", when=300.0)
        self.assertEqual(livereload.latest_mtime([self.td]), 500.0)

    def test_pattern_filter_excludes_unmatched(self):
        self._touch("app.py", when=100.0)
        self._touch("README.md", when=999.0)
        # only .py in defaults, so .md is ignored even though newer
        self.assertEqual(
            livereload.latest_mtime([self.td], patterns=(".py",)),
            100.0)

    def test_default_patterns_include_frontend_files(self):
        self._touch("a.js", when=200.0)
        self._touch("b.html", when=300.0)
        self._touch("c.css", when=400.0)
        self._touch("d.py", when=100.0)
        self.assertEqual(livereload.latest_mtime([self.td]), 400.0)

    def test_pycache_skipped(self):
        self._touch("real.py", when=100.0)
        self._touch("__pycache__/real.cpython-39.pyc", when=999.0)
        # .pyc isn't in default patterns anyway, but even if it were, the
        # __pycache__ walk-skip should keep us at 100.0
        self.assertEqual(
            livereload.latest_mtime([self.td],
                                    patterns=(".py", ".pyc")),
            100.0)

    def test_dotdirs_skipped(self):
        self._touch(".git/HEAD", when=999.0)
        self._touch("app.py", when=100.0)
        self.assertEqual(
            livereload.latest_mtime([self.td], patterns=(".py", "HEAD")),
            100.0)

    def test_multiple_dirs_take_max(self):
        d2 = tempfile.mkdtemp()
        try:
            self._touch("a.py", when=100.0)
            with open(os.path.join(d2, "b.py"), "w") as f:
                f.write("")
            os.utime(os.path.join(d2, "b.py"), (500.0, 500.0))
            self.assertEqual(
                livereload.latest_mtime([self.td, d2]), 500.0)
        finally:
            shutil.rmtree(d2, ignore_errors=True)

    def test_change_detection_flow(self):
        # This mirrors what the supervisor's poll loop does: snapshot,
        # touch, snapshot again. The second snapshot must be strictly
        # greater when a file is written.
        self._touch("a.py", when=100.0)
        baseline = livereload.latest_mtime([self.td])
        self.assertEqual(baseline, 100.0)
        # Bump mtime by an explicit later timestamp so this doesn't depend
        # on wall-clock resolution.
        self._touch("a.py", when=200.0)
        after = livereload.latest_mtime([self.td])
        self.assertGreater(after, baseline)


if __name__ == "__main__":
    unittest.main()
