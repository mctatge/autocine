"""Pure-geometry tests for the pre-record un-overlap (autocine/arrange.py).

Everything here runs with no permissions, no Quartz and no Accessibility:
`plan_separation` is deliberately split out from the AX calls precisely so
the part with the interesting decisions is the part that can be tested.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autocine import arrange  # noqa: E402


SCREEN = (0.0, 0.0, 1440.0, 900.0)


def _overlaps(rects):
    return arrange.any_overlap([list(r) for r in rects])


class NothingToDo(unittest.TestCase):
    """None means "don't touch the user's windows", and it has to be the
    answer whenever moving them would buy nothing."""

    def test_non_overlapping_windows_are_left_alone(self):
        rects = [[0, 0, 600, 400], [700, 0, 600, 400]]
        self.assertIsNone(arrange.plan_separation(rects, SCREEN))

    def test_touching_edges_is_not_overlapping(self):
        rects = [[0, 0, 600, 400], [600, 0, 600, 400]]
        self.assertIsNone(arrange.plan_separation(rects, SCREEN))

    def test_single_window_is_never_a_plan(self):
        self.assertIsNone(arrange.plan_separation([[0, 0, 600, 400]], SCREEN))

    def test_empty_is_never_a_plan(self):
        self.assertIsNone(arrange.plan_separation([], SCREEN))

    def test_malformed_rect_gives_up_rather_than_guessing(self):
        self.assertIsNone(
            arrange.plan_separation([[0, 0, 600], [10, 10, 600, 400]], SCREEN))

    def test_degenerate_screen_gives_up(self):
        rects = [[0, 0, 600, 400], [100, 100, 600, 400]]
        self.assertIsNone(arrange.plan_separation(rects, (0, 0, 0, 0)))


class SeparatesOverlaps(unittest.TestCase):
    def test_two_overlapping_windows_end_up_apart(self):
        rects = [[100, 100, 700, 500], [600, 300, 700, 500]]
        plan = arrange.plan_separation(rects, SCREEN)
        self.assertIsNotNone(plan)
        self.assertFalse(_overlaps(plan))

    def test_three_way_pile_ends_up_apart(self):
        rects = [[100, 100, 500, 400],
                 [250, 200, 500, 400],
                 [400, 300, 500, 400]]
        plan = arrange.plan_separation(rects, SCREEN)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan), 3)
        self.assertFalse(_overlaps(plan))

    def test_everything_stays_on_screen(self):
        rects = [[100, 100, 700, 500], [600, 300, 700, 500]]
        plan = arrange.plan_separation(rects, SCREEN)
        for x, y, w, h in plan:
            self.assertGreaterEqual(x, -0.01)
            self.assertGreaterEqual(y, -0.01)
            self.assertLessEqual(x + w, SCREEN[2] + 0.01)
            self.assertLessEqual(y + h, SCREEN[3] + 0.01)

    def test_identical_rects_are_deterministic(self):
        rects = [[200, 200, 400, 300], [200, 200, 400, 300]]
        first = arrange.plan_separation(rects, SCREEN)
        second = arrange.plan_separation(rects, SCREEN)
        self.assertIsNotNone(first)
        self.assertEqual(first, second)
        self.assertFalse(_overlaps(first))

    def test_shallow_overlap_moves_along_the_shallow_axis(self):
        """Two side-by-side windows overlapping by a sliver must be pushed
        SIDEWAYS. Shoving them apart vertically would separate them just as
        well and destroy the arrangement the user is looking at."""
        rects = [[100, 100, 600, 700], [660, 100, 600, 700]]
        plan = arrange.plan_separation(rects, SCREEN)
        self.assertIsNotNone(plan)
        self.assertAlmostEqual(plan[0][1], plan[1][1], places=1)
        self.assertNotAlmostEqual(plan[0][0], plan[1][0], places=1)


class ShrinksOnlyWhenItMust(unittest.TestCase):
    def test_windows_that_fit_are_not_resized(self):
        rects = [[100, 100, 600, 400], [400, 200, 600, 400]]
        plan = arrange.plan_separation(rects, SCREEN)
        self.assertIsNotNone(plan)
        for src, got in zip(rects, plan):
            self.assertAlmostEqual(src[2], got[2], places=1)
            self.assertAlmostEqual(src[3], got[3], places=1)

    def test_oversized_pair_shrinks_to_fit(self):
        rects = [[0, 0, 1000, 500], [200, 100, 1000, 500]]
        plan = arrange.plan_separation(rects, SCREEN)
        self.assertIsNotNone(plan)
        self.assertFalse(_overlaps(plan))
        self.assertLess(plan[0][2], 1000.0)

    def test_gives_up_rather_than_returning_an_overlapping_plan(self):
        """Four windows that cannot possibly be separated on a small screen.
        Returning a plan that still overlaps would move the user's windows
        for no benefit at all -- the one outcome with no upside."""
        tiny = (0.0, 0.0, 400.0, 300.0)
        rects = [[0, 0, 380, 280]] * 4
        self.assertIsNone(arrange.plan_separation(rects, tiny))


class AnyOverlap(unittest.TestCase):
    def test_detects_a_real_overlap(self):
        self.assertTrue(arrange.any_overlap([[0, 0, 100, 100],
                                             [50, 50, 100, 100]]))

    def test_touching_is_not_overlap(self):
        self.assertFalse(arrange.any_overlap([[0, 0, 100, 100],
                                              [100, 0, 100, 100]]))

    def test_gap_inflation_catches_near_misses(self):
        rects = [[0, 0, 100, 100], [110, 0, 100, 100]]
        self.assertFalse(arrange.any_overlap(rects))
        self.assertTrue(arrange.any_overlap(rects, gap=24.0))


class SoftFailsWithoutAccessibility(unittest.TestCase):
    """The AX half must degrade, never raise: a window that won't move is a
    worse recording, not a failed one."""

    def test_apply_with_no_targets_is_a_no_op(self):
        self.assertEqual(arrange.apply_arrangement([], []), (0, None))

    def test_restore_of_nothing_is_a_no_op(self):
        arrange.restore(None)
        arrange.restore([])

    def test_apply_soft_fails_when_the_framework_is_missing(self):
        real = arrange._ax
        arrange._ax = lambda: None
        try:
            moved, token = arrange.apply_arrangement(
                [{"id": 1, "x": 0, "y": 0, "w": 10, "h": 10}],
                [[100, 100, 10, 10]])
        finally:
            arrange._ax = real
        self.assertEqual((moved, token), (0, None))


if __name__ == "__main__":
    unittest.main()
