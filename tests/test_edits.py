"""Unit tests for persisted session edits."""

import os
import tempfile
import unittest

from autocine import edits


class EditsModel(unittest.TestCase):
    def test_default_load_uses_full_duration_trim(self):
        with tempfile.TemporaryDirectory() as td:
            got = edits.load_edits(td, duration=12.5)
            self.assertEqual(got["version"], 1)
            self.assertAlmostEqual(got["trim"]["start"], 0.0)
            self.assertAlmostEqual(got["trim"]["end"], 12.5)
            self.assertAlmostEqual(got["render"]["zoom"], 2.2)
            self.assertEqual(len(got["presets"]), 1)
            self.assertEqual(got["active_preset_id"], got["presets"][0]["id"])
            self.assertEqual(got["render"], got["presets"][0]["render"])

    def test_background_forces_framed_style(self):
        raw = {"render": {"style": "clean", "background": "sunset"}}
        got = edits.normalize_edits(raw, duration=10.0)
        self.assertEqual(got["render"]["style"], "framed")
        self.assertEqual(got["render"]["background"], "sunset")

    def test_merge_clamps_trim_to_duration(self):
        base = edits.normalize_edits({}, duration=8.0)
        patch = {"trim": {"start": 7.9, "end": 7.9}}
        got = edits.merge_edits(base, patch, duration=8.0)
        self.assertLess(got["trim"]["start"], got["trim"]["end"])
        self.assertLessEqual(got["trim"]["end"], 8.0)

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            saved = edits.save_edits(
                td,
                {
                    "trim": {"start": 1.25, "end": 4.5},
                    "render": {"zoom": 2.8, "style": "framed", "click_fx": False},
                },
                duration=9.0,
            )
            self.assertTrue(os.path.isfile(edits.edits_path(td)))
            loaded = edits.load_edits(td, duration=9.0)
            self.assertEqual(loaded, saved)
            self.assertAlmostEqual(loaded["trim"]["start"], 1.25)
            self.assertAlmostEqual(loaded["trim"]["end"], 4.5)
            self.assertFalse(loaded["render"]["click_fx"])

    def test_duplicate_preset_selects_new_copy(self):
        base = edits.normalize_edits(
            {"render": {"zoom": 2.4, "style": "framed"}},
            duration=10.0,
        )
        dup = edits.duplicate_preset(
            base,
            name="Punchy",
            source_render={"zoom": 3.1, "style": "framed"},
            duration=10.0,
        )
        self.assertEqual(len(dup["presets"]), 2)
        self.assertEqual(dup["active_preset_id"], dup["presets"][-1]["id"])
        self.assertEqual(dup["presets"][-1]["name"], "Punchy")
        self.assertAlmostEqual(dup["render"]["zoom"], 3.1)

    def test_cursor_fx_defaults_and_clamping(self):
        got = edits.normalize_edits({}, duration=10.0)
        self.assertFalse(got["render"]["cursor_fx"])
        self.assertAlmostEqual(got["render"]["cursor_size"], 1.0)

        got2 = edits.normalize_edits(
            {"render": {"cursor_fx": True, "cursor_size": 50.0}}, duration=10.0)
        self.assertTrue(got2["render"]["cursor_fx"])
        self.assertAlmostEqual(got2["render"]["cursor_size"], 5.0)   # clamped

    def test_aspect_defaults_to_auto_and_validates(self):
        got = edits.normalize_edits({}, duration=10.0)
        self.assertEqual(got["render"]["aspect"], "auto")

        for value, expected in [
            ("9:16", "9:16"), ("1080x1920", "1080x1920"), (None, "auto"),
            ("garbage", "auto"), ("", "auto"),
        ]:
            got = edits.normalize_edits({"render": {"aspect": value}}, duration=10.0)
            self.assertEqual(got["render"]["aspect"], expected)

    def test_resolution_normalizes_and_maps_to_max_height(self):
        # default is auto -> no height cap
        base = edits.normalize_edits({}, duration=10.0)
        self.assertEqual(base["render"]["resolution"], "auto")
        self.assertIsNone(edits.resolution_max_height("auto"))
        self.assertIsNone(edits.resolution_max_height(None))
        for value, expected in [
            ("1080", "1080"), ("1080p", "1080"), ("720P", "720"),
            ("2160", "2160"), ("1440", "1440"),
            ("native", "auto"), ("", "auto"), ("999", "auto"),
            ("garbage", "auto"),
        ]:
            got = edits.normalize_edits(
                {"render": {"resolution": value}}, duration=10.0)
            self.assertEqual(got["render"]["resolution"], expected)
        self.assertEqual(edits.resolution_max_height("1080"), 1080)
        self.assertEqual(edits.resolution_max_height("720p"), 720)

    def test_always_zoomed_and_gif_options_defaults_and_clamping(self):
        got = edits.normalize_edits({}, duration=10.0)
        self.assertFalse(got["render"]["always_zoomed"])
        self.assertIsNone(got["render"]["click_sound"])
        self.assertEqual(got["render"]["gif_fps"], 15)
        self.assertEqual(got["render"]["gif_width"], 1000)

        got2 = edits.normalize_edits(
            {"render": {"always_zoomed": True, "gif_fps": 999, "gif_width": 1}},
            duration=10.0)
        self.assertTrue(got2["render"]["always_zoomed"])
        self.assertEqual(got2["render"]["gif_fps"], 50)     # clamped
        self.assertEqual(got2["render"]["gif_width"], 64)   # clamped

    def test_click_sound_path_normalizes_to_clean_string(self):
        got = edits.normalize_edits(
            {"render": {"click_sound": "  /tmp/click.wav  "}},
            duration=10.0)
        self.assertEqual(got["render"]["click_sound"], "/tmp/click.wav")

    def test_activate_preset_switches_render_payload(self):
        base = edits.normalize_edits({}, duration=6.0)
        dup = edits.duplicate_preset(
            base,
            source_render={"zoom": 2.9, "style": "clean"},
            duration=6.0,
        )
        first_id = dup["presets"][0]["id"]
        activated = edits.set_active_preset(dup, first_id, duration=6.0)
        self.assertEqual(activated["active_preset_id"], first_id)
        self.assertAlmostEqual(activated["render"]["zoom"],
                               activated["presets"][0]["render"]["zoom"])


class AutoZoomProposals(unittest.TestCase):
    """The web/MCP auto-zoom materialization path -- must stay in step
    with `camera.cluster_to_range` on the trailing-cluster contract."""

    def test_lone_trailing_click_dropped(self):
        # A single click within min_room of the end has no zoom-out room.
        props = edits.auto_zoom_proposals([29.9], duration=30.0)
        self.assertEqual(props, [])

    def test_trailing_multiclick_cluster_holds_to_end(self):
        # Regression: a whole cluster whose LAST click lands near the end
        # must NOT be discarded -- it zooms and holds through clip end.
        # (Clicks kept within chain_gap so they stay one cluster.)
        props = edits.auto_zoom_proposals([26.0, 27.0, 28.0, 29.9],
                                          duration=30.0)
        self.assertEqual(len(props), 1)
        self.assertAlmostEqual(props[0]["end"], 30.0)   # held to clip end

    def test_midclip_cluster_zooms_out_normally(self):
        props = edits.auto_zoom_proposals([10.0, 11.0], duration=30.0)
        self.assertEqual(len(props), 1)
        self.assertLess(props[0]["end"], 30.0)          # normal tail + out


class AutoCameraFeatureDefaults(unittest.TestCase):
    """Pin the default-ON contract of the camera polish options (mutation-
    proven gap: flipping a default silently changed every render)."""

    def test_feature_options_default_on(self):
        r = edits.normalize_edits({})["render"]
        self.assertTrue(r["motion_blur"])
        self.assertTrue(r["overview"])
        self.assertTrue(r["typing_zoom"])
        self.assertTrue(r["drag_hold"])
        self.assertTrue(r["scroll_zoom"])

    def test_zoom_speed_normalization(self):
        self.assertEqual(edits.normalize_edits({})["render"]["zoom_speed"],
                         "normal")
        for good in ("slow", "fast", "Slow", " FAST "):
            r = edits.normalize_edits({"render": {"zoom_speed": good}})["render"]
            self.assertEqual(r["zoom_speed"], good.strip().lower())
        r = edits.normalize_edits({"render": {"zoom_speed": "warp"}})["render"]
        self.assertEqual(r["zoom_speed"], "normal")

    def test_feature_options_can_be_disabled_and_survive_normalize(self):
        r = edits.normalize_edits({"render": {
            "motion_blur": False, "overview": False, "typing_zoom": False,
            "drag_hold": False, "scroll_zoom": False,
        }})["render"]
        self.assertFalse(r["motion_blur"])
        self.assertFalse(r["overview"])
        self.assertFalse(r["typing_zoom"])
        self.assertFalse(r["drag_hold"])
        self.assertFalse(r["scroll_zoom"])


class SpeedupDefaults(unittest.TestCase):
    """render.speedup* defaults + normalization + top-level 'speedups' list."""

    def test_speedup_defaults_off(self):
        r = edits.normalize_edits({})["render"]
        self.assertFalse(r["speedup"])
        self.assertEqual(r["speedup_rate"], 6.0)
        self.assertTrue(r["speedup_silence_gate"])

    def test_speedup_rate_clamped_to_bounds(self):
        r = edits.normalize_edits({"render": {"speedup_rate": 100.0}})["render"]
        self.assertEqual(r["speedup_rate"], 12.0)
        r = edits.normalize_edits({"render": {"speedup_rate": 0.1}})["render"]
        self.assertEqual(r["speedup_rate"], 1.5)

    def test_default_edits_has_empty_speedups_array(self):
        got = edits.default_edits()
        self.assertEqual(got["speedups"], [])
        self.assertEqual(edits.normalize_edits({})["speedups"], [])

    def test_speedup_range_normalization(self):
        raw = {"speedups": [
            {"start": 5, "end": 10, "mode": "force", "rate": 4},
            {"start": 20, "end": 25, "mode": "off"},
            {"start": 15, "end": 30, "mode": "banana"},   # invalid -> off
            {"start": "bad"},                              # dropped: no end
        ]}
        got = edits.normalize_edits(raw, duration=60.0)
        speedups = got["speedups"]
        self.assertEqual(len(speedups), 4)   # every item survives with a fallback
        # sorted by start
        starts = [s["start"] for s in speedups]
        self.assertEqual(starts, sorted(starts))
        # ids unique
        ids = [s["id"] for s in speedups]
        self.assertEqual(len(set(ids)), len(ids))
        # mode fallback
        modes = {s["start"]: s["mode"] for s in speedups}
        self.assertEqual(modes.get(15.0), "off")
        # rate 4 preserved
        for s in speedups:
            if s["start"] == 5.0:
                self.assertEqual(s["rate"], 4.0)

    def test_speedup_survives_merge_and_preset_activate(self):
        base = edits.normalize_edits({}, duration=60.0)
        with_span = edits.merge_edits(base, {
            "speedups": [{"start": 5, "end": 15, "mode": "force"}]
        }, duration=60.0)
        self.assertEqual(len(with_span["speedups"]), 1)
        # duplicate a preset -- speedups is timeline content, so it stays
        dup = edits.duplicate_preset(with_span, duration=60.0)
        self.assertEqual(len(dup["speedups"]), 1)

    def test_merge_replaces_whole_speedups_array(self):
        base = edits.merge_edits(edits.default_edits(),
            {"speedups": [{"start": 1, "end": 4, "mode": "off"}]},
            duration=10.0)
        after = edits.merge_edits(base, {"speedups": []}, duration=10.0)
        self.assertEqual(after["speedups"], [])


class CutDefaults(unittest.TestCase):
    """Top-level 'cuts' list (ripple delete): {id, start, end} in source
    seconds, shaped like suppressed ranges. NO overlap merging in edits.py
    (ids must survive a save for remove_cut; the union happens inside
    retime.TimeMap at render time)."""

    def test_default_edits_has_empty_cuts_array(self):
        self.assertEqual(edits.default_edits()["cuts"], [])
        self.assertEqual(edits.normalize_edits({})["cuts"], [])

    def test_cut_normalization_clamps_sorts_and_ids(self):
        raw = {"cuts": [
            {"start": 20, "end": 25},
            {"start": 5, "end": 10, "id": "cut-keep"},
            {"start": 55, "end": 99},          # end clamps to duration
            {"start": "bad"},                  # falls back, never dropped
        ]}
        got = edits.normalize_edits(raw, duration=60.0)
        cuts = got["cuts"]
        self.assertEqual(len(cuts), 4)
        starts = [c["start"] for c in cuts]
        self.assertEqual(starts, sorted(starts))
        ids = [c["id"] for c in cuts]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertIn("cut-keep", ids)
        by_start = {c["start"]: c for c in cuts}
        self.assertEqual(by_start[55.0]["end"], 60.0)

    def test_sub_150ms_cut_is_preserved_exactly(self):
        """REGRESSION (review finding): the 0.15s _MIN_RANGE_SPAN floor is a
        manual-zoom UX rule; applied to a cut it WIDENED the range -- content
        loss vs. the authored intent, and it desynced the MCP snapped echo
        from what persisted. A 0.1s cut must round-trip at exactly 0.1s."""
        got = edits.normalize_edits(
            {"cuts": [{"start": 0.3, "end": 0.4}]}, duration=1.0)
        self.assertEqual(len(got["cuts"]), 1)
        self.assertEqual(got["cuts"][0]["start"], 0.3)
        self.assertEqual(got["cuts"][0]["end"], 0.4)

    def test_non_finite_cut_values_fall_back_without_widening(self):
        got = edits.normalize_edits(
            {"cuts": [{"start": float("nan"), "end": 5.0},
                      {"start": 1.0, "end": float("inf")}]},
            duration=10.0)
        # entries survive (defensive normalize) but collapse to zero-length
        # rather than becoming real removals; the render's union drops them
        self.assertEqual(len(got["cuts"]), 2)
        for c in got["cuts"]:
            self.assertEqual(c["start"], c["end"])

    def test_overlapping_cuts_are_kept_not_merged(self):
        """Merging would destroy ids (stale MCP cut_id, editor adopt churn).
        The union is TimeMap's job at render time."""
        got = edits.normalize_edits(
            {"cuts": [{"start": 1, "end": 5, "id": "cut-a"},
                      {"start": 3, "end": 8, "id": "cut-b"}]},
            duration=10.0)
        self.assertEqual(len(got["cuts"]), 2)
        self.assertEqual([c["id"] for c in got["cuts"]], ["cut-a", "cut-b"])

    def test_merge_replaces_whole_cuts_array(self):
        base = edits.merge_edits(edits.default_edits(),
                                 {"cuts": [{"start": 1, "end": 4}]},
                                 duration=10.0)
        self.assertEqual(len(base["cuts"]), 1)
        after = edits.merge_edits(base, {"cuts": []}, duration=10.0)
        self.assertEqual(after["cuts"], [])

    def test_merge_without_cuts_key_preserves_them(self):
        """THE editor-save regression pin: the editor's payload does not
        carry a cuts key in v1, and merge_edits patches on key MEMBERSHIP
        -- so an editor save must never wipe MCP-authored cuts."""
        base = edits.merge_edits(edits.default_edits(),
                                 {"cuts": [{"start": 1, "end": 4}]},
                                 duration=10.0)
        after = edits.merge_edits(base, {"render": {"zoom": 3.0},
                                         "zooms": [],
                                         "trim": {"start": 0.5}},
                                  duration=10.0)
        self.assertEqual(len(after["cuts"]), 1)
        self.assertEqual(after["cuts"][0]["start"], 1.0)

    def test_cuts_survive_preset_activate_and_duplicate(self):
        base = edits.merge_edits(edits.default_edits(),
                                 {"cuts": [{"start": 2, "end": 3}]},
                                 duration=10.0)
        dup = edits.duplicate_preset(base, duration=10.0)
        self.assertEqual(len(dup["cuts"]), 1)
        back = edits.set_active_preset(dup, "preset-default", duration=10.0)
        self.assertEqual(len(back["cuts"]), 1)

    def test_cuts_save_load_roundtrip_bumps_rev(self):
        with tempfile.TemporaryDirectory() as td:
            saved = edits.save_edits(
                td, {"cuts": [{"start": 1.0, "end": 2.5}]}, duration=9.0)
            loaded = edits.load_edits(td, duration=9.0)
            self.assertEqual(loaded, saved)
            self.assertEqual(len(loaded["cuts"]), 1)
            self.assertAlmostEqual(loaded["cuts"][0]["start"], 1.0)
            self.assertAlmostEqual(loaded["cuts"][0]["end"], 2.5)
            self.assertGreaterEqual(loaded["rev"], 1)


class ZoomAndSuppressedRanges(unittest.TestCase):
    def test_default_edits_has_empty_ranges(self):
        got = edits.normalize_edits({}, duration=10.0)
        self.assertEqual(got["zooms"], [])
        self.assertEqual(got["suppressed"], [])
        self.assertEqual(got["markers"], [])

    def test_zoom_range_normalizes_and_assigns_id(self):
        raw = {"zooms": [{"start": 1.0, "end": 3.0, "x": 100.0, "y": 50.0, "level": 3.0}]}
        got = edits.normalize_edits(raw, duration=10.0)
        self.assertEqual(len(got["zooms"]), 1)
        z = got["zooms"][0]
        self.assertTrue(z["id"])
        self.assertAlmostEqual(z["start"], 1.0)
        self.assertAlmostEqual(z["end"], 3.0)
        self.assertAlmostEqual(z["x"], 100.0)
        self.assertAlmostEqual(z["y"], 50.0)
        self.assertAlmostEqual(z["level"], 3.0)

    def test_zoom_range_follow_mode_when_point_missing(self):
        raw = {"zooms": [{"start": 1.0, "end": 2.0, "level": 2.5}]}
        got = edits.normalize_edits(raw, duration=10.0)
        z = got["zooms"][0]
        self.assertIsNone(z["x"])
        self.assertIsNone(z["y"])

    def test_zoom_range_partial_point_falls_back_to_follow(self):
        raw = {"zooms": [{"start": 1.0, "end": 2.0, "x": 10.0}]}
        got = edits.normalize_edits(raw, duration=10.0)
        z = got["zooms"][0]
        self.assertIsNone(z["x"])
        self.assertIsNone(z["y"])

    def test_zoom_level_clamped_to_valid_band(self):
        raw = {"zooms": [
            {"start": 0.0, "end": 1.0, "level": 0.2},
            {"start": 2.0, "end": 3.0, "level": 50.0},
        ]}
        got = edits.normalize_edits(raw, duration=10.0)
        levels = sorted(z["level"] for z in got["zooms"])
        self.assertAlmostEqual(levels[0], 1.0)
        self.assertAlmostEqual(levels[1], edits._MAX_ZOOM_LEVEL)

    def test_zoom_and_suppressed_ranges_clamp_to_duration(self):
        raw = {
            "zooms": [{"start": 9.5, "end": 50.0, "level": 2.0}],
            "suppressed": [{"start": -3.0, "end": 2.0}],
        }
        got = edits.normalize_edits(raw, duration=10.0)
        self.assertLessEqual(got["zooms"][0]["end"], 10.0)
        self.assertGreaterEqual(got["suppressed"][0]["start"], 0.0)
        self.assertLess(got["suppressed"][0]["start"], got["suppressed"][0]["end"])

    def test_zoom_range_enforces_minimum_span(self):
        raw = {"zooms": [{"start": 1.0, "end": 1.001, "level": 2.0}]}
        got = edits.normalize_edits(raw, duration=10.0)
        z = got["zooms"][0]
        self.assertGreaterEqual(z["end"] - z["start"], edits._MIN_RANGE_SPAN - 1e-9)

    def test_duplicate_ids_get_reassigned(self):
        raw = {"zooms": [
            {"id": "same", "start": 0.0, "end": 1.0, "level": 2.0},
            {"id": "same", "start": 2.0, "end": 3.0, "level": 2.0},
        ]}
        got = edits.normalize_edits(raw, duration=10.0)
        ids = [z["id"] for z in got["zooms"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_markers_normalize_clamp_and_sort(self):
        raw = {"markers": [
            {"id": "m-late", "time": 9.5, "label": "Finish"},
            {"id": "m-early", "time": -2.0, "label": "  Intro  "},
            {"time": 100.0, "label": "x" * 200},
        ]}
        got = edits.normalize_edits(raw, duration=10.0)
        self.assertEqual(len(got["markers"]), 3)
        self.assertAlmostEqual(got["markers"][0]["time"], 0.0)
        self.assertEqual(got["markers"][0]["label"], "Intro")
        self.assertAlmostEqual(got["markers"][1]["time"], 9.5)
        self.assertAlmostEqual(got["markers"][2]["time"], 10.0)
        self.assertEqual(len(got["markers"][2]["label"]), edits._MAX_MARKER_LABEL_LEN)

    def test_marker_duplicate_ids_get_reassigned(self):
        raw = {"markers": [
            {"id": "marker-dup", "time": 1.0, "label": "A"},
            {"id": "marker-dup", "time": 2.0, "label": "B"},
        ]}
        got = edits.normalize_edits(raw, duration=10.0)
        ids = [m["id"] for m in got["markers"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_ranges_sorted_by_start(self):
        raw = {
            "zooms": [{"start": 5.0, "end": 6.0, "level": 2.0},
                     {"start": 1.0, "end": 2.0, "level": 2.0}],
            "suppressed": [{"start": 8.0, "end": 9.0}, {"start": 0.5, "end": 1.0}],
        }
        got = edits.normalize_edits(raw, duration=10.0)
        self.assertAlmostEqual(got["zooms"][0]["start"], 1.0)
        self.assertAlmostEqual(got["zooms"][1]["start"], 5.0)
        self.assertAlmostEqual(got["suppressed"][0]["start"], 0.5)
        self.assertAlmostEqual(got["suppressed"][1]["start"], 8.0)

    def test_save_and_load_roundtrip_preserves_ranges(self):
        with tempfile.TemporaryDirectory() as td:
            saved = edits.save_edits(
                td,
                {
                    "zooms": [{"start": 1.0, "end": 2.0, "level": 3.0}],
                    "suppressed": [{"start": 4.0, "end": 5.0}],
                },
                duration=10.0,
            )
            loaded = edits.load_edits(td, duration=10.0)
            self.assertEqual(loaded["zooms"], saved["zooms"])
            self.assertEqual(loaded["suppressed"], saved["suppressed"])

    def test_merge_patch_replaces_whole_zoom_array(self):
        base = edits.normalize_edits(
            {"zooms": [{"start": 1.0, "end": 2.0, "level": 2.0}]}, duration=10.0)
        patch = {"zooms": [{"start": 3.0, "end": 4.0, "level": 4.0}]}
        merged = edits.merge_edits(base, patch, duration=10.0)
        self.assertEqual(len(merged["zooms"]), 1)
        self.assertAlmostEqual(merged["zooms"][0]["start"], 3.0)

    def test_merge_patch_replaces_whole_marker_array(self):
        base = edits.normalize_edits(
            {"markers": [{"time": 1.0, "label": "Old"}]}, duration=10.0)
        patch = {"markers": [{"time": 4.0, "label": "New"}]}
        merged = edits.merge_edits(base, patch, duration=10.0)
        self.assertEqual(len(merged["markers"]), 1)
        self.assertAlmostEqual(merged["markers"][0]["time"], 4.0)
        self.assertEqual(merged["markers"][0]["label"], "New")

    def test_merge_without_zooms_key_preserves_existing_ranges(self):
        base = edits.normalize_edits(
            {"zooms": [{"start": 1.0, "end": 2.0, "level": 2.0}]}, duration=10.0)
        merged = edits.merge_edits(base, {"trim": {"start": 0.5}}, duration=10.0)
        self.assertEqual(len(merged["zooms"]), 1)
        self.assertAlmostEqual(merged["zooms"][0]["start"], 1.0)

    def test_activate_and_duplicate_preset_preserve_ranges(self):
        base = edits.normalize_edits(
            {"zooms": [{"start": 1.0, "end": 2.0, "level": 2.0}],
             "suppressed": [{"start": 4.0, "end": 5.0}],
             "markers": [{"time": 3.0, "label": "Beat"}]},
            duration=10.0,
        )
        dup = edits.duplicate_preset(base, source_render={"zoom": 3.0}, duration=10.0)
        self.assertEqual(len(dup["zooms"]), 1)
        self.assertEqual(len(dup["suppressed"]), 1)
        self.assertEqual(len(dup["markers"]), 1)

        first_id = dup["presets"][0]["id"]
        activated = edits.set_active_preset(dup, first_id, duration=10.0)
        self.assertEqual(len(activated["zooms"]), 1)
        self.assertEqual(len(activated["suppressed"]), 1)
        self.assertEqual(len(activated["markers"]), 1)


class WindowsList(unittest.TestCase):
    def test_default_edits_has_empty_windows(self):
        got = edits.normalize_edits({}, duration=10.0)
        self.assertEqual(got["windows"], [])

    def test_window_normalizes_and_assigns_id(self):
        raw = {"windows": [{"x": 10.0, "y": 20.0, "w": 300.0, "h": 200.0}]}
        got = edits.normalize_edits(raw)
        self.assertEqual(len(got["windows"]), 1)
        w = got["windows"][0]
        self.assertTrue(w["id"])
        self.assertAlmostEqual(w["x"], 10.0)
        self.assertAlmostEqual(w["y"], 20.0)
        self.assertAlmostEqual(w["w"], 300.0)
        self.assertAlmostEqual(w["h"], 200.0)

    def test_no_frame_size_parameter_accepted_or_required(self):
        # Unlike zoom/suppressed/marker ranges, window rects are NOT
        # clamped against a frame size here -- spatial safety is enforced
        # downstream at render time (mirrors how the zoom pin's x/y are
        # unclamped in edits.py too). A wildly out-of-bounds rect should
        # pass through unclamped.
        raw = {"windows": [{"x": 99999.0, "y": -99999.0, "w": 5.0, "h": 5.0}]}
        got = edits.normalize_edits(raw, duration=10.0)
        w = got["windows"][0]
        self.assertAlmostEqual(w["x"], 99999.0)
        self.assertAlmostEqual(w["y"], 0.0)  # only floored at 0, not clamped to any max

    def test_negative_coordinates_floored_at_zero(self):
        raw = {"windows": [{"x": -50.0, "y": -50.0, "w": 100.0, "h": 100.0}]}
        got = edits.normalize_edits(raw)
        w = got["windows"][0]
        self.assertAlmostEqual(w["x"], 0.0)
        self.assertAlmostEqual(w["y"], 0.0)

    def test_degenerate_size_floored_to_minimum(self):
        raw = {"windows": [{"x": 0.0, "y": 0.0, "w": 0.0, "h": -5.0}]}
        got = edits.normalize_edits(raw)
        w = got["windows"][0]
        self.assertAlmostEqual(w["w"], edits._MIN_WINDOW_DIM)
        self.assertAlmostEqual(w["h"], edits._MIN_WINDOW_DIM)

    def test_duplicate_ids_get_reassigned(self):
        raw = {"windows": [
            {"id": "same", "x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0},
            {"id": "same", "x": 200.0, "y": 0.0, "w": 100.0, "h": 100.0},
        ]}
        got = edits.normalize_edits(raw)
        ids = [w["id"] for w in got["windows"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_capped_at_four_entries(self):
        raw = {"windows": [
            {"x": float(i), "y": 0.0, "w": 20.0, "h": 20.0} for i in range(6)
        ]}
        got = edits.normalize_edits(raw)
        self.assertEqual(len(got["windows"]), edits._MAX_WINDOWS)

    def test_order_is_preserved_not_sorted(self):
        # Order is grid position, not a timeline -- this must NOT get a
        # `.sort()` the way zooms/suppressed/markers do.
        raw = {"windows": [
            {"x": 500.0, "y": 0.0, "w": 20.0, "h": 20.0},
            {"x": 10.0, "y": 0.0, "w": 20.0, "h": 20.0},
            {"x": 250.0, "y": 0.0, "w": 20.0, "h": 20.0},
        ]}
        got = edits.normalize_edits(raw)
        xs = [w["x"] for w in got["windows"]]
        self.assertEqual(xs, [500.0, 10.0, 250.0])

    def test_merge_patch_replaces_whole_windows_array(self):
        base = edits.normalize_edits(
            {"windows": [{"x": 1.0, "y": 1.0, "w": 100.0, "h": 100.0}]})
        patch = {"windows": [{"x": 9.0, "y": 9.0, "w": 50.0, "h": 50.0}]}
        merged = edits.merge_edits(base, patch)
        self.assertEqual(len(merged["windows"]), 1)
        self.assertAlmostEqual(merged["windows"][0]["x"], 9.0)

    def test_merge_without_windows_key_preserves_existing(self):
        base = edits.normalize_edits(
            {"windows": [{"x": 1.0, "y": 1.0, "w": 100.0, "h": 100.0}]})
        merged = edits.merge_edits(base, {"trim": {"start": 0.5}})
        self.assertEqual(len(merged["windows"]), 1)

    def test_save_and_load_roundtrip_preserves_windows(self):
        with tempfile.TemporaryDirectory() as td:
            saved = edits.save_edits(
                td, {"windows": [{"x": 1.0, "y": 2.0, "w": 300.0, "h": 200.0}]})
            loaded = edits.load_edits(td)
            self.assertEqual(loaded["windows"], saved["windows"])

    def test_activate_and_duplicate_preset_preserve_windows(self):
        base = edits.normalize_edits(
            {"windows": [{"x": 1.0, "y": 1.0, "w": 100.0, "h": 100.0}]})
        dup = edits.duplicate_preset(base, source_render={"zoom": 3.0})
        self.assertEqual(len(dup["windows"]), 1)

        first_id = dup["presets"][0]["id"]
        activated = edits.set_active_preset(dup, first_id)
        self.assertEqual(len(activated["windows"]), 1)


class WindowLayoutArrangements(unittest.TestCase):
    """`render.window_layout`: the five arrangement names.

    `edits._WINDOW_LAYOUTS` is the canonical accepted set -- argparse's
    `--window-layout` choices, the MCP tool's schema enum and validator, and
    both render-kwarg resolvers all read this one tuple. So a name that
    survives normalization here is a name every surface honours, and a name
    that doesn't is one the whole app silently grids.
    """

    def test_the_accepted_set_is_exactly_the_five_canonical_names(self):
        # framing.py dispatches on these exact strings and treats a miss as
        # the grid, so a rename doesn't error -- it quietly re-grids every
        # session that saved the old name.
        self.assertEqual(edits._WINDOW_LAYOUTS,
                         ("grid", "desktop", "feature", "row", "column"))

    def test_every_arrangement_round_trips(self):
        for name in edits._WINDOW_LAYOUTS:
            got = edits.normalize_edits({"render": {"window_layout": name}},
                                        duration=10.0)
            self.assertEqual(got["render"]["window_layout"], name)

    def test_junk_falls_back_to_grid(self):
        for bad in ("spiral", "fit", "", None, 7, 2.5, True, [], {}, ["row"]):
            got = edits.normalize_edits({"render": {"window_layout": bad}},
                                        duration=10.0)
            self.assertEqual(got["render"]["window_layout"], "grid", repr(bad))

    def test_matching_is_exact_not_forgiving(self):
        # Unlike `zoom_speed`, this value gets no strip()/lower() repair: every
        # writer (editor, CLI, MCP) sends a canonical string, so a near miss is
        # a caller bug and reading as the grid is how you notice it.
        for near in ("Feature", "ROW", "Column", " row", "row ", "row\n"):
            got = edits.normalize_edits({"render": {"window_layout": near}},
                                        duration=10.0)
            self.assertEqual(got["render"]["window_layout"], "grid", repr(near))

    def test_save_and_load_roundtrip_preserves_the_arrangement(self):
        with tempfile.TemporaryDirectory() as td:
            saved = edits.save_edits(
                td, {"render": {"window_layout": "feature"}}, duration=10.0)
            loaded = edits.load_edits(td, duration=10.0)
            self.assertEqual(saved["render"]["window_layout"], "feature")
            self.assertEqual(loaded["render"]["window_layout"], "feature")

    def test_merge_switches_the_arrangement_and_junk_regrids_it(self):
        base = edits.normalize_edits({"render": {"window_layout": "feature"}},
                                     duration=10.0)
        merged = edits.merge_edits(base, {"render": {"window_layout": "column"}},
                                   duration=10.0)
        self.assertEqual(merged["render"]["window_layout"], "column")
        # A patch that doesn't mention it leaves the saved arrangement alone...
        untouched = edits.merge_edits(base, {"render": {"zoom": 3.0}},
                                      duration=10.0)
        self.assertEqual(untouched["render"]["window_layout"], "feature")
        # ...but a patch carrying a BAD name doesn't keep the old one either:
        # normalization runs on the merged doc, so the fallback wins. This is
        # why the MCP tool validates before saving instead of relying on it.
        regridded = edits.merge_edits(base, {"render": {"window_layout": "spiral"}},
                                      duration=10.0)
        self.assertEqual(regridded["render"]["window_layout"], "grid")


class MarkerChapters(unittest.TestCase):
    def test_marker_chapters_follow_next_marker_boundary(self):
        raw = {"markers": [
            {"id": "m1", "time": 1.0, "label": "Intro"},
            {"id": "m2", "time": 4.5, "label": "Demo"},
            {"id": "m3", "time": 8.0, "label": "Outro"},
        ]}
        got = edits.marker_chapters(raw, duration=10.0)
        self.assertEqual([c["id"] for c in got], ["m1", "m2", "m3"])
        self.assertAlmostEqual(got[0]["start"], 1.0)
        self.assertAlmostEqual(got[0]["end"], 4.5)
        self.assertEqual(got[0]["next_marker_id"], "m2")
        self.assertAlmostEqual(got[1]["duration"], 3.5)
        self.assertAlmostEqual(got[2]["end"], 10.0)

    def test_marker_chapters_respect_trim_window(self):
        raw = {
            "trim": {"start": 3.0, "end": 6.0},
            "markers": [
                {"id": "before", "time": 1.0, "label": "Before"},
                {"id": "inside", "time": 4.0, "label": "Inside"},
                {"id": "after", "time": 7.0, "label": "After"},
            ],
        }
        got = edits.marker_chapters(raw, duration=10.0)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["id"], "inside")
        self.assertAlmostEqual(got[0]["start"], 4.0)
        self.assertAlmostEqual(got[0]["end"], 6.0)
        self.assertIsNone(got[0]["next_marker_id"])

    def test_marker_chapters_fallback_label_and_skip_duplicate_times(self):
        raw = {"markers": [
            {"id": "dup-a", "time": 2.0},
            {"id": "dup-b", "time": 2.0, "label": "Should skip duplicate-time chapter"},
            {"id": "next", "time": 5.0, "label": "Next"},
        ]}
        got = edits.marker_chapters(raw, duration=7.0)
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0]["id"], "dup-a")
        self.assertEqual(got[0]["label"], "Chapter 1")
        self.assertAlmostEqual(got[0]["end"], 5.0)
        self.assertEqual(got[1]["id"], "next")

    def test_marker_chapters_empty_when_no_markers_in_trim(self):
        raw = {
            "trim": {"start": 8.0, "end": 9.0},
            "markers": [{"id": "m1", "time": 2.0, "label": "Too early"}],
        }
        got = edits.marker_chapters(raw, duration=10.0)
        self.assertEqual(got, [])


class CutWriteContract(unittest.TestCase):
    """The shared cuts guard — the rule every authoring surface enforces.

    It moved here from mcp_server so the web editor and the CLI could not
    write a cut set the renderer refuses. tests/test_mcp_server.py is the
    other half of this pin: it still passes unmodified, which is what says
    the move was faithful rather than merely similar.
    """

    def _cuts(self, *spans):
        return [{"id": "c{}".format(i), "start": a, "end": b}
                for i, (a, b) in enumerate(spans)]

    def test_summary_reports_what_the_encoder_removes(self):
        merged, removed, out_dur = edits.cut_summary(
            self._cuts((1.0, 2.0), (4.0, 5.0)),
            {"start": 0.0, "end": 10.0}, 10.0, 30.0)
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(removed, 2.0, places=6)
        self.assertAlmostEqual(out_dur, 8.0, places=6)

    def test_overlapping_cuts_are_counted_once(self):
        merged, removed, _ = edits.cut_summary(
            self._cuts((1.0, 3.0), (2.0, 4.0)),
            {"start": 0.0, "end": 10.0}, 10.0, 30.0)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(removed, 3.0, places=6)

    def test_removed_counts_only_the_overlap_with_the_trim(self):
        # A cut reaching outside the trim window removes only what was
        # going to be exported anyway.
        _, removed, out_dur = edits.cut_summary(
            self._cuts((0.0, 4.0)), {"start": 2.0, "end": 8.0}, 10.0, 30.0)
        self.assertAlmostEqual(removed, 2.0, places=6)
        self.assertAlmostEqual(out_dur, 4.0, places=6)

    def test_too_many_ranges_is_a_loud_reject_not_a_truncation(self):
        spans = [(float(i) * 2.0, float(i) * 2.0 + 0.5)
                 for i in range(edits.CUT_MAX_RANGES + 1)]
        merged, _, out_dur = edits.cut_summary(
            self._cuts(*spans), {"start": 0.0, "end": 500.0}, 500.0, 30.0)
        with self.assertRaises(edits.CutsError) as ctx:
            edits.validate_cuts(merged, out_dur)
        self.assertIn("too many cut ranges", str(ctx.exception))

    def test_exactly_the_cap_is_allowed(self):
        spans = [(float(i) * 2.0, float(i) * 2.0 + 0.5)
                 for i in range(edits.CUT_MAX_RANGES)]
        merged, _, out_dur = edits.cut_summary(
            self._cuts(*spans), {"start": 0.0, "end": 500.0}, 500.0, 30.0)
        edits.validate_cuts(merged, out_dur)      # must not raise

    def test_cutting_away_the_whole_window_is_rejected(self):
        merged, _, out_dur = edits.cut_summary(
            self._cuts((0.0, 10.0)), {"start": 0.0, "end": 10.0}, 10.0, 30.0)
        with self.assertRaises(edits.CutsError) as ctx:
            edits.validate_cuts(merged, out_dur)
        self.assertIn("whole trim window", str(ctx.exception))

    def test_empty_cut_set_is_never_rejected(self):
        # plan_cuts keys its bypass on the MERGED set, so a document with no
        # effective cuts cannot fail for a trim window the trim handles are
        # themselves allowed to create (their floor is below CUT_MIN_KEPT_SEC).
        merged, removed, out_dur = edits.plan_cuts(
            [], {"start": 0.0, "end": 0.05}, 10.0, 30.0)
        self.assertEqual(merged, [])
        self.assertAlmostEqual(removed, 0.0)
        self.assertLess(out_dur, edits.CUT_MIN_KEPT_SEC)

    def test_degenerate_cuts_merge_away_and_do_not_trip_the_guard(self):
        # end <= start collapses to nothing in the quantizer, so a document
        # full of them is an empty cut set, not a 64-range one.
        merged, _, _ = edits.plan_cuts(
            self._cuts(*[(1.0, 1.0)] * 200),
            {"start": 0.0, "end": 10.0}, 10.0, 30.0)
        self.assertEqual(merged, [])

    def test_cuts_outside_the_trim_window_do_not_reject(self):
        # `merged` is non-empty (a real 5-6s span) but nothing is REMOVED
        # from a 0-0.1s trim window, so there is nothing to reject. Keying
        # the bypass on `merged` instead of `removed` fails this.
        merged, removed, _ = edits.plan_cuts(
            self._cuts((5.0, 6.0)), {"start": 0.0, "end": 0.1}, 12.0, 30.0)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(removed, 0.0)

    def test_plan_cuts_validates(self):
        with self.assertRaises(edits.CutsError):
            edits.plan_cuts(self._cuts((0.0, 10.0)),
                            {"start": 0.0, "end": 10.0}, 10.0, 30.0)

    def test_edits_stays_import_light(self):
        # cut_summary imports retime (and therefore numpy) lazily on purpose:
        # load_edits is on the library-listing path.
        import ast
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "autocine", "edits.py")
        with open(path) as fh:
            tree = ast.parse(fh.read())
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                # `from . import retime` has module None and carries the
                # name in names[] -- collecting only `module` would miss
                # exactly the import this test exists to prevent.
                top.add(node.module or "")
                top.update(a.name for a in node.names)
        self.assertNotIn("numpy", top)
        self.assertNotIn("retime", top)


class ChannelLayouts(unittest.TestCase):
    """`channel_layouts` — manual per-card placement for a multi-native take
    (docs/architecture.md P1). POSITIONAL and null-preserving, unlike the
    compacting window list."""

    def test_normalizer_preserves_interior_nulls_and_trims_trailing(self):
        n = edits._normalize_channel_layouts(
            [None, {"x": 0.05, "y": 0.05, "w": 0.44, "h": 0.5},
             {"x": "bad"}, None])
        # interior null kept at index 0; malformed -> None; trailing None trimmed
        self.assertEqual(n, [None, {"x": 0.05, "y": 0.05, "w": 0.44, "h": 0.5}])

    def test_absent_and_all_null_canonicalize_to_empty(self):
        self.assertEqual(edits.normalize_edits({})["channel_layouts"], [])
        self.assertEqual(
            edits._normalize_channel_layouts([None, None]), [])

    def test_capped_at_max_windows(self):
        self.assertEqual(
            len(edits._normalize_channel_layouts(
                [{"x": 0, "y": 0, "w": 0.1, "h": 0.1}] * 7)), 4)

    def test_min_frac_floor_and_clamp(self):
        n = edits._normalize_channel_layouts(
            [{"x": -1, "y": 2, "w": 0.001, "h": 0.5}])
        self.assertEqual(n[0]["x"], 0.0)          # clamped to [0,1]
        self.assertEqual(n[0]["y"], 1.0)
        self.assertEqual(n[0]["w"], edits._MIN_CARD_FRAC)   # floor

    def test_merge_clears_and_sets_by_membership(self):
        base = {"channel_layouts": [{"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}]}
        # explicit [] clears
        self.assertEqual(
            edits.merge_edits(base, {"channel_layouts": []})["channel_layouts"],
            [])
        # a patch that omits the key leaves it unchanged
        self.assertEqual(
            edits.merge_edits(base, {})["channel_layouts"],
            [{"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}])
        # set from empty
        self.assertEqual(
            edits.merge_edits({}, {"channel_layouts":
                                   [None, {"x": 0.5, "y": 0.5, "w": 0.4,
                                           "h": 0.4}]})["channel_layouts"],
            [None, {"x": 0.5, "y": 0.5, "w": 0.4, "h": 0.4}])

    def test_survives_preset_switch(self):
        # channel_layouts is take-level spatial content, not a per-preset look:
        # switching the active preset must not drop it.
        d = edits.normalize_edits(
            {"channel_layouts": [{"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.3}]})
        d = edits.duplicate_preset(d, select_new=True)
        pid = d["presets"][0]["id"]
        d = edits.set_active_preset(d, pid)
        self.assertEqual(d["channel_layouts"],
                         [{"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.3}])


class SceneLayouts(unittest.TestCase):
    """`scene_layouts` — per-scene manual card placement for a scene take
    (docs/architecture.md P2). A dict keyed by scene-index string of
    null-preserving positional lists."""

    def test_normalizer_drops_empty_scenes_and_bad_keys(self):
        n = edits._normalize_scene_layouts({
            "0": [{"x": 0.02, "y": 0.06, "w": 0.6, "h": 0.8}],
            "1": [None, {"x": 0.5, "y": 0.1, "w": 0.4, "h": 0.6}, None],
            "2": [None, None],                      # all-null -> dropped
            "bad": [{"x": 0, "y": 0, "w": 0.1, "h": 0.1}],   # non-int -> dropped
        })
        self.assertEqual(set(n), {"0", "1"})
        self.assertEqual(n["1"], [None, {"x": 0.5, "y": 0.1, "w": 0.4,
                                         "h": 0.6}])   # interior null + trim

    def test_absent_is_empty_dict(self):
        self.assertEqual(edits.normalize_edits({})["scene_layouts"], {})

    def test_merge_clears_and_sets_whole_key(self):
        base = {"scene_layouts": {"0": [{"x": 0.1, "y": 0.1, "w": 0.2,
                                         "h": 0.2}]}}
        self.assertEqual(
            edits.merge_edits(base, {"scene_layouts": {}})["scene_layouts"], {})
        self.assertEqual(
            edits.merge_edits(base, {})["scene_layouts"],
            {"0": [{"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}]})   # unchanged
        self.assertEqual(
            edits.merge_edits({}, {"scene_layouts":
                                   {"1": [{"x": 0.3, "y": 0.3, "w": 0.3,
                                           "h": 0.3}]}})["scene_layouts"],
            {"1": [{"x": 0.3, "y": 0.3, "w": 0.3, "h": 0.3}]})

    def test_survives_preset_switch(self):
        d = edits.normalize_edits(
            {"scene_layouts": {"0": [{"x": 0.2, "y": 0.2, "w": 0.3,
                                      "h": 0.3}]}})
        d = edits.duplicate_preset(d, select_new=True)
        d = edits.set_active_preset(d, d["presets"][0]["id"])
        self.assertEqual(d["scene_layouts"],
                         {"0": [{"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.3}]})


class HiddenChannels(unittest.TestCase):
    """`hidden_channels` — cards the user REMOVED from the render (a reversible
    hide, keyed by channel FILE). Absent == [] is the byte-identical off state."""

    def test_normalizer_dedupes_keeps_order_and_caps(self):
        n = edits._normalize_hidden_channels(
            ["raw_2.mov", "raw_2.mov", "raw_1.mov", "", 5,
             "a.mov", "b.mov", "c.mov"])   # dupes/blank/non-str dropped; cap 4
        self.assertEqual(n, ["raw_2.mov", "raw_1.mov", "a.mov", "b.mov"])

    def test_absent_and_junk_are_empty(self):
        self.assertEqual(edits.normalize_edits({})["hidden_channels"], [])
        self.assertEqual(edits._normalize_hidden_channels(None), [])
        self.assertEqual(edits._normalize_hidden_channels("raw_1.mov"), [])

    def test_merge_clears_and_sets_whole_key(self):
        base = {"hidden_channels": ["raw_2.mov"]}
        self.assertEqual(
            edits.merge_edits(base, {"hidden_channels": []})["hidden_channels"],
            [])   # explicit [] clears
        self.assertEqual(
            edits.merge_edits(base, {})["hidden_channels"],
            ["raw_2.mov"])   # absent from patch -> unchanged
        self.assertEqual(
            edits.merge_edits({}, {"hidden_channels":
                                   ["raw_3.mov"]})["hidden_channels"],
            ["raw_3.mov"])

    def test_survives_preset_switch(self):
        d = edits.normalize_edits({"hidden_channels": ["raw_2.mov"]})
        d = edits.duplicate_preset(d, select_new=True)
        d = edits.set_active_preset(d, d["presets"][0]["id"])
        self.assertEqual(d["hidden_channels"], ["raw_2.mov"])


if __name__ == "__main__":
    unittest.main()
