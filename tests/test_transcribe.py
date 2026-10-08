"""Unit tests for the transcript read model (autocine/transcribe.py).

No ASR binary, no ffmpeg, no permissions — the whisper.cpp JSON is a fixture
and the cache is exercised on a temp dir. Run with:
    python3 -m unittest discover -s tests
"""
import json
import os
import shutil
import tempfile
import unittest

from autocine import transcribe as tr


def _tok(text, from_ms, to_ms, p=0.9):
    return {"text": text, "offsets": {"from": from_ms, "to": to_ms}, "p": p}


# Shaped exactly like real `whisper-cli -oj -ojf` output, including the
# [_BEG_] special token and the sub-word split ("Bou|levard").
FIXTURE = {
    "result": {"language": "en"},
    "transcription": [
        {
            "offsets": {"from": 0, "to": 2000},
            "text": " All right, the demo.",
            "tokens": [
                _tok("[_BEG_]", 0, 0, 0.68),
                _tok(" All", 0, 300, 0.5),
                _tok(" right", 300, 800),
                _tok(",", 800, 1000, 0.2),
                _tok(" the", 1000, 1300),
                _tok(" de", 1300, 1600, 0.8),
                _tok("mo", 1600, 1900, 0.6),
                _tok(".", 1900, 2000, 0.3),
            ],
        },
        {
            "offsets": {"from": 6000, "to": 7000},
            "text": " The demo again.",
            "tokens": [
                _tok(" The", 6000, 6300),
                _tok(" demo", 6300, 6700),
                _tok(" again", 6700, 6950),
                _tok(".", 6950, 7000, 0.4),
            ],
        },
    ],
}


class ParseWhisperJson(unittest.TestCase):
    def setUp(self):
        self.words, self.segments, self.language = tr.parse_whisper_json(FIXTURE)

    def test_language_and_segments(self):
        self.assertEqual(self.language, "en")
        self.assertEqual([s["text"] for s in self.segments],
                         ["All right, the demo.", "The demo again."])
        self.assertAlmostEqual(self.segments[1]["t"], 6.0)
        self.assertAlmostEqual(self.segments[1]["dur"], 1.0)

    def test_special_tokens_dropped(self):
        self.assertNotIn("[_BEG_]", [w["text"] for w in self.words])

    def test_subword_tokens_merge_into_one_word(self):
        # " de" + "mo" + "." is ONE word, spanning both tokens' timings.
        texts = [w["text"] for w in self.words]
        self.assertEqual(texts, ["All", "right,", "the", "demo.",
                                 "The", "demo", "again."])
        demo = self.words[3]
        self.assertAlmostEqual(demo["t"], 1.3)
        self.assertAlmostEqual(demo["dur"], 0.7)   # 1.3 -> 2.0

    def test_conf_is_the_mean_over_the_word_s_tokens(self):
        self.assertAlmostEqual(self.words[3]["conf"], round((0.8 + 0.6 + 0.3) / 3, 3))

    def test_garbage_never_raises(self):
        for junk in (None, [], {}, {"transcription": [None, 3, {}]}):
            self.assertEqual(tr.parse_whisper_json(junk), ([], [], ""))


class DtwTimings(unittest.TestCase):
    """`-dtw` output wins over `offsets` — it is the timing you can cut on."""

    def _doc(self, dtws, seg_to=3000):
        toks = []
        for i, (text, dtw) in enumerate(dtws):
            tok = _tok(text, 0, 0)      # deliberately useless offsets
            tok["t_dtw"] = dtw
            toks.append(tok)
        return {"transcription": [{"offsets": {"from": 0, "to": seg_to},
                                   "text": "hello there", "tokens": toks}]}

    def test_dtw_centiseconds_become_seconds_and_close_each_other(self):
        words, _, _ = tr.parse_whisper_json(
            self._doc([(" hello", 100), (" there", 155)]))
        self.assertEqual([(w["t"], w["dur"]) for w in words],
                         [(1.0, 0.55), (1.55, 1.45)])   # last closes at seg end

    def test_offsets_are_used_when_no_token_is_dtw_aligned(self):
        doc = {"transcription": [{
            "offsets": {"from": 0, "to": 3000}, "text": "hello",
            "tokens": [dict(_tok(" hello", 500, 900), t_dtw=-1)]}]}
        words, _, _ = tr.parse_whisper_json(doc)
        self.assertEqual([(w["t"], w["dur"]) for w in words], [(0.5, 0.4)])

    def test_inverted_offsets_never_make_a_negative_duration(self):
        # measured on a real take: whisper reports from=10000 to=9070
        doc = {"transcription": [{
            "offsets": {"from": 10000, "to": 11000}, "text": "blood",
            "tokens": [_tok(" blood", 10000, 9070)]}]}
        words, _, _ = tr.parse_whisper_json(doc)
        self.assertEqual(words[0]["dur"], 0.0)

    def test_dtw_preset_only_for_names_the_flag_knows(self):
        self.assertEqual(tr.dtw_preset("/m/ggml-small.en.bin"), "small.en")
        self.assertEqual(tr.dtw_preset("/m/ggml-base.en-q5_1.bin"), "base.en")
        self.assertIsNone(tr.dtw_preset("/m/ggml-large-v3-turbo.bin"))
        self.assertIsNone(tr.dtw_preset(None))


class DerivedViews(unittest.TestCase):
    def setUp(self):
        words, segments, language = tr.parse_whisper_json(FIXTURE)
        self.doc = {"status": "ok", "words": words, "segments": segments,
                    "language": language}

    def test_text(self):
        self.assertEqual(tr.transcript_text(self.doc),
                         "All right, the demo. The demo again.")

    def test_find_phrase_ignores_case_and_punctuation(self):
        hits = tr.find_phrase(self.doc, "the DEMO")
        self.assertEqual(len(hits), 2)
        self.assertAlmostEqual(hits[0]["t"], 1.0)
        self.assertAlmostEqual(hits[0]["dur"], 1.0)     # "the" 1.0 -> "demo." 2.0
        self.assertEqual(hits[0]["text"], "the demo.")
        self.assertAlmostEqual(hits[1]["t"], 6.0)

    def test_find_phrase_misses_are_empty(self):
        self.assertEqual(tr.find_phrase(self.doc, "kubernetes"), [])
        self.assertEqual(tr.find_phrase(self.doc, "   "), [])

    def test_find_phrase_limit(self):
        self.assertEqual(len(tr.find_phrase(self.doc, "demo", limit=1)), 1)

    def test_silence_spans_include_head_and_tail(self):
        spans = tr.silence_spans(self.doc, duration=10.0)
        self.assertEqual([(s["t"], s["dur"]) for s in spans],
                         [(2.0, 4.0), (7.0, 3.0)])

    def test_silence_ignores_breaths(self):
        # the 0.2s tail is a breath, not dead air
        self.assertEqual(tr.silence_spans(self.doc, duration=7.2),
                         [{"t": 2.0, "dur": 4.0}])

    def test_silence_with_no_words_is_the_whole_take(self):
        self.assertEqual(tr.silence_spans({"words": []}, duration=5.0),
                         [{"t": 0.0, "dur": 5.0}])

    def test_slice_words_overlaps(self):
        got = [w["text"] for w in tr.slice_words(self.doc, 1.5, 6.4)]
        self.assertEqual(got, ["demo.", "The", "demo"])
        self.assertEqual(len(tr.slice_words(self.doc)), 7)


class Availability(unittest.TestCase):
    def setUp(self):
        self.env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env)

    def test_missing_binary_explains_the_install(self):
        os.environ[tr.ENV_BIN] = "/nope/whisper-cli"
        os.environ[tr.ENV_MODEL] = "/nope/model.bin"
        avail = tr.availability()
        self.assertFalse(avail["available"])
        self.assertIn("whisper-cpp", avail["reason"])

    def test_engine_id_names_the_model(self):
        self.assertEqual(tr.engine_id(model="/x/ggml-small.en.bin"),
                         "whisper.cpp:ggml-small.en.bin")


class Cache(unittest.TestCase):
    """load_transcript is the non-blocking read: cache hit or None, no ASR."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        with open(os.path.join(self.td, "raw.mov"), "wb") as f:
            f.write(b"x")
        self.mtime = os.path.getmtime(os.path.join(self.td, "raw.mov"))
        self._find_model = tr.find_model
        self._find_binary = tr.find_binary
        tr.find_model = lambda: "/models/ggml-small.en.bin"
        tr.find_binary = lambda: "/bin/whisper-cli"

    def tearDown(self):
        tr.find_model = self._find_model
        tr.find_binary = self._find_binary
        shutil.rmtree(self.td, ignore_errors=True)

    def _write(self, doc):
        with open(os.path.join(self.td, tr.TRANSCRIPT_FILENAME), "w") as f:
            json.dump(doc, f)

    def _doc(self, **over):
        doc = {"status": "ok", "mtime": self.mtime,
               "engine": "whisper.cpp:ggml-small.en.bin",
               "words": [{"t": 0.0, "dur": 0.2, "text": "hi", "conf": 0.9}],
               "segments": []}
        doc.update(over)
        return doc

    def test_hit(self):
        self._write(self._doc())
        self.assertEqual(len(tr.load_transcript(self.td)["words"]), 1)

    def test_missing_is_none(self):
        self.assertIsNone(tr.load_transcript(self.td))

    def test_stale_mtime_is_none(self):
        self._write(self._doc(mtime=self.mtime - 100))
        self.assertIsNone(tr.load_transcript(self.td))

    def test_other_model_is_none(self):
        self._write(self._doc(engine="whisper.cpp:ggml-tiny.en.bin"))
        self.assertIsNone(tr.load_transcript(self.td))

    def test_cached_unavailable_goes_stale_once_asr_is_installed(self):
        self._write(self._doc(status="unavailable", words=[], engine=""))
        self.assertIsNone(tr.load_transcript(self.td))

    def test_a_real_transcript_survives_losing_the_binary(self):
        tr.find_binary = lambda: None
        tr.find_model = lambda: None
        self._write(self._doc())
        self.assertEqual(tr.load_transcript(self.td)["status"], "ok")

    def test_corrupt_cache_is_none(self):
        with open(os.path.join(self.td, tr.TRANSCRIPT_FILENAME), "w") as f:
            f.write("{not json")
        self.assertIsNone(tr.load_transcript(self.td))

    def test_no_asr_installed_writes_an_unavailable_document(self):
        tr.find_binary = lambda: None
        tr.find_model = lambda: None
        doc = tr.transcribe_session(self.td)
        self.assertEqual(doc["status"], "unavailable")
        self.assertEqual(doc["words"], [])
        self.assertTrue(doc["reason"])
        # ...and it is persisted, so the editor does not re-probe every paint.
        self.assertTrue(os.path.isfile(os.path.join(self.td,
                                                    tr.TRANSCRIPT_FILENAME)))

    def test_missing_recording_never_raises(self):
        os.remove(os.path.join(self.td, "raw.mov"))
        self.assertEqual(tr.transcribe_session(self.td)["status"], "failed")


class WordEndTimes(unittest.TestCase):
    """The removal-repaired word ends the editor cuts on.

    Deliberately NOT the same unit as find_phrase's repair: a phrase LOOKUP
    wants the tightest honest span for the run it matched, while a REMOVAL
    must cover every word the user selected, including a final collapsed
    one. Same numbers, different unit -- pinned here so neither drifts.
    """

    def test_normal_words_end_at_t_plus_dur(self):
        words = [{"t": 0.0, "dur": 0.4, "text": "hello"},
                 {"t": 0.5, "dur": 0.3, "text": "there"}]
        self.assertEqual(tr.word_end_times(words), [0.4, 0.8])

    def test_a_collapsed_word_ends_at_the_next_word(self):
        words = [{"t": 1.0, "dur": 0.0, "text": "um"},
                 {"t": 1.4, "dur": 0.3, "text": "so"}]
        self.assertEqual(tr.word_end_times(words)[0], 1.4)

    def test_a_run_sharing_one_timestamp_ends_at_the_first_later_word(self):
        # whisper collapses several words onto one instant; they can only be
        # removed together, and each of them must reach past the run.
        words = [{"t": 2.0, "dur": 0.0, "text": "a"},
                 {"t": 2.0, "dur": 0.0, "text": "b"},
                 {"t": 2.0, "dur": 0.0, "text": "c"},
                 {"t": 3.0, "dur": 0.5, "text": "next"}]
        ends = tr.word_end_times(words)
        self.assertEqual(ends[:3], [3.0, 3.0, 3.0])

    def test_the_repair_is_bounded_before_a_long_silence(self):
        words = [{"t": 0.0, "dur": 0.0, "text": "word"},
                 {"t": 40.0, "dur": 0.4, "text": "later"}]
        self.assertEqual(tr.word_end_times(words)[0], 2.0)

    def test_a_trailing_collapsed_word_keeps_its_own_time(self):
        # Nothing later to reach for: the end must not go backwards.
        words = [{"t": 5.0, "dur": 0.4, "text": "end"},
                 {"t": 5.6, "dur": 0.0, "text": "."}]
        ends = tr.word_end_times(words)
        self.assertEqual(ends[1], 5.6)

    def test_the_collapsed_threshold_is_where_it_says_it_is(self):
        # The crux of the repair: _MIN_WORD_SPAN = 0.05. Just under it the
        # timing is treated as collapsed and reaches the next word; just
        # over it the word keeps its own end.
        under = [{"t": 1.0, "dur": 0.04, "text": "a"},
                 {"t": 1.4, "dur": 0.3, "text": "b"}]
        over = [{"t": 1.0, "dur": 0.06, "text": "a"},
                {"t": 1.4, "dur": 0.3, "text": "b"}]
        self.assertEqual(tr.word_end_times(under)[0], 1.4)
        self.assertEqual(tr.word_end_times(over)[0], 1.06)

    def test_words_with_ends_attaches_without_mutating(self):
        words = [{"t": 1.0, "dur": 0.0, "text": "um"},
                 {"t": 1.4, "dur": 0.3, "text": "so"}]
        out = tr.words_with_ends(words)
        self.assertEqual(out[0]["end"], 1.4)
        self.assertEqual(out[0]["text"], "um")
        self.assertNotIn("end", words[0])       # the caller's list is untouched
        self.assertEqual(tr.words_with_ends(None), [])

    def test_empty_input(self):
        self.assertEqual(tr.word_end_times([]), [])
        self.assertEqual(tr.word_end_times(None), [])

    def test_it_differs_from_find_phrase_on_a_run_ending_collapsed(self):
        # The case that makes the two units genuinely different, and the
        # reason cutting does not reuse find_phrase's span: the RUN is long
        # (so find_phrase does not repair and its span ends at the last
        # word's own t), while the last WORD is collapsed (so a cut built
        # from find_phrase's span would leave that word in the video).
        words = [{"t": 0.0, "dur": 0.5, "text": "the"},
                 {"t": 0.6, "dur": 0.0, "text": "demo"},
                 {"t": 1.2, "dur": 0.4, "text": "here"}]
        doc = {"words": words, "segments": []}
        hit = tr.find_phrase(doc, "the demo")[0]
        run_end = hit["t"] + hit["dur"]
        word_end = tr.word_end_times(words)[1]
        self.assertAlmostEqual(run_end, 0.6, places=3)   # leaves "demo" in
        self.assertAlmostEqual(word_end, 1.2, places=3)  # removes it
        self.assertGreater(word_end, run_end)


if __name__ == "__main__":
    unittest.main()
