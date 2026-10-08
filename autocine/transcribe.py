"""Speech-to-text sidecar: `transcript.json` beside the recording.

The transcript is a READ MODEL, nothing else. Nothing here writes to
`edits.json`, and render.py never reads it -- it exists so a human (the
editor's transcript panel) or a model (the MCP `get_transcript` /
`find_in_transcript` tools) can name a moment in the take by what was SAID
instead of guessing a timestamp. Every edit still lands in edits.json through
the normal spec primitives and the `rev` CAS.

Same cache-sidecar pattern as `waveform.json` / `thumb.jpg`: keyed on the
raw file's mtime (plus the engine identity, since a different model produces
a different transcript from identical bytes), regenerated when stale.

ASR is an OPTIONAL, EXTERNAL binary -- whisper.cpp's `whisper-cli`, shelled
out to, exactly like ffmpeg. That keeps the stdlib-first invariant intact (no
new Python deps, nothing imported at module scope beyond the stdlib) and
keeps the work offline. A machine without the binary or without a model is
NOT an error state: `availability()` says why, and every entry point returns
a document with `status != "ok"` that the UI and the MCP tools render as
"no transcript" rather than a failure.

Timeline note: whisper timestamps are relative to the start of the extracted
audio, and `raw.mov`'s audio starts at 0 (record.py pins
`aresample=async=1:first_pts=0`), so word `t` is session time -- the SAME
clock as `describe_session`'s `click_times`, `edits.trim` and every zoom
range. No pairing step is needed here, unlike the cross-process timestamps
described in docs/architecture.md's sync model, because this process runs the clock.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile

TRANSCRIPT_FILENAME = "transcript.json"

# Env overrides (both optional; discovery below is the normal path).
ENV_BIN = "AUTOCINE_WHISPER_BIN"
ENV_MODEL = "AUTOCINE_WHISPER_MODEL"

_BIN_CANDIDATES = ("whisper-cli", "whisper-cpp", "whisper.cpp")
_MODEL_DIRS = (
    os.path.expanduser("~/.cache/whisper-cpp/models"),
    os.path.expanduser("~/.local/share/whisper-cpp/models"),
    "/opt/homebrew/share/whisper-cpp/models",
    "/usr/local/share/whisper-cpp/models",
)
# Preference order among discovered ggml-*.bin models: good-enough accuracy
# first, then anything. English-only variants are preferred at equal size --
# they are smaller and more accurate on English, which is what a screen
# recording narration overwhelmingly is.
_MODEL_PREFERENCE = (
    "ggml-small.en.bin", "ggml-small.bin",
    "ggml-medium.en.bin", "ggml-medium.bin",
    "ggml-base.en.bin", "ggml-base.bin",
    "ggml-tiny.en.bin", "ggml-tiny.bin",
)

_DEFAULT_TIMEOUT = 1800.0   # s; a wedged decoder must not hang the editor
_MIN_SILENCE = 1.2          # s; shorter gaps are breaths, not dead air
# Repairing whisper's collapsed word timings. Shared by `find_phrase` (which
# repairs a matched RUN) and `word_end_times` (which repairs each WORD, for
# removal) so the two differ only in unit, never in numbers.
_MIN_WORD_SPAN = 0.05       # s; below this the timing is collapsed, not real
_MAX_WORD_REPAIR = 2.0      # s; bound, so a word before a silence stays short
_SPECIAL_TOKEN = re.compile(r"^\s*(\[_[A-Z_]+_\]|<\|[^|]*\|>)\s*$")
_WORDISH = re.compile(r"[0-9a-z]+")


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def find_binary():
    """Path to a whisper.cpp CLI, or None."""
    override = os.environ.get(ENV_BIN)
    if override:
        return override if os.path.isfile(override) and os.access(override, os.X_OK) else None
    for name in _BIN_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    return None


def find_model():
    """Path to a ggml model file, or None."""
    override = os.environ.get(ENV_MODEL)
    if override:
        return override if os.path.isfile(override) else None
    found = {}
    for directory in _MODEL_DIRS:
        try:
            names = os.listdir(directory)
        except OSError:
            continue
        for name in names:
            if name.startswith("ggml-") and name.endswith(".bin"):
                found.setdefault(name, os.path.join(directory, name))
    for name in _MODEL_PREFERENCE:
        if name in found:
            return found[name]
    for name in sorted(found):
        return found[name]
    return None


def availability():
    """{"available": bool, "binary", "model", "reason"} -- never raises.

    `reason` is written for a human staring at a missing transcript panel, so
    it names the install, not the function that failed.
    """
    binary = find_binary()
    model = find_model()
    if binary and model:
        return {"available": True, "binary": binary, "model": model, "reason": ""}
    if not binary:
        reason = ("no whisper.cpp CLI found -- install it (brew install "
                  "whisper-cpp) or set {}".format(ENV_BIN))
    else:
        reason = ("no whisper model found -- download a ggml model into "
                  "~/.cache/whisper-cpp/models or set {}".format(ENV_MODEL))
    return {"available": False, "binary": binary, "model": model, "reason": reason}


_DTW_PRESETS = (
    "tiny.en", "tiny", "base.en", "base", "small.en", "small",
    "medium.en", "medium", "large.v1", "large.v2", "large.v3",
)


def dtw_preset(model_path):
    """whisper.cpp's `-dtw` alignment-preset name for a model file, or None.

    DTW is what makes a word's timestamp trustworthy enough to CUT on: without
    it whisper reports token offsets that are quantized and, in runs of fast
    speech, collapse several words onto one instant. The preset must name the
    model's own architecture, and the flag hard-errors on a name it doesn't
    know -- hence the explicit list rather than a slice of the filename.
    """
    name = os.path.basename(model_path or "")
    if not (name.startswith("ggml-") and name.endswith(".bin")):
        return None
    stem = name[len("ggml-"):-len(".bin")]
    for suffix in ("-q5_0", "-q5_1", "-q8_0", "-q4_0", "-q4_1"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    return stem if stem in _DTW_PRESETS else None


def engine_id(binary=None, model=None):
    """Identity of the ASR setup, stored in the cache key.

    A cached transcript produced by a different model is stale even though
    raw.mov has not changed, so the mtime alone cannot key this cache.
    """
    return "whisper.cpp:{}".format(os.path.basename(model or "") or "unknown")


# ---------------------------------------------------------------------------
# Parsing whisper.cpp JSON  (pure -- the part worth unit-testing)
# ---------------------------------------------------------------------------

def _ms(node, key):
    try:
        return float(node["offsets"][key]) / 1000.0
    except (KeyError, TypeError, ValueError):
        return None


def _seg_tokens(seg, seg_end):
    """One segment's real tokens as [(raw_text, t0, t1, p)], seconds.

    Two timing sources, and the better one is not always there:

    - `t_dtw` (centiseconds) is whisper.cpp's DTW alignment, only present when
      the decoder ran with `-dtw`. It is a token START only, so a token's end
      is the NEXT token's start — the segment end closes the last one.
    - `offsets` is the fallback. It is coarser and, measured on a real take,
      occasionally arrives INVERTED (`from` 10000 > `to` 9070), so an
      unguarded `to - from` would produce negative durations.

    Tokens carrying no text ([_BEG_], <|...|>) are dropped before any of this,
    so they can never take a word's start with them.
    """
    kept = []
    for tok in (seg.get("tokens") or []):
        if not isinstance(tok, dict):
            continue
        raw = str(tok.get("text") or "")
        if not raw or _SPECIAL_TOKEN.match(raw) or not raw.strip():
            continue
        try:
            prob = float(tok.get("p", 0.0))
        except (TypeError, ValueError):
            prob = 0.0
        try:
            dtw = float(tok.get("t_dtw", -1))
        except (TypeError, ValueError):
            dtw = -1.0
        kept.append((raw, _ms(tok, "from"), _ms(tok, "to"), prob,
                     dtw / 100.0 if dtw >= 0 else None))

    use_dtw = any(k[4] is not None for k in kept)
    out = []
    for i, (raw, off0, off1, prob, dtw) in enumerate(kept):
        if use_dtw:
            t0 = dtw
            if t0 is None:
                continue
            t1 = None
            for nxt in kept[i + 1:]:
                if nxt[4] is not None:
                    t1 = nxt[4]
                    break
            if t1 is None:
                t1 = seg_end if seg_end is not None else t0
        else:
            t0, t1 = off0, off1
            if t0 is None:
                continue
        if t1 is None or t1 < t0:
            t1 = t0
        out.append((raw, t0, t1, prob))
    return out


def parse_whisper_json(doc):
    """whisper.cpp `-oj -ojf` output -> (words, segments, language).

    Words are assembled from TOKENS, not from segments: whisper.cpp's segment
    timestamps are second-granular, while token offsets are ~100ms, and
    word-level timing is the whole point of this file (a language edit names a
    span of speech, and a one-second-quantized span cuts mid-syllable).

    Tokens are sub-word pieces; a leading space starts a new word and anything
    else (suffix pieces, trailing punctuation) glues onto the current one.
    Special tokens ([_BEG_], <|...|>) carry no text and are dropped.
    """
    words = []
    segments = []
    language = ""
    if not isinstance(doc, dict):
        return words, segments, language
    result = doc.get("result")
    if isinstance(result, dict):
        language = str(result.get("language") or "")

    for seg in (doc.get("transcription") or []):
        if not isinstance(seg, dict):
            continue
        start = _ms(seg, "from")
        end = _ms(seg, "to")
        text = str(seg.get("text") or "").strip()
        if start is not None and text:
            segments.append({
                "t": round(start, 3),
                "dur": round(max(0.0, (end if end is not None else start) - start), 3),
                "text": text,
            })
        current = None
        for raw, t0, t1, prob in _seg_tokens(seg, end):
            piece = raw.strip()
            starts_word = raw[:1].isspace() or current is None
            if starts_word:
                current = {"t": t0, "end": t1, "text": piece, "_p": [prob]}
                words.append(current)
            else:
                current["text"] += piece
                current["end"] = max(current["end"], t1)
                current["_p"].append(prob)

    out_words = []
    for w in words:
        probs = w.pop("_p")
        conf = sum(probs) / len(probs) if probs else 0.0
        out_words.append({
            "t": round(w["t"], 3),
            "dur": round(max(0.0, w["end"] - w["t"]), 3),
            "text": w["text"],
            "conf": round(conf, 3),
        })
    return out_words, segments, language


# ---------------------------------------------------------------------------
# Derived views (pure)
# ---------------------------------------------------------------------------

def transcript_text(doc):
    """The whole take as one string of speech."""
    segs = (doc or {}).get("segments") or []
    if segs:
        return " ".join(s.get("text", "") for s in segs).strip()
    return " ".join(w.get("text", "") for w in (doc or {}).get("words") or []).strip()


def _norm(text):
    return _WORDISH.findall(str(text).lower())


def _segment_at(doc, t):
    """The sentence a moment falls in -- context, so a caller can tell which
    of three "the demo"s it is looking at without a second round trip."""
    best = ""
    for seg in (doc or {}).get("segments") or []:
        t0 = float(seg.get("t", 0.0))
        if t0 <= t + 0.001:
            best = seg.get("text", "")
        else:
            break
    return best


def word_end_times(words):
    """Per-word end times for REMOVAL, as a list parallel to `words`.

    Whisper's token timing collapses a word to (near) zero length often
    enough that it is the common case, not an edge one -- measured at ~48%
    of words on a real take. `t + dur` is therefore not a usable end for
    cutting: select six words, and the sixth is not removed.

    So each word's end is repaired the same way `find_phrase` repairs a
    span -- fall back to the next word that actually starts later, bounded
    so a word before a long silence does not swallow the silence -- but
    applied PER WORD rather than per run.

    That difference from `find_phrase` is deliberate, not drift. A phrase
    LOOKUP wants the tightest honest span for the run it matched; a
    REMOVAL must cover every word the user selected, including a final
    collapsed one. Same repair, different unit, and both are computed here
    so no consumer has to reimplement either. The editor gets these ends
    served on `/api/transcript` precisely so the browser never carries a
    copy of this rule.
    """
    words = words or []
    ends = []
    for i, w in enumerate(words):
        t0 = float(w.get("t", 0.0))
        t1 = t0 + float(w.get("dur", 0.0) or 0.0)
        if t1 - t0 < _MIN_WORD_SPAN:
            for j in range(i + 1, len(words)):
                nxt = float(words[j].get("t", 0.0))
                if nxt > t0:
                    t1 = min(nxt, t0 + _MAX_WORD_REPAIR)
                    break
        ends.append(round(max(t0, t1), 3))
    return ends


def words_with_ends(words):
    """`words` with a derived `end` on each — what a CUT should use.

    Derived on read and never written to transcript.json: the file stays
    the ASR's own output, so this rule can be corrected without
    invalidating anybody's cache. Every surface that hands words to
    something that will remove time goes through here, so none of them
    has to carry a copy of the repair (see `word_end_times`).
    """
    words = words or []
    return [dict(w, end=e) for w, e in zip(words, word_end_times(words))]


def find_phrase(doc, query, limit=20):
    """Spans where `query` is spoken, as [{t, dur, text, word_index}].

    Matching is on normalized word tokens (case, punctuation and the token
    split whisper chose are all noise), so "the demo" finds "The demo," and
    the returned span is the real word timing, ready to hand to add_zoom /
    add_marker / set_trim.
    """
    words = (doc or {}).get("words") or []
    needle = _norm(query)
    if not needle or not words:
        return []
    # One normalized token list per word; a word can normalize to nothing
    # (a bare "--"), which must not silently shift the alignment.
    norm = [_norm(w.get("text", "")) for w in words]
    flat = []          # (token, word_index)
    for i, toks in enumerate(norm):
        for tok in toks:
            flat.append((tok, i))
    hits = []
    n = len(needle)
    for start in range(0, max(0, len(flat) - n + 1)):
        if [tok for tok, _ in flat[start:start + n]] != needle:
            continue
        first = flat[start][1]
        last = flat[start + n - 1][1]
        t0 = float(words[first]["t"])
        t1 = float(words[last]["t"]) + float(words[last].get("dur", 0.0))
        # whisper's token timing occasionally collapses a word to zero length.
        # A zero-length span is useless to every consumer downstream (you
        # cannot trim, zoom or mark it), so fall back to the next word's start
        # -- bounded, because the next word may be a long silence away.
        if t1 - t0 < _MIN_WORD_SPAN:
            for j in range(last + 1, len(words)):
                nxt = float(words[j]["t"])
                if nxt > t0:
                    t1 = min(nxt, t0 + _MAX_WORD_REPAIR)
                    break
        hits.append({
            "t": round(t0, 3),
            "dur": round(max(0.0, t1 - t0), 3),
            "text": " ".join(words[i].get("text", "") for i in range(first, last + 1)),
            "word_index": first,
            "context": _segment_at(doc, t0),
        })
        if len(hits) >= limit:
            break
    return hits


def silence_spans(doc, duration=None, min_gap=_MIN_SILENCE):
    """Stretches with no speech, as [{t, dur}] -- the "dead air" read model.

    Includes the head and tail of the take (nothing said before the first word
    / after the last), which is usually exactly what "cut the dead air before
    the demo" means.
    """
    words = (doc or {}).get("words") or []
    if not words:
        if duration and duration >= min_gap:
            return [{"t": 0.0, "dur": round(float(duration), 3)}]
        return []
    spans = []
    prev_end = 0.0
    for w in words:
        t0 = float(w.get("t", 0.0))
        if t0 - prev_end >= min_gap:
            spans.append({"t": round(prev_end, 3), "dur": round(t0 - prev_end, 3)})
        prev_end = max(prev_end, t0 + float(w.get("dur", 0.0)))
    if duration is not None and float(duration) - prev_end >= min_gap:
        spans.append({"t": round(prev_end, 3),
                      "dur": round(float(duration) - prev_end, 3)})
    return spans


def slice_words(doc, start=None, end=None):
    """Words overlapping [start, end] (either bound optional)."""
    words = (doc or {}).get("words") or []
    if start is None and end is None:
        return list(words)
    lo = float(start) if start is not None else float("-inf")
    hi = float(end) if end is not None else float("inf")
    out = []
    for w in words:
        t0 = float(w.get("t", 0.0))
        t1 = t0 + float(w.get("dur", 0.0))
        if t1 >= lo and t0 <= hi:
            out.append(w)
    return out


# ---------------------------------------------------------------------------
# Cache sidecar
# ---------------------------------------------------------------------------

def _cache_path(session_dir):
    return os.path.join(session_dir, TRANSCRIPT_FILENAME)


def _raw_path(session_dir):
    return os.path.join(session_dir, "raw.mov")


def _raw_mtime(session_dir):
    try:
        return os.path.getmtime(_raw_path(session_dir))
    except OSError:
        return None


def _blank(status, reason="", **extra):
    doc = {"status": status, "reason": reason, "words": [], "segments": [],
           "language": "", "engine": "", "model": ""}
    doc.update(extra)
    return doc


def _engine_still_matches(doc):
    """False when the cached document was produced by a setup this machine no
    longer has -- a DIFFERENT model transcribes the same bytes differently, so
    mtime alone cannot key this cache.

    Deliberately asymmetric, in both directions that matter:
    - a cached "unavailable"/"failed" document goes stale the moment ASR
      becomes available, so installing whisper fixes the panel by itself;
    - a real transcript is KEPT when no model can be found at all (the binary
      was uninstalled, the cache was cleared), because throwing away good text
      we cannot regenerate is strictly worse than serving it.
    """
    model = find_model()
    if not (model and find_binary()):
        return bool(doc.get("words"))
    return doc.get("engine", "") == engine_id(model=model)


def load_transcript(session_dir):
    """The cached transcript document, or None -- never runs ASR.

    Returns None when there is no cache OR when the cache is stale against
    raw.mov, so a caller that must not block (the editor's first paint) can
    tell "not transcribed yet" from "transcribed, here it is".
    """
    path = _cache_path(session_dir)
    try:
        with open(path) as f:
            doc = json.load(f)
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    mtime = _raw_mtime(session_dir)
    if mtime is not None and doc.get("mtime") != mtime:
        return None
    if not _engine_still_matches(doc):
        return None
    doc.setdefault("words", [])
    doc.setdefault("segments", [])
    doc.setdefault("status", "ok" if doc["words"] else "empty")
    return doc


def _save(session_dir, doc):
    try:
        with open(_cache_path(session_dir), "w") as f:
            json.dump(doc, f)
    except OSError:
        pass
    return doc


def _extract_wav(raw_path, wav_path, timeout):
    """16 kHz mono PCM -- the only input format whisper.cpp accepts."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", raw_path, "-map", "a:0?",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if proc.returncode != 0 or not os.path.isfile(wav_path):
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        return False, (tail[-1] if tail else "ffmpeg could not extract audio")
    if os.path.getsize(wav_path) <= 44:   # header only == silence/no stream
        return False, "the recording has no audio track"
    return True, ""


def transcribe_session(session_dir, force=False, timeout=_DEFAULT_TIMEOUT,
                       threads=None):
    """Transcribe (or return the cached transcript for) one session.

    SLOW on the first call for a session -- this is a real ASR pass over the
    whole take, minutes on a long one -- and cheap on every call after. Never
    raises for an environment problem: a missing binary, a missing model, a
    session with no audio and a decoder that fails all come back as a
    document with `status` set and empty `words`, which is what every caller
    renders as "no transcript".
    """
    cached = None if force else load_transcript(session_dir)
    if cached is not None:
        return cached

    mtime = _raw_mtime(session_dir)
    if mtime is None:
        return _blank("failed", "the recording is missing or unreadable")

    avail = availability()
    if not avail["available"]:
        return _save(session_dir, _blank("unavailable", avail["reason"],
                                         mtime=mtime))

    tmpdir = tempfile.mkdtemp(prefix="autocine-asr-")
    wav_path = os.path.join(tmpdir, "audio.wav")
    out_prefix = os.path.join(tmpdir, "out")
    try:
        try:
            ok, why = _extract_wav(_raw_path(session_dir), wav_path, timeout)
        except Exception as exc:
            ok, why = False, str(exc)
        if not ok:
            return _save(session_dir, _blank("no_audio", why, mtime=mtime))

        base = [avail["binary"], "-m", avail["model"], "-oj", "-ojf",
                "-of", out_prefix, "-np"]
        if threads:
            base += ["-t", str(int(threads))]
        # DTW first, plain second. `-dtw` rejects a preset it doesn't know and
        # exits, so the plain run is a real fallback, not belt-and-braces.
        attempts = []
        preset = dtw_preset(avail["model"])
        if preset:
            attempts.append(base + ["-dtw", preset, "-nfa", wav_path])
        attempts.append(base + [wav_path])

        json_path = out_prefix + ".json"
        raw_doc = None
        why = "whisper failed"
        for cmd in attempts:
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, timeout=timeout)
            except subprocess.TimeoutExpired:
                return _save(session_dir, _blank(
                    "failed",
                    "transcription timed out after {:.0f}s".format(timeout),
                    mtime=mtime))
            except Exception as exc:
                return _save(session_dir, _blank("failed", str(exc), mtime=mtime))
            if proc.returncode != 0 or not os.path.isfile(json_path):
                tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
                why = tail[-1] if tail else why
                continue
            try:
                with open(json_path) as f:
                    raw_doc = json.load(f)
                break
            except Exception as exc:
                why = str(exc)
        if raw_doc is None:
            return _save(session_dir, _blank("failed", why, mtime=mtime))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    words, segments, language = parse_whisper_json(raw_doc)
    doc = {
        "status": "ok" if words else "empty",
        "reason": "" if words else "no speech detected",
        "mtime": mtime,
        "engine": engine_id(avail["binary"], avail["model"]),
        "model": os.path.basename(avail["model"]),
        "language": language,
        "words": words,
        "segments": segments,
    }
    return _save(session_dir, doc)
