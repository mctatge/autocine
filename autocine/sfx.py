"""Event sound effects: the click / keystroke audio bed.

The renderer turns the recorded event stream into ONE pre-mixed mono WAV --
a "bed" -- which the encoder takes as a single input. That shape is the
whole point of this module, and it is not the obvious one, so:

WHY NOT `adelay` TAPS. The original `--click-sound` built the SFX inside
ffmpeg: `asplit` the sound N ways, `adelay` each tap to its click time,
`amix` them back together. `amix` normalizes by the number of still-ACTIVE
inputs, and an `adelay` tap stays active (emitting silence) until its own
delay elapses -- so the first tap is mixed at ~1/N while the last, once
every sibling has hit EOF, plays at ~1/1. Measured on ffmpeg 8.1 with 50
taps: first click peak 0.0035, last 0.0875 -- a 25x ramp across the take,
and at the old 1200-click cap the opening clicks are inaudible. Summing the
sounds ourselves has no such coupling: every event is placed at unit gain,
the cap can be generous, and per-event pitch/gain variation becomes free
(the same tap N times is what makes synthetic typing sound like a machine
gun rather than a keyboard).

WHY SYNTHESIZED, NOT SHIPPED. The built-in sounds are generated from noise
plus damped sinusoids at import-time cost of a few ms. No binary assets in
the repo, no sample licensing to reason about before open-sourcing (see
docs/architecture.md), and the character is tunable in code. Determinism matters
as much as the sound: the noise comes from a fixed-seed
`np.random.RandomState` (numpy's legacy generator, whose stream is
version-stable by policy) and the per-event variation from an integer hash
of the event index -- so re-rendering a take twice produces byte-identical
audio, which is what lets the render tests compare output at all.

Levels are deliberately conservative (click peaks 0.42, keystroke 0.26):
the bed is mixed against the mic track with `amix=normalize=0`, so the mic
keeps EXACTLY its own level and the bed adds on top. Halving the voice to
make room for a click would be the wrong trade -- and so would running the
voice through a limiter to buy headroom, which is why the headroom is
bought here instead, by keeping the loudest thing in the bed under -7 dBFS.
A click landing on a speech peak can still sum past full scale on an
unusually hot mic; `sfx_volume` is the knob for that, in both directions.
"""

import os
import subprocess
import wave

import numpy as np

SAMPLE_RATE = 48000

# Resolved choices for a `click_sound` / `key_sound` option.
AUTO = "auto"    # use the built-in synthesized sound
OFF = "off"      # play nothing for this event kind

_OFF_WORDS = ("off", "none", "no", "false", "0")

# Per-event pitch/gain variation is picked from a small pre-rendered bank
# rather than resampled per event -- 6 variants is enough that a typing
# burst never audibly repeats, and the cost is paid once.
_KEY_VARIANTS = 6
_CLICK_VARIANTS = 3

# Runaway guard on the EVENT count. Note that this is not what bounds the
# bed's memory -- `build_bed` allocates from the take's DURATION, not from
# the number of events (see its docstring).
MAX_SFX_EVENTS = 20000

# Hard ceiling on a user-supplied sound file, in seconds. A click or a
# keystroke is tens of milliseconds; anything past a second or two is a
# mistake (someone pointing `--click-sound` at a podcast). Without a cap the
# decode buffers the WHOLE file in memory, and `build_bed` then sizes its
# tail -- and pays a per-event copy -- from that length, so one wrong path
# turns every click into a multi-megabyte memcpy.
MAX_SOUND_FILE_SEC = 2.0
_DECODE_TIMEOUT_SEC = 30


def resolve(value):
    """Normalize a `click_sound` / `key_sound` option to AUTO, OFF, or a path.

    None and "" mean AUTO (the built-in sound) -- sounds are on by default,
    so the absence of an opinion is not the absence of sound. "off" (and its
    obvious synonyms) is the explicit off switch; anything else is a file
    path, checked by the caller so it can warn with the path in hand.
    """
    if value is None:
        return AUTO
    text = str(value).strip()
    if not text:
        return AUTO
    if text.lower() in _OFF_WORDS:
        return OFF
    if text.lower() == AUTO:
        return AUTO
    return text


# ---- synthesis ---------------------------------------------------------


def _rng(seed):
    return np.random.RandomState(seed)


def _env(n, tau, sample_rate=SAMPLE_RATE):
    """Exponential decay envelope of `n` samples with time constant `tau`."""
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    return np.exp(-t / float(tau))


def _damped_sine(n, freq, tau, phase=0.0, sample_rate=SAMPLE_RATE):
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    return np.sin(2.0 * np.pi * freq * t + phase) * np.exp(-t / float(tau))


def _soften(noise, cutoff_hz, sample_rate=SAMPLE_RATE):
    """Roll the top off a white-noise burst.

    Raw white noise puts the spectral centroid near 10 kHz, which reads as
    hiss -- a "tss", not a "tick". A short Hann-window convolution is a
    crude but perfectly adequate low-pass here (the burst is a few hundred
    samples and the filter runs once, at bank-build time), and it is what
    moves the built-ins from sounding like static to sounding like plastic.
    """
    k = int(max(2, round(float(sample_rate) / max(1.0, float(cutoff_hz)))))
    if k < 2 or k >= noise.size:
        return noise
    win = np.hanning(k + 2)[1:-1]
    win = win / win.sum()
    return np.convolve(noise, win, mode="same")


def _attack(buf, ms=0.35, sample_rate=SAMPLE_RATE):
    """Ramp the first fraction of a millisecond in.

    Every component here starts at full amplitude on sample 0, which is a
    step -- audible as a thin extra tick on top of the sound we designed.
    """
    k = max(1, int(round(sample_rate * ms / 1000.0)))
    k = min(k, buf.size)
    buf[:k] *= np.linspace(0.0, 1.0, k)
    return buf


def _peak_to(buf, target):
    peak = float(np.max(np.abs(buf))) if buf.size else 0.0
    if peak <= 1e-9:
        return buf
    return buf * (float(target) / peak)


def _click_sample(seed=0x5EED, peak=0.42, pitch=1.0, length_s=0.014):
    """A mouse-switch click: broadband transient + a little body.

    A real switch is mostly a very fast noise transient; the damped
    sinusoids give it the 2-5 kHz "snap" that reads as plastic rather than
    as a pop, and the 190 Hz term adds just enough weight that it does not
    disappear under speech.
    """
    n = int(round(SAMPLE_RATE * length_s))
    noise = _soften(_rng(seed).uniform(-1.0, 1.0, n), 6500.0)
    buf = (noise * _env(n, 0.0016) * 1.00
           + _damped_sine(n, 2400.0 * pitch, 0.0025) * 0.75
           + _damped_sine(n, 5200.0 * pitch, 0.0010) * 0.22
           + _damped_sine(n, 190.0 * pitch, 0.0060) * 0.30)
    return _peak_to(_attack(buf), peak).astype(np.float32)


def _release_sample(seed=0x5EED + 1, peak=0.185):
    """The mouse coming back UP: same switch, shorter and brighter.

    Only the built-in click layer plays this. It is what separates "a tick
    happened" from "someone pressed a button"; at 0.185 peak it sits well
    under the press and never competes with it.
    """
    n = int(round(SAMPLE_RATE * 0.009))
    noise = _soften(_rng(seed).uniform(-1.0, 1.0, n), 8000.0)
    buf = (noise * _env(n, 0.0010) * 1.00
           + _damped_sine(n, 3600.0, 0.0014) * 0.55
           + _damped_sine(n, 6400.0, 0.0008) * 0.18)
    return _peak_to(_attack(buf), peak).astype(np.float32)


def _key_sample(seed=0xC0FFEE, peak=0.26, pitch=1.0, length_s=0.034):
    """A keystroke "thock": transient plus a short wooden body.

    Softer and longer-tailed than the mouse click on purpose -- a take can
    hold hundreds of these, and a bright, clicky keystroke repeated at
    10 Hz is genuinely unpleasant. The 620/1450 Hz resonances carry the
    body; the 3800 Hz term is the keycap tick.
    """
    n = int(round(SAMPLE_RATE * length_s))
    noise = _soften(_rng(seed).uniform(-1.0, 1.0, n), 3200.0)
    buf = (noise * _env(n, 0.0018) * 0.90
           + _damped_sine(n, 620.0 * pitch, 0.0110) * 1.00
           + _damped_sine(n, 1450.0 * pitch, 0.0045) * 0.45
           + _damped_sine(n, 3800.0 * pitch, 0.0012) * 0.14)
    return _peak_to(_attack(buf), peak).astype(np.float32)


def _bank(make, count, spread=0.12):
    """`count` pitch-shifted variants of a generator, centered on 1.0."""
    if count <= 1:
        return [make()]
    steps = np.linspace(1.0 - spread, 1.0 + spread, count)
    return [make(pitch=float(p), seed=0x5EED + 17 * i)
            for i, p in enumerate(steps)]


_CACHE = {}


def builtin_bank(kind):
    """The variant bank for "click", "release" or "key" (built once)."""
    if kind not in _CACHE:
        if kind == "click":
            _CACHE[kind] = _bank(
                lambda pitch=1.0, seed=0x5EED: _click_sample(seed=seed,
                                                             pitch=pitch),
                _CLICK_VARIANTS, spread=0.07)
        elif kind == "release":
            _CACHE[kind] = [_release_sample()]
        elif kind == "key":
            _CACHE[kind] = _bank(
                lambda pitch=1.0, seed=0xC0FFEE: _key_sample(seed=seed,
                                                             pitch=pitch),
                _KEY_VARIANTS, spread=0.14)
        else:
            raise ValueError("unknown builtin sound: " + str(kind))
    return _CACHE[kind]


# ---- user-supplied sound files ----------------------------------------


def decode_file(path, sample_rate=SAMPLE_RATE):
    """Decode any ffmpeg-readable audio file to a mono float32 array.

    Goes through ffmpeg rather than `wave` so a user can point
    `--click-sound` at an mp3/m4a/aiff exactly as they always could -- the
    old graph handed the path straight to ffmpeg, and narrowing that to
    PCM-WAV-only would be a regression. Returns None on any failure; the
    caller warns and drops the layer.
    """
    if not path or not os.path.isfile(path):
        return None
    # `-t` before the output caps the DECODE, so a huge file never lands in
    # this process's memory; the timeout covers a stream that never ends.
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-t", "{:.3f}".format(MAX_SOUND_FILE_SEC),
           "-f", "f32le", "-ac", "1", "-ar", str(int(sample_rate)), "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              timeout=_DECODE_TIMEOUT_SEC)
    except Exception:
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    buf = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if buf.size == 0 or not np.all(np.isfinite(buf)):
        return None
    return buf


# ---- the bed -----------------------------------------------------------


def _pick(index, count):
    """Deterministic variant index for event `index`.

    Knuth's multiplicative hash rather than `index % count`: a cyclic
    pattern at a fixed period is audible in a fast typing burst, which is
    the exact thing the variants exist to hide.
    """
    h = (int(index) * 2654435761) & 0xFFFFFFFF
    return h % max(1, int(count))


def _gain_jitter(index, amount):
    if amount <= 0.0:
        return 1.0
    h = ((int(index) * 40503) >> 3) & 0x3FF
    return 1.0 + amount * ((h / 1023.0) * 2.0 - 1.0)


class Layer(object):
    """One event kind's contribution to the bed.

    `times` are OUTPUT-timeline seconds (post-trim, post-cut, post-warp --
    the renderer hands us the same array it hands the encoder). `bank` is a
    list of variants; `jitter` is the +/- fraction of per-event gain wobble.
    """

    def __init__(self, times, bank, gain=1.0, jitter=0.0):
        self.times = times
        self.bank = [np.asarray(b, dtype=np.float32) for b in bank if
                     np.asarray(b).size]
        self.gain = float(gain)
        self.jitter = float(jitter)


def build_bed(duration_s, layers, volume=1.0, sample_rate=SAMPLE_RATE):
    """Sum every layer's events into one mono float32 buffer.

    Returns None when there is nothing to play, so the caller can keep the
    encoder command byte-identical to the no-SFX one. The buffer runs a
    little past `duration_s` so a sound landing on the final frame keeps its
    tail instead of being cut off mid-decay; the encoder's `-shortest`
    equivalents trim it back.

    MEMORY IS O(DURATION), NOT O(EVENTS). One float32 sample per 1/48000 s
    of video -- ~11.5 MB per minute, so ~690 MB for a one-hour take with a
    single click in it. `MAX_SFX_EVENTS` does not bound this. `write_wav`
    is chunked so the conversion adds only a block, not another two or
    three full-length copies.
    """
    volume = float(volume)
    if volume <= 0.0:
        return None
    duration_s = max(0.0, float(duration_s or 0.0))
    if duration_s <= 0.0:
        return None
    usable = [ly for ly in layers
              if ly is not None and ly.bank and len(ly.times or [])]
    if not usable:
        return None

    tail = max((max(b.size for b in ly.bank) for ly in usable), default=0)
    n = int(round(duration_s * sample_rate)) + tail + 1
    buf = np.zeros(n, dtype=np.float32)
    placed = 0
    for ly in usable:
        count = len(ly.bank)
        for i, t in enumerate(ly.times):
            try:
                tf = float(t)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(tf) or tf < 0.0:
                continue
            start = int(round(tf * sample_rate))
            if start >= n:
                continue
            sample = ly.bank[_pick(i, count)]
            end = min(n, start + sample.size)
            take = end - start
            if take <= 0:
                continue
            buf[start:end] += (sample[:take]
                               * (ly.gain * _gain_jitter(i, ly.jitter)))
            placed += 1
    if not placed:
        return None
    buf *= volume
    # Safety net only: individual peaks are low enough that this clips
    # nothing in practice (two simultaneous keystrokes reach ~0.6), but a
    # user-supplied sound file can be anything at all.
    np.clip(buf, -1.0, 1.0, out=buf)
    return buf


# Samples converted per write. Big enough that the syscall overhead is
# irrelevant, small enough that the temporaries are a rounding error.
_WAV_CHUNK = 1 << 20


def write_wav(path, samples, sample_rate=SAMPLE_RATE):
    """Write a mono float array as 16-bit PCM. Returns `path`.

    Chunked deliberately. The bed is a full-video-length buffer (see
    `build_bed`), and the obvious one-liner --
    `(np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()` -- makes three
    more copies of it, so an hour-long take peaked around 2.2 GB to write
    690 MB of samples. This holds one block at a time instead.
    """
    data = np.asarray(samples, dtype=np.float32)
    handle = wave.open(path, "wb")
    try:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        for start in range(0, data.size, _WAV_CHUNK):
            block = np.clip(data[start:start + _WAV_CHUNK], -1.0, 1.0)
            handle.writeframes((block * 32767.0).astype("<i2").tobytes())
    finally:
        handle.close()
    return path
