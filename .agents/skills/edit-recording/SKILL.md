---
name: edit-recording
description: "Edit an AutoCine screen recording through the autocine MCP server — trim it, zoom on the important parts, speed up dead air, crop it, lay windows out, and export. Use whenever the user wants something done to a take in recordings/ (\"clean this up\", \"zoom where I click the terminal\", \"cut the boring part\", \"make a vertical version\", \"export this\"), and whenever a request names a moment by what happened rather than by a timestamp. Also use before answering questions about what is IN a recording. Not for changing the recorder or the renderer — that is ordinary code work in autocine/."
---

# Editing a recording over MCP

A session dir is a project: `raw.mov` + `events.jsonl` + `meta.json` (+ optional
`edits.json`, `transcript.json`). Rendering is deterministic from that spec, so
**editing = adjust the spec and re-render**. You never touch pixels; you write
`edits.json` through the MCP tools and let render resolve it.

The whole job is a loop: **orient → locate → edit → verify.** The failure mode
this skill exists to prevent is skipping straight to *edit* — guessing a
timestamp, authoring a zoom in the wrong units, and reporting confidently that
it worked.

## 1. Orient — always `describe_session` first

```
describe_session(session)
```

**Read `beats` before anything else.** It is a time-ordered account of what
actually happened, derived from the recorded input events — click clusters,
scroll runs, typing, idle holes, and windows coming to front / opening /
closing. It is what lets you answer "where does the interesting part start"
without decoding a single frame.

| beat | means | reach for |
|---|---|---|
| `clicks` | a cluster of clicks; `n`, a source-px `bbox`, and `zoom` or `zoom_proposal` | `add_zoom` |
| `scroll` | a scroll run; `n` and `rate` | `add_speedup` if it's a long skim |
| `typing` | key activity (no position — see below) | `add_marker` |
| `idle` | dead air ≥3s | `add_speedup`, or `set_trim` if it's at an end |
| `front` | a window came to the front | **the closest thing a silent recording has to a scene cut** |
| `open` / `close` | a window appeared / went away | chapter boundaries |

Also read, every time:

- **`beats.notes`** — non-empty means this session cannot support part of the
  above (no geometry track, no z-rank, no materialized zooms). Say so; do not
  quietly produce a worse answer.
- **`beats.truncated`** — non-zero means beats were left out of a long take.
  The kept ones are spread evenly across it, so the shape is right but the
  detail is not. Raise `max_beats` if you need them all.
- **`has_audio` / `has_transcript`** — decide step 2.

`click_times` / `scroll_times` / `key_times` are **behind `detail: true`** and
you almost never want them. A flat list of 300 floats tells you something
happened 300 times and nothing about what.

## 2. Locate — turn "the part where…" into a timestamp

**If the take has audio**, name the moment by what was said:

```
find_in_transcript(session, query)   → matches with `context` (the surrounding sentence)
get_transcript(session)              → also `silences` and sentence `segments`
```

Both return times on the same clock as `beats`, `trim` and every zoom range, so
a match feeds straight into `set_trim` / `add_zoom` / `add_speedup` /
`add_marker`. Note the first call on a session runs a real ASR pass (minutes on
a long take; `has_transcript` tells you which you're about to get). A non-`ok`
`status` means there is no transcript at all — **report that reason and stop
guessing**; inventing timings is the exact failure the transcript exists to
remove.

**If the take is silent** (no audio, or `has_transcript: false`), the beat sheet
is your index and your eyes are the fallback:

1. Pick candidate times from `beats` — `front` beats are scene cuts, big
   `clicks` clusters are where the work happened.
2. Look at them: `preview_frame(session, time, source: true, max_width: 700)`.

**Window ids are opaque on purpose.** The event log records geometry and an id,
never an app name or window title — so a beat says `window_id 42`, never
"Excel". To find out what a window is, look at a `source: true` frame during
its `front` beat. Do not infer app identity from window size or position and
state it as fact. `list_recorded_windows(session)` gives each id's median rect
and `occluded_sec` (real seconds where something else covered it — a genuine
quality signal, since capture is display-capture-plus-crop).

## 3. Edit

Exact signatures (`*` = required):

```
set_trim(*session, start, end)              add_marker(*session, *time, label)
add_zoom(*session, *start, *end, level, x, y)   remove_marker(*session, *marker_id)
adjust_zoom(*session, *at, change, level)   set_crop(*session, *crop)
remove_zoom(*session, *zoom_id)
add_speedup(*session, *start, *end, mode, rate)  set_windows(*session, *windows)
remove_speedup(*session, *speedup_id)       fit_windows(*session)
add_cut(*session, *start, *end)             remove_cut(*session, *cut_id)
set_cuts(*session, *cuts)                   reset_edits(*session)
set_render_options(*session, ...)
```

The remove tools take `zoom_id` / `speedup_id` / `marker_id` / `window_id` /
`cut_id` — **not `id`**. Passing `id` fails with "missing required argument".

**Coordinates are source-video pixels** — the space `describe_session` reports
as `width`/`height`. Two safe sources, no conversion needed:

- a `clicks` beat's `bbox` (already in source px; its centre is a good pin), or
- a coordinate read off `preview_frame(source: true)`, multiplied by the
  `source_scale` in that call's text block.

Never measure off the **rendered** preview. It is output-canvas pixels after
the camera's zoom/pan and aspect framing, so no scale maps it back to source —
that reply reports `is_source: false` and a `frame_scale`, and says so.

### `zoom` vs `zoom_proposal` — do not conflate these

Outside a bare CLI render, the camera is planned **only** from `edits.zooms`.

- A `clicks` beat with **`zoom`** → a real zoom in the timeline; the `ids` are
  the entries `remove_zoom` takes. This *will* render.
- A `clicks` beat with **`zoom_proposal`** → nothing is in the timeline there.
  It is what auto-zoom *would* suggest. It renders **nothing** until you
  `add_zoom`.

If `notes` says `edits.zooms is empty`, the session will render with **no
camera movement at all**, however many proposals you see. Say that plainly
rather than describing proposals as though they were the plan.

### "That zoom at 1:23 was too aggressive" — `adjust_zoom`

Reach for this whenever the note is about ONE moment being too much or not
enough, instead of removing and re-adding a range, and instead of
`set_render_options(zoom=…)`, which retunes the ceiling for the whole take.

`adjust_zoom(at, change)` — `at` is seconds or `'m:ss'`, `change` is
`softer` / `stronger` / `off`. It finds the move covering that time (or the
nearest within 3s), works out which camera this take actually has, and
reports `matched`, `before` and `after`. **Report those numbers**, not the
ones you asked for, and note when it says it removed the move rather than
softening it.

Two things to know before you call it:

- `at` is **source-media time**, like every other tool. A timestamp the user
  read off an exported video with a trim, cuts or speed-ups is a different
  clock; the response says so in `notes` when they differ, so read that before
  claiming the edit landed where they meant.
- On a **multi-window** take the move is the composition camera, so the rungs
  are `full` (the whole frame pushes in) → `focus` (the card just grows) →
  gone. `softer` on a move already at `focus` deliberately does nothing and
  says `change='off'` is the next step: nothing can add a focus move back.

To switch which multi-window camera runs at all, use one word —
`set_render_options(zoom_style=…)`: `frame` (grow the card, then push the
frame in — what a take recorded with a window pick already defaults to),
`inside` (zoom the footage inside each card), `both`, `off`.

### Cuts (ripple delete) — "remove the part where I…"

`add_cut(start, end)` removes that range from the EXPORT: what follows slides
earlier, mic audio stays in sync, and no zoom or click sound fires for events
inside it. Times are the same clock as the transcript and every other tool, so
a `find_in_transcript` hit feeds `add_cut` directly; for several cuts at once
("remove all three stutters") use `set_cuts` — one call, whole-array REPLACE.
The response's `snapped` range (boundaries snap outward to the frame grid) and
`output_duration` are what the export will actually do — report those numbers,
not the ones you asked for. `trim` stays the single in/out endpoints tool;
cuts are for ranges in the middle.

### Known limits — state them, don't work around them silently

- **The editor preview does not ripple.** Cut ranges show as dimmed regions
  and playback skips them, but SCRUBBING still shows removed frames — only the
  export ripples. A v1 scope limit, not a bug; don't report it as one.
- **Cuts don't apply to scene or multi-window takes yet** (the render prints a
  note and exports un-cut). Check `describe_session` first; on those takes say
  the limit instead of quietly leaving the cuts in place.
- The **whole-screen** camera is off in multi-window mode; what moves there is
  `zoom_style` (a take recorded with a window pick opens on `frame`, hand-drawn
  cards on `off`). `add_zoom` is refused on a fleet take, and per-card zoom has
  no per-move surface — `adjust_zoom` retunes composition moves only.
- `set_crop` rejects degenerate rects (under 16px a side, or the whole frame)
  rather than silently ignoring them — a crop that did nothing can't be
  mistaken for one that worked.

## 4. Verify — look before you report

After editing, **re-read what you wrote**:

- `describe_session` again: the affected `clicks` beat should now carry `zoom`
  with your new id, and `edits.rev` should have advanced. `rev` is a
  compare-and-swap — a stale write is rejected rather than clobbering the
  editor, so re-read after any conflict instead of retrying blind.
- `preview_frame(session, time)` (rendered, *not* source) inside the span you
  changed, and actually look at the image.

Only then say it's done. If you did not look, say you did not look.

Export with `render_video(session, out: "final.mp4")`. It resolves the same
`edits.json` the web app's Export does. The output name must stay inside the
session directory; parent traversal and symlink escapes are rejected.

## Worked example (public-safe pattern)

For a narrated sample take named `demo-product-tour`:

1. Call `describe_session(session: "demo-product-tour")` and read `beats`
   before choosing a timestamp.
2. If the user names spoken words, call
   `find_in_transcript(session: "demo-product-tour", query: "watch this")`
   and use the returned span. If transcription is unavailable, report the
   returned reason instead of guessing.
3. Use `preview_frame(source: true)` to identify an opaque window id or inspect
   the unedited source. Then make the smallest requested change with
   `adjust_zoom`, `add_cut`, or another edit tool.
4. Re-read `describe_session`, then inspect a normal rendered
   `preview_frame` inside the affected span. Export only after the returned
   edit and the rendered frame agree.

The important part is the evidence chain, not these example words: a beat or
transcript match locates the moment, an edit changes the shared spec, and a
rendered preview verifies what the export will do.

## Guardrails

- **The user's project is real.** `edits.json` is their work. Prefer additive
  edits; never call `reset_edits` without being asked. If you must experiment,
  back the file up and restore it, and say that you did.
- **Report what you verified**, not what you expect. "I added the zoom and the
  preview at 1:48 shows it framed on the formula bar" — or "I added it but did
  not preview it."
- **Never invent a timestamp.** Every time you state should trace to a beat, a
  transcript match, or a frame you looked at.
