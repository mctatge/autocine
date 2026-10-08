# Editing AutoCine over MCP

AutoCine exposes its local, non-destructive editing engine as a stdio MCP
server. It does not provide a separate cloud service. A compatible AI client
starts `studio.py mcp`, discovers the available tools, and reads or updates the
same project files used by the editor.

```text
MCP client  <->  stdio server  <->  recordings/<session>/
                                      raw media (read)
                                      edits.json (read/write)
                                      rendered exports (write)
```

## Connect a client

Install AutoCine from source first, then start the server directly to confirm
that the environment can import its dependencies:

```bash
/absolute/path/to/AutoCine/.venv/bin/python \
  /absolute/path/to/AutoCine/studio.py mcp \
  --recordings-root /absolute/path/to/AutoCine/recordings
```

Register the same executable and arguments in any MCP client that supports a
local stdio server. Generate a configuration entry with resolved absolute
paths:

```bash
.venv/bin/python studio.py mcp --print-config
```

Paste or translate the generated entry into the client's MCP settings. Clients
use different configuration files and surrounding field names, but the
executable and argument array are the same. Pass `--recordings-root` or set
`AUTOCINE_RECORDINGS_ROOT` before generating it when projects live somewhere
other than the checkout's `recordings/` directory.

Do not copy a static tool list from this guide into an integration. Print the
live schemas directly from a checkout:

```bash
.venv/bin/python studio.py mcp --print-tools
```

The command and MCP `tools/list` return schemas generated from the same
canonical tool definitions.

## The editing loop

Use the tools as a feedback loop, not as a one-shot renderer.

### 1. Orient

Start with `list_sessions`, then call `describe_session` on the selected take.
Its `beats` field gives a time-ordered summary of click clusters, scrolling,
typing, idle spans, and available window transitions. Use `get_edits` to see
the existing non-destructive decisions before replacing any array.

Representative tools:

- `list_sessions`
- `describe_session`
- `get_edits`
- `list_recorded_windows`

### 2. Locate

Prefer semantic evidence over guessed timestamps:

- Use `find_in_transcript` when the user identifies a moment by what was said.
- Use beat times when they identify an interaction or idle span.
- Use `preview_frame(source=true)` to measure a crop, window, or pinned zoom in
  source coordinates.
- Use ordinary `preview_frame` to judge how the current edit will look.

The first transcript request can run a full local whisper.cpp transcription.
`describe_session.has_transcript` tells the client whether the cached read
model already exists. If transcription is unavailable, report the reason; do
not invent a timestamp.

### 3. Edit

Choose the smallest tool that expresses the requested change. Examples:

- `set_trim` for one in/out range.
- `add_cut` or `set_cuts` for ripple-delete ranges on a supported take.
- `adjust_zoom` for “softer,” “stronger,” or “off” at a named time.
- `add_zoom` for an explicit time range and optional source-space target.
- `add_speedup` for a deliberate fast-forward span.
- `set_render_options` for aspect, background, camera, cursor, audio, or
  presentation choices.
- The window tools for a supported card-layout take.

Every edit is written to `edits.json`; the source media is not rewritten.
Saves use a revision value so the MCP client and web editor do not silently
overwrite each other's concurrent changes.

### 4. Verify

Read the saved edit back with `get_edits`. Use `preview_frame` near every
material visual change and self-correct before starting a full export. Finish
with `render_video`, which resolves the same render choices as the web
editor's Export action.

A useful client instruction is:

> Read `describe_session.beats` first. Locate moments from transcript or source
> frames instead of guessing. After every edit, read it back and inspect a
> rendered preview before exporting.

## Time and coordinates

MCP edit times use the recording's **source-media clock**. Trim, cuts, and
speed-ups can make a timestamp shown in the exported video differ from its
source time. Tool responses report the matched or snapped source range; report
those returned values instead of assuming the requested numbers were exact.

`preview_frame(source=true)` returns an unedited source-space image with
`source_width`, `source_height`, and `source_scale`. Use that image for crops,
window rectangles, and pinned camera targets. A rendered preview has already
been cropped, framed, panned, and zoomed, so no single scale maps it back to
source coordinates.

Recorded window IDs are intentionally opaque. Event logs do not store app
names or window titles. Match an ID through its geometry or inspect a source
frame.

## Data and privacy boundary

The AutoCine MCP process runs locally over stdio. It reads local recording
files and writes local edit specs and exports. AutoCine does not upload the
recording on its own.

The connected client decides what it sends to its model provider. Tool results
can include:

- transcript text and matched phrases;
- timing and interaction metadata;
- source or rendered image pixels from `preview_frame`;
- local paths and project names.

Use the client's own provider, retention, and local-model controls for
sensitive material. “AutoCine runs locally” is not a claim that every MCP
client or model also runs locally.

AutoCine records only quantized key-activity timing, never the key identity.
`record --no-key-log` disables even those activity ticks.

## Supported-surface limits

- Native multi-window and scene takes do not have one shared source coordinate
  space. Tools such as a non-null `set_crop`, `add_zoom`, and card-window edits
  refuse where their coordinates would be ambiguous.
- Ripple cuts are currently applied only to ordinary single-source renders.
  Scene, native multi-window, and card-layout renders report that cuts are not
  applied.
- `preview_frame` uses source time and does not simulate the output-time shift
  introduced by cuts or speed-ups.
- `render_video.out` must resolve inside the selected session directory. Use a
  session-relative filename such as `final.mp4`; parent traversal and symlink
  escapes are rejected.
- Transcription requires an audio track, `whisper.cpp`, and a compatible local
  model. The rest of the MCP server works without them.

These are explicit refusals or reported limitations. A client should preserve
the user's existing edits and present the returned reason instead of silently
substituting a different operation.
