# AutoCine

AutoCine is a free, open-source, experimental macOS screen recorder and
deterministic video editor. It records the screen and interaction timing, then
builds a virtual camera from deliberate signals such as click clusters and
selected windows.

[Website](https://autocine.pages.dev/) · [MCP guide](docs/mcp.md)

The same local editing engine is available through the command line, the
browser-based editor, and a stdio MCP server. An MCP-capable AI client can
inspect a take, find a spoken phrase, write non-destructive cuts or camera
changes, preview the result, and export through the same renderer as the UI.

```text
record  ->  raw media + events + metadata
edit    ->  edits.json (UI, CLI, or MCP)
render  ->  MP4 or GIF
```

> **Alpha status:** AutoCine currently ships as source for developers and
> early testers. There is no signed, notarized consumer download yet. Expect
> rough edges, macOS permission setup, and changes to the project format.

## Why AutoCine

- **Camera motion from real activity.** Click clusters produce intentional
  zoom ranges instead of a camera that chases the pointer every frame.
- **Non-destructive projects.** A recording stays untouched. Trims, cuts,
  zooms, layouts, and presentation settings live in `edits.json` and can be
  rendered again.
- **MCP-native editing.** AI clients use structured beats, transcript spans,
  source frames, and explicit edit tools rather than guessing from a flattened
  video.
- **Local source media.** Recording, transcription with whisper.cpp, editing,
  and rendering run on the Mac. See [Data and privacy](#data-and-privacy) for
  the boundary when an external AI client is connected.
- **One engine, several surfaces.** The local editor, CLI, and MCP server share
  the same project files and renderer.

AutoCine also supports facecam capture, synthetic-cursor replacement,
multi-window layouts, occlusion-free ScreenCaptureKit recording, ripple cuts,
idle speed-up, vertical and square exports, backgrounds, click and keyboard
sounds, and GIF export. Some combinations remain limited; see
[Current limitations](#current-limitations).

## Install from source

Requirements:

- macOS
- Python 3.9 or newer
- `ffmpeg`

From an AutoCine checkout:

```bash
brew install ffmpeg

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 studio.py devices
python3 studio.py app
```

`studio.py app` starts the local project library and editor. `studio.py bar`
opens the compact recorder, and `studio.py open recordings/<session>` opens a
project directly.

## macOS permissions

The app or terminal that starts AutoCine needs these grants in **System
Settings -> Privacy & Security**:

1. **Screen Recording** for video capture.
2. **Input Monitoring** for global click and pointer activity.
3. **Accessibility** for the capture controls and event listener.
4. **Camera**, only when recording a facecam.

Grant all required permissions to the same launching app, then fully quit and
reopen it. A recording with video but no click events usually means Input
Monitoring or Accessibility is missing from that launcher.

AutoCine never records key identity. By default it records a 100 ms-quantized
activity tick that says only that typing occurred. Use `record --no-key-log`
to disable those ticks too.

## Record, edit, and render

```bash
# Record until Ctrl+C, then render the new take.
python3 studio.py record --render

# Record a fixed-length take with microphone and facecam.
python3 studio.py record --duration 30 --mic 1 --face --render

# Record selected windows into independent ScreenCaptureKit buffers.
python3 studio.py devices
python3 studio.py record --occlusion-free \
  --capture-window <id-a> --capture-window <id-b>

# Re-render a project for a vertical canvas with a framed background.
python3 studio.py render recordings/<session> \
  --aspect 9:16 --background midnight

# Remove ranges from an ordinary single-source take.
python3 studio.py render recordings/<session> \
  --cut 12.5-19 --cut 44-51.2
```

Run `python3 studio.py --help` and the subcommand `--help` pages for the
complete option set.

## Edit with an AI client over MCP

Start the stdio server, implemented without an MCP SDK, with:

```bash
python3 studio.py mcp
```

Then register that command in any client that supports local stdio MCP
servers. AutoCine can print a portable configuration with resolved absolute
paths:

```bash
python3 studio.py mcp --print-config
```

Paste or translate that generated entry into the client's MCP settings.
Configuration file names and surrounding fields vary by client; the executable
and argument array are the portable part. Use `--recordings-root` or
`AUTOCINE_RECORDINGS_ROOT` when projects live somewhere else.

Print the live JSON tool schemas with:

```bash
python3 studio.py mcp --print-tools
```

That generated catalog contains 26 tools in this release. MCP clients receive
the same schemas through `tools/list`; generated output, rather than a
hand-written documentation list, is authoritative.

A reliable editing loop is:

1. **Orient** with `list_sessions`, then `describe_session`. Read its `beats`
   before requesting frames one by one.
2. **Locate** a moment with `find_in_transcript`, beat times, or
   `preview_frame(source=true)` when source coordinates matter.
3. **Edit** with explicit tools such as `add_cut`, `adjust_zoom`,
   `set_render_options`, or the window-layout tools.
4. **Verify** with `get_edits` and `preview_frame`, then use `render_video` for
   the final export.

For example:

> Find where I say “watch this,” cut the pause before it, soften the next zoom,
> preview the result, and render a vertical version.

See [docs/mcp.md](docs/mcp.md) for setup details, time and coordinate rules,
the privacy boundary, and common workflows.

## Data and privacy

AutoCine stores source recordings, event logs, transcripts, and edit specs on
the local filesystem. Its editor accepts only an explicit loopback bind, the
MCP server uses stdio, and AutoCine itself does not upload a recording or
require an account. When you explicitly select windows, project metadata can
include the application name and window title; those titles may contain
document or message subjects.

An attached MCP client is a separate trust boundary. It may send tool results
to its configured model provider. Those results can include transcript
excerpts, recording metadata, and image pixels returned by `preview_frame`.
Connecting an AI client therefore does **not** mean that everything remains
on-device. Review that client's provider and data controls before exposing a
sensitive take.

Local transcription is optional. Install `whisper.cpp` and provide a compatible
GGML model to use transcript tools without a cloud transcription service.

## Project format

Each directory under `recordings/` is one project. A normal project contains:

```text
raw.mov                 captured screen media
events.jsonl            pointer, click, scroll, and key-activity timing
meta.json               capture and coordinate metadata
edits.json              non-destructive timeline and render choices
transcript.json          optional local speech-to-text cache
face.mov                 optional facecam track
project.json             optional display-name sidecar
```

Occlusion-free multi-window and scene takes use manifests plus multiple
`raw_*.mov` files instead of one `raw.mov`.

## Current limitations

- AutoCine is an alpha source release, not a polished consumer application.
- Capture and interaction mapping currently target the main macOS display.
- macOS permission setup belongs to the launching app and can be confusing
  when AutoCine is started through an IDE or nested shell.
- Ripple cuts work on ordinary single-source takes. Scene, native multi-window,
  and card-layout renders currently leave cuts unapplied and report that
  limitation.
- Some spatial MCP edits intentionally refuse native multi-window or scene
  takes because those projects do not have one shared source coordinate
  space.
- Render speed depends heavily on source resolution, frame rate, and enabled
  effects.

## Tests

The test suite runs without Screen Recording or Input Monitoring permission:

```bash
python3 -m unittest discover -s tests
```

The suite covers camera planning, capture metadata, edit persistence, render
geometry and audio, the local web surfaces, multi-window and scene projects,
and the MCP protocol.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for the Python 3.9 compatibility and
privacy requirements. Report security issues through the private process in
[SECURITY.md](SECURITY.md), not through a public issue containing recording or
exploit details.

## License

AutoCine is licensed under the [Apache License 2.0](LICENSE). Dependencies and
external tools retain their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
