# Architecture notes

This is the public map of AutoCine's main contracts. It is deliberately
shorter than the source comments and tests, which remain the authority for
edge cases.

## Pipeline

```text
capture -> project files -> non-destructive edits -> preview or export
```

A normal project contains captured media, `events.jsonl`, and `meta.json`.
Optional files add facecam media, a local transcript, a display name, cached
preview data, and `edits.json`. AutoCine does not rewrite source media when an
edit changes. It resolves the edit document when it previews or renders.

The main modules are:

- `autocine/record.py`: process lifecycle, event capture, and manifests.
- `autocine/sck.py` and `_sck_worker.py`: ScreenCaptureKit workers.
- `autocine/camera.py`: offline click clustering and camera planning.
- `autocine/render.py`: single-source, card-layout, native multi-window, and
  scene render dispatch.
- `autocine/edits.py`: edit defaults, normalization, and revisioned saves.
- `autocine/studio_app.py`: local HTTP editor API and native-window bridges.
- `autocine/mcp_server.py`: stdio JSON-RPC/MCP editing surface.
- `studio_web/`: the project library, editor, recorder bar, and overlays.

## Capture modes

The default capture backend is FFmpeg's AVFoundation input. ScreenCaptureKit
is opt-in and is required for occlusion-free window capture.

Single-source takes use one `raw.mov`. An ordinary selected-window take still
has one source coordinate space. An occlusion-free multi-window take records
each window into its own `raw_i.mov` and stores a `capture_channels` manifest.
A scene take changes that channel set at a pause/resume boundary and stores a
`capture_scenes` manifest.

These shapes are not interchangeable. In particular, a native multi-window or
scene take has no single source rectangle. Spatial edits that require one
shared coordinate space must refuse those projects rather than guess.

## Time and process clocks

Events, capture workers, media streams, and exported frames use related but
distinct clocks. Worker timestamps are paired into the parent process clock
before they enter a project. Edit times use source-media seconds. Cuts and
speed-ups can make an output timestamp differ from its source timestamp.

Keep clock conversion at the boundary that owns both clocks. Do not infer a
cross-process offset from two unpaired monotonic timestamps.

## Interaction data

Pointer movement, click, and scroll events can be recorded. Keyboard capture
stores only quantized activity ticks. It must never serialize the key object,
character, key code, or modifier identity.

Window metadata can include geometry and, for explicitly selected windows,
an application name or window title. Treat project metadata and transcripts
as private local data even when the media itself looks harmless.

## Camera and rendering

The main camera is planned offline:

1. Cluster nearby clicks.
2. Convert clusters into intentional zoom ranges.
3. Choose overview and target framing.
4. Evaluate a closed-form spring for every frame.

This avoids a camera that follows every pointer update. Typing, scrolling, and
dragging are useful beat metadata, but they are not legacy per-frame camera
detectors.

Feature switches are compatibility boundaries. When an effect is disabled,
the output path should reproduce the previous disabled behavior exactly.
Tests pin this rule for camera overview, speed-up, motion blur, window zoom,
window focus, and layout fallbacks.

## Edits and concurrency

`edits.json` is shared by the web editor and MCP clients. Every saved document
has a `rev`. Writers use compare-and-swap semantics and receive the current
document on conflict. `rev` itself is never patchable.

Render options are normalized in one place and must reach each supported
output path. A new option is incomplete if it works in the CLI but is dropped
by the editor, MCP server, preview path, or a specialized renderer.

## Local web server

The Studio server accepts only an explicit IPv4 loopback bind. It validates the exact HTTP
authority and origin, injects a per-launch token into each HTML shell, and
requires that token for API access. Passive browser resources that cannot add
a header receive a tokenized local URL. Render output requested through the
web UI is contained to the selected project directory.
Wildcard, LAN, hostname, and IPv6 binds are rejected rather than treated as a
supported remote-access mode.

## MCP boundary

The MCP server speaks JSON-RPC over stdio. Standard output belongs exclusively
to the protocol, so application output is redirected to standard error.

MCP and the web editor use the same project loader, edit document, preview
renderer, and final renderer. `describe_session` provides structured beats so
a client can orient before requesting individual frames. `preview_frame` with
`source=true` is the measurement surface for source-space coordinates.

The connected MCP client is a separate privacy boundary. It can transmit tool
results, transcript text, metadata, paths, or image pixels to its configured
model provider.

## Tests

Run the permission-free suite with:

```bash
python3 -m unittest discover -s tests
```

Synthetic media tests cover render behavior without Screen Recording,
Accessibility, Input Monitoring, microphone, or camera permission. Those tests
cannot establish that a real capture works on a particular Mac. Capture-path
changes also need a permissioned recording and inspection of the resulting
media, events, and metadata.

The supported language baseline is Python 3.9. Keep syntax compatible with the
system Python used by the development workflow, and avoid new heavy runtime
dependencies unless the project explicitly accepts the distribution cost.
