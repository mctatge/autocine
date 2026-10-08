# Third-party notices

AutoCine's repository does not vendor third-party Python packages, model
weights, codecs, or executables. Developers install them separately. The
following inventory is based on `requirements.txt` and locally inspected
package metadata from the development environment as of October 7, 2026.

The AutoCine source is licensed under Apache-2.0. Each dependency remains
under its own license. Downstream distributors must review the exact packages
and binaries in their build and ship the license texts and notices those
artifacts require. This file is not a substitute for a binary-distribution
license audit.

## Direct Python dependencies

| Package | Declared license | Project |
|---|---|---|
| NumPy | BSD-3-Clause | <https://numpy.org/> |
| opencv-python-headless | Apache-2.0 | <https://github.com/opencv/opencv-python> |
| Pillow | MIT-CMU | <https://python-pillow.org/> |
| pynput | LGPLv3 | <https://github.com/moses-palmer/pynput> |
| pywebview | BSD-3-Clause | <https://pywebview.flowrl.com/> |
| PyObjC core and framework packages | MIT | <https://github.com/ronaldoussoren/pyobjc> |

The PyObjC entries cover the Quartz, ScreenCaptureKit, AVFoundation,
CoreMedia, Cocoa, CoreAudio, ApplicationServices, WebKit, Security, and
UniformTypeIdentifiers framework wrappers reached by AutoCine's declared
dependencies on macOS.

## Transitive Python dependencies

The installed dependency graph also includes these runtime packages:

| Package | Declared license | Project |
|---|---|---|
| six | MIT | <https://github.com/benjaminp/six> |
| bottle | MIT | <https://bottlepy.org/> |
| proxy_tools | MIT | <https://github.com/jtushman/proxy_tools> |
| typing_extensions | PSF-2.0 | <https://github.com/python/typing_extensions> |

Package versions and transitive dependencies can change. Treat the package
metadata installed into a release build as authoritative for that build.

## External tools

- **FFmpeg** is required by the current development workflow but is not
  included in this repository. FFmpeg's effective license depends on its build
  configuration. The Homebrew build used during development reports
  GPL-3.0-or-later because it enables GPL components. See
  <https://ffmpeg.org/legal.html>.
- **whisper.cpp** is an optional, separately installed transcription backend.
  It is MIT-licensed and is not included here. See
  <https://github.com/ggml-org/whisper.cpp>.
- Apple frameworks are operating-system components and are not redistributed
  by this source repository.

The `opencv-python-headless` wheel can contain additional compiled libraries,
including an FFmpeg build. Anyone redistributing that wheel must preserve its
`LICENSE.txt` and `LICENSE-3RD-PARTY.txt` and audit the exact wheel selected for
the release platform.
