"""AutoCine: record your macOS screen and render it with
automatic, click-driven zoom — an automatic cinematic camera.

Pipeline:
  record  -> raw.mov (ffmpeg/avfoundation) + events.jsonl (pynput) + meta.json
  render  -> output.mp4 with a smooth virtual camera that eases toward clicks
"""

__version__ = "0.1.0"
