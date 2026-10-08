"""Standalone child process: reports bare key-activity timestamps on stdout.

WHY THIS IS ITS OWN PROCESS. pynput's macOS keyboard backend decodes every
keypress through Carbon/TIS (`TISCopyCurrentKeyboardInputSource`,
`TISGetInputSourceProperty`, `LMGetKbdType` -- see the installed
`pynput/_util/darwin.py:keycode_context`), reached via raw `ctypes`, not
PyObjC's Objective-C bridge. On this machine (macOS 15.6.1, the Xcode-bundled
Python 3.9.6, pynput 1.8.2) that call reproduces a hard, native
EXC_BREAKPOINT/SIGTRAP crash -- confirmed 6/6 across independent real
recordings, including a fully manual one, with an identical faulting address
every time (ctypes -> libffi -> HIToolbox -> libdispatch). Matches known
pynput issues (moses-palmer/pynput #511/#512) hit by other users running a
keyboard.Listener inside an active native GUI run loop, which is exactly
what `studio.py bar`'s WKWebView-hosted process is.

A native trap is not a Python exception -- no `try/except` in the parent can
survive it. Running the decode in a disposable child process is the only way
to keep a fresh confirmed-safe-in-this-environment crash from taking the
whole recording down with it: if this process dies, the parent just marks
`key_capture: "failed"` (a state that already exists and is already handled
end to end) and the screen recording is completely unaffected -- the same
posture the facecam capture already uses for exactly the same reason.

PRIVACY. We only ever need to know THAT a key was pressed, never WHICH one --
`Recorder._on_key` in record.py has never read the key object. So the
decoded value pynput hands to `on_press` is dropped in the very same
statement it's received; nothing about it is retained, and nothing but a
raw `time.monotonic()` float ever crosses the pipe to the parent.

CLOCK. Our timestamps are NOT directly comparable to the parent's. This
file used to claim the opposite -- that `time.monotonic()` is
`mach_absolute_time`-backed system uptime and so process-independent -- and
that is FALSE on this stack (measured 2026-07-30, macOS 15.6 / Xcode-bundled
Python 3.9.6): three fresh interpreters each read ~0.007 s while the machine
had been up ~189,700 s, and a child spawned 0.5 s into a parent's life read
0.0058 against the parent's 0.55. The reference point is process start, which
CPython's docs explicitly leave undefined. Untranslated, our readings land
one whole parent-process-age in the past -- and the parent is usually the
long-lived `studio.py bar`, so every key tick would fall far before the take
even began. So the READY line now carries OUR clock at the moment we send it,
letting the parent pair the two domains once and translate every later
timestamp. If the two clocks ever DO share a domain, the measured offset is
simply ~0 and the translation is a no-op -- correct either way, on any
platform, without this file having to know which case it is in.

WIRE PROTOCOL, deliberately tiny (less here is less that can go wrong in the
one process whose whole job is to isolate a known-flaky call):
  stdout, line-buffered, one line per event:
    "READY <float>\n"  -- printed once the tap is confirmed up (mirrors the
                           existing wait()+is_alive() check the in-process
                           listener used to do). The float is our own
                           time.monotonic() as we write the line; see CLOCK.
    "<float>\n"         -- one key-press timestamp, in the child's own
                            time.monotonic()
  No output at all (silent exit) means the tap could not be created --
  macOS's Input Monitoring denial fails this way (pynput's tap creation
  returns None instead of raising), the same silent-failure shape the old
  in-process path had to defend against.
Never argv, never stderr for anything payload-shaped -- keep the surface
a parent has to trust to a minimum.
"""
import sys
import time


def main():
    from pynput import keyboard

    # Undo any inherited SIG_BLOCK on our stop signals. A parent that blocks
    # SIGINT/SIGTERM process-wide (cli._install_signal_quit, so the bar's
    # sigwait thread can quit the Cocoa-owned main thread) leaks that blocked
    # mask across fork+exec -- and this worker has no handler, so a blocked
    # SIGINT can neither run a handler nor take its default (terminate). The
    # parent's `_KeyWorker.stop()` SIGINT would then be ignored until its 2s
    # SIGKILL, delaying every stop. Unblock so the parent's SIGINT stops us at
    # once. Mirrors _sck_worker.py; see record.Recorder for the harvest side.
    import signal
    if hasattr(signal, "pthread_sigmask"):
        try:
            signal.pthread_sigmask(
                signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
        except (ValueError, OSError):
            pass

    def on_press(_key):
        # `_key` is intentionally unused -- see the privacy note above.
        try:
            sys.stdout.write("{}\n".format(time.monotonic()))
            sys.stdout.flush()
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    wait = getattr(listener, "wait", None)
    if callable(wait):
        wait()
    time.sleep(0.05)   # let a silently-failed tap finish exiting
    if not listener.is_alive():
        return   # exit with no READY line -- the parent reads this as failed
    # Read the clock as late as possible -- the parent pairs it against its
    # own reading of THIS line, so anything between the two calls (the write,
    # the pipe hop) shows up as offset error. See CLOCK above.
    sys.stdout.write("READY {}\n".format(time.monotonic()))
    sys.stdout.flush()
    listener.join()


if __name__ == "__main__":
    main()
