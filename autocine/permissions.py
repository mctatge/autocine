"""macOS permission checks for recording readiness."""

import sys


def _mk_check(label, state, detail):
    return {
        "label": label,
        "state": state,  # granted | missing | unknown
        "detail": detail,
    }


def _check_screen_recording():
    label = "Screen Recording"
    try:
        import Quartz
        fn = getattr(Quartz, "CGPreflightScreenCaptureAccess", None)
        if fn is None:
            return _mk_check(label, "unknown",
                             "Unable to preflight Screen Recording access on this macOS build.")
        granted = bool(fn())
        if granted:
            return _mk_check(label, "granted", "Permission is granted.")
        return _mk_check(label, "missing",
                         "Grant Screen Recording to your terminal app, then fully quit and reopen it.")
    except Exception as exc:
        return _mk_check(label, "unknown",
                         "Check failed: {}".format(exc))


def _check_input_monitoring():
    label = "Input Monitoring"
    try:
        import Quartz
        fn = getattr(Quartz, "CGPreflightListenEventAccess", None)
        if fn is None:
            return _mk_check(label, "unknown",
                             "Unable to preflight Input Monitoring access on this macOS build.")
        granted = bool(fn())
        if granted:
            return _mk_check(label, "granted", "Permission is granted.")
        return _mk_check(label, "missing",
                         "Grant Input Monitoring to your terminal app to capture global clicks.")
    except Exception as exc:
        return _mk_check(label, "unknown",
                         "Check failed: {}".format(exc))


def _check_accessibility():
    label = "Accessibility"
    try:
        import Quartz
        fn = getattr(Quartz, "AXIsProcessTrusted", None)
        if fn is None:
            return _mk_check(label, "unknown",
                             "Unable to preflight Accessibility trust on this macOS build.")
        granted = bool(fn())
        if granted:
            return _mk_check(label, "granted", "Permission is granted.")
        return _mk_check(label, "missing",
                         "Grant Accessibility to your terminal app for pynput click listeners.")
    except Exception as exc:
        return _mk_check(label, "unknown",
                         "Check failed: {}".format(exc))


def build_report(checks, platform):
    """Build a stable permission report envelope used by the web UI."""
    required = ("screen_recording", "input_monitoring", "accessibility")
    missing = [k for k in required if checks.get(k, {}).get("state") == "missing"]
    unknown = [k for k in required if checks.get(k, {}).get("state") == "unknown"]

    if missing:
        summary = "Recording is blocked: missing {}.".format(
            ", ".join(checks[k]["label"] for k in missing))
    elif unknown:
        summary = "Some checks are unavailable; recording may still work."
    else:
        summary = "All required permissions are granted."

    return {
        "platform": platform,
        "checks": checks,
        "required": list(required),
        "missing_required": missing,
        "unknown_required": unknown,
        "all_required_granted": not missing and not unknown,
        "can_attempt_record": not missing,
        "summary": summary,
        "next_steps": [
            "System Settings -> Privacy & Security -> grant all missing permissions to this terminal app.",
            "Fully quit and reopen the terminal after granting Screen Recording.",
        ],
    }


def check_permissions():
    """Return a macOS-focused permission report."""
    platform = sys.platform
    if platform != "darwin":
        checks = {
            "screen_recording": _mk_check(
                "Screen Recording", "unknown",
                "This checker currently supports macOS only."),
            "input_monitoring": _mk_check(
                "Input Monitoring", "unknown",
                "This checker currently supports macOS only."),
            "accessibility": _mk_check(
                "Accessibility", "unknown",
                "This checker currently supports macOS only."),
        }
        return build_report(checks, platform)

    checks = {
        "screen_recording": _check_screen_recording(),
        "input_monitoring": _check_input_monitoring(),
        "accessibility": _check_accessibility(),
    }
    return build_report(checks, platform)
