#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 DESTINATION" >&2
  echo "Create a new, history-free public-source snapshot." >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
destination="$(python3 -c \
  'import os, sys; print(os.path.realpath(os.path.abspath(os.path.expanduser(sys.argv[1]))))' \
  "$1")"

case "$destination" in
  "$repo_root"|"$repo_root"/*)
    echo "Refusing to create a snapshot inside the source repository." >&2
    exit 2
    ;;
esac

if [[ -e "$destination" ]]; then
  echo "Destination already exists; choose a new empty path: $destination" >&2
  exit 2
fi

mkdir -p "$destination"

is_public_path() {
  case "$1" in
    .gitignore|LICENSE|THIRD_PARTY_NOTICES.md|SECURITY.md|CONTRIBUTING.md|README.md|requirements.txt|studio.py)
      return 0
      ;;
    autocine/*|studio_web/*|site/*|tests/*)
      return 0
      ;;
    .agents/skills/edit-recording/*|.claude/skills/edit-recording/*)
      return 0
      ;;
    .github/workflows/tests.yml)
      return 0
      ;;
  docs/architecture.md|docs/mcp.md|docs/publishing.md|scripts/prepare_public_snapshot.sh)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

copied=0
while IFS= read -r -d '' path; do
  if ! is_public_path "$path"; then
    continue
  fi
  if [[ -L "$repo_root/$path" || ! -f "$repo_root/$path" ]]; then
    echo "Refusing non-regular public path: $path" >&2
    exit 1
  fi
  mkdir -p "$destination/$(dirname "$path")"
  cp -p "$repo_root/$path" "$destination/$path"
  copied=$((copied + 1))
done < <(git -C "$repo_root" ls-files --cached --others --exclude-standard -z)

# Private working notes are intentionally not part of the public snapshot.
# Source comments historically named those topic files, so point every copied
# reference at the consolidated public contract instead of leaving dead paths.
python3 - "$destination" <<'PY'
import os
import sys

root = sys.argv[1]
names = (
    "architecture.md",
    "camera.md",
    "features.md",
    "testing.md",
    "roadmap.md",
    "distribution.md",
    "window-native-capture.md",
    "multi-window-native-capture.md",
    "segmented-takes.md",
    "scene-takes.md",
    "manual-card-layout.md",
    "CLAUDE.md",
)
extensions = (".py", ".js", ".css", ".html")
marker = "__AUTOCINE_PUBLIC_ARCHITECTURE_DOC__"
for top in ("autocine", "studio_web", "tests"):
    base = os.path.join(root, top)
    for directory, _subdirs, files in os.walk(base):
        for filename in files:
            if not filename.endswith(extensions):
                continue
            path = os.path.join(directory, filename)
            with open(path, encoding="utf-8") as handle:
                before = handle.read()
            after = before
            for name in sorted(names, key=len, reverse=True):
                after = after.replace(".claude/docs/" + name, marker)
                after = after.replace(name, marker)
            after = after.replace(marker, "docs/architecture.md")
            if after != before:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(after)
PY

required=(
  LICENSE
  THIRD_PARTY_NOTICES.md
  SECURITY.md
  CONTRIBUTING.md
  README.md
  requirements.txt
  studio.py
  autocine/mcp_server.py
  docs/architecture.md
  docs/mcp.md
  .github/workflows/tests.yml
  .claude/skills/edit-recording/SKILL.md
)

for path in "${required[@]}"; do
  if [[ ! -f "$destination/$path" ]]; then
    echo "Snapshot is incomplete; missing $path" >&2
    exit 1
  fi
done

for path in .git recordings dist .wrangler tools packaging .claude/docs CLAUDE.md AGENTS.md .mcp.json; do
  if [[ -e "$destination/$path" ]]; then
    echo "Snapshot contains prohibited path: $path" >&2
    exit 1
  fi
done

user_home_pattern="/""Users""/[^/]+/"
if grep -R -I -n -E "$user_home_pattern" "$destination" >/dev/null; then
  echo "Snapshot contains a machine-specific /Users/... path:" >&2
  grep -R -I -n -E "$user_home_pattern" "$destination" >&2
  exit 1
fi

# Catch common credential formats without ever printing a matched value. The
# pattern fragments keep this script from matching its own scanner source.
python3 - "$destination" <<'PY'
import os
import re
import sys

root = sys.argv[1]
rules = {
    "private-key header": re.compile("-----BEGIN " +
                                      "(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "AWS access key": re.compile("AKIA" + "[0-9A-Z]{16}"),
    "GitHub token": re.compile("gh" + "[pousr]_[A-Za-z0-9]{30,}"),
    "OpenAI-style key": re.compile("sk-" + "(?:proj-)?[A-Za-z0-9_-]{32,}"),
    "Anthropic key": re.compile("sk-ant-" + "[A-Za-z0-9_-]{24,}"),
    "Slack token": re.compile("xox" + "[abprs]-[A-Za-z0-9-]{16,}"),
    "Google API key": re.compile("AIza" + "[A-Za-z0-9_-]{30,}"),
    "credential in URL": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
}
bad_names = {".env", "credentials", "credentials.json", "id_rsa", "id_ed25519"}
bad_suffixes = (".pem", ".p12", ".mobileprovision")
findings = []
for directory, subdirs, files in os.walk(root):
    subdirs[:] = [name for name in subdirs if name != ".git"]
    for filename in files:
        rel = os.path.relpath(os.path.join(directory, filename), root)
        lower = filename.lower()
        if lower in bad_names or lower.endswith(bad_suffixes):
            findings.append((rel, "sensitive filename"))
            continue
        try:
            with open(os.path.join(directory, filename), encoding="utf-8") as handle:
                content = handle.read()
        except (OSError, UnicodeDecodeError):
            continue
        for label, pattern in rules.items():
            if pattern.search(content):
                findings.append((rel, label))
if findings:
    print("Snapshot may contain credential material:", file=sys.stderr)
    for rel, label in sorted(set(findings)):
        print("  {} ({})".format(rel, label), file=sys.stderr)
    raise SystemExit(1)
PY

chmod +x "$destination/scripts/prepare_public_snapshot.sh"

echo "Created public snapshot with $copied files:"
echo "  $destination"
echo "No Git history was copied. Review docs/publishing.md before publishing."
echo "Private design-note references were normalized to docs/architecture.md."
echo "Set a public-safe, repository-local Git author email before the first commit."
