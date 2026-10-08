# Publishing a source snapshot

Publish AutoCine from a new snapshot, not by changing the visibility of this
working repository. The existing Git history contains private development
notes, former names, and machine-specific context that are not part of the
public product.

## Create the snapshot

Choose a destination that does not already exist and is outside this checkout:

```bash
./scripts/prepare_public_snapshot.sh /tmp/autocine-public
```

The exporter copies only an explicit allowlist from the current working tree.
It includes the application source, web UI, tests, site, public architecture
notes, license files, contribution/security policies, and the MCP editing
skill.

Some source comments name private design-note files. In the snapshot, those
names are normalized to the consolidated `docs/architecture.md` contract so
the published tree does not contain dead internal references.

It excludes:

- all Git history;
- recordings, rendered media, transcripts, notes, settings, and cached state;
- `.claude/docs/`, `CLAUDE.md`, and `AGENTS.md` internal project records;
- local MCP/client configuration and Cloudflare account state;
- developer-only app/DMG packaging scaffolding;
- development probes in `tools/`; and
- build output and Python caches.

The script refuses to overwrite an existing destination or write inside this
repository. It also fails if a copied text file contains a `/Users/...`
absolute path, a sensitive credential filename, or one of several common
credential formats. That fast check reduces accidents but does not replace a
dedicated secret scanner before the first push.

## Verify the result

Run these commands from the new snapshot:

```bash
bash -n scripts/prepare_public_snapshot.sh
python3 -m unittest discover -s tests
grep -R -I -n -E "$(printf '/%s/[^/]+/' Users)" .
```

The final `grep` must produce no output. Also inspect the full tree manually,
check every dependency and media asset against `THIRD_PARTY_NOTICES.md`, and
run a dedicated secret scanner before upload.

Do not copy a local `.mcp.json` or `.codex/config.toml` into the public tree.
Their absolute paths are machine-specific. Generate a client configuration for
each checkout with:

```bash
python3 studio.py mcp --print-config
```

Initialize a new repository only after that review:

```bash
git init -b main
git config --local user.name "YOUR PUBLIC NAME"
git config --local user.email "YOUR GITHUB NOREPLY ADDRESS"
git var GIT_AUTHOR_IDENT
git add --all
git diff --cached --check
git status --short
```

Replace both placeholders. Use the GitHub-provided `@users.noreply.github.com`
address if the account's personal email should remain private. Do not make the
first commit until `git var GIT_AUTHOR_IDENT` prints the intended public name
and address; commit metadata is public and persists in repository history.

Review the staged file list before committing. Creating the GitHub repository,
pushing, and changing its visibility are separate external actions and should
happen only after explicit approval.

## Release boundary

This process prepares a source repository. It does not make the current `.app`
or DMG suitable for public distribution. A public binary still needs a
self-contained runtime, a release-specific dependency/license inventory,
Developer ID signing, notarization, and the remaining packaging/security work.
