# Contributing to AutoCine

AutoCine is an early macOS developer preview. Focused bug fixes, tests,
documentation corrections, and small, well-scoped features are welcome.

## Before opening a pull request

1. Search the existing issues and pull requests.
2. Explain the user-visible problem before proposing a large design change.
3. Keep changes compatible with Python 3.9 unless the project explicitly
   changes that baseline.
4. Do not add heavy dependencies without discussing the tradeoff first.
5. Run the test suite:

   ```bash
   python3 -m unittest discover -s tests
   ```

6. Run `git diff --check` and review the exact files you are submitting.

Hardware capture changes also need a real, permissioned recording test. Unit
tests cannot prove Screen Recording, Accessibility, Input Monitoring, camera,
audio synchronization, or ScreenCaptureKit behavior.

Read [docs/architecture.md](docs/architecture.md) before changing capture
manifests, clock conversion, camera planning, edit persistence, the local web
server, or the MCP boundary.

## Privacy requirements

- Never commit recordings, screenshots, transcripts, local notes, or device
  inventories from a real user.
- Never commit credentials, signing material, model-provider tokens, or
  machine-specific absolute paths.
- The keyboard listener may record activity timing only. It must never store
  key values.
- Use synthetic sessions and invented transcript text in tests and examples.
- Keep MCP stdout reserved for JSON-RPC protocol messages.

## Change discipline

- Preserve existing behavior when an effect or camera option is disabled.
- Add or update tests for behavior changes.
- Keep edits to `edits.json` compatible with its compare-and-swap `rev`
  contract.
- Keep the MCP editing skill in sync with MCP tool names and argument schemas.
- Avoid drive-by formatting or unrelated refactors.

## Licensing

By submitting a contribution, you agree that it may be distributed under the
Apache License 2.0 in [LICENSE](LICENSE). Do not submit code, media, model
weights, or other material unless you have the right to license it for this
project.
