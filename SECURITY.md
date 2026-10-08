# Security policy

AutoCine is an early developer preview. Only the latest revision on the
default branch receives security fixes. There are no supported binary releases
yet.

## Report a vulnerability

Do not publish exploit details, private recordings, transcripts, access
tokens, or other sensitive material in a GitHub issue.

Use GitHub's **Report a vulnerability** action in the repository's Security
tab. If private vulnerability reporting is unavailable, open an issue that
contains no exploit details and asks the maintainers to establish a private
channel.

Include:

- the affected revision or version;
- the macOS and Python versions;
- a concise description of the impact;
- the smallest safe reproduction you can provide; and
- whether the report involves a recording, transcript, local file, or network
  request.

You should receive an acknowledgment within seven days. A fix timeline depends
on severity and the maturity of the affected feature. Please allow a reasonable
time for a fix before public disclosure.

## Security and privacy boundaries

AutoCine records the screen and observes mouse activity. Its keyboard listener
records only quantized activity timestamps; it must never serialize the key
that was pressed. Recordings, transcripts, and edit metadata are local files
and are excluded from version control.

The MCP server gives the connected AI client tools to inspect and modify local
AutoCine projects. The client may send tool results, transcript excerpts, or
preview images to its configured model provider. Review that client's data
policy before connecting private recordings.

The source-development server accepts only an explicit IPv4 loopback bind. It
rejects wildcard, LAN, hostname, and IPv6 binds, validates the HTTP authority
and origin, and uses a per-launch token for its local APIs. Remote serving is
not supported. A packaged consumer release still has signing, dependency,
notarization, and sandboxing work; an ad-hoc developer build is not a supported
public download.
