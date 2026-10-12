# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/pulserver/pypulseqpp/security/advisories/new).
Do not open a public issue.

Include the affected version or commit, operating system and architecture, the
smallest input or sequence that reproduces the issue, its observed impact, and
any known mitigation. The maintainers aim to acknowledge a report within seven
days and will communicate the assessment and remediation plan through the
private advisory. Coordinated public disclosure follows release of a fix or an
agreed disclosure date. Reporters are credited unless they request otherwise.

## Supported versions

pypulseqpp is pre-1.0. Security fixes land on the default branch and are
included in the next release; older releases and backport branches are not
maintained.

| Version | Supported |
| --- | --- |
| Default branch and latest release | Yes |
| Older releases | No |

## Scope

In scope: anything that turns data into execution or memory corruption -- a
text or binary `.seq` file, a VOP, PNS-model, forbidden-band or acoustic
response file, or event values and waveforms passed to the native extension,
which indexes raw buffers.

Out of scope: a sequence script, which is Python that runs in the caller's
process by design; and resource exhaustion from inputs a caller chose
themselves. A check that returns a wrong estimate is a
[bug report](https://github.com/pulserver/pypulseqpp/issues/new/choose).
