# User guide

Installing the package, the platforms it is supported on, and how the project
is used and cited. Sequence physics and design concepts are covered in
{doc}`../explanations/index`; executable workflows are in
{doc}`../examples/index`.

## Documentation sections

| Section | Purpose |
| --- | --- |
| This page | Installation, supported platforms, and how to cite the project. |
| {doc}`../developer-guide/index` | Development setup and contribution workflow. |
| {doc}`../explanations/index` | Pulseq representation, sequence design and constraint models. |
| {doc}`../examples/index` | A course in sequence design, and a page per shipped sequence. |
| {doc}`../sequences` | Catalogue of the shipped sequences, one reference page each. |
| {doc}`../api/index` | Exact interfaces, units and defaults. |
| {doc}`../misc/index` | Licensing, related projects and contributors. |
| [Source](https://github.com/pulserver/pypulseqpp) | Repository, issues and discussions. |
| [PDF manual](https://github.com/pulserver/pypulseqpp/releases/latest/download/pypulseqpp-docs.pdf) | Single-file documentation from the latest release. |

## Prerequisites and supported platforms

pypulseqpp supports Python 3.10 through 3.13. CI tests the lower and upper
bounds on Linux, macOS and Windows.

Published wheels cover:

| Platform | Architectures |
| --- | --- |
| Linux | x86-64, glibc (`manylinux`) |
| macOS | Apple silicon and x86-64 |
| Windows | AMD64 |

The native core is compiled into each wheel. Source builds may work on other
C++17 platforms, but those configurations are not release-tested. A source
build requires a C++17 compiler, CMake, Python development headers and the
checked-out Git submodules.

## Installation

Install the core package from PyPI:

```bash
pip install pypulseqpp
```

Optional facilities are installed as extras:

| Extra | Command | Purpose |
| --- | --- | --- |
| SeqEyes viewer | `pip install 'pypulseqpp[plot]'` | Interactive sequence viewing through the separately distributed GPL viewer. |
| Intel MKL | `pip install 'pypulseqpp[mkl]'` | Optional FFT backend for mechanical-resonance analysis on x86-64. |
| FSE design | `pip install 'pypulseqpp[design]'` | `torchsim`, used by optimized fast-spin-echo refocusing schedules. |

Developer installation is documented in {doc}`../developer-guide/index`.

## Reporting issues

Use the [GitHub issue tracker](https://github.com/pulserver/pypulseqpp/issues)
for a reproducible defect, a documentation error, or a narrowly scoped feature
request. A useful report includes the package and Python versions, operating
system, relevant PyPulseq version, the affected subsystem and the smallest
reproducer.

Questions and broad design proposals belong in Discussions, and vulnerabilities
follow the private process stated under Security.

## Discussions

[GitHub Discussions](https://github.com/pulserver/pypulseqpp/discussions) is the
public forum for usage questions and broad design ideas. Search existing topics
before opening a new Q&A or idea. Reproducible defects belong in the issue
tracker.

## Security

The canonical policy is [`SECURITY.md`](https://github.com/pulserver/pypulseqpp/blob/main/SECURITY.md).
Vulnerabilities must be reported through
[GitHub private vulnerability reporting](https://github.com/pulserver/pypulseqpp/security/advisories/new),
not through a public issue.
