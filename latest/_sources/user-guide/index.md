# User guide

These pages get pypulseqpp onto your machine and a sequence out of it. If you
are new to Pulseq, the {doc}`Course <../examples/course>` is the place to start;
if you already have PyPulseq scripts, {doc}`from-pypulseq` shows what changes.

| Page | Contents |
| --- | --- |
| This page | Supported platforms, installation, shipped sequences, issues and security. |
| {doc}`from-pypulseq` | A PyPulseq script run unchanged, then rewritten as a sequence function. |
| {doc}`../explanations/index` | The concepts: the Pulseq file, the C++ core, design, modules, sampling, analysis and checks. |
| {doc}`../examples/index` | The Course, the Tours and a page per shipped sequence. |
| {doc}`../sequences` | The catalogue of shipped sequences. |
| {doc}`../api/index` | Exact interfaces, units and defaults. |
| {doc}`../developer-guide/index` | Development setup and contribution workflow. |
| {doc}`../misc/index` | Licensing, related projects and contributors. |
| [Source](https://github.com/pulserver/pypulseqpp) | Repository, issues and discussions. |
| [PDF manual](https://github.com/pulserver/pypulseqpp/releases/latest/download/pypulseqpp-docs.pdf) | This documentation as one file, from the latest release. |

```{toctree}
:hidden:

from-pypulseq
```

## Supported platforms

You need Python 3.10 to 3.14. CI tests Python 3.14 on Linux, macOS and Windows,
and Python 3.10 on Linux.

Wheels are published for:

| Platform | Architectures |
| --- | --- |
| Linux | x86-64, glibc (`manylinux`) |
| macOS | Apple silicon and x86-64 |
| Windows | AMD64 |

Each wheel carries the compiled C++ core. On any other platform pip builds from
source, which needs a C++17 compiler, CMake and the Python development headers;
those builds are not release-tested.

## Installation

```bash
pip install pypulseqpp
```

Three extras add optional features:

| Extra | Command | Adds |
| --- | --- | --- |
| `plot` | `pip install 'pypulseqpp[plot]'` | The interactive SeqEyes viewer, distributed separately under the GPL. |
| `mkl` | `pip install 'pypulseqpp[mkl]'` | Intel MKL as the FFT backend of `mech_resonance_spectrum`, on x86-64. |
| `design` | `pip install 'pypulseqpp[design]'` | `blochsim`, which designs the optimized refocusing schedules of the fast spin echo. |

To work on pypulseqpp itself, follow {doc}`../developer-guide/installation`.

## Shipped sequences

Each shipped sequence is a module of `pypulseqpp.sequences`. You call the
module with the prescription as keyword arguments and get the sequence back;
the parameters each one takes, with their units, are on its page in the
{doc}`catalogue <../sequences>`.

```python
>>> from pypulseqpp import sequences
>>> seq = sequences.gre2D_sequence(n_x=64, n_y=64, n_slices=1)
>>> signature = seq.write("gre_2d.seq")

```

Every module is also a command-line program, with one option per parameter:

```bash
python -m pypulseqpp.sequences.sequence.gre2D_sequence --help
```

## Issue reports

Report a reproducible defect, a documentation error or a narrowly scoped
feature request on the [issue tracker](https://github.com/pulserver/pypulseqpp/issues).
Include the pypulseqpp, PyPulseq and Python versions, your operating system and
the smallest script that reproduces the problem.

## Discussions

Ask usage questions and propose broad design ideas in
[GitHub Discussions](https://github.com/pulserver/pypulseqpp/discussions).
Search the existing topics first.

## Security

Report a vulnerability through
[GitHub private vulnerability reporting](https://github.com/pulserver/pypulseqpp/security/advisories/new),
never in a public issue. The policy is in
[`SECURITY.md`](https://github.com/pulserver/pypulseqpp/blob/main/SECURITY.md).
