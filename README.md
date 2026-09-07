# pypulseqpp

Fast drop-in PyPulseq replacement over a C++ sequence core, with hardware safety checks.

[![Tests](https://github.com/pulserver/pypulseqpp/actions/workflows/test-ci.yml/badge.svg)](https://github.com/pulserver/pypulseqpp/actions/workflows/test-ci.yml)
[![codecov](https://codecov.io/gh/pulserver/pypulseqpp/branch/main/graph/badge.svg)](https://codecov.io/gh/pulserver/pypulseqpp)
[![PyPI](https://img.shields.io/pypi/v/pypulseqpp.svg)](https://pypi.org/project/pypulseqpp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

pypulseqpp is the sequence core of [Pulserver](https://github.com/pulserver/pulserver),
as a package of its own. The contract is [PyPulseq](https://github.com/imr-framework/pypulseq)'s
API: a design script written against it runs here unchanged, with the same
functions, the same signatures and the same `.seq` output. Underneath, events
are compact compiled objects, `add_block` is one compiled call, and reading
and writing, in text and in binary, are C++.

## Scope

The package owns what is true about a sequence in isolation:

- reading and writing `.seq`, text and binary, with event deduplication;
- structural TR and base-block detection;
- hardware checks: gradient amplitude, slew and continuity, PNS, mechanical
  resonance;
- k-space trajectory and gradient-moment calculation;
- sequence-level operations such as FOV transformation and tiling.

It does not own segmentation, the scanner-side execution stream, protocol
contracts or consoles, which live in Pulserver, nor pulse, trajectory and
sampling design, which lives in its own package.

The runtime dependency is NumPy alone. Wheels ship the compiled core for
Linux, macOS and Windows, so no compiler is needed to install.

## Status

Scaffold. The core is being moved here from Pulserver; nothing is importable
beyond the version yet.

## Install

```bash
pip install pypulseqpp
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).
