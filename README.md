# pypulseqpp

Pulseq sequence design and analysis with a C++ core and a PyPulseq-compatible
Python interface.

[![Tests](https://github.com/pulserver/pypulseqpp/actions/workflows/test-ci.yml/badge.svg)](https://github.com/pulserver/pypulseqpp/actions/workflows/test-ci.yml)
[![codecov](https://codecov.io/gh/pulserver/pypulseqpp/branch/main/graph/badge.svg)](https://codecov.io/gh/pulserver/pypulseqpp)
[![PyPI](https://img.shields.io/pypi/v/pypulseqpp.svg)](https://pypi.org/project/pypulseqpp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/pulserver/pypulseqpp/blob/main/LICENSE)

pypulseqpp combines compiled event storage, block registration and analysis
with [PyPulseq](https://github.com/imr-framework/pypulseq) event factories.
It also provides RF and gradient design, sampling patterns, and composable
excitation, preparation and readout modules.

## Scope

- Pulseq text and binary reading/writing, signatures and event deduplication.
- Waveform expansion, k-space trajectories, sequence reports and structural
  repetition detection.
- Timing, gradient amplitude, slew and boundary-continuity checks, a
  mechanical-resonance check against forbidden gradient bands, and a PNS
  check under the SAFE or the rheobase-chronaxie model.
- Logical-frame FOV scaling, rotation and translation.
- Pulse, trajectory and sampling design, with reusable sequence modules.

This is an alpha package, not a complete replacement for every PyPulseq
feature. SAR assessment is not provided by the core checks; passing the checks
does not establish scanner or patient safety.

Scanner execution, protocol orchestration and reconstruction integration
belong to [Pulserver](https://github.com/pulserver/pulserver).

## Install

Requires Python 3.10 or later. Runtime dependencies are NumPy, SciPy and
PyPulseq; the native core is included in platform wheels.

```bash
pip install pypulseqpp
```

The optional viewer is a separate GPL-licensed package:

```bash
pip install 'pypulseqpp[plot]'
```

## Basic use

```python
import pypulseqpp as pp

system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
seq = pp.Sequence(system)
gx = pp.make_trapezoid("x", area=100, system=system)
seq.add_block(gx)
ok, errors = seq.check_timing()
seq.write("example.seq")
```

Sequence-module classes are available from `pypulseqpp.sequences`.
The [API reference](https://pulserver.github.io/pypulseqpp/latest/api/index.html)
groups sequence operations, event design, sampling, modules and checks.

## Development and documentation

See the [contribution guide](https://github.com/pulserver/pypulseqpp/blob/main/CONTRIBUTING.md)
for installation and checks. Build the local Markdown/Sphinx documentation with:

```bash
pip install -e '.[doc]'
bash scripts/build_docs.sh
```

The script compiles the checkout into `docs/build/site` and generates the pages
from that build.

Open `docs/build/html/index.html`. The API reference is populated; the user
guide, developer guide and examples sections are scaffolds.
