# pypulseqpp

[![Tests](https://github.com/pulserver/pypulseqpp/actions/workflows/test-ci.yml/badge.svg)](https://github.com/pulserver/pypulseqpp/actions/workflows/test-ci.yml)
[![Documentation](https://github.com/pulserver/pypulseqpp/actions/workflows/docs.yml/badge.svg)](https://pulserver.github.io/pypulseqpp/)
[![codecov](https://codecov.io/gh/pulserver/pypulseqpp/branch/main/graph/badge.svg)](https://codecov.io/gh/pulserver/pypulseqpp)
[![PyPI](https://img.shields.io/pypi/v/pypulseqpp.svg)](https://pypi.org/project/pypulseqpp/)
[![Python](https://img.shields.io/pypi/pyversions/pypulseqpp.svg)](https://pypi.org/project/pypulseqpp/)
[![Wheels](https://img.shields.io/badge/wheels-Linux%20x86--64%20%7C%20macOS%20arm64%2Fx86--64%20%7C%20Windows%20AMD64-2b76ad)](https://github.com/pulserver/pypulseqpp/actions/workflows/wheels.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-ffbd28.svg)](https://github.com/pulserver/pypulseqpp/blob/main/LICENSE)

<p align="center"><img src="https://raw.githubusercontent.com/pulserver/pypulseqpp/main/docs/_static/pypulseqpp-logo.svg" alt="pypulseqpp" width="620"></p>

pypulseqpp provides Pulseq sequence design and analysis through a
PyPulseq-compatible Python interface over a C++ core. It includes RF, gradient
and trajectory design, reusable sequence modules, complete sequence
applications, and timing, gradient, PNS, mechanical-resonance and SAR checks.
Unsupported PyPulseq features are not presented as available. Passing these
checks does not establish scanner or patient safety.

## Features

- Pulseq text and binary I/O, event storage, deduplication and repetition analysis.
- RF, gradient and non-Cartesian trajectory design.
- Composable excitation, preparation and readout `SequenceModule` classes.
- Waveform, ADC sampling-location and k-space analysis with publication figures.
- Timing, hardware-limit, PNS, mechanical-resonance and VOP-based SAR estimates.

<p align="center"><img src="https://raw.githubusercontent.com/pulserver/pypulseqpp/main/docs/_static/architecture.svg" alt="pypulseqpp architecture" width="900"></p>

## Quick start

```bash
pip install pypulseqpp
```

```python
from pypulseqpp.sequences import gre2D_sequence

seq = gre2D_sequence(n_x=64, n_y=64, n_slices=1, tr=None)
ok, errors = seq.check_timing()
seq.write("gre2d.seq")
```

## Documentation

| Section | Purpose |
| --- | --- |
| [User guide](https://pulserver.github.io/pypulseqpp/latest/user-guide/index.html) | Installation, support and project-use logistics. |
| [Explanations](https://pulserver.github.io/pypulseqpp/latest/explanations/index.html) | Pulseq representation, sequence design and constraint models. |
| [Examples](https://pulserver.github.io/pypulseqpp/latest/examples/index.html) | Executable sequence workflows and design studies. |
| [API reference](https://pulserver.github.io/pypulseqpp/latest/api/index.html) | Exact interfaces, units and defaults. |
| [Developer guide](https://pulserver.github.io/pypulseqpp/latest/developer-guide/index.html) | Development setup and contribution workflow. |
| [Source](https://github.com/pulserver/pypulseqpp) | Repository, issues and discussions. |
| [PDF manual](https://github.com/pulserver/pypulseqpp/releases/latest/download/pypulseqpp-docs.pdf) | Single-file documentation from the latest release. |

## Citation

pypulseqpp has no project publication. Cite the foundational formats and APIs
used in work built with it:

1. Layton KJ, Kroboth S, Jia F, et al. Pulseq: A rapid and hardware-independent
   pulse sequence prototyping framework. *Magnetic Resonance in Medicine*.
   2017;77(4):1544–1552. [doi:10.1002/mrm.26235](https://doi.org/10.1002/mrm.26235).
2. Ravi KS, Geethanath S, Vaughan JT. PyPulseq: A Python package for MRI pulse
   sequence design. *Journal of Open Source Software*. 2019;4(42):1725.
   [doi:10.21105/joss.01725](https://doi.org/10.21105/joss.01725).

## License

MIT, except bundled or vendored third-party components that retain their own
licences. See [License and third-party notices](https://pulserver.github.io/pypulseqpp/latest/misc/license.html).
