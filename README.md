# pypulseqpp

[![Tests](https://github.com/pulserver/pypulseqpp/actions/workflows/test-ci.yml/badge.svg)](https://github.com/pulserver/pypulseqpp/actions/workflows/test-ci.yml)
[![codecov](https://codecov.io/gh/pulserver/pypulseqpp/branch/main/graph/badge.svg)](https://codecov.io/gh/pulserver/pypulseqpp)
[![CodeFactor](https://www.codefactor.io/repository/github/pulserver/pypulseqpp/badge)](https://www.codefactor.io/repository/github/pulserver/pypulseqpp)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docs: stable](https://img.shields.io/badge/docs-stable-2b76ad)](https://pulserver.github.io/pypulseqpp/stable/)
[![Docs: latest](https://img.shields.io/badge/docs-latest-6b7684)](https://pulserver.github.io/pypulseqpp/latest/)

[![PyPI](https://img.shields.io/pypi/v/pypulseqpp.svg)](https://pypi.org/project/pypulseqpp/)
[![Downloads](https://img.shields.io/pypi/dm/pypulseqpp.svg)](https://pypistats.org/packages/pypulseqpp)
[![Python](https://img.shields.io/pypi/pyversions/pypulseqpp.svg)](https://pypi.org/project/pypulseqpp/)
[![Wheels](https://img.shields.io/badge/wheels-Linux%20x86--64%20%7C%20macOS%20arm64%2Fx86--64%20%7C%20Windows%20AMD64-2b76ad)](https://github.com/pulserver/pypulseqpp/actions/workflows/wheels.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-ffbd28.svg)](https://github.com/pulserver/pypulseqpp/blob/main/LICENSE)
[![Source](https://img.shields.io/badge/source-GitHub-181717?logo=github)](https://github.com/pulserver/pypulseqpp)
[![Stars](https://img.shields.io/github/stars/pulserver/pypulseqpp?style=flat&logo=github&color=ffbd28)](https://github.com/pulserver/pypulseqpp/stargazers)
[![FAIR checklist badge](https://fairsoftwarechecklist.net/badge.svg)](https://fairsoftwarechecklist.net/v0.2?f=31&a=32113&i=32322&r=133)

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

The [user guide](https://pulserver.github.io/pypulseqpp/latest/user-guide/index.html)
covers installation and support, and lists what each documentation section
holds. The
[examples](https://pulserver.github.io/pypulseqpp/latest/examples/index.html)
are a course in sequence design, from a pulse-acquire experiment to echo planar
and non-Cartesian acquisitions, followed by a page for each shipped sequence.
Every version of the documentation is published at
<https://pulserver.github.io/pypulseqpp/>.

## Citation

pypulseqpp has no project publication. Cite the foundational formats and APIs
used in work built with it:

```bibtex
@article{layton2017pulseq,
  title   = {Pulseq: a rapid and hardware-independent pulse sequence prototyping framework},
  author  = {Layton, Kelvin J and Kroboth, Stefan and Jia, Feng and Littin, Sebastian
             and Yu, Huijun and Leupold, Jochen and Nielsen, Jon-Fredrik
             and St{\"o}cker, Tony and Zaitsev, Maxim},
  journal = {Magnetic Resonance in Medicine},
  volume  = {77},
  number  = {4},
  pages   = {1544--1552},
  year    = {2017},
  doi     = {10.1002/mrm.26235}
}

@article{ravi2019pypulseq,
  title   = {PyPulseq: A Python Package for MRI Pulse Sequence Design},
  author  = {Ravi, Keerthi Sravan and Geethanath, Sairam and Vaughan, John Thomas},
  journal = {Journal of Open Source Software},
  volume  = {4},
  number  = {42},
  pages   = {1725},
  year    = {2019},
  doi     = {10.21105/joss.01725}
}
```

## License

MIT, except bundled or vendored third-party components that retain their own
licences. See [License and third-party notices](https://pulserver.github.io/pypulseqpp/latest/misc/license.html).
