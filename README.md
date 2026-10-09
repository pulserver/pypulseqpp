[![Tests](https://github.com/pulserver/pypulseqpp/actions/workflows/test-ci.yml/badge.svg)](https://github.com/pulserver/pypulseqpp/actions/workflows/test-ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pypulseqpp.svg)](https://pypi.org/project/pypulseqpp/)
[![Python](https://img.shields.io/pypi/pyversions/pypulseqpp.svg)](https://pypi.org/project/pypulseqpp/)
[![Docs: stable](https://img.shields.io/badge/docs-stable-2b76ad)](https://pulserver.github.io/pypulseqpp/stable/)
[![License: MIT](https://img.shields.io/badge/license-MIT-ffbd28.svg)](https://github.com/pulserver/pypulseqpp/blob/main/LICENSE)

<p align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pulserver/pypulseqpp/main/docs/_static/pypulseqpp-logo-dark.svg">
  <img src="https://raw.githubusercontent.com/pulserver/pypulseqpp/main/docs/_static/pypulseqpp-logo.svg" alt="pypulseqpp" width="620">
</picture></p>

pypulseqpp is a fast drop-in replacement for PyPulseq, written in Python with
a C++ backend. You write MR pulse sequences exactly as you write them in
PyPulseq, and get back a sequence that is fast to build, check and write,
however long it is. Change one line of your script,

```python
import pypulseqpp as pp  # in place of: import pypulseq as pp
```

and it runs: the event factories, `Opts` and the `Sequence` methods keep
PyPulseq's names, arguments and units. Underneath, the sequence is stored by a
C++ core, so a scan of a million blocks is built, deduplicated and written in
seconds.

You write the sequence; pypulseqpp does the rest. It stores and deduplicates
your events, writes Pulseq text and binary files that any Pulseq interpreter
plays, and checks the result against your scanner: timing, gradient amplitude
and slew rate, peripheral nerve stimulation, mechanical resonances, acoustic
noise and SAR.

<p align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pulserver/pypulseqpp/main/docs/_static/architecture-dark.svg">
  <img src="https://raw.githubusercontent.com/pulserver/pypulseqpp/main/docs/_static/architecture.svg" alt="pypulseqpp architecture" width="900">
</picture></p>

## How it works

A Pulseq sequence is a list of blocks played back to back, and each block
holds at most one event per channel: an RF pulse, a gradient on each axis, an
ADC window, a delay. You build it in five steps:

1. Describe your scanner: field strength, gradient amplitude and slew-rate
   limits, raster times and dead times. This is the `system` object, a
   `pp.Opts`.
2. Design the events with the factories (`pp.make_sinc_pulse`,
   `pp.make_trapezoid`, `pp.make_adc`...). Each one is solved against
   `system`, so a gradient you ask for by its area comes back as the shortest
   trapezoid your hardware plays.
3. Add the events to the sequence block by block, in a loop over the lines of
   k-space.
4. Check it. `seq.check_timing()` checks every block against the rasters and
   dead times; the gradient, PNS, mechanical-resonance and SAR checks look at
   the whole scan.
5. Write the `.seq` file, and play it on the scanner through your vendor's
   Pulseq interpreter.

Here is a 2D gradient echo, written the PyPulseq way:

```python
import numpy as np

import pypulseqpp as pp

system = pp.Opts(max_grad=32, grad_unit="mT/m", max_slew=130, slew_unit="T/m/s")
fov, n = 256e-3, 64

rf, gz, gz_reph = pp.make_sinc_pulse(
    flip_angle=np.deg2rad(15),
    duration=2e-3,
    slice_thickness=5e-3,
    time_bw_product=4,
    system=system,
    return_gz=True,
)
gx = pp.make_trapezoid("x", flat_area=n / fov, flat_time=3.2e-3, system=system)
adc = pp.make_adc(n, duration=gx.flat_time, delay=gx.rise_time, system=system)
gx_pre = pp.make_trapezoid("x", area=-gx.area / 2, duration=1e-3, system=system)

seq = pp.Sequence(system)
for line in range(n):
    gy_pre = pp.make_trapezoid(
        "y", area=(line - n / 2) / fov, duration=1e-3, system=system
    )
    seq.add_block(rf, gz)
    seq.add_block(gx_pre, gy_pre, gz_reph)
    seq.add_block(gx, adc)
    seq.add_block(pp.make_delay(10e-3))

ok, errors = seq.check_timing()
seq.write("gre.seq")
```

## Sequences as functions

A script designs one protocol. Put the same code in a function of `system`
and keyword arguments, and it designs any protocol you ask for:

```python
def gre(system, *, fov=256e-3, n=64, flip_angle_deg=15.0) -> pp.Sequence:
    ...
    return seq
```

- The keyword arguments are your protocol. Document each in a NumPy-style
  `Parameters` section, with its unit, and `pp.cli.run(gre)` turns them into a
  command line.
- Return the sequence, or a list of sequences when the scan has prescans; the
  main sequence goes last.
- This is the shape [pulserver](https://github.com/pulserver/pulserver)
  expects, so the same function becomes a sequence the operator edits in the
  scanner UI.

pypulseqpp ships 28 sequences written this way (gradient echo, spin echo, FSE,
MPRAGE, bSSFP, EPI, ZTE, with Cartesian, radial, spiral and PROPELLER
readouts), each one a module you can call or run from the command line:

```python
from pypulseqpp import sequences

seq = sequences.gre2D_sequence(n_x=64, n_y=64)
```

```bash
python -m pypulseqpp.sequences.sequence.gre2D_sequence --help
```

They are built from sequence modules, ready-made excitations, preparations
and readouts that you can reuse in your own functions.

The checks compute estimates. Passing them does not establish scanner or
patient safety.

## Install

```bash
pip install pypulseqpp
```

Wheels are published for Linux x86-64, macOS (Apple silicon and x86-64) and
Windows AMD64, Python 3.10 to 3.14.

## Learn more

- [Course](https://pulserver.github.io/pypulseqpp/stable/examples/index.html):
  ten lessons from a pulse-acquire experiment to your own sequence module.
- [Sequence catalogue](https://pulserver.github.io/pypulseqpp/stable/sequences.html):
  every shipped sequence, with its protocol and diagram.
- [From a PyPulseq script](https://pulserver.github.io/pypulseqpp/stable/user-guide/from-pypulseq.html):
  your script, and the same script as a sequence function.
- [Documentation](https://pulserver.github.io/pypulseqpp/stable/), for every
  option and API detail.

## How to cite

pypulseqpp has no publication of its own. If you use it, please cite Pulseq
and PyPulseq:

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

MIT, except bundled or vendored third-party components, which keep their own
licences. See [License and third-party notices](https://pulserver.github.io/pypulseqpp/stable/misc/license.html).
