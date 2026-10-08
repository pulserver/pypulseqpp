# From a PyPulseq script

A PyPulseq script runs with its import changed:

```python
import pypulseqpp as pp  # in place of: import pypulseq as pp
```

The event factories (`make_sinc_pulse`, `make_trapezoid`, `make_adc`,
`make_delay`, `make_label`), `calc_duration`, `Opts` and the `Sequence` methods
(`add_block`, `check_timing`, `test_report`, `calculate_kspace`, `plot`,
`write`, `read`) keep PyPulseq's signatures and units. The API reference lists
what is implemented.

## The script as a sequence function

A sequence function is the script with the system and the parameters as
arguments. The first parameter is `system`, the keyword parameters after it are
the protocol, and the function returns the sequence, or a list of sequences when
the scan has prescans, with the main sequence last:

| PyPulseq script | Sequence function |
| --- | --- |
| `system = pp.Opts(...)` | the `system` argument; `pp.cap_system` lowers its limits to those the design is made for, `max_grad` in mT/m and `max_slew` in T/m/s |
| parameters at the top | keyword parameters after `system`, each documented with its unit in a NumPy-style `Parameters` section |
| events built before the loop, `seq = pp.Sequence(system)`, the loop | the body of the function |
| `pp.make_label` before each block | `sequences.Labels`, which writes the label events a block changes |
| `seq.set_definition(...)` | unchanged, before the function returns `seq` |
| `seq.write("gre.seq")` | `sequences.write`, or `cli.run`, which also derives the command-line options |

`write_gre.py`, at a 64 matrix:

```pycon
>>> import numpy as np
>>> import pypulseqpp as pp
>>> from pypulseqpp import cli, sequences
>>> def gre(system, *, fov: float = 256e-3, n: int = 64, te: float = 4.3e-3,
...         tr: float = 10e-3) -> pp.Sequence:
...     """Gradient echo with RF spoiling.
...
...     Parameters
...     ----------
...     fov : float, default=0.256
...         Field of view (m).
...     n : int, default=64
...         Matrix size.
...     te : float, default=0.0043
...         Echo time (s).
...     tr : float, default=0.01
...         Repetition time (s).
...     """
...     system = pp.cap_system(system, max_grad=28.0, max_slew=150.0)
...     dk = 1 / fov
...     rf, gz, _ = pp.make_sinc_pulse(
...         flip_angle=np.deg2rad(10), duration=3e-3, slice_thickness=3e-3,
...         apodization=0.5, time_bw_product=4, system=system, return_gz=True)
...     gx = pp.make_trapezoid("x", flat_area=n * dk, flat_time=3.2e-3, system=system)
...     adc = pp.make_adc(n, duration=gx.flat_time, delay=gx.rise_time, system=system)
...     gx_pre = pp.make_trapezoid("x", area=-gx.area / 2, duration=1e-3, system=system)
...     gz_reph = pp.make_trapezoid("z", area=-gz.area / 2, duration=1e-3, system=system)
...     gx_spoil = pp.make_trapezoid("x", area=2 * n * dk, system=system)
...     gz_spoil = pp.make_trapezoid("z", area=4 / 3e-3, system=system)
...     delay_te = pp.round_to_raster(
...         te - pp.calc_duration(gx_pre) - gz.fall_time
...         - gz.flat_time / 2 - pp.calc_duration(gx) / 2, system.grad_raster_time)
...     delay_tr = pp.round_to_raster(
...         tr - pp.calc_duration(gz) - pp.calc_duration(gx_pre)
...         - pp.calc_duration(gx) - delay_te, system.grad_raster_time)
...     seq, labels, phase = pp.Sequence(system), sequences.Labels(), 0.0
...     for line in range(n):
...         phase += np.deg2rad(117) * line
...         rf.phase_offset = adc.phase_offset = phase % (2 * np.pi)
...         gy_pre = pp.make_trapezoid("y", area=(line - n / 2) * dk, duration=1e-3,
...                                    system=system)
...         seq.add_block(rf, gz)
...         seq.add_block(gx_pre, gy_pre, gz_reph)
...         seq.add_block(pp.make_delay(delay_te))
...         seq.add_block(gx, adc, *labels(LIN=line))
...         seq.add_block(pp.make_delay(delay_tr), gx_spoil,
...                       pp.scale_grad(gy_pre, -1), gz_spoil)
...     return seq
>>> seq = gre(pp.Opts(), te=5e-3)
>>> seq.check_timing()[0]
True

```

`labels(LIN=line)` returns the event that sets the line counter a
reconstruction places the readout by, and nothing for a value that has not
changed; `pp.make_label` does the same as in PyPulseq.

{func}`~pypulseqpp.sequences.write` writes the sequence, or a list of sequences
as linked files. {func}`pypulseqpp.cli.run` derives one command-line option from
each keyword parameter, with the help text from the `Parameters` section,
builds `system` from `--max-grad-mtm` and `--max-slew-tm-s`, and writes the
result in the same way:

```pycon
>>> sequences.write("gre.seq", seq)
['gre.seq']
>>> cli.run(gre, ["-o", "gre_32.seq", "--n", "32", "--te", "0.005"])
Wrote sequence: gre_32.seq
0

```

As the last lines of the script, the same call takes the options from the
command line:

```python
if __name__ == "__main__":
    raise SystemExit(cli.run(gre, sys.argv[1:], default_output="gre.seq"))
```
