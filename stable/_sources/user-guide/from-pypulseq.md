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

## The script as a sequence application

A {class}`~pypulseqpp.sequences.SequenceApp` is the same script split in
three methods, so that a prescription can be checked without playing the scan:

| PyPulseq script | `SequenceApp` |
| --- | --- |
| `system = pp.Opts(...)` | `MAX_GRAD` (mT/m) and `MAX_SLEW` (T/m/s), which cap the `system` passed in |
| parameters at the top | arguments of `init_sequence`, in SI units |
| events built before the loop | `init_sequence`, stored on `self` |
| `seq = pp.Sequence(system)` | `self.seq`, created by `design()` |
| the loop | `loop` |
| the loop body | `kernel`, one repetition |
| `seq.set_definition(...)` | `finalize` |

`write_gre.py`, at a 64 matrix:

```pycon
>>> import numpy as np
>>> import pypulseqpp as pp
>>> from pypulseqpp import sequences
>>> class Gre(sequences.SequenceApp):
...     MAX_GRAD, MAX_SLEW = 28.0, 150.0
...     def init_sequence(self, fov: float = 256e-3, n: int = 64,
...                       te: float = 4.3e-3, tr: float = 10e-3):
...         sys, dk = self.system, 1 / fov
...         self.rf, self.gz, _ = pp.make_sinc_pulse(
...             flip_angle=np.deg2rad(10), duration=3e-3, slice_thickness=3e-3,
...             apodization=0.5, time_bw_product=4, system=sys, return_gz=True)
...         self.gx = pp.make_trapezoid("x", flat_area=n * dk, flat_time=3.2e-3, system=sys)
...         self.adc = pp.make_adc(n, duration=self.gx.flat_time,
...                                delay=self.gx.rise_time, system=sys)
...         self.gx_pre = pp.make_trapezoid("x", area=-self.gx.area / 2, duration=1e-3, system=sys)
...         self.gz_reph = pp.make_trapezoid("z", area=-self.gz.area / 2, duration=1e-3, system=sys)
...         self.gx_spoil = pp.make_trapezoid("x", area=2 * n * dk, system=sys)
...         self.gz_spoil = pp.make_trapezoid("z", area=4 / 3e-3, system=sys)
...         self.phase_areas = (np.arange(n) - n / 2) * dk
...         self.delay_te = pp.round_to_raster(
...             te - pp.calc_duration(self.gx_pre) - self.gz.fall_time
...             - self.gz.flat_time / 2 - pp.calc_duration(self.gx) / 2, sys.grad_raster_time)
...         self.delay_tr = pp.round_to_raster(
...             tr - pp.calc_duration(self.gz) - pp.calc_duration(self.gx_pre)
...             - pp.calc_duration(self.gx) - self.delay_te, sys.grad_raster_time)
...         self.n = n
...     def loop(self):
...         phase = 0.0
...         for line in range(self.n):
...             phase += np.deg2rad(117) * line
...             self.kernel(line, phase % (2 * np.pi))
...     def kernel(self, line, phase):
...         self.rf.phase_offset = self.adc.phase_offset = phase
...         gy_pre = pp.make_trapezoid("y", area=self.phase_areas[line], duration=1e-3,
...                                    system=self.system)
...         self.seq.add_block(self.rf, self.gz)
...         self.seq.add_block(self.gx_pre, gy_pre, self.gz_reph)
...         self.seq.add_block(pp.make_delay(self.delay_te))
...         self.seq.add_block(self.gx, self.adc, *self.labels(LIN=line))
...         self.seq.add_block(pp.make_delay(self.delay_tr), self.gx_spoil,
...                            pp.scale_grad(gy_pre, -1), self.gz_spoil)
>>> seq = Gre(te=5e-3).design()
>>> seq.check_timing()[0]
True

```

`self.labels(LIN=line)` sets the line counter a reconstruction places the
readout by; `pp.make_label` does the same as in PyPulseq.
