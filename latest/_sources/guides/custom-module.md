# Build a sequence from modules

This guide assembles an inversion-prepared gradient echo from excitation,
preparation and readout modules, and then turns the assembly into a
{class}`~pypulseqpp.sequences.SequenceApp`. The abstractions themselves are
described in {doc}`../explanations/design/index`; this page is the procedure.

## Construct the modules

Each module solves its own timing and gradient waveforms at construction. The
readout is given the excitation's pulse and gradients, which is what lets it
measure the echo time from the pulse centre and place the rephaser inside an
interval the repetition already has to wait out.

```python
>>> import numpy as np
>>> import pypulseqpp as pp
>>> from pypulseqpp import sequences
>>> system = pp.Opts(
...     max_grad=32.0, grad_unit="mT/m", max_slew=130.0, slew_unit="T/m/s",
...     adc_dead_time=10e-6,
... )
>>> fov, matrix = (220e-3, 220e-3), (128, 96)
>>> excitation = sequences.SpatialSelectiveExcitation(
...     system, flip_angle_deg=8.0, thickness_m=5e-3, duration_s=3e-3
... )
>>> readout = sequences.LineReadout2D(
...     system,
...     excitation.rf,
...     excitation.gz,
...     excitation.gz_reph,
...     fov=fov,
...     matrix=matrix,
...     te=None,
...     spoiling_cycles=4.0,
... )
>>> inversion = sequences.InversionPreparation(system, duration_s=10e-3)

```

A module reports what it achieved rather than what was requested.

```python
>>> f"TE {readout.echo_time * 1e3:.2f} ms, bandwidth {readout.bandwidth_hz * 1e-3:.0f} kHz"
'TE 2.86 ms, bandwidth 100 kHz'
>>> sorted(vars(readout.events))
['adc', 'gx', 'gx_pre', 'gx_spoil', 'gy_pre', 'gy_rew', 'gz', 'gz_reph', 'lobe', 'rf']

```

## Compose them at a prescribed interval

An inversion time is defined between two pulse centres. Each module's `center`
is the position of its own reference relative to its start, so the recovery
interval
subtracts the part of the inversion module after its pulse and the part of the
excitation module before its pulse.

```python
>>> ti = 900e-3
>>> recovery = pp.make_delay(
...     pp.round_to_raster(
...         ti - (inversion.duration - inversion.center) - excitation.center,
...         system.block_duration_raster,
...     )
... )

```

## Write the scan loop

{meth}`~pypulseqpp.sequences.SequenceModule.blocks` returns a module's blocks
in play order, so a module needing no per-view modification is added as it
stands. The readout's phase encode is scaled per line.

```python
>>> etl = 32
>>> shots = np.array_split(np.arange(matrix[1]), matrix[1] // etl)
>>> seq = pp.Sequence(system=system)
>>> phase = increment = 0.0
>>> for shot in shots:
...     for block in inversion.blocks:
...         _ = seq.add_block(*block)
...     _ = seq.add_block(recovery)
...     for line in shot:
...         increment += np.deg2rad(117.0)
...         phase = (phase + increment) % (2 * np.pi)
...         excitation.rf.phase_offset = phase
...         readout.adc.phase_offset = phase
...         step = (line - matrix[1] // 2) / (matrix[1] / 2)
...         _ = seq.add_block(excitation.rf, excitation.gz,
...                           pp.make_label(type="SET", label="LIN", value=int(line)))
...         _ = seq.add_block(readout.gx_pre,
...                           pp.scale_grad(readout.gy_pre, step),
...                           readout.gz_reph)
...         _ = seq.add_block(readout.gx, readout.adc)
...         _ = seq.add_block(readout.gx_spoil, pp.scale_grad(readout.gy_rew, step))
>>> seq.check_timing()[0]
True

```

The interval the design asked for is the one the block table plays.
{meth}`~pypulseqpp.Sequence.rf_times` with `compat=False` returns each pulse's
centre time together with its `use`.

```python
>>> pulses = seq.rf_times(compat=False)
>>> uses = np.asarray(pulses.use)
>>> inverted = pulses.t[uses == "inversion"][0]
>>> excited = pulses.t[uses == "excitation"][0]
>>> f"{(excited - inverted) * 1e3:.1f} ms"
'900.0 ms'

```

## Turn the assembly into an application

A {class}`~pypulseqpp.sequences.SequenceApp` separates the prescription, the
sampling order and one repetition. `MAX_GRAD` and `MAX_SLEW` have no default,
so every implementation states the gradient ceilings it is designed under.

```python
>>> class InversionRecoveryGre(sequences.SequenceApp):
...     """Segmented inversion-prepared 2D Cartesian gradient echo."""
...
...     NAME = "ir_gre_2d"
...     MAX_GRAD = 32.0
...     MAX_SLEW = 130.0
...
...     def init_sequence(self, fov: float = 220e-3, n_x: int = 128,
...                       n_y: int = 96, ti: float = 900e-3, etl: int = 32) -> None:
...         """Design the inversion, the excitation, the readout and the shots.
...
...         Parameters
...         ----------
...         fov : float, optional
...             Field of view along both encoded axes (m).
...         n_x, n_y : int, optional
...             Matrix size along the readout and the phase encode.
...         ti : float, optional
...             Inversion time (s), between the inversion and excitation centres.
...         etl : int, optional
...             Phase-encode lines read after each inversion.
...         """
...         self.matrix = (n_x, n_y)
...         self.inversion = sequences.InversionPreparation(self.system, duration_s=10e-3)
...         self.excitation = sequences.SpatialSelectiveExcitation(
...             self.system, 8.0, 5e-3, duration_s=3e-3
...         )
...         self.readout = sequences.LineReadout2D(
...             self.system, self.excitation.rf, self.excitation.gz,
...             self.excitation.gz_reph, fov=(fov, fov), matrix=(n_x, n_y),
...             te=None, spoiling_cycles=4.0,
...         )
...         self.recovery = pp.make_delay(pp.round_to_raster(
...             ti - (self.inversion.duration - self.inversion.center)
...             - self.excitation.center, self.system.block_duration_raster))
...         self.shots = np.array_split(np.arange(n_y), max(1, n_y // etl))
...
...     def loop(self) -> None:
...         """Play every shot: the inversion, the recovery interval, then its lines."""
...         phase = increment = 0.0
...         for index, shot in enumerate(self.shots):
...             for block in self.inversion.blocks:
...                 self.seq.add_block(*block)
...             self.seq.add_block(self.recovery)
...             for line in shot:
...                 increment += np.deg2rad(117.0)
...                 phase = (phase + increment) % (2 * np.pi)
...                 self.kernel(int(line), index, phase)
...
...     def kernel(self, line: int, shot: int, phase: float) -> None:
...         """One excitation reading phase-encode line ``line``."""
...         excitation, readout = self.excitation, self.readout
...         excitation.rf.phase_offset = phase
...         readout.adc.phase_offset = phase
...         step = (line - self.matrix[1] // 2) / (self.matrix[1] / 2)
...         self.seq.add_block(excitation.rf, excitation.gz,
...                            *self.labels(LIN=line, SEG=shot))
...         self.seq.add_block(readout.gx_pre,
...                            pp.scale_grad(readout.gy_pre, step), readout.gz_reph)
...         self.seq.add_block(readout.gx, readout.adc)
...         self.seq.add_block(readout.gx_spoil, pp.scale_grad(readout.gy_rew, step))

```

Constructing the application designs it; nothing is played until
{meth}`~pypulseqpp.sequences.SequenceApp.design`.

```python
>>> application = InversionRecoveryGre(ti=1100e-3, etl=24)
>>> designed = application.design()
>>> designed.check_timing()[0]
True
>>> application.protocol()
{'fov': 0.22, 'n_x': 128, 'n_y': 96, 'ti': 0.9, 'etl': 32}

```

## Next steps

* {doc}`command-line` — the command-line interface an application derives from
  its prescription.
* {doc}`../explanations/design/sequence-application` — why the three concerns
  are separated.
