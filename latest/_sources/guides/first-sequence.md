# Write your first sequence

This guide builds a two-dimensional spoiled gradient-echo sequence from
individual Pulseq events and writes it as a `.seq` file. It assumes the
vocabulary of {doc}`../explanations/pulseq/events-and-blocks`.

## System limits

Every factory solves its waveforms against a set of system limits, and every
event time is quantized to the rasters they declare.

```python
>>> import numpy as np
>>> import pypulseqpp as pp
>>> system = pp.Opts(
...     max_grad=32.0,
...     grad_unit="mT/m",
...     max_slew=130.0,
...     slew_unit="T/m/s",
...     rf_dead_time=100e-6,
...     rf_ringdown_time=20e-6,
...     adc_dead_time=10e-6,
... )

```

## The excitation

{func}`~pypulseqpp.make_slr_pulse` designs the pulse; `return_gz=True` also
returns the slice-selection gradient and the rephaser that unwinds its second
half.

```python
>>> rf, gz, gz_reph = pp.make_slr_pulse(
...     np.deg2rad(12.0),
...     duration=3e-3,
...     slice_thickness=5e-3,
...     time_bw_product=4.0,
...     return_gz=True,
...     system=system,
...     use="excitation",
...     delay=system.rf_dead_time,
... )

```

## The readout

The acquisition window must be a whole number of gradient raster periods while
the dwell stays on the ADC raster. {func}`~pypulseqpp.calc_adc_timing` finds
the shortest dwell satisfying both, so the receiver bandwidth it returns is at
or below the one requested — see
{doc}`../explanations/pulseq/timing-and-rasterization` for why the sample count
participates in that bound.

```python
>>> fov, n_x, n_y = 220e-3, 128, 64
>>> dwell, readout_duration = pp.calc_adc_timing(
...     n_x,
...     1.0 / 250e3,
...     grad_raster_time=system.grad_raster_time,
...     adc_raster_time=system.adc_raster_time,
... )
>>> round(1e-3 / dwell)          # achieved receiver bandwidth, kHz
100

```

The readout gradient's flat top must traverse `n_x / fov` in k-space. The
prewinder covers the half of that moment preceding the echo, and is negative so
that the echo falls at the centre of the window.

```python
>>> gx = pp.make_trapezoid(
...     "x", flat_area=n_x / fov, flat_time=readout_duration, system=system
... )
>>> adc = pp.make_adc(
...     num_samples=n_x, dwell=dwell, delay=gx.rise_time, system=system
... )
>>> gx_pre = pp.make_trapezoid("x", area=-gx.area / 2, system=system)
>>> gy_pre = pp.make_trapezoid("y", area=n_y / (2 * fov), system=system)

```

## Spoiling

Four cycles of dephasing across one readout voxel leave no coherent transverse
magnetization for the next repetition to refocus. Combined with a quadratic RF
phase increment, this is the standard spoiled gradient echo; Zur et al.
(Magn Reson Med 1991, doi:10.1002/mrm.1910210210) report 117° as robust against
the residual coherences a linear increment leaves.

```python
>>> gx_spoil = pp.make_trapezoid("x", area=4.0 * n_x / fov, system=system)
>>> gz_spoil = pp.make_trapezoid("z", area=4.0 / 5e-3, system=system)

```

## Delays

The echo time runs from the pulse centre to the centre of the acquisition
window, and the repetition time from one excitation to the next. Each delay is
whatever remains once the events between those points are accounted for,
rounded up onto the block-duration raster.

```python
>>> te, tr = 8e-3, 20e-3
>>> te_fill = pp.ceil_to_raster(
...     te
...     - (pp.calc_duration(gz) - rf.center - rf.delay)
...     - pp.calc_duration(gz_reph, gx_pre, gy_pre)
...     - gx.rise_time
...     - gx.flat_time / 2,
...     system.block_duration_raster,
... )
>>> tr_fill = pp.ceil_to_raster(
...     tr
...     - pp.calc_duration(gz)
...     - pp.calc_duration(gz_reph, gx_pre, gy_pre)
...     - pp.calc_duration(gx)
...     - pp.calc_duration(gx_spoil, gz_spoil)
...     - te_fill,
...     system.block_duration_raster,
... )

```

## The scan loop

One repetition is six blocks. {func}`~pypulseqpp.scale_grad` returns the
phase-encode template at this line's step, and
{func}`~pypulseqpp.make_label` writes the `LIN` counter a reconstruction sorts
by. The RF phase offset is applied to the ADC as well, so the receiver
demodulates in the frame the pulse transmitted in.

```python
>>> seq = pp.Sequence(system=system)
>>> phase = increment = 0.0
>>> for line in range(n_y):
...     rf.phase_offset = phase
...     adc.phase_offset = phase
...     increment += np.deg2rad(117.0)
...     phase = (phase + increment) % (2 * np.pi)
...     step = (line - n_y // 2) / (n_y / 2)
...     _ = seq.add_block(rf, gz, pp.make_label(type="SET", label="LIN", value=line))
...     _ = seq.add_block(gz_reph, gx_pre, pp.scale_grad(gy_pre, step))
...     _ = seq.add_block(pp.make_delay(te_fill))
...     _ = seq.add_block(gx, adc)
...     _ = seq.add_block(gx_spoil, gz_spoil)
...     _ = seq.add_block(pp.make_delay(tr_fill))

```

## Check the timing and write the file

{meth}`~pypulseqpp.Sequence.check_timing` establishes that every event time is
addressable and that the transmit and receive dead times are respected. It also
records the sequence duration as the `TotalDuration` definition.

```python
>>> seq.check_timing()[0]
True
>>> seq.num_blocks
384

```

The definitions written beside the block table are what a reconstruction reads
the prescription from, so record anything downstream needs before writing.

```python
>>> seq.set_definition(key="FOV", value=[fov, fov, 5e-3])
>>> seq.set_definition(key="Name", value="gre_2d")
>>> import tempfile, pathlib
>>> with tempfile.TemporaryDirectory() as folder:
...     path = str(pathlib.Path(folder, "gre_2d.seq"))
...     _ = seq.write(path)
...     restored = pp.Sequence()
...     restored.read(path)
>>> restored.num_blocks
384
>>> restored.get_definition("FOV")
[0.22, 0.22, 0.005]

```

## Next steps

* {doc}`analysing-a-sequence` — the trajectory, the waveforms and the report.
* {doc}`checking-constraints` — gradient, stimulation and SAR checks.
* {doc}`custom-module` — the same sequence assembled from sequence modules.
