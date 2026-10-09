# Sequence-analysis computations

What the analysis functions of {class}`pypulseqpp.Sequence` and
{class}`~pypulseqpp.TransformFOV` compute, in which frame and units, and what a
caller can rely on. The concepts are in
{doc}`../../explanations/sequence-analysis`.

```{figure} ../../generated/figures/kspace_reset_and_unbroken.png
One gradient, two integrals: the unbroken area that shift phases use and the
trajectory that echo anchoring uses.
```

## Waveforms and times

{meth}`~pypulseqpp.Sequence.waveforms_and_times` returns gradient channels as
`(2, n)` arrays of time (s) and amplitude (Hz/m), and an optional complex RF
channel in Hz. Each block's rotation is applied, so the channels are the
logical axes. A trapezoid is its corner points. A time range keeps times
measured from the start of the scan; a block range restarts them at its first
block, and the two are not given together.

With `compat=True` the five values are upstream's: `wave_data`,
`tfp_excitation`, `tfp_refocusing`, `t_adc`, `fp_adc`. With `compat=False` a
named result carries all seven Pulseq RF uses, per-sample ADC phases and the
1-based block each pulse and sample belongs to; the five values cannot hold
the uses other than excitation and refocusing.

## k-space

{meth}`~pypulseqpp.Sequence.calculate_kspace` returns, in order, the k-space
location of each ADC sample `(3, n)` in 1/m, the full trajectory sampled
through ramps and at event times, the centres of the excitation and refocusing
pulses and the ADC sample times, all in seconds from the start of the selected
range. `trajectory_delay` is a per-axis timing correction in seconds, positive
advancing the gradient; `gradient_offset` is a background gradient in Hz/m.
`calculate_kspacePP` is the same calculation under upstream's second spelling.

The trajectory restarts at the centre of an excitation pulse, or of a pulse
with no use recorded, and is inverted at the centre of a refocusing pulse.
{meth}`~pypulseqpp.Sequence.adc_kspace` returns the first result alone, with
the trajectory between the samples left unbuilt; it integrates identically and
accepts the same options. With `readouts=(first, stop)`, two 0-based indices in
the order {meth}`~pypulseqpp.Sequence.adc_echoes` lists them, it returns the
samples of those readouts, each where the whole sequence puts it. The
integration starts at the last block before readout `first` that plays an
excitation, or a pulse with no use recorded, and does not acquire, so a
trajectory delay that moves an earlier block's gradient past that pulse's
centre is not seen. `readouts` is not combined with `block_range`.

{meth}`~pypulseqpp.Sequence.adc_echoes` returns one entry per block that
acquires: the 1-based `block`, `num_samples`, `first_sample` (the column in
`adc_kspace`), `moving` (whether the readout's k-space spans more than 1e-6 of
its widest span along x, y and z) and `echo`, the first and last 0-based
sample no further from the centre of k-space, over the moving axes, than the
nearest sample plus 1 % of the larger k step beside it; -1 for a readout that
does not move or has fewer than two samples. `start`, `path`, `paths`,
`rotation` and `rotations` give each readout's k-space as its first sample plus
one of the distinct paths from there, turned by the block's rotation. They are
None when an RF pulse that excites or refocuses, or has no use recorded, plays
in a block that acquires, or when more than one readout in 16 has a path of its
own.

Two integrals of the gradients are kept distinct. The unbroken integral is
the cumulative gradient area without RF resets and is the reference of the
RF and ADC shift phases. The excitation- and refocusing-aware k-space is the
one that anchors echoes. {class}`~pypulseqpp.TransformFOV` stores them as
`swept_k` and `block_k_origin`, both in 1/m in the frame of the translation,
and preserves both across consecutive ranges.

## Repetition

{meth}`~pypulseqpp.Sequence.repetition` returns `(size, start)`: blocks per
repetition, and the 1-based first block, always 1. The size is, in order of
preference:

1. the shortest period of the block-definition stream from the first block
   that every later block repeats, the last copy possibly cut short;
2. the shortest period dividing the table over which blocks match in duration
   and in the channels they play;
3. the whole sequence.

A slice acquired with its own preparation and dummy shots is one repetition,
and a block played once makes the whole sequence one. A sequence with no
blocks has size zero. A `TRSize` definition shorter than the sequence, dividing
it and repeated by the blocks is taken instead, so a longer hyper-TR can be
declared; one the blocks contradict is ignored. Nothing is written into the
sequence, and the answer is recomputed after a native mutation (a block added,
a block retimed, duplicates collapsed).

Because the stream is of definitions, shots that differ only in amplitudes,
rotations, shifted gradient waveforms or phase-encode steps repeat, and a
phase-encode table is one definition at many amplitudes. Labelling a pulse
splits the definition it shared.

## RF power

{func}`~pypulseqpp.calc_rf_power` returns one event's energy (Hz² s), peak
power (Hz²) and RMS amplitude (Hz), as MATLAB Pulseq's `calcRfPower`, resampled
at `dt`; the energy does not depend on `dt`.
{meth}`~pypulseqpp.Sequence.calc_rf_power` returns mean power, peak power, RMS
and energy over all blocks or a 1-based inclusive `block_range`; with
`window_duration` the energy, mean power and RMS are the largest over runs of
whole blocks no longer than the window, each divided by it. A dynamic pTx
pulse is read as the root-sum-square of its channels, so channel powers add and
never cancel. The values are relative: divide the RMS by gamma for tesla and
the powers by gamma squared. SAR is {func}`~pypulseqpp.safety.check_sar`.

## Field-of-view transform

{class}`~pypulseqpp.TransformFOV` takes at least one of `rotation`,
`translation`, `scale` or a 4-by-4 `transform`, which excludes the other two;
its translation is stated in the output frame and is converted to the frame of
`translation` with the transpose of its rotation. {meth}`~pypulseqpp.TransformFOV.apply_to_sequence`
applies scaling, then translation, then rotation, to a copy unless
`in_place=True`, over `block_range` (1-based, inclusive) or `time_range`,
not both.

| Operation | Frame and units |
| --- | --- |
| scale | multipliers on the channel axes; field of view scales inversely, zero disables encoding on an axis |
| translation | metres, along the channel axes; along the logical axes with `through_rotation`, where a block with rotation $R$ is moved by the gradients it plays, $R g$ |
| rotation | 3-by-3 matrix or SciPy rotation, composed after the rotation each block holds; stored as a scalar-first quaternion |

Translation moves RF and ADC by a phase from the unbroken gradient area: a
steady gradient under a pulse or readout costs two numbers, a moving one a
phase shape, and a readout is referenced to its window centre. Repeated
readouts share one reference and register one modulation shape. A rotation
given with a translation does not turn it. RF phase shapes store cycles; ADC
modulation and phase offsets are in radians.

A matrix of determinant -1 is played as `rotation @ diag(1, 1, -1)` after
negating the gradient on channel axis z, each block's own rotation conjugated
by that reflection (`reflected_axis` is 2). A rotation is always an
annotation; `use_rotation_extension=False`, which would bake it into new
waveforms, is refused.

`NOSCL`, `NOROT` and `NOPOS` labels exempt blocks from scaling, rotation and
translation, evaluated from the selected range's start without inheriting
earlier values. A nonzero translation still integrates exempt blocks, because
where k stands depends on everything played before it. A nonzero translation
updates `swept_k` and `block_k_origin`; reuse them only for consecutive
ranges, initialised to the state entering the first selected block. A block
range does not integrate the blocks before it.
{meth}`~pypulseqpp.TransformFOV.trajectories` returns `(block, (3, n))` ADC
trajectories on the channel axes from `block_k_origin`, without updating it and
without applying rotation extensions.

## Figures

{func}`~pypulseqpp.plot.plot` needs the optional SeqEyes viewer
(`pip install 'pypulseqpp[plot]'`) and is also `Sequence.plot`.
{func}`~pypulseqpp.plot.paper_plot` draws one repetition over the others, up to
`max_underlays`, as `Sequence.paper_plot`.
{func}`~pypulseqpp.plot.plot_kspace` draws the ADC locations in 1/m after each
block's rotation.
