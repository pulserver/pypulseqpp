# Analyse a finished sequence

This guide covers what a sequence can be asked about once it has been built or
read from a file: its structure, its timing, the waveforms it plays and the
k-space trajectory it samples.

The sequence used here is one of the shipped implementations; the same calls
apply to a sequence assembled by hand or read with
{meth}`~pypulseqpp.Sequence.read`.

```python
>>> from pypulseqpp import sequences
>>> seq = sequences.gre2D_sequence(n_x=64, n_y=32, n_slices=1, tr=None)

```

## Structure and encoding

{meth}`~pypulseqpp.Sequence.test_report` states the event counts, the timing,
the resolution the k-space positions imply, and the gradient extremes reached
on each physical axis. {meth}`~pypulseqpp.Sequence.test_report_dict` returns
the same content as data.

```python
>>> report = seq.test_report_dict()
>>> report["num_blocks"], report["dimensions"]
(288, 2)
>>> report["event_count"]["adc"], report["event_count"]["rf"]
(32, 48)
>>> sorted(report["libraries"])[:4]
['ADC', 'Extension', 'Gradient', 'Label inc']

```

The encoding indices are recorded as labels, so the sampling order is
recoverable without re-deriving it.
{meth}`~pypulseqpp.Sequence.evaluate_labels` with `evolution="adc"` returns one
value per acquisition.

```python
>>> labels = seq.evaluate_labels(evolution="adc")
>>> sorted(labels)
['IMA', 'LIN', 'ONCE', 'SEG', 'SLC']
>>> int(labels["LIN"].min()), int(labels["LIN"].max()), labels["LIN"].size
(0, 31, 32)

```

## Timing

{meth}`~pypulseqpp.Sequence.check_timing` is a separate question from the
constraint checks: it establishes that every event time is addressable on the
raster its event is played on, and that the declared dead times are respected.

```python
>>> is_ok, report = seq.check_timing()
>>> is_ok
True

```

When it fails, `report` is a list of entries naming the block, the event, the
field and the error type. See
{doc}`../explanations/pulseq/timing-and-rasterization` for what each type
means.

## Waveforms and k-space

{meth}`~pypulseqpp.Sequence.waveforms_and_times` expands the block table into
one time-and-amplitude array per physical axis.
{meth}`~pypulseqpp.Sequence.calculate_kspace` integrates the gradient
waveforms, applies each block's rotation, and returns the k-space location of
every ADC sample in 1/m, together with the full trajectory and the pulse and
sample times.

```python
>>> k_adc, k_full, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
>>> k_adc.shape[0], k_adc.shape[1]
(3, 4096)
>>> t_excitation.size
48

```

There are more excitations than acquisitions because the sequence plays dummy
repetitions to approach a steady state before the first acquired line; they
excite without an ADC event, which is why `event_count` above reports 48 RF
events and 32 ADC events.

Amplitudes are reported in the file's own units; divide by `gamma` to convert
to physical ones.

```python
>>> waveforms = seq.waveforms_and_times()[0]
>>> gx_times, gx_amplitudes = waveforms[0]
>>> float(round(abs(gx_amplitudes).max() * 1e3 / seq.system.gamma, 1))   # mT/m
39.5

```

## Figures

{meth}`~pypulseqpp.Sequence.paper_plot` draws one repetition as a timing
diagram with the others underneath, which shows what varies between
repetitions. {func}`~pypulseqpp.plot.plot_kspace` draws the ADC sampling
locations, optionally coloured by shot or echo index.
{func}`~pypulseqpp.plot.plot_rf` draws a pulse's envelope beside the
magnetization profile it produces.

```python
>>> import matplotlib
>>> matplotlib.use("Agg")
>>> import pypulseqpp as pp
>>> drawn = seq.paper_plot()
>>> drawn.tr
1
>>> figure = pp.plot.plot_kspace(seq, color_by="shot", plot_now=False)
>>> figure.__class__.__name__
'Figure'

```

{meth}`~pypulseqpp.Sequence.plot` opens the sequence in SeqEyes, an optional
viewer installed with `pip install 'pypulseqpp[plot]'`.

## Next steps

* {doc}`checking-constraints` — gradient, stimulation and SAR checks.
* {doc}`../api/plotting` — the figure functions and their options.
