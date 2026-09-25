# Plotting

Figures of a sequence, of its ADC sampling locations in k-space and of its RF
pulses. Each figure is a function of the object it draws. {func}`plot` and
{func}`paper_plot` are also {class}`pypulseqpp.Sequence` methods, for scripts
written against upstream PyPulseq; the other figures are functions only.

```{eval-rst}
.. currentmodule:: pypulseqpp.plot
```

## Sequence views

{func}`plot` requires SeqEyes, an optional viewer installed with
`pip install 'pypulseqpp[plot]'`.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.plot.plot` | Sequence, arguments of `Sequence.plot` | SeqEyes viewer process | Interactive view of the sequence. |
| {obj}`~pypulseqpp.plot.paper_plot` | Sequence, arguments of `Sequence.paper_plot` | Matplotlib `Axes` | Publication diagram of one repetition over the others. |

## Colormaps

The colormaps the figures on this page are drawn with, for figures drawn beside
them.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.plot.SAMPLING` | Normalised value in [0, 1] | RGBA colour | Acquisition order, first to last. |
| {obj}`~pypulseqpp.plot.MAGNITUDE` | Normalised value in [0, 1] | RGBA colour, transparent at zero | Magnitude from zero. |
| {obj}`~pypulseqpp.plot.SIGNED` | Normalised value in [0, 1] | RGBA colour, transparent at 0.5 | Signed quantity about zero. |

## k-space and RF profiles

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.plot.plot_kspace` | Sequence, range, projection plane, `color_by` | Matplotlib `Figure` | ADC sampling locations in k-space (1/m), after each block's rotation. |
| {obj}`~pypulseqpp.plot.plot_rf` | Sequence, module or RF event; pulse; profile axis or plane | Matplotlib `Figure` | RF envelope beside its magnetisation profile. |
