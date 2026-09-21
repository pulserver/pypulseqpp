# Plotting

Each figure is a function of the object it draws. {func}`plot` and
{func}`paper_plot` are also {class}`pypulseqpp.Sequence` methods, for scripts
written against upstream PyPulseq; the other figures are functions only.

```{eval-rst}
.. currentmodule:: pypulseqpp.plot
```

## Sequence views

{func}`plot` opens the sequence in SeqEyes, an optional viewer installed with
`pip install 'pypulseqpp[plot]'`. {func}`paper_plot` draws one repetition as a
publication diagram, overlaid on the others.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.plot.plot` | Open ``seq`` in the SeqEyes viewer; see :meth:`pypulseqpp.Sequence.plot`. |
| {obj}`~pypulseqpp.plot.paper_plot` | Draw a publication diagram of ``seq``; see :meth:`pypulseqpp.Sequence.paper_plot`. |

## k-space and RF profiles

{func}`plot_kspace` draws the ADC sampling locations in physical-axis k-space,
in 1/m, optionally coloured by shot or echo index. {func}`plot_rf` draws an RF
pulse's envelope beside the magnetisation profile it produces, against position
or off-resonance, or over a plane.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.plot.plot_kspace` | Plot the ADC sampling locations in k-space. |
| {obj}`~pypulseqpp.plot.plot_rf` | Draw an RF pulse's ``|B1|`` envelope beside the profile it produces. |
