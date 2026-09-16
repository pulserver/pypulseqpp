# Plotting

`pypulseqpp.plot`: each figure is a function of the object it draws.
{func}`plot` and {func}`paper_plot` are also {class}`pypulseqpp.Sequence`
methods, for scripts written against upstream PyPulseq; the other figures are
functions only.

```{eval-rst}
.. currentmodule:: pypulseqpp.plot
```

## Sequence views

{func}`plot` opens the sequence in SeqEyes, an optional viewer installed with
`pip install 'pypulseqpp[plot]'`. {func}`paper_plot` draws one repetition as a
publication diagram, overlaid on the others.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   plot
   paper_plot
```

## k-space and RF profiles

{func}`plot_kspace` draws the ADC sampling locations in physical-axis k-space,
in 1/m, optionally coloured by shot or echo index. {func}`plot_rf` draws an RF
pulse's envelope beside the magnetisation profile it produces, against position
or off-resonance, or over a plane.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   plot_kspace
   plot_rf
```
