# Plotting

```{eval-rst}
.. currentmodule:: pypulseqpp.plot
```

Every figure is a function in {mod}`pypulseqpp.plot` that takes what it draws.
{func}`plot` and {func}`paper_plot` call the {class}`pypulseqpp.Sequence`
methods of the same name, which remain for scripts written against upstream
PyPulseq; the other figures are functions only.

{func}`plot` opens the sequence in SeqEyes, an optional viewer installed with
`pip install 'pypulseqpp[plot]'`. {func}`paper_plot` draws one repetition as a
publication diagram over the others. {func}`plot_kspace` draws where the ADC
samples land in physical-axis k-space, optionally coloured by shot or echo.
{func}`plot_rf` draws a pulse's envelope beside the magnetisation profile it
produces, against position or off-resonance, or over a plane.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   plot
   paper_plot
   plot_kspace
   plot_rf
```
