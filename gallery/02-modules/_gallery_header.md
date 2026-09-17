# Sequence modules

Composing a sequence from reusable block layouts instead of individual events.

The first page assembles an inversion-prepared gradient echo from an
excitation, a preparation and a readout module, and simulates the slice profile
of the pulse it excites with. The second turns that assembly into a
{class}`~pypulseqpp.sequences.SequenceApp`, which carries the sampling order,
the scan loop and the command-line interface the shipped sequences are built
on.
