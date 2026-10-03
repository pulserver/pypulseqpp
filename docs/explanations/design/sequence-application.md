# Sequence functions

```{admonition} TL;DR
:class: tldr

- A complete sequence is a function `sequence(system, **protocol)` of the system
  limits and a protocol, the keyword parameters after `system`. It returns the
  designed {class}`~pypulseqpp.Sequence`, or a list of sequences, the prescans
  first and the main sequence last.
- The protocol is read from the signature and the NumPy-style Parameters
  section: types and defaults from the signature, units and descriptions from
  the docstring. {func}`~pypulseqpp.sequences.parameters` returns it, and
  {func}`pypulseqpp.cli.run` derives the command-line options from it.
- {class}`~pypulseqpp.sequences.Labels` writes the label events a block changes,
  so a loop records encoding indices as Pulseq labels without tracking label
  state. {meth}`~pypulseqpp.Sequence.evaluate_labels` recovers their ADC order
  from the sequence, so sampling figures can use the implemented acquisition
  order.
- {func}`~pypulseqpp.sequences.write` writes a list of sequences as separate
  files, each naming the next with `NextSequence`, so each file retains a single
  repeating unit. {func}`~pypulseqpp.sequences.duration` is the time the chain
  plays.
```

A complete sequence is designed from the system limits and a prescription: the
field of view, matrix, timing, flip angle and sampling order of the
acquisition. A sequence function is that design as a callable, from which a
script, the command line and a protocol editor obtain the sequence and the
prescription.

## Sequence function

A sequence function has the signature `sequence(system, **protocol)`. `system`
is the {class}`~pypulseqpp.Opts` the sequence is designed under. A function
designed for lower gradient or slew limits lowers the limits of `system` to them
with {func}`~pypulseqpp.cap_system`, which never raises a limit. The keyword
parameters after `system` are the protocol, and each states its unit in its
description. The function returns a {class}`~pypulseqpp.Sequence`, or, for a
scan with prescans, a list of sequences in play order.

| Concern | Sequence function |
| --- | --- |
| Prescription | Keyword parameters after `system`, documented in the Parameters section |
| System limits | `system`, capped to the design's limits with `cap_system` |
| Modules, timing and sampling arrays | The body, before the loop |
| Sampling order | The loop |
| One repetition | The body of the loop |
| Pulseq definitions required by reconstruction | `set_definition` after the loop |
| Prescans | The elements of the returned list before the main sequence |

A shipped sequence module defines its design limits as the constants
`MAX_GRAD` (mT/m) and `MAX_SLEW` (T/m/s). Changing one redesigns the gradient
waveforms and may alter echo spacing, acquisition duration, and constraint
estimates.

## Protocol

The protocol is read from the function. A parameter's type and default come
from its annotation and default value, and its unit, choices and description
from the NumPy-style Parameters section. The unit is the parenthesised group in
the first sentence of a description, `Echo time (s).`, and the choices of a
string parameter are the values its documented type lists in braces. An
annotation that admits `None` leaves the value to the design.
{func}`~pypulseqpp.sequences.parameters` returns one
{class}`~pypulseqpp.sequences.ProtocolParameter` per keyword parameter, in
signature order, which is what a protocol editor presents.

{func}`pypulseqpp.cli.run` derives one command-line option from each scalar
keyword parameter and takes its help text from the first sentence of the
parameter's description. The limit options `--max-grad-mtm` (mT/m) and
`--max-slew-tm-s` (T/m/s) build the `Opts` passed as `system`, and the result is
written with {func}`~pypulseqpp.sequences.write`.

## Encoding labels

A loop records encoding indices with Pulseq label extensions. The principal
labels are `LIN`, `PAR`, `ECO`, `SEG`, `REP`, and `SET`. A label keeps its value
across blocks until an event changes it, so a block carries the events of the
labels it changes and no others. {class}`~pypulseqpp.sequences.Labels` holds
that state: calling it with the new value of each label returns the events to
add to the block. An unchanged value writes nothing, a change equal to the
label's previous change is an INC, and any other change is a SET.
{meth}`~pypulseqpp.Sequence.evaluate_labels` recovers the ADC order directly
from the sequence. Sampling figures can therefore use the implemented
acquisition order rather than reconstructing it from the prescription.

## Prescans

A prescan is a sequence played before the main sequence, such as a calibration
acquisition. A sequence function returns it ahead of the main sequence in a
list. {func}`~pypulseqpp.sequences.write` writes the list as separate files: the
first at the given path, and each later one beside it as `<stem>_<Name>.seq`,
where `Name` is the sequence's `Name` definition, or its position in the list,
counted from 0 at the first, when it has none. A later file whose name an
earlier file has taken is written as `<stem>_<Name>_<position>.seq`, so every
sequence has a file of its own. Each file but the last names the next with
`NextSequence`. The chain represents one acquisition while retaining
a single repeating unit per file for repetition-based analyses.
{func}`~pypulseqpp.sequences.duration` is the sum of the durations of the
sequences in the chain.

## Checking a prescription

Calling a sequence function designs every block, and the number of blocks grows
with the matrix. A prescription the design cannot meet, such as an echo time
shorter than the readout admits, raises `ValueError`. The call does not evaluate
the waveforms against gradient, PNS or SAR limits, which take the designed
sequence ({doc}`../safety/index`).

The function records the prescription as designed in the definitions it sets
after the loop. A value the design chooses, such as the shortest echo time for
`te=None`, is recorded as `TE`, and prescribing the recorded value as `te`
designs a sequence that records the same `TE`. The receiver bandwidth as
designed is the reciprocal of the dwell time of the ADC events, which is
rounded to the ADC raster.

## See also

* {doc}`sequence-module` — reusable block layouts.
* {doc}`sampling` — acquired support and temporal ordering.
* {doc}`../../user-guide/from-pypulseq` — a PyPulseq script as a sequence
  function.
* {doc}`/generated/gallery/05-sequence-modules/03_sequence_app` — a sequence
  function assembled from modules.
* {doc}`../../api/apps` — exact interface.
* {doc}`../../sequences` — shipped complete sequences.
