# Sequence applications

A complete sequence combines a prescription, sampling order and scan loop with
its constituent modules. {class}`~pypulseqpp.sequences.SequenceApp` is
the contract that separates these three concerns, and every sequence
implementation the package ships is written against it.

## Prescription, sampling order and repetition

| Concern | Method | Responsibility |
|---|---|---|
| Prescription | `init_sequence` | Design modules and sampling schedule |
| Sampling order | `loop` | Iterate over repetitions and encoding states |
| Repetition | `kernel` | Add the blocks for one repetition |

`init_sequence` receives the prescription as keyword arguments and designs the
modules and the sampling order from it. `loop` calls `kernel` once per
repetition. `finalize` records the definitions a reconstruction reads.

Constructing an application runs `init_sequence` and therefore designs it, but
plays nothing. {meth}`~pypulseqpp.sequences.SequenceApp.design` starts a fresh
sequence, runs `loop`, then `finalize`, and returns the result.

Because `kernel` adds exactly one repetition, the same implementation supports
complete acquisition design and repetition-level inspection. Calling the
application directly appends one repetition to the current sequence.

## Prescription against settings

The signature of `init_sequence` is the prescription: its keyword parameters,
their defaults and its NumPy-style `Parameters` section are what
{meth}`~pypulseqpp.sequences.SequenceApp.protocol`, the command line and a
protocol editor present. Documenting a parameter once therefore documents it
everywhere it is offered.

Everything a user does not prescribe is a class attribute. `MAX_GRAD` and
`MAX_SLEW`, the gradient ceilings the sequence is designed under, have no
default, so every concrete application states them; the system limits are
capped to them at construction. A subclass that changes one class attribute and
nothing else is the same sequence designed under a different limit:

```python
class GentleEpi(Epi2DApp):
    MAX_SLEW = 60.0
```

A design constraint is imposed from outside the prescription in this way.
Lowering the slew ceiling lengthens every gradient ramp and therefore the echo
spacing, which reduces the peripheral-nerve-stimulation estimate of
{doc}`../safety/pns` at a cost in acquisition time that the redesigned sequence
reports.

## Encoding indices

Acquisitions record their encoding indices as `LABELSET` and `LABELINC`
extension rows, rather than leaving a reconstruction to re-derive them.
{meth}`~pypulseqpp.sequences.SequenceApp.labels` returns the label events that
move each counter to its new value, and applies the format's semantics: label
state is sticky, so an unchanged value writes nothing; a change equal to the
previous change is written as an increment, which a repeating scan reuses as a
single event; any other change is written as a set.

The consequence is that the sampling order chosen in `loop` is recoverable from
the file by {meth}`~pypulseqpp.Sequence.evaluate_labels` without re-deriving it,
which the echo-ordering example in the gallery reads to plot which view
was acquired at which echo.

## Prescans and sequence chains

A calibration prescan is not a mode of the imaging scan. Loops listed by
{meth}`~pypulseqpp.sequences.SequenceApp.prescans` are written by
{meth}`~pypulseqpp.sequences.SequenceApp.write` as separate files ahead of the
main one, each naming the next through the `NextSequence` definition described
in {doc}`../pulseq/libraries-and-shapes`.

An interpreter plays the chain as one acquisition, while each file remains a
single repeating unit, which the repetition detection underlying the
SAR check requires, and what would be lost if a prescan and an imaging scan
shared one file distinguished by a flag.

## See also

* {doc}`sequence-module` — the layer the applications are built from.
* {doc}`../../api/apps` — the `SequenceApp` interface.
* {doc}`../../sequences` — the sequence implementations that ship with the package.
* {doc}`../../guides/command-line` — running an application from a shell.
