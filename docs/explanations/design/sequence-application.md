# Sequence applications

A complete sequence is more than a set of modules. It also has a prescription —
the parameters a user sets — a sampling order, and a scan loop that plays the
modules in that order. {class}`~pypulseqpp.sequences.SequenceApp` is the
contract that separates these three concerns, and every sequence implementation
the package ships is written against it.

## Three separable concerns

| Concern | Where it lives | Question it answers |
|---|---|---|
| Prescription | `init_sequence` | What is being acquired? |
| Sampling order | `loop` | In what order are the repetitions played? |
| One repetition | `kernel` | What blocks does a single repetition contain? |

`init_sequence` receives the prescription as keyword arguments and designs the
modules and the sampling order from it. `loop` calls `kernel` once per
repetition. `finalize` records the definitions a reconstruction reads.

Constructing an application runs `init_sequence` and therefore designs it, but
plays nothing. {meth}`~pypulseqpp.sequences.SequenceApp.design` starts a fresh
sequence, runs `loop`, then `finalize`, and returns the result.

The separation is not merely tidy. Because `kernel` adds exactly one
repetition, calling the application plays one repetition into the current
sequence, so a single repetition and a whole scan are written by the same code.
A caller that needs a chunk of a scan — to inspect it, or to hand it to an
analysis that does not need the whole thing — is not writing a second code
path.

## Prescription against settings

The signature of `init_sequence` *is* the prescription: its keyword parameters,
their defaults and its NumPy-style `Parameters` section are what
{meth}`~pypulseqpp.sequences.SequenceApp.protocol`, the command line and a
protocol editor present. Documenting a parameter once therefore documents it
everywhere it is offered.

Everything a user does not prescribe is a class attribute. `MAX_GRAD` and
`MAX_SLEW` — the gradient ceilings the sequence is designed under — have no
default, so every concrete application states them; the system limits are
capped to them at construction. A subclass that changes one class attribute and
nothing else is the same sequence designed under a different limit:

```python
class GentleEpi(Epi2DApp):
    MAX_SLEW = 60.0
```

This is the mechanism by which a design constraint is imposed from outside the
prescription. Lowering the slew ceiling lengthens every gradient ramp and
therefore the echo spacing, which is what makes it an effective control on the
peripheral-nerve-stimulation estimate of {doc}`../safety/pns` — at a cost in
acquisition time that the redesign makes visible.

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
which is what the echo-ordering example in the gallery reads to plot which view
was acquired at which echo.

## Prescans and sequence chains

A calibration prescan is not a mode of the imaging scan. Loops listed by
{meth}`~pypulseqpp.sequences.SequenceApp.prescans` are written by
{meth}`~pypulseqpp.sequences.SequenceApp.write` as separate files ahead of the
main one, each naming the next through the `NextSequence` definition described
in {doc}`../pulseq/libraries-and-shapes`.

An interpreter plays the chain as one acquisition, while each file remains a
single repeating unit — which is what the repetition detection underlying the
SAR check requires, and what would be lost if a prescan and an imaging scan
shared one file distinguished by a flag.

## Related pages

* {doc}`sequence-module` — the layer the applications are built from.
* {doc}`../../api/apps` — the `SequenceApp` interface.
* {doc}`../../sequences` — the sequence implementations that ship with the package.
* {doc}`../../guides/command-line` — running an application from a shell.
