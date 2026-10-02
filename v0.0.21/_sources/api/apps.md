# Complete sequences

A complete sequence is a function `sequence(system, **protocol)` that returns a
{class}`~pypulseqpp.Sequence`, or a list of them with the prescans first and the
main sequence last. The functions below read its protocol, label its blocks,
write what it returns and time it; {class}`~pypulseqpp.sequences.SequenceApp`
is the base class the shipped sequences are written against. The sequences
themselves are listed in {doc}`../sequences`; the structure of a sequence
function, its protocol and its prescan chain are described in
{doc}`../explanations/design/sequence-application`.

```{eval-rst}
.. currentmodule:: pypulseqpp.sequences
```

## Sequence functions

The protocol of a sequence function is its keyword parameters after `system`.
`parameters` reads each one's type and default from the signature, and its unit,
choices and description from the NumPy-style Parameters section.
{func}`pypulseqpp.cli.run` derives the command-line options from the same
signature.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.parameters` | Sequence function | `dict` of `ProtocolParameter`, in signature order | Protocol of a sequence function. |
| {obj}`~pypulseqpp.sequences.ProtocolParameter` | Returned by `parameters` and `SequenceApp.parameters()` | Frozen record | One protocol parameter: its type, default, unit, choices and description. |
| {obj}`~pypulseqpp.sequences.Labels` | The new value of each label, by Pulseq name | Label events to add to the block | Writer of the label events a block changes. |
| {obj}`~pypulseqpp.sequences.write` | Path, a sequence or a list of sequences | Written paths, in play order | Write a sequence, or a chain of linked sequences, as Pulseq files. |
| {obj}`~pypulseqpp.sequences.duration` | A sequence or a list of sequences | Duration (s) | Return the time a sequence, or a chain, plays. |

## Base class

Every concrete subclass sets `MAX_GRAD` (mT/m) and `MAX_SLEW` (T/m/s);
`init_sequence`'s signature and Parameters section are the prescription.
`parameters()` returns that prescription one parameter at a time, and a
constructed application reports its resolved prescription, `resolved`, and its
scan time, `scan_time()`, without playing the loop. `function()` returns the
class as a sequence function.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.SequenceApp` | System limits, prescription keywords for `init_sequence` | Application; `design()` returns the `Sequence`, `function()` the sequence function | Base class of complete sequence implementations. |
