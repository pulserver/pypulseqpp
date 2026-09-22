# Complete sequences

{class}`SequenceApp`, the base class every shipped sequence is written
against.

```{eval-rst}
.. currentmodule:: pypulseqpp.sequences
```

A {class}`SequenceApp` derives event modules and sampling order from its
`init_sequence` prescription. {meth}`~SequenceApp.design` executes the scan
`loop`, with one `kernel` call per repetition. Prescans listed by
{meth}`~SequenceApp.prescans` are written by {meth}`~SequenceApp.write` as
separate files linked through `NextSequence`. Each example sequence reached as
`sequences.<name>` defines one subclass. The sequences themselves are listed
under {doc}`../sequences`.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.sequences.SequenceApp` | A complete sequence, designed from a prescription and played one repetition at a time. |
