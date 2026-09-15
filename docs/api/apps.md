# Sequence apps

`pypulseqpp.sequences`: the base class of the complete sequences in the zoo.

```{eval-rst}
.. currentmodule:: pypulseqpp.sequences
```

A {class}`SequenceApp` designs its events and sampling order from the
prescription its `init_sequence` takes; {meth}`~SequenceApp.design` runs its
scan `loop`, one `kernel` call per repetition. Prescans listed by
{meth}`~SequenceApp.prescans` are written by {meth}`~SequenceApp.write` as
files linked through `NextSequence`. Each zoo script reached as
`sequences.<name>` defines one.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SequenceApp
```
