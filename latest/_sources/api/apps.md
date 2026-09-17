# Complete sequences

`pypulseqpp.sequences`: {class}`SequenceApp`, the base class of the shipped
example sequences.

```{eval-rst}
.. currentmodule:: pypulseqpp.sequences
```

A {class}`SequenceApp` designs its events and sampling order from the
prescription its `init_sequence` takes; {meth}`~SequenceApp.design` runs its
scan `loop`, one `kernel` call per repetition. Prescans listed by
{meth}`~SequenceApp.prescans` are written by {meth}`~SequenceApp.write` as
separate files linked through `NextSequence`. Each example sequence reached as
`sequences.<name>` defines one subclass. The sequences themselves are listed
under {doc}`../sequences`.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SequenceApp
```
