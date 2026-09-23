# Complete sequences

The base class every shipped complete sequence is written against. The
sequences themselves are listed in {doc}`../sequences`; the structure of an
application, its scan loop and its prescan chain are described in
{doc}`../explanations/design/sequence-application`.

```{eval-rst}
.. currentmodule:: pypulseqpp.sequences
```

## Base class

Every concrete subclass sets `MAX_GRAD` (mT/m) and `MAX_SLEW` (T/m/s);
`init_sequence`'s signature and Parameters section are the prescription.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.SequenceApp` | System limits, prescription keywords for `init_sequence` | Application; `design()` returns the `Sequence` | Base class of complete sequence implementations. |
