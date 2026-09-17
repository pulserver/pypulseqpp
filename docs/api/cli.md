# Command-line helpers

`pypulseqpp.cli`: {func}`run` derives command-line options from a sequence
function's signature and its NumPy-style `Parameters` section;
{func}`write_sequence` deduplicates the finished sequence and writes it as
signed Pulseq text or in the binary form.

```{eval-rst}
.. currentmodule:: pypulseqpp.cli
```

```{eval-rst}
.. autosummary::
   :nosignatures:

   run
   write_sequence
```
