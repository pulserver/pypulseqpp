# Command-line helpers

{func}`run` derives the command-line options from a sequence function's
signature and its NumPy-style `Parameters` section. {func}`write_sequence`
deduplicates the finished sequence and writes it as signed Pulseq text or in
the binary form.

```{eval-rst}
.. currentmodule:: pypulseqpp.cli
```

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.cli.run` | Run a sequence script's ``main`` from the command line. |
| {obj}`~pypulseqpp.cli.write_sequence` | Write a deduplicated copy as Pulseq text or binary. |
