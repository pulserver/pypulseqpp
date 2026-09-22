# Command-line helpers

{func}`run` derives the command-line options from a sequence function's
signature and its NumPy-style `Parameters` section, runs it, and writes what
it produces through {func}`pypulseqpp.io.write`.

```{eval-rst}
.. currentmodule:: pypulseqpp.cli
```

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.cli.run` | Run a sequence script's ``main`` from the command line. |
