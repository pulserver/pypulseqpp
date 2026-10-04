# Command-line helpers

The command-line entry point of a sequence function and of the sequence scripts.
The options are derived from the callable's signature and its NumPy-style
`Parameters` section.

```{eval-rst}
.. currentmodule:: pypulseqpp.cli
```

## Entry point

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.cli.run` | Sequence function or script's `main`, argument list | Exit status; writes the `.seq` file or file chain | Run a sequence function, or a sequence script, from the command line. |
