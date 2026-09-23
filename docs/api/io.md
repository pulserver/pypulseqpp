# Reading and writing files

Reading a `.seq` file into a sequence whose system is built from the file, and
writing a sequence in the text or the binary form.
{doc}`../explanations/pulseq/libraries-and-shapes` describes what a file
records, including which system limits, and the two forms.

```{eval-rst}
.. currentmodule:: pypulseqpp.io
```

## File input and output

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.io.read` | File path; optional gradient and slew limits | `Sequence` on a system built from the file | Read with file-derived rasters and limits. |
| {obj}`~pypulseqpp.io.write` | `Sequence`, file path, `binary` | Text signature, or `None` for binary | Write Pulseq text or binary. |
