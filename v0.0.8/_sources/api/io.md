# Reading and writing files

Reading a `.seq` file into a sequence whose system is built from the file,
writing a sequence in the text or the binary form, and the tables a file of a
sequence holds, as arrays.
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

## Library export

{meth}`~pypulseqpp.Sequence.libraries` returns the block table and every
library of a sequence as the tables a text file of it holds: the rows
{meth}`~pypulseqpp.Sequence.write` writes with `remove_duplicates=False`,
under the same ids, in read-only arrays. Times are in seconds and the other
columns are in the file's units.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.io.SequenceLibraries` | — | Block table; RF, gradient, ADC, shape and extension tables; `layout` | A sequence's libraries without decoding its blocks. |
| {obj}`~pypulseqpp.io.Shape` | Sample count, stored values | `Shape`; `decompressed` returns the samples | One row of the shape library. |
