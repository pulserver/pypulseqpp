# Reading and writing files

Reading a `.seq` file into a sequence whose system is built from the file,
writing a sequence in the text or the binary form, and the tables a file of a
sequence holds, as arrays.
{doc}`../explanations/pulseq-file` describes what a file
records, including which system limits, and the two forms.

```{eval-rst}
.. currentmodule:: pypulseqpp.io
```

## File input and output

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.io.read` | File path; optional gradient and slew limits | `Sequence` on a system built from the file | Read with file-derived rasters and limits. |
| {obj}`~pypulseqpp.io.read_chain` | First file path; `detect_rf_use`, `verify` | `(Path, Sequence)` pairs in play order | Read a file and the `NextSequence` files it names, each on a system built from it. |
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

## Diffusion tables

A diffusion table gives one b-value and one direction per volume. A sequence
reads the table from its file and records where the file is, not the table, so
that a reader of the sequence, such as a reconstruction proxy, reads the same
table from the same file. Directions are along the sequence's gradient
channels, which turn with the prescription.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.io.read_diffusion_table` | Path; `b_value` and `directions` for a block direction table | b-values (s/mm²) and unit directions, one per volume | Read an `x y z b` table, an FSL pair or a block direction table. |
| {obj}`~pypulseqpp.io.set_diffusion_table` | `Sequence`, then as `read_diffusion_table` | The table | Read a table and record its path and arguments in `[DEFINITIONS]`. |
| {obj}`~pypulseqpp.io.get_diffusion_table` | `Sequence` | The table, or `None` | Read the table a sequence's definitions name. |
