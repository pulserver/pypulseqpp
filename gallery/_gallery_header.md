# Examples

Executable examples, grouped by the level at which a sequence is described.

**Pulseq basics** works at the level of the file format: system limits, RF,
gradient and ADC events, the blocks that play them, and the analyses and
hardware checks a finished sequence is inspected with.

**Sequence modules** works at the level of a reusable block layout. A module
solves the timing and gradient waveforms of an excitation, a preparation or a
readout once and exposes the resulting events; a
{class}`~pypulseqpp.sequences.SequenceApp` adds the sampling order and the scan
loop that plays them.

**Complete sequences** works at the level of a prescription. Each page takes
one of the sequences the package ships and varies the parameters that separate
its applications, showing what changes in the timing diagram, in the sampling
pattern and in the scan duration.

Conceptual background is in {doc}`../../explanations/index`, and the exhaustive
interface semantics in {doc}`../../api/index`.
