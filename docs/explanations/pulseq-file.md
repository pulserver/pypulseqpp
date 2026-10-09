# The Pulseq file

```{admonition} TL;DR
:class: tldr

- A `.seq` file is a flat list of blocks indexing shared event and shape libraries, every time on one of four rasters.
- pypulseqpp reads, deduplicates, signs and writes it, in text and binary, as Pulseq 1.5.1 or 1.4.1.
- Repetitions and shots are not stored in the file; pypulseqpp derives them from the block table.
```

A scanner plays a `.seq` file block after block, so everything you design has
to end up as a table of blocks, the events they play and the shapes those
events sample. pypulseqpp keeps that table and the libraries behind it, so
that the file you write is the one MATLAB Pulseq would write, and a file you
read is the one you can analyse. Every event time also has to fall on a raster
the scanner can address, which the timing check verifies.

## What pypulseqpp does

:::{container} capabilities

- **Writes a flattened schedule of blocks, each at most one event per channel, played concurrently and back to back.**

  <details><summary>Show code and tests</summary>

  Code: {class}`~pypulseqpp.Sequence`, {meth}`~pypulseqpp.Sequence.add_block`, {meth}`~pypulseqpp.Sequence.write`. Tests: *a file is written back as it was read*; *writing a file twice writes the same bytes*; *the written file is byte identical to the reference* (`test_corpus.py`, `test_parity.py`).

  </details>
- **Stores each distinct event once and refers to it by 1-based id, so an event played ten thousand times is one row.**

  <details><summary>Show code and tests</summary>

  Code: {class}`~pypulseqpp.io.SequenceLibraries`, {class}`~pypulseqpp.io.Shape`. Tests: *every gradient id is one trapezoid or one arbitrary gradient*; *the libraries come back the size they went in* (`test_libraries.py`, `test_read.py`).

  </details>
- **Splits every event into a definition and its instances, so a phase-encode table is one definition at many amplitudes.**

  <details><summary>Show code and tests</summary>

  Code: {class}`~pypulseqpp.RfInstances`, {class}`~pypulseqpp.RfDefinition`. Tests: *a phase encode table is one definition at many amplitudes*; *gradients differing only in amplitude share a definition*; *arbitrary gradients of different lengths are different definitions* (`test_repetition.py`, `test_libraries.py`).

  </details>
- **Merges equal shapes and events when writing, on a copy, and renumbers the blocks.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.remove_duplicates`, {meth}`~pypulseqpp.Sequence.write`. Tests: *writing collapses duplicates without touching the sequence*; *collapsing duplicates twice changes nothing the second time*; *collapsing duplicates merges the roles of the shapes it merges* (`test_facade.py`, `test_structure.py`, `test_shape.py`).

  </details>
- **Finds the repetition from the block table, or takes the declared `TRSize`.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.repetition`. Tests: *a phase encode does not make a shot different*; *blocks played once before the loop make the whole sequence one*; *a declared hyper TR the blocks repeat with is taken* (`test_repetition.py`).

  </details>
- **Writes one `ROTATIONS` row per orientation instead of a rotated copy of every waveform.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.make_rotation`. Tests: *turning by extension plays what turning the events plays*; *a turned sequence survives a file*; *the file is the one the toolbox writes* (`test_rotation_extension.py`).

  </details>
- **Writes labels as sticky `LABELSET`/`LABELINC` rows, and evaluates their running values.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.make_label`, {meth}`~pypulseqpp.Sequence.evaluate_labels`. Tests: *a label is recorded where it is asked for*; *a custom label comes back by its name* (`test_evaluate_labels.py`, `test_labels.py`).

  </details>
- **Carries triggers, RF shims and soft delays as extensions.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.make_trigger`, {func}`~pypulseqpp.make_rf_shim`, {func}`~pypulseqpp.make_soft_delay`. Tests: *an extended sequence survives the text form*; *an extended sequence survives the binary form* (`test_binary.py`).

  </details>
- **Reads text and binary files, including Pulseq 1.2 to 1.4, into the same sequence.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.read`, {func}`pypulseqpp.io.read`, {func}`pypulseqpp.io.read_chain`. Tests: *a file survives the binary form*; *an older file holds the events the 1 5 file holds*; *a chain is read in play order each file on its own system* (`test_corpus.py`, `test_io.py`).

  </details>
- **Writes the binary form, with an optional MD5 signature in either form.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.write_binary`. Tests: *a written file carries the signature it reports*; *a patched file is refused*; *a shape comes back within a single precision sample* (`test_binary_signature.py`, `test_binary.py`).

  </details>
- **Writes Pulseq 1.4.1 on request, folding ppm offsets and refusing what it cannot express.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.write_v141`. Tests: *a rotating sequence is refused rather than flattened*; *a ppm offset becomes the hertz it stands for*; *a soft delay is left out with a warning* (`test_v141.py`).

  </details>
- **Builds the system from a file's rasters and stated limits, or keeps the one you constructed.**

  <details><summary>Show code and tests</summary>

  Code: {func}`pypulseqpp.io.read`, {meth}`~pypulseqpp.Sequence.read`. Tests: *the rasters of the system built are the ones the file carries*; *a limit passed in wins over the one the file would give* (`test_io.py`).

  </details>
- **Reports every event time off its raster, and the dead, ringdown and duration conditions.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.check_timing`. Tests: *a ramp between two gradient ticks is reported*; *a dwell between two ADC ticks is reported*; *a pulse starting inside the RF dead time is reported*; *a gradient left on before the block ends is reported* (`test_check_timing.py`).

  </details>
- **Finds the shortest dwell on both the ADC and gradient rasters, and reports the bandwidth achieved.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.calc_adc_timing`, {func}`~pypulseqpp.round_to_raster`, {func}`~pypulseqpp.ceil_to_raster`. Test: *the ADC lands on both rasters at once* (`test_pypulseq_base_factories.py`).

  </details>

:::

## What PyPulseq and MATLAB Pulseq do

- **MATLAB Pulseq defines the format.** `pypulseq-matlab-like`, a transcription
  of it, is the authority for the file; the written bytes are held to it, with
  deduplication on and off.
- **PyPulseq defines the Python vocabulary.** Signatures and event conventions
  are preserved; not every upstream feature is implemented.
- **Neither records repetitions, shots or limits beyond the rasters.** The file
  is a playout order and nothing above it.

## How it works

```{figure} ../generated/figures/block_table_and_libraries.png
The block table of a written eight-line gradient-echo file, the libraries its
cells index, and the shape library the RF and gradient rows index in turn. Each
library holds one row per distinct event, however many blocks play it, so the
tables below the block table are far shorter than it is.
```

### Blocks, not loops

Two rules fix the timing: the events of a block start together, each after its
own delay, and blocks follow one another with no gap. A block lasts at least as
long as its events and may last longer; the excess is a delay, which is how an
echo time or a repetition time is written. Amplitudes are in Hz and Hz/m, so
the file does not depend on the nucleus; `gamma` of {class}`~pypulseqpp.Opts`
enters only where a physical amplitude is needed.

### Structure derived, not stored

A phase encode changes an instance and not its definition, so the lines of a
Cartesian scan are one repetition. That is what {meth}`~pypulseqpp.Sequence.repetition`
reads, and what the SAR check of {doc}`safety-checks` averages over. The same
split lets one interleaf serve every shot of a radial or spiral scan: the shape
library stays the size of one repetition, and an analysis of played gradients
applies each block's rotation first.

### Times on a grid

Every time is a multiple of one of four rasters: RF, gradient, ADC and block
duration. The ADC window has to end on the gradient raster and its dwell on the
ADC raster, so the sample count limits the receiver bandwidth: at the default
rasters, 128 samples admit at most 100 kHz and 100 samples admit 500 kHz.

The rules, formats and formulas are in
{doc}`../developer-guide/internals/pulseq-file`; the stored layout of the
libraries in the C++ core is in {doc}`cpp-core`.

## See it run

- {doc}`../generated/gallery/01-course/01_first_gradient_echo`: the first sequence, written and read back.
- {doc}`../generated/gallery/01-course/02_echo_and_repetition_time`: echo and repetition times as block durations.
- {doc}`../generated/gallery/01-course/04_labels_and_metadata`: labels and definitions.
- {doc}`../generated/gallery/01-course/07_hardware_and_safety_checks`: `check_timing` on a finished sequence.
- {doc}`../api/sequence`, {doc}`../api/io` and {doc}`../api/timing`: the container, the readers and writers, and the timing helpers.
