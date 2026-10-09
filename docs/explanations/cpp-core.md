# The C++ core

```{admonition} TL;DR
:class: tldr

- A script written for PyPulseq runs on a C++ sequence store after one changed import.
- Block insertion, deduplication, file I/O, timing and analysis run natively; event design and some helpers remain PyPulseq's.
- A few upstream names are withheld, each with the reason it is not here.
```

A PyPulseq script builds a sequence in Python objects, which is slow when the
sequence holds a million blocks. pypulseqpp keeps the PyPulseq vocabulary and
moves the storage and the heavy loops into a C++ core, so the script you wrote
for PyPulseq runs unchanged. The mental model: Python hands events across once
per block, and everything after that happens in compiled code.

## What pypulseqpp does

:::{container} capabilities

- **Runs a PyPulseq script after `import pypulseqpp as pp` replaces `import pypulseq as pp`.**

  <details><summary>Show code and tests</summary>

  Code: {class}`~pypulseqpp.Sequence`, {class}`~pypulseqpp.Opts`, {func}`~pypulseqpp.make_trapezoid`. Tests: *every upstream name a script could write still resolves*; *the amplitude and slew defaults are upstreams* (`test_facade.py`).

  </details>
- **Registers every event of a block in one native call, bound by hand to skip argument conversion.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.add_block`. Test: *adding a block goes through the fast calling convention* (`test_sequence.py`).

  </details>
- **Stores shapes, events and blocks natively, with 1-based libraries and block indices.**

  <details><summary>Show code and tests</summary>

  Code: {class}`~pypulseqpp.Sequence`. Test: *the block table is read back without copying* (`test_sequence.py`).

  </details>
- **Returns block-table arrays as snapshots that later edits do not reach.**

  <details><summary>Show code and tests</summary>

  Code: `Sequence.block_durations`, `Sequence.block_events`. Tests: *a view is a snapshot that later writes do not reach*; *a view stays valid when the block table grows*; *a view outlives the sequence it was taken from* (`test_sequence.py`).

  </details>
- **Deduplicates equal shapes and events, and detects the repetition from the result.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.remove_duplicates`, {meth}`~pypulseqpp.Sequence.repetition`. Tests: *collapsing duplicates makes the answer stale*; *dedup does not change RF instances* (`test_repetition.py`, `test_rf_instances.py`).

  </details>
- **Drops a derived result when the sequence is edited.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.repetition`. Tests: *adding a block makes the answer stale*; *retiming a block that plays events makes the answer stale* (`test_repetition.py`).

  </details>
- **Writes Pulseq text and binary files byte-identical to the reference writer.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.write`, {meth}`~pypulseqpp.Sequence.write_binary`, {meth}`~pypulseqpp.Sequence.read`. Tests: *the written file is byte identical to the reference*; *a file read back writes the same bytes* (`test_parity.py`, `test_read.py`).

  </details>
- **Lets PyPulseq's own functions take and return compiled events.**

  <details><summary>Show code and tests</summary>

  Code: {func}`~pypulseqpp.calc_duration`, {func}`~pypulseqpp.align`, {func}`~pypulseqpp.split_gradient`, {func}`~pypulseqpp.rotate`. Tests: *upstreams own helpers take a compiled event*; *a helper gives back what the rest of the package wants*; *calc duration agrees with upstream* (`test_facade.py`).

  </details>
- **Reads and writes files that PyPulseq reads and writes.**

  <details><summary>Show code and tests</summary>

  Code: {meth}`~pypulseqpp.Sequence.write_binary`. Tests: *they read a binary file we wrote*; *we read a binary file they wrote* (`test_interoperability.py`).

  </details>
- **Refuses a name it withholds with the reason, not a bare `AttributeError`.**

  <details><summary>Show code and tests</summary>

  Code: `add_ramps`, `add_custom_label`. Test: *a name that is not here says why* (`test_facade.py`).

  </details>

:::

## What PyPulseq does

- **Event design that is not reimplemented.** Upstream's own code answers where
  pypulseqpp has no replacement of its own; the interoperability wrapper
  converts events on the way in and out.
- **Unsupported areas stay unsupported.** Automatic labelling and tiling are not
  implemented, and `add_ramps` is withheld: it needs `calc_ramp`, which raises
  on any join needing intermediate points. {func}`~pypulseqpp.traj_to_grad`
  solves the same problem. `add_custom_label` has no counterpart because
  {func}`~pypulseqpp.make_label` takes any name.
- **Shape codecs** (`compress_shape`, `decompress_shape`) are imported from
  `pypulseq`.

## How it works

```{figure} ../generated/figures/block_loop_time.png
Time of a loop adding one trapezoid block per iteration, as the number of
blocks grows, in pypulseqpp and in PyPulseq when it is installed. Measured
when the page is built; the absolute times depend on the machine.
```

### One crossing per block

`Sequence.add_block` converts the events once, registers them and appends the
row to the block table in a single native call. The loop over blocks is yours;
every loop over the blocks of a finished sequence, such as timing, waveforms or
checks, is native. Substantial native work releases the GIL, so other Python
threads run meanwhile.

### Snapshots and revisions

The block table is read back without copying: an array shares the table's
buffer, and the core copies the buffer before a later write while an array is
out. The core also counts its edits. Derived analyses, such as the repetition,
record the count they were computed at, so a cached result is checked by one
comparison rather than by a per-block update.

### Events across the boundary

A compiled event is not a PyPulseq namespace and fails PyPulseq's type checks
and `deepcopy`. The wrapper converts arguments to namespaces and results back,
which is what lets the upstream namespace be used without forking it.

The rules, and how insertion cost is measured, are in
{doc}`../developer-guide/internals/cpp-core`.

## See it run

- {doc}`../generated/gallery/01-course/01_first_gradient_echo`: a PyPulseq-style
  script on the compiled `Sequence`.
- {doc}`../generated/gallery/01-course/02_echo_and_repetition_time`: the
  repetition the core detects.
- {doc}`../api/sequence`: the container and its methods.
