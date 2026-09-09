# What is left

`pypulseq-matlab-like` is the authority for the format, and the tests are now
written against it. Reading and writing are complete: every revision from
1.2.0 up is read, in text and in binary, and a file can be written as 1.5.1 or
as 1.4.1. `check_timing` judges every event time against the system's rasters
and dead times. What remains is here.

## The rest of the reference suite

`tests/` there is 6809 lines over 80 files, most of it covering event
factories this package does not have yet. Of the thirteen that bear on what is
here, three are ported: `test_block` as `tests/test_block.py`,
`test_rotation_extension` and `test_make_rf_shim` under their own names. The
other ten are not worth porting as they stand, and why is worth writing down:

- `test_md5` and `test_quaternion` test `hashlib` and `scipy`, not the
  toolbox. What they would test here -- the C++ md5 behind a signature, and
  what a quaternion does to a block's gradients -- is held by
  `tests/test_corpus.py` and `tests/test_rotation_extension.py`.
- `test_opts`, `test_get_supported_labels` and `test_block2events` test
  upstream PyPulseq, which this re-exports unchanged; `tests/test_facade.py`
  holds the re-export itself.
- `test_compress_shape` and `test_decompress_shape` are property tests of the
  codec. `tests/test_shape.py` compares our encoder against the reference
  encoder sample for sample, which is stronger.
- `test_add_custom_label` needs `add_supported_label`, which upstream PyPulseq
  does not have either.
- `test_aux_version` reads `version_major`, `version_minor` and
  `version_revision` off a Sequence. The core holds all three; the facade
  exposes none.
- `test_binary_signature` needs a signed `.bseq`. See below.

### What the port turned up

Four differences from the authority, none of them found by anything else:

- **A binary file is not signed here.** The toolbox writes an md5 over the
  binary form and reports it as `signature_type='md5'`, `signature_file='bin'`;
  `write_binary` here writes no signature section and reading a signed one
  reports none. Files still pass both ways -- `tests/test_interoperability.py`
  holds that -- so what is missing is the integrity check, not the format.
- **`make_rotation` takes one call form.** The toolbox takes six: an angle, an
  angle and a polar angle, an axis and an angle, a quaternion, a 3x3 matrix,
  and a stack of them. Here it takes an object with `as_quat`, which is a
  SciPy `Rotation`. Upstream PyPulseq has no `make_rotation`, so the toolbox is
  the authority for this one.
- **`make_rf_shim` keeps the shape it is given.** The toolbox reshapes the
  weights to a column, so `shim.shim_vector[k, 0]` is how a script written
  against it reads a weight; here a list gives `(n,)` and a scalar gives a
  0-d array. One weight per channel is the simpler convention, but the
  scalar case is a wart either way.
- **`add_block(None)` is refused.** The toolbox takes it and adds no block.
  Upstream PyPulseq raises, and upstream is the API this stands in for, so
  this raises too -- with a message that says what was expected.

## The rest of the timing check

`check_timing` covers what the authority's `check_timing` module covers, and
two of the three further questions its `Sequence.check_timing` asks. Gradient
continuity is in: a waveform picked up after a delay, one left on before its
block ends, one starting where the block before did not leave the axis, and a
sequence that never ramps its axes down are each reported. The step between
two blocks is judged against the system's slew limit rather than against a
flat tolerance -- the raster runs on between blocks, so a step of one raster
is a slew like any other, and the reference toolbox refuses two of these four
when the block is added rather than when the sequence is checked.

`TotalDuration` is in with it. The first check records what the blocks add up
to and every later one holds the record to them, and a file that declares a
duration arrives already held to it, so one that does not add up is reported
rather than quietly corrected. The write is a side effect of the check, as it
is in MATLAB; `write(check_timing=True)` therefore produces a file carrying
`TotalDuration` where `write()` does not.

One question is left:

- **Frequency offsets.** An RF or ADC offset, in Hz or as a ppm shift through
  gamma, must stay inside `system.max_freq_offset`.

## The rest of `Sequence`

The one-line methods, the aliases and the no-ops are in. What is left, by
what it would take:

**Open.** `sound`, deferred with `plot`.

## Promoted extensions

The block table carries two columns the file format knows nothing about: the
rotation a block turns its gradients by, and the RF shim its pulse is played
through. Both are still extensions -- written as ones, read as ones, named by
the chain -- and the column is filled on the way in, so a round trip is byte
for byte what it was.

What earns a column is being wanted before anything else can happen: a
gradient cannot be drawn without knowing which way it faces, a pulse cannot
be materialised without its shim. `Sequence::Promoted` holds the list, so a
third is an entry there and a wider `BLOCK_WIDTH`.

Nothing yet reads the shim column: block decoding walks the chain once for
every extension kind at once, which the column does not shorten. It is there
for the RF materialisation path, which will want it the way the waveform
expansion wants the rotation.

The candidates that were weighed and left alone: `TRIGGERS`, which already
has a per-chain-node cache and so is a lookup already; `DELAYS`, wanted only
by `apply_soft_delay`, which is one pass over the whole sequence; and the
control flags (`NOPOS`, `NOROT`, `NOSCL`, `PMC`, `ONCE`, `TRID`), which are
not extension types at all but entries in the label table, carried by
`LABELSET`/`LABELINC` like `LIN` and `SLC`. A label is running state -- set
at one block, in force until changed -- so a column for one would be
materialising an accumulated scan, which any earlier `set_block` invalidates.
That is a pass to run when something wants it, not a column.

## Safety

`pypulseqpp.safety` has `check_max_grad`, `check_max_slew` and
`check_grad_continuity`. All three read the libraries rather than an expanded
waveform: a gradient is a normalised shape and one amplitude, so the steepest
step belongs to the shape and an instance's is that times its own amplitude.

The slew limit within a block and the joins between blocks are separate
questions and are asked separately: `check_timing` asks the second, which is
two numbers per gradient, and leaves the first to whoever wants it.

The vector peaks are exact rather than upper bounds, and the per-axis peaks
account for how a block is turned. Both come from the same walk: a gradient
is a handful of points with straight lines between them, so what the three
axes ask for together is decided at the moments any of them turns a corner.
The cheap combination of the axes' own peaks is kept as a filter -- a block
that cannot beat what has been found already is never walked -- which leaves
the passes at tens of nanoseconds a block.

What is weighed is the waveform an interpreter draws, corner to corner, and
not the samples the file stores. The two differ where a shape is kept at the
centre of each raster interval: the corners are half a raster from any sample,
and the drawn waveform passes outside all of them. `restore_shape_corners`
puts them back, once per gradient rather than once per block, and `corners.cpp`
is the one place that says what a gradient draws -- the waveform expansion
reads it too. Across the reference zoo and `tests/seq` the checks now agree
with the peaks taken from the expanded waveforms to within a part in a
million, where they were three parts in ten thousand out on an amplitude and
a part in a thousand on a slew.

What that costs is the corner restoration, which is a pass over a shape and
belongs to the gradient: on the corpus, twenty nanoseconds a block. A sequence
that registers a waveform per shot instead of scaling one pays for each of
them, the same way the waveform expansion does -- `remove_duplicates` is what
makes it once. A rotated sequence of long waveforms is the one case that
walks: a turned block's per-axis peaks are a support function over the
instants it plays, and the bound that skips a block is loose for a rotation
that sweeps. Twenty milliseconds for a six-hundred-shot rotated spiral. The
convex hull of those instants, taken once per gradient triple, would answer
every rotation from a handful of points; it is not there, and nothing has
wanted it yet.

Still to write: `calculate_pns` and `calc_rf_power`, and the mechanical
resonance check. pulserver's C library has all three.

## The analysis family

`test_report` and `test_report_dict` are in, and with them `waveforms`,
`waveforms_and_times`, `adc_times`, `rf_times`, `get_gradients` and
`calculate_kspace`. The report answers the toolbox's report, entry for entry
and line for line, wherever the toolbox answers at all: it looks for the echo
in the sampled trajectory, so a sequence that acquires nothing stops it, where
here that sequence reports an undefined TE and the repetition time between its
last two excitations.

Two of its answers are worked out rather than read off, and both are compiled.
A flip angle is the integral of a pulse's envelope, so it belongs to the RF
library row and a pulse played ten thousand times is integrated once. What the
encoding covers -- the distinct positions along each axis, how often one is
revisited, whether they fill a grid -- is a pass over every sample the scan
takes, binned onto a lattice of the trajectory's extent over four million.

The FOV transform reads the same trajectory, and is still to write.

### What a shot shares, and what it does not

A sequence built the way one should be -- the gradient made once, outside
the loop, and scaled per shot -- gives every shot the same corner times, a
zero phase encode included, because a scaled event keeps its ramps. Measured
on a sixty-four line gradient echo: one corner-time pattern on every axis.

What does differ per shot is which ramps are *followed through at the
raster*, since a ramp is only sampled when it is ramping and that depends on
its amplitude. The same sixty-four lines give five different moment counts,
229 to 233. So a trajectory cannot be one base repeated -- but those extra
moments are only where the trajectory is *reported*, never where a sample
sits, which is why `samples_only` can skip all of them and still be exact.

What is left for the repeat to save is the merge, a third of the full pass.
It would have to merge a prologue, a base, and the shots that differ from
it: more machinery than the merge it replaces, on a pass already 26x the
toolbox and 85x with `samples_only`.

The repeating unit is already found (`_detect_tr`), so an analysis that only
needs one shot does not have to look at the whole scan to find it.

## Where a name differs from the toolbox

`waveforms_and_times` and its family take `block_range`, where
`pypulseq-matlab-like` spells it `blockRange`. Upstream PyPulseq has no such
parameter at all -- only `time_range` -- so there is no drop-in contract to
keep, and every other name here is snake_case.

`waveforms_and_times` returns six values, as the toolbox does; upstream
returns five, having no `pm_adc`. A script unpacking upstream's five breaks
on six. The sixth carries the ADC phase modulation, which is a 1.5 feature
upstream's return predates and which `calculate_kspace` needs, so it is kept
-- but it is a difference from upstream worth knowing about.

## What a block reads back as

`get_block` answers with the compiled event types rather than with
namespaces, so one object serves twice: in Python it reads the way an event
from a factory reads, and handed to `add_block` or `set_block` it takes the
fast path. It carries the shapes it was stored under, so a sequence read out
block by block and put back registers no waveform twice and writes the same
file. A whole block can be passed on as it stands, which is how a block moves
between sequences with its duration intact -- the only place a block that
plays nothing keeps how long it waits.

That is what every method reading a sequence needs, so the ones still to
write -- the plotting and analysis below among them -- have what they read
from.

**Deferred.** Safety -- `calculate_pns`, `calculate_gradient_spectrum`,
`calc_rf_power` -- and plotting -- `plot`, `paper_plot`, `sound` -- and the
label readers, `auto_label` and `evaluate_labels`.
