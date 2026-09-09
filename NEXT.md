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

Four differences from the authority, none of them found by anything else.
Two are closed:

- **A binary file is signed**, as the toolbox signs one: an MD5 over
  everything above the section that carries it, checked before the bytes are
  parsed rather than after. `write_binary(create_signature=False)` writes an
  unsigned file, as `write` does. Ours verifies theirs and theirs verifies
  ours.
- **`version_major`, `version_minor` and `version_revision`** are on the
  Sequence. They say what the sequence *is*: a file older than 1.5 is
  converted as it is read, so it is held as 1.5 whatever it declared.

- **`make_rotation` takes all six of the toolbox's call forms** -- an angle,
  an angle and a polar angle, an axis and an angle, a quaternion, a 3x3 matrix
  and a stack of them -- as well as a SciPy `Rotation`. The angle forms are
  worked out here rather than asked of SciPy, which is what a scan turning a
  base spoke per shot actually spends: a turn about z is a cosine and a sine.
  Measured against the block it is attached to, on a golden-angle loop:

  | rotation per shot | cost | against the block |
  | --- | --- | --- |
  | none | 271 ns | 1.0x |
  | `make_rotation(angle)` | 2.1 us | 7.8x |
  | `make_rotation(Rotation.from_euler(...))` | 35.3 us | 130x |

  In C++ it would be slower, not faster: a binding round trip costs about
  700 ns against the 800 ns the whole Python constructor costs, and what the
  scipy path spends is inside scipy either way.

One is open, and it is a decision rather than a gap:

- **`add_block(None)` is refused.** The toolbox takes it and adds no block.
  Upstream PyPulseq raises, and upstream is the API this stands in for, so
  this raises too -- with a message that says what was expected. This one is
  a decision rather than a gap.

And one difference that is deliberate: **`make_rf_shim` keeps the shape it is
given** where the toolbox reshapes the weights to a column. One weight per
channel, indexed `shim_vector[k]`, is what a Python caller expects; the
toolbox's `shim_vector[k, 0]` is MATLAB's column convention.

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

The last is in too. An RF or ADC offset is recorded twice over -- in hertz,
and as a shift in parts per million of the Larmor frequency -- and either can
be inside `system.max_freq_offset` while the two together are not, so all
three are weighed. A scanner that names no limit is asking nothing, which is
what upstream's `Opts` does: it carries no such field.

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

## Where the answer differs from the toolbox

The toolbox holds a gradient axis at zero in front of what it plays by putting
a knot a picosecond ahead of the axis's own first corner. Its rule for merging
coincident knots works to a nanosecond, so the corner behind that pad is lost
to it -- and the ramp behind the corner is a picosecond longer than the
sequence asked for. At full amplitude that is an area of amplitude times half
a picosecond, added every time an axis starts anywhere but the beginning of
the sequence.

Four parts in a hundred million of a phase encode, and it does not accumulate,
which is why nothing had noticed. It is not nothing: a phase encode and its
rewinder stop cancelling, and a sequence whose repetitions are identical stops
reading as though they are. Here the pad is placed only where a waveform
begins away from zero, which is where a step onto it is needed at all, so a
trapezoid encloses its area wherever it is played.

Two consequences are in the tests, and neither is skipped: where the toolbox
is wrong the expectation says what is right and why.

`tests/test_kspace.py` compares the trajectory as a set of moments rather than
a list, because the toolbox reports one raster tick this package does not --
the tick its own pad brings a sloping stretch back across -- and allows the
width of the leak where the two differ.

Two reference reports are decided by it outright. The four repetitions of
`inversion_recovery_train` have k-space centres agreeing to twelve figures,
and which is nearest the origin fixes the echo; the spacing after it is what
the report calls TR, and the train's spacings are 0.537, 0.557 and 0.607
seconds -- it has no one TR. The leak biases the first repetition, so the
toolbox settles on the second and answers 0.557; without it the four agree and
the first wins, 0.537. And `gre_with_noise_scan` resolves exactly 9.375 mm
along one axis, which is what this package computes to the bit: printed to two
places that is a tie and rounds to even, 9.38, where the toolbox's extent
carries the leak and comes out a hair under the tie, 9.37.

## Where a name differs from the toolbox

`waveforms_and_times` and its family take `block_range`, where
`pypulseq-matlab-like` spells it `blockRange`. Upstream PyPulseq has no such
parameter at all -- only `time_range` -- so there is no drop-in contract to
keep, and every other name here is snake_case.

`waveforms_and_times` and `rf_times` take `compat`, which is `True`: what they
return is what upstream returns, five values and four, because that is what a
script written against upstream unpacks. `compat=False` returns
`WaveformsAndTimes` and `RfTimes` instead, which are `pulserver`'s shapes and
carry three things the tuples cannot:

- *Every* RF use. Pulseq has seven; the tuple carries two, and an inversion, a
  saturation or a preparation pulse is not in it at all. `rf.of("inversion")`
  asks for one, `rf.of("excitation", "undefined")` reproduces upstream's
  bucket.
- The per-sample ADC phase and phase modulation -- the phase a sample is
  actually acquired with, which is what a simulation wants. The reference
  toolbox returns the modulation as a sixth value; upstream returns neither.
- Which block each pulse and each ADC window is in, and how many samples a
  window takes.

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

`evaluate_labels` is in, and it answers the toolbox's answer for every way it
can be asked -- at the end, at every block, at every block that acquires, at
every block that sets or increments one -- across the whole reference zoo. It
is a walk over the extension chains rather than over decoded blocks, so a
block carrying no label costs a column read, which is most of them.

One rule of the toolbox's is reproduced rather than improved on: asked for an
evolution that records fewer than two points, it hands back what the labels
finish at instead of an array. A sequence with no ADC asked for the `adc`
evolution therefore reports final values.

**Deferred.** Safety -- `calculate_pns`, `calculate_gradient_spectrum`,
`calc_rf_power` -- and plotting -- `plot`, `paper_plot`, `sound` -- and
`auto_label`.

## What the design layer is waiting on

The module toolbox is in, and `tests/test_design_*.py` say what is still owed
it. Each blocked test names the one thing it waits for:

- **`TransformFOV`**, which the fat-saturation module uses to place a band on
  an oblique slab. Ten tests skip.
- **A block's rotation extension read back from `get_block`**, so a module can
  ask whether a block turns. One test skips.
- **A host to drive a sequence from**, which the navigator's tests reach for
  and which belongs to `pulserver` rather than here. Six tests skip.
- **`tile`**, so a scan can write its averages out rather than leave them to
  an interpreter's repeat count, and **`Sequence.plot`**. Both are named where
  the sequence that would use them stands.
