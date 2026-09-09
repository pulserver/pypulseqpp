# What is left

`pypulseq-matlab-like` is the authority for the format, and the tests are now
written against it. Reading and writing are complete: every revision from
1.2.0 up is read, in text and in binary, and a file can be written as 1.5.1 or
as 1.4.1. `check_timing` judges every event time against the system's rasters
and dead times. What remains is here.

## The rest of the reference suite

`tests/` there is 6809 lines over 80 files, most of it covering event
factories this package does not have yet. The ones that bear on what is here,
to port as idiomatic pytest rather than `unittest.TestCase`:

`test_binary_signature`, `test_md5`, `test_compress_shape`,
`test_decompress_shape`, `test_block2events`, `test_block`, `test_opts`,
`test_get_supported_labels`, `test_add_custom_label`, `test_make_rf_shim`,
`test_rotation_extension`, `test_quaternion`, `test_aux_version`.

`test_read_write_binary_roundtrip` and `test_sequence_backwards_compatibility`
are already covered by `tests/test_corpus.py`, which runs over the same files.

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

## Safety

`pypulseqpp.safety` has `check_max_grad` and `check_max_slew`, the second
covering continuity. Both read the libraries rather than an expanded
waveform: a gradient is a normalised shape and one amplitude, so the steepest
step belongs to the shape and an instance's is that times its own amplitude.

One difference from a waveform-based answer is worth knowing. What is weighed
is the samples the sequence stores, and an interpreter draws between them --
where a waveform's samples sit at the centre of each raster interval, that
drawing can pass a little outside the outermost of them. Three parts in ten
thousand across the reference sequences. Making it exact means a second shape
statistic, the peak of the restored corners, which is computable once per
shape the same way the slew is; it is not there yet.

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
