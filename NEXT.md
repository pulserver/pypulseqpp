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

`check_timing` covers what the authority's `check_timing` module covers. Its
`Sequence.check_timing`, the transcription of MATLAB's `checkTiming`, asks
three further questions, none of which is a raster question:

- **Gradient continuity.** A waveform starting at a non-zero value must
  continue one that ended there on the same axis in the previous non-empty
  block, and must not also carry a delay; one ending at a non-zero value must
  last to the end of its block; and the last block must ramp to zero. The
  library already carries the `first` and `last` of every arbitrary gradient,
  so this is a scan over the block table rather than a decode.
- **Frequency offsets.** An RF or ADC offset, in Hz or as a ppm shift through
  gamma, must stay inside `system.max_freq_offset`.
- **TotalDuration.** A `TotalDuration` already in `[DEFINITIONS]` must equal
  what the blocks add up to. MATLAB rewrites it as a side effect of checking;
  a check that writes to the sequence it is judging is a surprise, so if this
  is wanted it should be split into a check and a `set_definition` the caller
  makes.

## The rest of `Sequence`

The one-line methods, the aliases and the no-ops are in. What is left, by
what it would take:

**Open.**

- `detect_rf_use` when reading a file older than 1.5.0. The flag is accepted
  and warned about; honouring it means guessing what a pulse is for from its
  flip angle, the way the toolbox does.
- `install`, `sound`, `test_report` and `test_report_dict`, which nobody has
  ruled on yet.

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
`calc_rf_power` -- and plotting and analysis -- `plot`, `paper_plot`,
`calculate_kspace`, `auto_label`, `evaluate_labels`, `get_gradients`,
`waveforms`, `waveforms_and_times`, `adc_times`, `rf_times`.
