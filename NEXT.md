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
