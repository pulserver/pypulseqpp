# Reading a file older than 1.5.0, and what is left after it

`pypulseq-matlab-like` is the authority for the format, and the tests are now
written against it. What that move found and fixed is in the git history; what
remains is here.

## 1. Reading a file older than 1.5.0

The corpus has one sequence written at 1.2.0, 1.3.0, 1.3.1, 1.4.0, 1.4.1 and
1.4.2 beside its 1.5.0 form, which is the fixture: what the 1.5.0 file says is
what the others have to be read as. `tests/test_corpus.py` currently asserts
that an older file is refused by name, and those assertions become the
conversion's tests once it exists.

`read_seq.py` in the reference toolbox is the specification. Three things are
missing from an older file and none of them can be defaulted:

- an RF pulse's `center`, which comes from `calc_rf_center` over the
  decompressed magnitude;
- an arbitrary gradient's `first` and `last`, which come from walking the
  block table: `first` is the previous gradient's `last` on that axis, reset
  to zero where there is a delay or a gap, and `last` is the waveform's last
  sample for an extended trapezoid or a linear extrapolation otherwise;
- the ppm frequency and phase terms, which are zero.

1.3 and older also carry no gradient `time_shape_id`, and their `[BLOCKS]`
duration column is a delay id rather than a raster count, so the duration has
to be recovered from the events.

## 2. The 1.4.1 writer

`write_v141` in the reference toolbox, which folds the ppm offsets back into
absolute hertz with `1e-6 * gamma * B0`, drops `center`, `first` and `last`,
and refuses soft delays.

## 3. The rest of the reference suite

`tests/` there is 6809 lines over 80 files, most of it covering event
factories this package does not have yet. The ones that bear on what is here,
to port as idiomatic pytest rather than `unittest.TestCase`:

`test_binary_signature`, `test_md5`, `test_compress_shape`,
`test_decompress_shape`, `test_block2events`, `test_block`, `test_opts`,
`test_get_supported_labels`, `test_add_custom_label`, `test_make_rf_shim`,
`test_rotation_extension`, `test_quaternion`, `test_aux_version`.

`test_read_write_binary_roundtrip` and `test_sequence_backwards_compatibility`
are already covered by `tests/test_corpus.py`, which runs over the same files.
