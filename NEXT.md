# Reading a file older than 1.5.0, and what is left after it

`pypulseq-matlab-like` is the authority for the format, and the tests are now
written against it. What that move found and fixed is in the git history; what
remains is here.

## 1. Reading a file older than 1.4.0

1.4 is read now. Below it, two more things move: a gradient carries no time
shape column at all, and a block's duration is an index into a `[DELAYS]`
section rather than a count of rasters, so the duration has to be recovered
from the events the block plays. `simple_mprage120.seq`, `130` and `131` are
the fixtures, against `150` as before. There is also a trapezoid fix-up in
the reference reader for that era: a zero-amplitude trap with no rise gets one
grad raster moved out of its flat time.

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
