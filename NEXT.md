# What is left

`pypulseq-matlab-like` is the authority for the format, and the tests are now
written against it. Reading is complete: every revision from 1.2.0 up is read,
in text and in binary. What remains is here.

## 1. The 1.4.1 writer

`write_v141` in the reference toolbox, which folds the ppm offsets back into
absolute hertz with `1e-6 * gamma * B0`, drops `center`, `first` and `last`,
and refuses soft delays.

## 2. The rest of the reference suite

`tests/` there is 6809 lines over 80 files, most of it covering event
factories this package does not have yet. The ones that bear on what is here,
to port as idiomatic pytest rather than `unittest.TestCase`:

`test_binary_signature`, `test_md5`, `test_compress_shape`,
`test_decompress_shape`, `test_block2events`, `test_block`, `test_opts`,
`test_get_supported_labels`, `test_add_custom_label`, `test_make_rf_shim`,
`test_rotation_extension`, `test_quaternion`, `test_aux_version`.

`test_read_write_binary_roundtrip` and `test_sequence_backwards_compatibility`
are already covered by `tests/test_corpus.py`, which runs over the same files.
