# Migrating the reference toolbox: what is done and what is left

`pypulseq-matlab-like` is the authority for the file format. The tests were
written against upstream `pypulseq`, which is a different toolbox, and moving
them found real defects. This file is the state of that move.

## Done

- The reference corpus is vendored at `tests/seq/` and `tests/test_corpus.py`
  is the reader's test of record: 39 files, revisions 1.2.0 to 1.5.1, written
  by MATLAB and by two Python toolboxes. Every 1.5.x file reads and writes
  back identically and survives the binary form.
- Three writer divergences the corpus caught: the RF header said `freqPPm`
  where the authority says `freqPPM`, the `use` comment was one line of
  double-quoted text rather than two of single-quoted, and rotation rows were
  padded to twelve columns rather than written with `%g`.
- `required_revision` no longer returns 2. There is no revision 1.5.2; 1.5.1
  is the newest there is.
- `RequiredExtensions ROTATIONS` is declared when the sequence rotates.
- The test fixtures build against `pypulseq_matlab_like`, and `reference.py`
  now covers rotations and RF shims through its own `make_rotation` and
  `make_rf_shim` rather than through hand-built events.

## Left, in the order that matters

1. **The builtin label table is wrong, and it changes what a file means.**
   Ours is upstream's 22 labels; the authority has 23. `OFF` is missing
   entirely and `TRID` sits at the end instead of at index 11, so every label
   from index 11 on resolves to the wrong name: a file that says `NOISE`
   reads back here as `NOROT`. Fix `builtin_labels()` in `sequence.cpp` to
   the authority's order:

       SLC SEG REP AVG SET ECO PHS LIN PAR ACQ TRID NAV REV SMS REF IMA OFF
       NOISE PMC NOROT NOPOS NOSCL ONCE

   This is in code that predates the IO branch, so it affects the merged
   package and `structural-fork` too.

2. **The soft delay row.** The authority writes `1 0 1e+06 1 TE`; we write
   `1 0 1000000 1 1.0`. The offset wants `%g` rather than `%.0f`, and the
   hint is arriving as the factor, which is a column mapping error in
   `tests/convert.py` for that library's layout.

3. **A block goes missing.** `gre_with_noise_scan` writes 132 blocks there
   and 131 here, with the numbering diverging from block 5. Something in the
   conversion drops one.

4. **The `[SIGNATURE]` comment block.** The authority wraps it over five
   lines at about eighty columns; ours is three long lines.

5. **The LABELINC comment.** "for increasing labels" and `# id inc
   labelstring`, against our "for setting labels" for both sections.

6. **Reading a file older than 1.5.0.** The corpus has one sequence at
   1.2.0, 1.3.0, 1.3.1, 1.4.0, 1.4.1 and 1.4.2 beside its 1.5.0 form, which
   is exactly the fixture for the conversion. `read_seq.py` in the reference
   toolbox is the specification: the RF `center` comes from
   `calc_rf_center`, and a gradient's `first` and `last` from walking the
   block table.

7. **The 1.4.1 writer**, from that toolbox's `write_v141`.

8. **The rest of its test suite.** `tests/` there is 6809 lines over 80
   files. Most of it covers event factories this package does not have yet;
   the IO-related ones are `test_read_write_binary_roundtrip`,
   `test_binary_signature`, `test_md5`, `test_compress_shape`,
   `test_decompress_shape`, `test_sequence_backwards_compatibility`,
   `test_make_rf_shim`, `test_rotation_extension` and `test_add_custom_label`.

## Two places we deliberately differ, both visible in `test_parity.py`

The authority stamps its own version on every file it writes and records
`TotalDuration` in `test_report` rather than in `write`. This package writes
the oldest revision that can read the file back, and records the duration on
write. Keeping the declared revision is what lets a 1.5.0 file read here and
write back unchanged, which the corpus holds over 33 files; the reference
suite makes the same allowance by letting a file match either its source or
its own canonical rewrite. Both are one-line changes if you would rather
match exactly.
