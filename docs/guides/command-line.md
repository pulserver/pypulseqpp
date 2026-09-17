# Run a sequence from the command line

Every sequence implementation the package ships is a command-line entry point
as well as an importable module. This guide covers both, and how a sequence of
your own acquires the same interface.

## Run a shipped sequence

Each module under `pypulseqpp.sequences.sequence` exposes `main`, and is
callable as that `main`. Its options are derived from the signature and the
NumPy-style `Parameters` section of `init_sequence`, so the prescription is
documented once and presented identically in Python, on the command line and to
a protocol editor.

```bash
python -m pypulseqpp.sequences.sequence.gre2D_sequence --help
python -m pypulseqpp.sequences.sequence.gre2D_sequence \
    --n-x 128 --n-y 128 --n-slices 5 --output gre_2d.seq
```

Three options are added beyond the prescription: `--output`, `--report` and
`--binary`, together with `--max-grad-mtm` and `--max-slew-tm-s`, which lower
the gradient ceilings the implementation declares.

The same sequence from Python:

```python
>>> from pypulseqpp import sequences
>>> seq = sequences.gre2D_sequence(n_x=64, n_y=32, n_slices=1, tr=None)
>>> seq.num_blocks
288

```

{doc}`../sequences` lists the implementations and what each one acquires.

## Write the file

{func}`~pypulseqpp.cli.write_sequence` deduplicates the finished sequence and
writes it, either as signed Pulseq text or in the binary form.

```python
>>> import tempfile, pathlib
>>> from pypulseqpp import cli
>>> with tempfile.TemporaryDirectory() as folder:
...     path = pathlib.Path(folder, "gre_2d.seq")
...     written = cli.write_sequence(seq, path)
...     size = path.stat().st_size
>>> size > 0
True

```

An implementation with prescans writes a chain instead:
{meth}`~pypulseqpp.sequences.SequenceApp.write` writes each prescan as its own
file ahead of the main one, linked through `NextSequence`, as
{doc}`../explanations/design/sequence-application` describes.

## Give your own sequence the same interface

A {class}`~pypulseqpp.sequences.SequenceApp` subclass acquires the command line
by exposing its `main` at module level. {doc}`custom-module` builds such a
subclass; the module defining it ends with:

```python
main = InversionRecoveryGre.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="ir_gre_2d.seq"))
```

{func}`~pypulseqpp.cli.run` parses the arguments, builds and designs the
application, and writes the result. Because the flags are derived from
`init_sequence`, a parameter added there appears on the command line with its
docstring as help text and requires no change here.

## Next steps

* {doc}`../api/cli` — `run` and `write_sequence`.
* {doc}`../sequences` — the shipped sequence implementations.
