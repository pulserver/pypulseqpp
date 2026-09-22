# Reading and writing files

Reading a `.seq` file into a sequence that carries the system it describes,
and writing one out in either form.

```{eval-rst}
.. currentmodule:: pypulseqpp.io
```

A `.seq` file records the rasters its events are addressed on and nothing
about the limits the sequence was designed against.
{meth}`~pypulseqpp.Sequence.read` therefore takes the rasters from the file
and leaves the sequence on whatever system it was constructed with, which is
the shared default unless the caller supplied one. The limits of that system
are what the design helpers solve against and what the checks in
{doc}`safety` compare a waveform with, so {func}`read` builds the system from
the file: the rasters as recorded, the limits and field strength the file
states if it states any, and otherwise a gradient limit large enough for the
waveforms the file holds.

{func}`write` is the other direction, and chooses between the text and the
binary form.

| Object | Description |
| --- | --- |
| {obj}`~pypulseqpp.io.read` | Read a Pulseq file into a sequence whose system describes what is in it. |
| {obj}`~pypulseqpp.io.write` | Write a sequence as Pulseq text or binary. |
