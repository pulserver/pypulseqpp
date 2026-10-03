# Pulseq representation

What a `.seq` file records, how it stores it, and the rasters its event times
are addressed on. These conventions appear throughout the Python interface, in
the units of amplitudes, the quantization of times and the distinction between
an event and the block that plays it. What the package adds above the format is
in {doc}`../design/index`.

| Concept | What it covers |
| --- | --- |
| {doc}`events-and-blocks` | What a block holds, what each kind of event carries, and how an extension is chained onto one. |
| {doc}`libraries-and-shapes` | How events and shapes are stored, deduplicated and signed, and what changes between format revisions. |
| {doc}`timing-and-rasterization` | The rasters an event time is addressed on, how a block duration follows from them, and what the timing check establishes. |

```{toctree}
:hidden:

events-and-blocks
libraries-and-shapes
timing-and-rasterization
```
