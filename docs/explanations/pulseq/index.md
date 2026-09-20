# Pulseq representation

A `.seq` file is a portable description of acquisition events, channel
assignments and timing. It is the object `pypulseqpp` builds,
analyses and writes, and its conventions are visible throughout the Python
interface — in the units amplitudes are reported in, in the rasters event times
are quantized to, and in the distinction between an event and the block that
plays it.

What the package adds on top of the representation is in
{doc}`../design/index`.

{doc}`events-and-blocks`
: The block as the unit of playout, the event kinds and their fields, the
  event libraries, gyromagnetic-ratio-free units, and the extension chain.

{doc}`libraries-and-shapes`
: How events are stored as definitions and instances, what deduplication
  merges, why a rotation extension keeps a non-Cartesian shape library small,
  the definition metadata and signature, and what each file revision can
  express.

{doc}`timing-and-rasterization`
: The four rasters, the relationship between block duration and event extent,
  the coupling between the ADC and gradient rasters that bounds receiver
  bandwidth, and what `check_timing` establishes.

```{toctree}
:hidden:

events-and-blocks
libraries-and-shapes
timing-and-rasterization
```
