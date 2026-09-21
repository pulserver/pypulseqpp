# Pulseq representation

A `.seq` file is a portable description of acquisition events, channel
assignments and timing. It is the object `pypulseqpp` builds,
analyses and writes, and its conventions are visible throughout the Python
interface — in the units amplitudes are reported in, in the rasters event times
are quantized to, and in the distinction between an event and the block that
plays it.

What the package adds on top of the representation is in
{doc}`../design/index`.

| Concept | Scope |
| --- | --- |
| {doc}`events-and-blocks` | Blocks, event fields, libraries, units and extension chains. |
| {doc}`libraries-and-shapes` | Definitions, instances, shape deduplication, metadata, signatures and format revisions. |
| {doc}`timing-and-rasterization` | Event rasters, block duration, ADC–gradient raster coupling and timing checks. |

```{toctree}
:hidden:

events-and-blocks
libraries-and-shapes
timing-and-rasterization
```
