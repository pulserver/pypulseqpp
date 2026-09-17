# Echo-planar imaging

One excitation followed by a train of readout lobes of alternating polarity, so
the phase-encode axis is covered after a single pulse and off-resonance
accumulates along it.

| Example | What it covers |
| --- | --- |
| {doc}`/generated/gallery/15-epi/epi2D_sequence` | One blipped echo train per excitation. |
| {doc}`/generated/gallery/15-epi/epi3D_sequence` | One blipped echo train per (shot, shell), skipped-CAIPI when undersampled. |
| {doc}`/generated/gallery/15-epi/epi-segmentation-and-acceleration` | Echo train length against geometric distortion and volume acquisition time, over segmentation and in-plane acceleration. |

```{toctree}
:hidden:

/generated/gallery/15-epi/epi2D_sequence
/generated/gallery/15-epi/epi3D_sequence
/generated/gallery/15-epi/epi-segmentation-and-acceleration
```
