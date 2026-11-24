# Sequence functions

A sequence function is a function `sequence(system, **protocol)` that returns a
{class}`~pypulseqpp.Sequence`, or a list of them with the prescans first and the
main sequence last. The objects below read its protocol, label its blocks,
write what it returns, time it, and build the excitation a volumetric
prescription chooses. The sequences themselves are listed in
{doc}`../sequences`; the structure of a sequence function, its protocol and its
prescan chain are described in
{doc}`../explanations/sequence-modules`.

```{eval-rst}
.. currentmodule:: pypulseqpp.sequences
```

## Sequence functions

The protocol of a sequence function is its keyword parameters after `system`.
`parameters` reads each one's type and default from the signature, and its unit,
choices and description from the NumPy-style Parameters section.
{func}`pypulseqpp.cli.run` derives the command-line options from the same
signature.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.parameters` | Sequence function | `dict` of `ProtocolParameter`, in signature order | Protocol of a sequence function. |
| {obj}`~pypulseqpp.sequences.ProtocolParameter` | Returned by `parameters` | Frozen record | One protocol parameter: its type, default, unit, choices and description. |
| {obj}`~pypulseqpp.sequences.Labels` | The new value of each label, by Pulseq name | Label events to add to the block | Writer of the label events a block changes. |
| {obj}`~pypulseqpp.sequences.write` | Path, a sequence or a list of sequences | Written paths, in play order | Write a sequence, or a chain of linked sequences, as Pulseq files. |
| {obj}`~pypulseqpp.sequences.duration` | A sequence or a list of sequences | Duration (s) | Return the time a sequence, or a chain, plays. |

## Excitation choice

A volumetric sequence function offers its excitation as one protocol parameter,
whose choices are `EXCITATIONS`, and builds the module with `make_excitation`
from one slab prescription.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.EXCITATIONS` | — | `("nonselective", "slab", "spsp")` | The excitation kinds `make_excitation` builds. |
| {obj}`~pypulseqpp.sequences.make_excitation` | System, kind, flip angle (deg), slab thickness (m) | Excitation module | Non-selective, slab-selective or water-selective spectral-spatial excitation of one slab. |

## Spatial saturation bands

A sequence function offers each spatial saturation band as `sat<k>_normal_x`,
`_y`, `_z`, `sat<k>_position` and `sat<k>_thickness`, in the physical frame,
and builds the bands it plays with `spatial_saturations`; a band of zero
thickness is off. `gre2D_sequence` and `se2D_sequence` offer six, `k` from 1
to 6, as many as a console places, played in that order before every
excitation.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.spatial_saturations` | System, `(normal, position, thickness)` per band | List of `SpatialSaturation` | The bands a sequence function plays, those with a thickness. |

## Dummy repetitions

A sequence function whose `n_dummy` is `None` plays as many dummy repetitions
as bring its spoiled steady-state signal within 1 % for a T1 of
`STEADY_STATE_T1_S` at `STEADY_STATE_B1` of its flip angle, from its
repetition time; a balanced
one counts for `STEADY_STATE_BALANCED_T1_S` and `STEADY_STATE_T2_S`.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.steady_state_dummies` | Repetition time (s), flip angle (deg), T1 (s), T2 (s) for a balanced steady state, tolerance | Repetitions | The repetitions to play before acquiring. |
| {obj}`~pypulseqpp.sequences.STEADY_STATE_T1_S` | — | 4.4 s | T1 a spoiled steady state's dummies are counted for: CSF at 7 T. |
| {obj}`~pypulseqpp.sequences.STEADY_STATE_B1` | — | 0.5 | Fraction of the flip a spoiled steady state is counted at: the 7 T transmit field in the temporal lobes and cerebellum. |
| {obj}`~pypulseqpp.sequences.STEADY_STATE_BALANCED_T1_S` | — | 1.5 s | T1 a balanced steady state's dummies are counted for: grey matter at 3 T. |

## Receive-gain prescan

A receive-gain prescan plays a sequence from its first block and stops after
the readouts its `NumGainCalibrationReadouts` definition states. Every
sequence function with a readout states the readouts through the end of its
first imaging repetition, the dummies before it included, so the gain is set
on the signal the scan acquires.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.gain_calibration_readouts` | Sequence, imaging repetitions | Readouts | The `NumGainCalibrationReadouts` of whole repetitions. |

## Diffusion weighting

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.diffusion_pair_b_value` | Diffusion lobe, gap between the lobes (s), raster (s) | b-value (s/mm²) | The b-value of a Stejskal–Tanner pair of lobes around a refocusing pulse. |

## Fat saturation timing

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.fat_null_flip_deg` | Delay to the excitation (s), period (s), fat T1 (s) | Flip angle (deg) | The fat-saturation flip that leaves fat at zero at the excitation, in the steady state. |
| {obj}`~pypulseqpp.sequences.fat_null_inversion_time` | Period (s), fat T1 (s) | Inversion time (s) | The time after a fat inversion at which fat crosses zero, in the steady state. |
| {obj}`~pypulseqpp.sequences.FAT_T1_S` | — | 0.37 s | T1 of fat at 3 T. |
| {obj}`~pypulseqpp.sequences.FAT_SAT_BANDWIDTH_PPM` | — | 2 ppm | Default passband of the fat saturation, as a span of the field. |
| {obj}`~pypulseqpp.sequences.FAT_INVERSION_BANDWIDTH_PPM` | — | 2.2 ppm | Default inverted band of the fat inversion, as a span of the field. |
