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
thickness is off.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.spatial_saturations` | System, `(normal, position, thickness)` per band | List of `SpatialSaturation` | The bands a sequence function plays, those with a thickness. |

## Dummy repetitions

A sequence function whose `n_dummy` is `None` plays as many dummy repetitions
as bring its spoiled steady state within 1 % for a T1 of
`STEADY_STATE_T1_S`, from its repetition time and flip angle.

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.steady_state_dummies` | Repetition time (s), flip angle (deg), T1 (s), tolerance | Repetitions | The repetitions to play before acquiring. |
| {obj}`~pypulseqpp.sequences.STEADY_STATE_T1_S` | — | 1.5 s | T1 the dummies are counted for: grey matter at 3 T. |

## Fat saturation timing

| Object | Input | Returns | Purpose |
| --- | --- | --- | --- |
| {obj}`~pypulseqpp.sequences.fat_null_flip_deg` | Delay to the excitation (s), period (s), fat T1 (s) | Flip angle (deg) | The fat-saturation flip that leaves fat at zero at the excitation, in the steady state. |
| {obj}`~pypulseqpp.sequences.FAT_T1_S` | — | 0.37 s | T1 of fat at 3 T. |
