# Example sequences

Complete sequences shipped with the package, each a {class}`~pypulseqpp.sequences.SequenceApp`
subclass in its own module. Import one as an attribute of `pypulseqpp.sequences`;
the module is callable as its own `main`, which designs the sequence and returns
it.

```python
from pypulseqpp import sequences

seq = sequences.gre2D_sequence(n_x=128, n_y=128, n_slices=5)
seq.write("gre_2d.seq")
```

Every module is also a command-line entry point. Its options are derived from
the signature and NumPy-style `Parameters` section of `init_sequence`, so
`--help` lists the prescription the sequence accepts:

```bash
python -m pypulseqpp.sequences.sequence.gre2D_sequence --help
```

See {doc}`../api/apps` for the {class}`~pypulseqpp.sequences.SequenceApp`
contract these modules implement, and {doc}`../api/modules` for the excitation,
preparation and readout modules they are built from.


## Cartesian gradient echo

| Module | Description |
| --- | --- |
| `gre2D_sequence` | RF-spoiled, multi-slice 2D Cartesian gradient echo. |
| `gre3D_sequence` | RF-spoiled 3D Cartesian gradient echo. |
| `gre_multiecho2D_sequence` | RF-spoiled, multi-slice multi-echo 2D Cartesian gradient echo. |
| `gre_multiecho3D_sequence` | RF-spoiled multi-echo 3D Cartesian gradient echo. |
| `mprage3D_sequence` | 3D MPRAGE: one inversion per partition, then a train of spoiled low-flip lines. |


## Cartesian spin echo

| Module | Description |
| --- | --- |
| `se2D_sequence` | Multi-slice 2D Cartesian spin echo: one line per excitation. |
| `se3D_sequence` | 3D Cartesian spin echo: one ``(line, partition)`` view per excitation. |
| `fse3D_sequence` | 3D Cartesian fast spin echo: one CPMG train per excitation over a (ky, kz) grid. |


## Balanced SSFP

| Module | Description |
| --- | --- |
| `bssfp2D_sequence` | Balanced SSFP 2D Cartesian: one complete train per slice. |
| `bssfp3D_sequence` | Balanced SSFP, 3D Cartesian: a hard pulse and a balanced line readout per TR. |


## Echo planar

| Module | Description |
| --- | --- |
| `epi2D_sequence` | Multi-slice 2D gradient-echo EPI: single-shot or segmented, optionally multiband. |
| `epi3D_sequence` | 3D gradient-echo EPI: one train per ``(shot, shell)``, skipped-CAIPI sampled. |


## Radial

| Module | Description |
| --- | --- |
| `gre_radial2D_sequence` | RF-spoiled, multi-slice 2D radial gradient echo: one full spoke per repetition. |
| `se_radial2D_sequence` | Multi-slice 2D radial spin echo: one full spoke per excitation. |
| `gre_stack_of_stars3D_sequence` | RF-spoiled 3D stack of stars: radial spokes in-plane, Cartesian partitions along z. |
| `se_stack_of_stars3D_sequence` | 3D stack-of-stars spin echo: one spoke at one partition per excitation. |
| `mprage_stack_of_stars3D_sequence` | 3D MPRAGE on a stack of stars: one inversion per partition, then its spokes. |


## Spiral

| Module | Description |
| --- | --- |
| `gre_spiral2D_sequence` | RF-spoiled, multi-slice 2D spiral gradient echo: one interleaf per repetition. |
| `se_spiral2D_sequence` | Multi-slice 2D spiral spin echo: one interleaf per excitation. |
| `gre_stack_of_spirals3D_sequence` | RF-spoiled 3D stack of spirals: spiral interleaves in-plane, Cartesian partitions along z. |
| `se_stack_of_spirals3D_sequence` | 3D stack-of-spirals spin echo: one interleaf at one partition per excitation. |
| `mprage_stack_of_spirals3D_sequence` | 3D MPRAGE on a stack of spirals: one inversion per partition, then its interleaves. |


## PROPELLER

| Module | Description |
| --- | --- |
| `gre_propeller2D_sequence` | RF-spoiled, multi-slice 2D PROPELLER gradient echo: one blade line per repetition. |
| `se_propeller2D_sequence` | Multi-slice 2D PROPELLER spin echo: one blade line per excitation. |
| `se_epi_propeller2D_sequence` | Multi-slice 2D PROPELLER spin echo: one EPI blade per excitation. |
| `gre_stack_of_blades3D_sequence` | RF-spoiled 3D stack of blades: PROPELLER blades in-plane, Cartesian partitions along z. |
| `se_stack_of_blades3D_sequence` | 3D stack-of-blades spin echo: one blade line at one partition per excitation. |


## Zero echo time

| Module | Description |
| --- | --- |
| `zte3D_sequence` | 3D zero echo time: hard pulses on a readout gradient held on across each shell. |

