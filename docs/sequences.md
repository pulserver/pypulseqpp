# Sequence catalogue

Complete sequences shipped with the package, grouped by sequence family. Each
is a {class}`~pypulseqpp.sequences.SequenceApp` subclass in its own module, and
each has a reference page giving its prescription and a representative
configuration. Import one as an attribute of `pypulseqpp.sequences`; the module
is callable as its own `main`, which designs the sequence and returns it.

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

Dimensionality and sampling distinguish the variants within a family: a
gradient echo is the same family whether it samples a Cartesian grid, radial
spokes or spiral interleaves. See {doc}`api/apps` for the
{class}`~pypulseqpp.sequences.SequenceApp` contract these modules implement,
and {doc}`api/modules` for the excitation, preparation and readout modules they
are built from.

## Gradient echo

One excitation per repetition with no refocusing pulse, spoiled between
repetitions.

```{eval-rst}
.. include:: generated/sequences/tables/gradient-echo.rst
```

## Spin echo

One excitation and one refocusing pulse per repetition, with the acquisition at
the refocused echo.

```{eval-rst}
.. include:: generated/sequences/tables/spin-echo.rst
```

## Fast spin echo

One excitation followed by a CPMG train of refocusing pulses, with one view
acquired per echo.

```{eval-rst}
.. include:: generated/sequences/tables/fast-spin-echo.rst
```

## MPRAGE

An inversion followed by a spoiled gradient-echo train. Each shot inverts once
and then acquires every sampled in-plane view of a single partition, so the
partition encode is constant within a shot and the number of shots is the
number of sampled partitions.

```{eval-rst}
.. include:: generated/sequences/tables/mprage.rst
```

## Balanced SSFP

Every gradient axis returns to zero moment within each repetition, so the
steady state depends on the off-resonance accumulated over one repetition time.

```{eval-rst}
.. include:: generated/sequences/tables/balanced-ssfp.rst
```

## Echo-planar imaging

One excitation followed by a train of readout lobes of alternating polarity,
with phase-encode blips between them.

```{eval-rst}
.. include:: generated/sequences/tables/echo-planar-imaging.rst
```

## Zero echo time

The readout gradient is at amplitude before the pulse is transmitted, so
acquisition begins without a ramp and the trajectory starts at the centre of
k-space.

```{eval-rst}
.. include:: generated/sequences/tables/zero-echo-time.rst
```

```{eval-rst}
.. include:: generated/sequences/index.rst
```
