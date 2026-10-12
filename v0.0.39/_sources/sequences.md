# Sequence catalogue

Shipped sequences by family, each a sequence function with a reference page.
Executed designs are under {doc}`examples/built-in-sequences/index`; the module
contract is in {doc}`api/sequence-functions`.

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
