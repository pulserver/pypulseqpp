# Gradient continuity

Adjacent Pulseq blocks have no implicit gap. A change between the physical-axis
endpoint of one block and the initial amplitude of the next must satisfy the
one-raster slew criterion

$$
|\Delta G| \leq \texttt{max\_slew}\,
                   \texttt{grad\_raster\_time}.
$$

```{figure} ../../generated/figures/continuity_seam.png
A boundary step below the one-raster limit passes; a larger step is reported.
The dashed line marks the block boundary and the red arrow marks $\Delta G$.
```

{func}`~pypulseqpp.safety.check_grad_continuity` evaluates every boundary and
reports the 1-based block index, physical axis, endpoint amplitudes, and implied
slew rate. The final physical gradient amplitude must also be zero; otherwise
`ends_at_zero` and the overall verdict are false.

Block rotations are applied before the comparison. Identical logical endpoint
vectors under different rotations can therefore be discontinuous on the
physical axes. Returning every interleaf to zero avoids this dependence.

## See also

* {func}`~pypulseqpp.safety.check_grad_continuity` — check and report fields.
* {doc}`slew_rate` — within-block slew-rate evaluation.
