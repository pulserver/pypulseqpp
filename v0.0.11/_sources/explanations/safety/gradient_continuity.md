# Gradient continuity

```{admonition} TL;DR
:class: tldr

- Adjacent Pulseq blocks have no implicit gap, so the change on each axis
  between the endpoint of one block and the initial amplitude of the next,
  after the blocks' rotations, must satisfy the one-raster slew criterion
  $|\Delta G| \leq \mathtt{max\_slew}\cdot\mathtt{grad\_raster\_time}$.
- {func}`~pypulseqpp.safety.check_grad_continuity` evaluates every boundary and
  reports the 1-based block index, axis, endpoint amplitudes and implied slew
  rate.
- The final gradient amplitude on every axis must be zero; otherwise
  `ends_at_zero` and the overall verdict are false.
- Each block's rotation is applied before the comparison, so endpoints equal
  on the channel axes of two blocks with different rotations can differ after
  them. Returning every interleaf to zero avoids this dependence.
```

Adjacent Pulseq blocks have no implicit gap. A change on each axis between the
endpoint of one block and the initial amplitude of the next, after the blocks'
rotations, must satisfy the one-raster slew criterion

$$
|\Delta G| \leq \mathtt{max\_slew} \cdot
                   \mathtt{grad\_raster\_time}.
$$

```{figure} ../../generated/figures/continuity_seam.png
A boundary step below the one-raster limit passes; a larger step is reported.
The dashed line marks the block boundary and the red arrow marks $\Delta G$.
```

{func}`~pypulseqpp.safety.check_grad_continuity` evaluates every boundary and
reports the 1-based block index, axis, endpoint amplitudes, and implied slew
rate. The final gradient amplitude on every axis must also be zero; otherwise
`ends_at_zero` and the overall verdict are false.

Each block's rotation is applied before the comparison. Endpoints equal on the
channel axes of two blocks with different rotations can therefore differ after
them. Returning every interleaf to zero avoids this dependence.

## See also

* {func}`~pypulseqpp.safety.check_grad_continuity` — check and report fields.
* {doc}`slew_rate` — within-block slew-rate evaluation.
