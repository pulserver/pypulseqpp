# Custom modules

Modules the package does not ship, written against the base-class
contract, and measured against the shipped design each of them departs
from.

| Page | What it establishes |
| --- | --- |
| {doc}`/generated/gallery/07-custom-modules/01_excitation_module` | A minimum-phase SLR excitation: a shorter interval from the pulse to the echo, at a larger peak transmit amplitude. |
| {doc}`/generated/gallery/07-custom-modules/02_cartesian_readout` | A readout sampled through the ramps: the same k-space extent in a shorter lobe, at sampling locations that are not evenly spaced. |
| {doc}`/generated/gallery/07-custom-modules/03_noncartesian_readout` | A twisting radial arm, stated as a k-space path and solved into a waveform under the gradient limits. |

```{toctree}
:hidden:

/generated/gallery/07-custom-modules/01_excitation_module
/generated/gallery/07-custom-modules/02_cartesian_readout
/generated/gallery/07-custom-modules/03_noncartesian_readout
```
