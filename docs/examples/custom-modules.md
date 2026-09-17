# Custom sequence modules

Implementing a module of your own against the base-class contract: what
`init_module` has to assign, how its events reach a scan loop, and how the
result is consumed by the readout modules and applications already in the
package.

| Example | What it covers |
| --- | --- |
| {doc}`/generated/gallery/40-custom-modules/custom-excitation-module` | A minimum-phase selective excitation, and the echo time its effective centre buys. |
| {doc}`/generated/gallery/40-custom-modules/custom-cartesian-readout` | A Cartesian line acquired across the whole readout lobe, and the non-uniform k-space spacing that follows. |
| {doc}`/generated/gallery/40-custom-modules/custom-noncartesian-readout` | A twisting radial arm designed as a k-space path, solved as an interleaf and played as a readout module. |

```{toctree}
:hidden:

/generated/gallery/40-custom-modules/custom-excitation-module
/generated/gallery/40-custom-modules/custom-cartesian-readout
/generated/gallery/40-custom-modules/custom-noncartesian-readout
```
