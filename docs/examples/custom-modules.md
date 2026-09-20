# Custom sequence modules

Implementing a module of your own against the base-class contract: what
`init_module` has to assign, how its events reach a scan loop, and how the
result is consumed by the readout modules and applications already in the
package.

| Example | Scope |
| --- | --- |
| {doc}`/generated/gallery/40-custom-modules/custom-excitation-module` | Minimum-phase selective excitation and its effect on echo time. |
| {doc}`/generated/gallery/40-custom-modules/custom-cartesian-readout` | A Cartesian line acquired across the whole readout lobe, and the non-uniform k-space spacing that follows. |
| {doc}`/generated/gallery/40-custom-modules/custom-noncartesian-readout` | A twisting radial k-space path solved under gradient constraints and implemented as a readout module. |

```{toctree}
:hidden:

/generated/gallery/40-custom-modules/custom-excitation-module
/generated/gallery/40-custom-modules/custom-cartesian-readout
/generated/gallery/40-custom-modules/custom-noncartesian-readout
```
