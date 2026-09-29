# Fast spin echo

A CPMG refocusing train acquires one Cartesian view per echo. Refocusing-angle
modulation and view ordering jointly determine the k-space weighting.

| Example | Acquisition |
| --- | --- |
| {doc}`/generated/gallery/13-fast-spin-echo/fse3D_sequence` | Fixed optimized train with radial view ordering and its k-space modulation. |
| {doc}`/generated/gallery/13-fast-spin-echo/fse3D_adaptive` | Individually optimized train length, TR, and adaptive radial ordering. |
| {doc}`/generated/gallery/13-fast-spin-echo/fse3D_shuffling` | Variable-density shuffled sampling for echo-resolved reconstruction. |

```{toctree}
:hidden:

/generated/gallery/13-fast-spin-echo/fse3D_sequence
/generated/gallery/13-fast-spin-echo/fse3D_adaptive
/generated/gallery/13-fast-spin-echo/fse3D_shuffling
```
