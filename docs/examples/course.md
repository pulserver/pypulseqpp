# Course

Ten lessons, read in order. You start from a 2D Cartesian gradient echo and
arrive at a 3D radial MPRAGE, then check it, make it a function of its
protocol, rebuild it from sequence modules and write a module of your own.

| Lesson | What you learn |
| --- | --- |
| {doc}`/generated/gallery/01-course/01_first_gradient_echo` | Events, blocks and rasters; designing every event before the loop; the `.seq` file. |
| {doc}`/generated/gallery/01-course/02_echo_and_repetition_time` | Setting TE and TR with delays, measuring them, and asymmetric readouts. |
| {doc}`/generated/gallery/01-course/03_spoiling` | Gradient and RF spoiling, and the steady state each one leaves. |
| {doc}`/generated/gallery/01-course/04_labels_and_metadata` | Several slices, labels on every readout, and the definitions a reconstruction reads. |
| {doc}`/generated/gallery/01-course/05_radial_sampling` | Radial spokes, the golden angle, and merging gradients to shorten the TR. |
| {doc}`/generated/gallery/01-course/06_radial_mprage` | An inversion-prepared 3D stack of stars, and the contrast along its train. |
| {doc}`/generated/gallery/01-course/07_hardware_and_safety_checks` | Gradient, slew-rate, nerve stimulation, mechanical resonance and SAR checks. |
| {doc}`/generated/gallery/01-course/08_sequence_functions` | A sequence as a function of the scanner and the protocol, on the command line. |
| {doc}`/generated/gallery/01-course/09_sequence_modules` | The same MPRAGE built from the shipped sequence modules. |
| {doc}`/generated/gallery/01-course/10_custom_sequence_module` | A T2 preparation written as a module and swapped in for the inversion. |

```{toctree}
:hidden:

/generated/gallery/01-course/01_first_gradient_echo
/generated/gallery/01-course/02_echo_and_repetition_time
/generated/gallery/01-course/03_spoiling
/generated/gallery/01-course/04_labels_and_metadata
/generated/gallery/01-course/05_radial_sampling
/generated/gallery/01-course/06_radial_mprage
/generated/gallery/01-course/07_hardware_and_safety_checks
/generated/gallery/01-course/08_sequence_functions
/generated/gallery/01-course/09_sequence_modules
/generated/gallery/01-course/10_custom_sequence_module
```
