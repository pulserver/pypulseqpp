# Examples

Executable pages, run when the documentation is built, so every figure and
printed number is produced by the code as it stands. Concepts are in
{doc}`/explanations/index`, interfaces in {doc}`/api/index`, and the
prescriptions of the shipped sequences in the {doc}`catalogue </sequences>`.

The Course is the shortest coherent path that gives a new user the framework's
core mental model and enough practical competence to work independently. Tours
are useful applications, advanced branches or specialised workflows that are
not necessary for that core competence.

## Course

Ten lessons, read in order, from a 2D Cartesian gradient echo to a 3D radial
MPRAGE built from sequence modules.

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

## Tours

Standalone examples that build on the Course; each names the lessons it
assumes.

| Tour | What you see |
| --- | --- |
| {doc}`/generated/gallery/02-tours/01_fid_and_spin_echo` | A free induction decay and a spin echo, and the shortest echo time a crusher pair admits. |
| {doc}`/generated/gallery/02-tours/02_slice_selective_excitation` | How time-bandwidth product and duration shape the slice profile, the selection gradient and the peak B1. |
| {doc}`/generated/gallery/02-tours/03_epi` | A single-shot echo planar readout, its blurring along the train and the ghost a gradient delay produces. |
| {doc}`/generated/gallery/02-tours/04_multi_echo` | A multi-echo gradient echo, and how many echoes fit in the repetition time. |
| {doc}`/generated/gallery/02-tours/05_segmented_epi` | Echo planar imaging over several shots, trading scan time for geometric fidelity. |
| {doc}`/generated/gallery/02-tours/06_spiral` | Which of the amplitude, slew-rate and receiver limits sets the length of a spiral arm. |
| {doc}`/generated/gallery/02-tours/07_ramp_sampled_readout` | A readout module sampled through its ramps. |
| {doc}`/generated/gallery/02-tours/08_twisting_radial_readout` | A twisting radial arm, from its k-space path to a waveform within the gradient limits. |
| {doc}`/generated/gallery/02-tours/09_minimum_phase_excitation` | A minimum-phase excitation module, and the echo time it saves. |

## Sequence catalogue

| Section | Covers |
| --- | --- |
| {doc}`/examples/built-in-sequences/index` | Each complete sequence at a representative prescription: diagram, sampling and acquisition order. |

```{toctree}
:hidden:

/examples/course
/examples/tours
/examples/built-in-sequences/index
```
