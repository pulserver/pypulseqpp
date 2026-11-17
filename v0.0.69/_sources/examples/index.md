# Examples

Executed when the documentation is built: every figure and number comes from
the code on the page. Concepts are in
{doc}`/explanations/index`, interfaces in {doc}`/api/index`, and the
prescriptions of the shipped sequences in the {doc}`catalogue </sequences>`.

## Course

Ten lessons, read in order, from a 2D Cartesian gradient echo to a 3D radial
MPRAGE built from sequence modules.

| Lesson | Contents |
| --- | --- |
| {doc}`/generated/gallery/01-course/01_first_gradient_echo` | Events, blocks, rasters and the `.seq` file |
| {doc}`/generated/gallery/01-course/02_echo_and_repetition_time` | TE and TR set with delays; asymmetric readouts |
| {doc}`/generated/gallery/01-course/03_spoiling` | Gradient and RF spoiling and the resulting steady state |
| {doc}`/generated/gallery/01-course/04_labels_and_metadata` | Multislice loops, labels and file definitions for reconstruction |
| {doc}`/generated/gallery/01-course/05_radial_sampling` | Radial spokes, golden-angle ordering and merged gradients |
| {doc}`/generated/gallery/01-course/06_radial_mprage` | Inversion-prepared 3D stack of stars and contrast along the train |
| {doc}`/generated/gallery/01-course/07_hardware_and_safety_checks` | Gradient, slew-rate, PNS, mechanical-resonance and SAR checks |
| {doc}`/generated/gallery/01-course/08_sequence_functions` | Sequence functions, protocol parameters and the command line |
| {doc}`/generated/gallery/01-course/09_sequence_modules` | MPRAGE rebuilt from the shipped sequence modules |
| {doc}`/generated/gallery/01-course/10_custom_sequence_module` | Custom T2-preparation module |

## Tours

Standalone examples beyond the Course. Each names the lessons it assumes.

| Tour | Contents |
| --- | --- |
| {doc}`/generated/gallery/02-tours/01_fid_and_spin_echo` | FID, spin echo and the minimum TE with crushers |
| {doc}`/generated/gallery/02-tours/02_slice_selective_excitation` | Slice profile and peak B1 versus time-bandwidth product and duration |
| {doc}`/generated/gallery/02-tours/03_epi` | Single-shot EPI: blurring along the echo train and Nyquist ghosting from gradient delays |
| {doc}`/generated/gallery/02-tours/04_multi_echo` | Multi-echo gradient echo and the number of echoes per TR |
| {doc}`/generated/gallery/02-tours/05_segmented_epi` | Segmented EPI: scan time versus geometric distortion |
| {doc}`/generated/gallery/02-tours/06_spiral` | Spiral arm length under gradient, slew-rate and receiver limits |
| {doc}`/generated/gallery/02-tours/07_ramp_sampled_readout` | Ramp-sampled readout module |
| {doc}`/generated/gallery/02-tours/08_twisting_radial_readout` | Twisting radial arm: k-space path and gradient waveform |
| {doc}`/generated/gallery/02-tours/09_minimum_phase_excitation` | Minimum-phase excitation and the shorter TE |

## Sequence catalogue

| Section | Contents |
| --- | --- |
| {doc}`/examples/built-in-sequences/index` | Every shipped sequence at a representative prescription: diagram, sampling and acquisition order |

```{toctree}
:hidden:

/examples/course
/examples/tours
/examples/built-in-sequences/index
```
