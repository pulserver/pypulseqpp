# Roadmap and compatibility notes

## Deferred work

Unimplemented areas include audio output, gradient-spectrum and PNS analysis,
RF-power analysis, automatic labelling and sequence tiling. The implemented
`Sequence` methods are listed in the API reference.

Scanner-host integration belongs to Pulserver. Tests requiring that host
remain skipped; implementing an execution stream is outside this package.

The promoted RF-shim column is available for an RF materialisation path.
Block decoding currently reads extension chains; gradient expansion uses
the promoted rotation column.

## Format and API compatibility

`pypulseq-matlab-like` is the file-format reference, while upstream PyPulseq
provides the Python API and event factories. Tests compare text bytes,
binary round trips and numerical results; format and API compatibility are
separate contracts.

- Text files from Pulseq 1.2 onward are converted to the internal 1.5 model.
  Conversion derives missing RF centres, gradient endpoints and durations.
  Missing RF uses remain undefined unless inference is requested.
- Default writers produce Pulseq 1.5.1. The 1.4.1 writer folds ppm offsets
  into absolute offsets, at the gamma and B0 of the sequence's system unless
  others are given, as the reference writer does; omits soft-delay extensions
  with a warning; and refuses rotations and RF shims.
- Binary shapes use float32; other numeric fields preserve their specified
  binary representation. Both text and binary support optional MD5 signatures.
- Custom labels need no registration by users. Binary IDs beyond the builtin
  table resolve through the ordered `CustomLabels` definition.
- A block carries at most one rotation and one RF shim, as in the reference
  toolbox, and `add_block` refuses a second. A file whose chain holds two,
  which the reference refuses to decode, is read with the first of each: the
  one the waveforms, k-space and checks play.
- `Sequence.read` and `pypulseqpp.io.read` also take a binary file object,
  such as `io.BytesIO` over a file's contents; upstream's `read` takes a path.
- `add_block(None)` raises, matching upstream PyPulseq.
  `make_rf_shim` preserves the supplied weight-array shape.
- `check_timing` records `TotalDuration` if absent and checks an existing
  value against the blocks. Writing does not independently add it.
- `make_slr_pulse` (and its alias `make_sigpy_pulse`) records the magnitude
  peak of the designed waveform as the pulse's centre, as `calc_rf_center`
  finds it, unless `center_pos` is given; upstream's `make_sigpy_pulse`
  records the midpoint. The two differ for the `'se'` and `'inv'` designs and
  for minimum- and maximum-phase filters. `make_ptx_pulse` takes the peak of
  the channels' root-sum-square magnitude.
- `calc_rf_bandwidth` returns upstream's values by default. `compat=False`
  returns an `RfBandwidth` that adds the bands of a multiband pulse and labels
  each spectral bin with its own frequency; upstream's axis, which the
  default return keeps, reads one bin low.

## Renamed sampling routines (breaking)

The sampling routines were renamed; the former names are removed.

| Former name | New name | Argument change |
| --- | --- | --- |
| `calc_sampled_lines` | `make_cartesian_axis_sampling` | `r` is `acceleration`. |
| `calc_sampled_pairs` | `make_cartesian_plane_sampling` | `shuffling=True` is `sampling='poisson'`. |
| `calc_traversal_order` | `make_traversal_order` | None. |
| `calc_epi_order` | `make_epi_shot_offsets` | None. |

The echo-train orderings (`make_linear_order`, `make_centric_order`,
`make_radial_order`, `make_radial_adaptive_order`, `make_shuffling_order`)
take centred coordinates with the k-space centre at the origin. Their
`center` argument, boolean-mask input and the integer-count form of
`make_linear_order` are removed.

## Storage and structural analysis

Stored blocks are snapshots: editing an event returned by `get_block` does
not change the sequence. Block-table views own a shared buffer and remain
valid after later mutations or destruction of the sequence.

`Sequence.libraries` exports the block table and every library as the tables
a text file of the sequence holds, under the sequence's own ids, with times in
seconds. Its `layout` field is 1 and rises when a table gains, loses or
reorders a column or a column changes unit, so a consumer such as a scanner
converter can refuse a layout it was not written for. Labels are exported by
name, because a label id indexes only the sequence's own table.

Event definitions separate reusable timing/shape information from per-playout
values. RF shapes belong to definitions; gradient waveform shapes belong to
instances. ADC and extension choices do not distinguish block definitions.
Pure delays share one definition regardless of duration; triggers and digital
outputs are not pure delays.

Deduplication rebuilds definitions after renumbering shapes and events.
Definition IDs must not be retained across it. Repetition detection uses this
definition stream; matching definitions do not imply identical amplitudes,
waveforms, ADC state or extensions.

## Trajectories and FOV transforms

Gradient corners represent piecewise-linear waveforms. Analytic integration
gives the same ADC coordinates whether the full trajectory grid is requested
or `samples_only=True`.

Gradient timing padding and floating-point ties can differ from the reference
toolbox. Tests compare trajectory moments and explicitly cover the resulting
echo-selection and report-rounding differences; do not loosen them solely
to force reference agreement.

FOV translation is in metres along the channel axes, or along the logical
axes with `through_rotation`. The unbroken gradient integral supplies RF and
ADC shift phase; excitation/reset-aware k-space supplies the ADC echo
reference. These are distinct state vectors and both must be carried between
consecutive processing ranges.

ADC phase is anchored to the nearest k-space approach shared by the
block/ADC definition within the selected range, not the midpoint of the
sampling window. Varying gradients require residual phase modulation.
RF phase shapes store cycles; ADC modulation and event phase offsets use
radians.

Prescription rotation is composed after a block's existing rotation and
stored as an extension, after the translation of the same transform.
Trajectories returned by `TransformFOV` are on the channel axes and do not
apply those rotations. Exemption labels are sticky within the selected range;
their state is not inherited from preceding blocks.

An improper prescription M (determinant -1) has no quaternion. The reference
toolbox plays one only by rotating the waveforms; here it is the rotation
M diag(1, 1, -1) stored as an extension, after the gradient on channel z is
negated and each block's own rotation is conjugated by diag(1, 1, -1), which
plays M R_b g. The negation is part of the rotation step, after the
translation, so the frequency and phase offsets of the translation are those
it gives without the prescription.

A block's rotation extension is taken, by default, for a prescription the
translation turns with: the block is moved along its channel axes, by the
gradients the file stores. The reference toolbox, which by default rotates the
waveforms first, moves it by the gradients it plays.
`through_rotation=True` does the same without new waveforms, and is what a
design whose rotations are its own, the spokes of a radial or ZTE readout,
needs to be moved as one object.

## Gradient checks

Amplitude and slew checks use the axes after each block's rotation and report
both per-axis peaks and simultaneous vector magnitudes. Within-block slew and
boundary continuity are separate checks. Boundary jumps are judged over one
gradient raster interval, and the final gradients must return to zero.

These checks do not cover SAR, PNS or mechanical resonance.
