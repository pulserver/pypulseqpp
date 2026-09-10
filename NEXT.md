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
  into absolute offsets, omits soft-delay extensions with a warning, and
  refuses rotations and RF shims.
- Binary shapes use float32; other numeric fields preserve their specified
  binary representation. Both text and binary support optional MD5 signatures.
- Custom labels need no registration by users. Binary IDs beyond the builtin
  table resolve through the ordered `CustomLabels` definition.
- `add_block(None)` raises, matching upstream PyPulseq.
  `make_rf_shim` preserves the supplied weight-array shape.
- `check_timing` records `TotalDuration` if absent and checks an existing
  value against the blocks. Writing does not independently add it.

## Storage and structural analysis

Stored blocks are snapshots: editing an event returned by `get_block` does
not change the sequence. Block-table views own a shared buffer and remain
valid after later mutations or destruction of the sequence.

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

FOV translation uses logical coordinates in metres. The unbroken gradient
integral supplies RF and ADC shift phase; excitation/reset-aware k-space
supplies the ADC echo reference. These are distinct state vectors and both
must be carried between consecutive processing ranges.

ADC phase is anchored to the nearest k-space approach shared by the
block/ADC definition within the selected range, not the midpoint of the
sampling window. Varying gradients require residual phase modulation.
RF phase shapes store cycles; ADC modulation and event phase offsets use
radians.

Prescription rotation is composed after a block's existing rotation and
stored as an extension. Logical trajectories returned by `TransformFOV`
do not apply those rotations. Exemption labels are sticky within the selected
range; their state is not inherited from preceding blocks.

## Gradient checks

Amplitude and slew checks use physical axes after block rotations and report
both per-axis peaks and simultaneous vector magnitudes. Within-block slew and
boundary continuity are separate checks. Boundary jumps are judged over one
gradient raster interval, and the final gradients must return to zero.

These checks do not cover SAR, PNS or mechanical resonance.
