# Check a sequence against hardware constraints

Apply the gradient-amplitude, slew-rate, gradient-continuity, PNS, mechanical-
resonance and SAR checks to a completed sequence. The criteria and limitations
are described in {doc}`../explanations/safety/index`. These design-time
estimates do not constitute a scanner or patient-safety assessment.

```python
>>> from pypulseqpp import safety, sequences
>>> seq = sequences.epi2D_sequence(
...     n_x=64, n_y=64, n_slices=3, fat_saturation=True, tr=None, n_dummy=0
... )

```

Every check returns a Boolean verdict and a report. Reports include the peak
quantity and limit regardless of the verdict.

## Gradient amplitude, slew rate and continuity

These three require nothing beyond the sequence and its system limits.

```python
>>> for check in (safety.check_max_grad, safety.check_max_slew,
...               safety.check_grad_continuity):
...     print(check.__name__, check(seq)[0])
check_max_grad True
check_max_slew True
check_grad_continuity True

```

Amplitudes in a report are in Hz/m and slew rates in Hz/m/s. Convert with the
gyromagnetic ratio on the system limits.

```python
>>> is_ok, report = safety.check_max_grad(seq)
>>> to_mt_per_m = 1e3 / seq.system.gamma
>>> f"{report.per_axis.value * to_mt_per_m:.1f} mT/m on {report.per_axis.axis}"
'39.7 mT/m on z'
>>> f"limit {report.limit * to_mt_per_m:.1f} mT/m"
'limit 40.0 mT/m'

```

The `vector` entry is the largest simultaneous magnitude over the three
physical axes. It is reported but not compared with a limit, and it is not the
root-sum-square of the three axis peaks.

### Checking against different limits

A check takes the limits it compares against from the sequence, or from an
{class}`~pypulseqpp.Opts` passed to it. Re-answering the question for the
system a sequence will actually be played on is the same call with a second
argument.

```python
>>> import pypulseqpp as pp
>>> derated = pp.apply_system_derates(seq.system, grad_derate=0.7, slew_derate=0.7)
>>> safety.check_max_slew(seq, derated)[0]
False

```

{func}`~pypulseqpp.apply_system_derates` scales the limits by a fraction and
{func}`~pypulseqpp.cap_system` lowers them to a stated ceiling; both return a
copy and retain the base limits, so repeated derating does not compound.

## Peripheral nerve stimulation

{func}`~pypulseqpp.safety.check_pns` needs a nerve model, which is site and coil
data no sequence file contains. Supply a SAFE description — upstream PyPulseq's
example hardware is used here — or state a rheobase–chronaxie model directly.

```python
>>> from pypulseq.utils.safe_pns_prediction import safe_example_hw
>>> is_ok, report = safety.check_pns(seq, safe_example_hw())
>>> f"{100 * report.peak.value:.0f}% of threshold at block {report.peak.block}"
'148% of threshold at block 150'
>>> [f"{axis.axis}: {100 * axis.value:.0f}%" for axis in report.axes]
['x: 83%', 'y: 108%', 'z: 74%']

```

The response is evaluated for an axial prescription unless a rotation is given.
Because a SAFE description has different coefficients per axis, an oblique
prescription changes the result; pass the prescription rotation for any scan
that will not be prescribed axially.

```python
>>> from scipy.spatial.transform import Rotation
>>> rotation = Rotation.from_euler("x", 30.0, degrees=True).as_matrix()
>>> f"{100 * safety.check_pns(seq, safe_example_hw(), rotation=rotation)[1].peak.value:.0f}%"
'126%'

```

## Mechanical resonance

{func}`~pypulseqpp.safety.check_mech_resonance` needs a forbidden-band table.
Bands are stated directly as {class}`~pypulseqpp.safety.ForbiddenBand` entries,
or read from a vendor table with
{func}`~pypulseqpp.safety.read_forbidden_bands`.

```python
>>> bands = [
...     safety.ForbiddenBand(axis=None, f_min=530.0, f_max=590.0, tolerance=2.0),
...     safety.ForbiddenBand(axis=None, f_min=1090.0, f_max=1180.0, tolerance=0.0),
... ]
>>> is_ok, report = safety.check_mech_resonance(seq, bands, min_threshold=10.0)
>>> is_ok
False
>>> for band in report.bands:
...     print(f"{band.f_min:.0f}-{band.f_max:.0f} Hz:"
...           f" peak {band.peak:.2f} mT/m on {band.peak_axis},"
...           f" threshold {band.threshold:.1f}, {band.violations} violations")
530-590 Hz: peak 2.47 mT/m on x, threshold 2.0, 5 violations
1090-1180 Hz: peak 1.11 mT/m on z, threshold 10.0, 0 violations

```

A band whose table states a tolerance is judged against it; a band that states
none is judged against `min_threshold`.

## Specific absorption rate

{func}`~pypulseqpp.safety.check_sar` needs a virtual-observation-point model
and the drive calibration that converts the file's Hz amplitudes to the unit
the model was computed in. {func}`~pypulseqpp.safety.example_vops` supplies a
synthetic eight-channel model for demonstration only; a measured model is read
with {func}`~pypulseqpp.safety.read_vops`.

```python
>>> example = safety.example_vops()
>>> is_ok, report = safety.check_sar(
...     seq,
...     example.model,
...     drive_per_hz=example.drive_per_hz,
...     default_shim=example.cp_shim,
... )
>>> f"worst local {report.worst_local.sar:.2f} W/kg (limit {report.local_limit:.1f})"
'worst local 0.96 W/kg (limit 10.0)'
>>> f"worst global {report.worst_global.sar:.2f} W/kg (limit {report.global_limit:.1f})"
'worst global 0.51 W/kg (limit 3.2)'

```

Absolute SAR is only as good as the calibration behind it. Passing `reference=`
a second sequence evaluated under the same model and calibration reports ratios
instead, in which the scale of the drive and of the VOPs cancels.

## Responses to a failing check

A sequence over a threshold is redesigned, not re-checked. The gradient
constraints are relieved by lowering the amplitude or lengthening a ramp; the
stimulation estimate additionally by lengthening the echo spacing or splitting
a single-shot train into more shots; SAR by lowering the flip angles or
lengthening the repetition time. {doc}`../explanations/safety/index` states
which quantity each check bounds, and
{doc}`../explanations/design/sequence-application` shows how a gradient ceiling
is imposed on a sequence implementation from outside its prescription.
