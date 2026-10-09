r"""
====================
Ramp-sampled readout
====================

The Cartesian readouts you have used so far sample on the flat top of the
readout lobe, so the area under the ramps is not sampled. Sampling through the
ramps as well covers the same extent of k-space in a shorter lobe, at the cost
of sample positions that are not evenly spaced. In this Tour you write a
Cartesian readout module that samples through the ramps, compare its duration
with the flat-top design at the same resolution, and measure the sample
spacing along the line from the k-space analysis.

**Prerequisites:** lessons 9 and 10 of the :doc:`course </examples/course>`.

The module interface, and the events a module publishes, are described in
:doc:`/explanations/sequence-modules`.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE_WIDTH = 7.8  # inches, the width of the documentation column


def sampling_figure(k_read, spacing, nyquist):
    """Sample positions along the read axis, and the spacing between them."""
    figure, (position_axis, spacing_axis) = plt.subplots(
        1, 2, figsize=(PAGE_WIDTH, 3.4)
    )
    position_axis.plot(k_read * 1e-3, lw=1.2)
    position_axis.set_xlabel("sample")
    position_axis.set_ylabel("$k_x$ (1/mm)")
    spacing_axis.plot(k_read[:-1] * 1e-3, spacing / nyquist, lw=1.2)
    spacing_axis.axhline(1.0, color="0.55", ls="--", lw=1.0)
    spacing_axis.set_xlabel("$k_x$ (1/mm)")
    spacing_axis.set_ylabel(r"$\Delta k_x$ / (1/FOV)")
    spacing_axis.set_title("Nyquist spacing dashed")
    spacing_axis.set_ylim(0.0, 1.15)
    figure.tight_layout()
    return figure


# sphinx_gallery_end_ignore

# %%
# Sampling through the ramps
# --------------------------
#
# A readout lobe of area :math:`N/\mathrm{FOV}` resolves an :math:`N`-point
# matrix over the field of view. A flat-top design samples only the plateau,
# which has to carry the whole area, so the lobe is as long as the plateau plus
# its two ramps. A ramp-sampled design puts the ADC window across the whole
# lobe: the plateau is shorter and the lobe with it, at the same gradient
# amplitude and slew rate.
#
# The ADC samples at a fixed dwell time. Where the gradient changes, equal steps
# in time are unequal steps in k, so the sample positions are denser on the
# ramps and the data need regridding before a Fourier transform.
#
# The readout module
# ------------------
#
# This is the same kind of class you wrote in lesson 10. ``init_module``
# assigns ``self.seq``, adds the blocks of the layout to it and sets
# :attr:`~pypulseqpp.sequences.SequenceModule.center`, which for a readout is
# the interval from the start of the module to the echo. Events bound to local
# variables are published under those names, so a scan loop reaches the phase
# encode as ``readout.gy_pre`` without the module returning anything.
#
# The prephaser has half the area of the whole lobe, ramps included, so the
# echo lands at the middle of the lobe rather than the middle of its flat top.
# The acquisition window is centred on the lobe, clears the ADC dead time at
# both ends, and is sampled at the requested rate.

import math

import numpy as np

import pypulseqpp as pp
from pypulseqpp import sequences


class RampSampledLineReadout(sequences.SequenceModule):
    """Cartesian line readout acquired across the whole readout lobe.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    rf : RfEvent
        The pulse that opens the repetition.
    gz : GradEvent, optional
        Its selection gradient, played in the same block.
    gz_reph : GradEvent, optional
        Its rephaser, played in the prewinder block.
    fov : float
        Isotropic in-plane field of view (m).
    matrix : int
        Isotropic in-plane matrix size.
    readout_bandwidth_hz : float, optional
        Requested ADC sampling rate (Hz). ``bandwidth_hz`` reports the rate
        the ADC raster admits.
    spoiling_cycles : float, optional
        Dephasing left on the read axis at the end of the repetition, in
        cycles across one voxel.

    Attributes
    ----------
    gx_pre : TrapEvent
        Readout prephaser, half the area of the lobe.
    gy_pre : TrapEvent
        Phase encode at its largest step, for the loop to scale.
    gx : TrapEvent
        Readout lobe, sampled from ramp to ramp.
    adc : AdcEvent
        The acquisition window, centred on the lobe.
    gx_spoil : TrapEvent
        Rewinds the second half of the line and adds the spoiling.
    gy_rew : TrapEvent
        The negated phase encode, scaled by the loop alongside ``gy_pre``.
    echo_time : float
        From the RF isodelay to the echo (s).
    center_sample : int
        Index of the sample at the echo.
    bandwidth_hz : float
        Achieved ADC sampling rate (Hz).
    """

    def init_module(
        self,
        system: pp.Opts,
        rf,
        gz=None,
        gz_reph=None,
        *,
        fov: float,
        matrix: int,
        readout_bandwidth_hz: float = 500e3,
        spoiling_cycles: float = 4.0,
    ) -> None:
        gx = pp.make_trapezoid("x", area=matrix / fov, system=system)
        span = pp.calc_duration(gx)

        dwell = (
            math.floor(1.0 / readout_bandwidth_hz / system.adc_raster_time)
            * system.adc_raster_time
        )
        # The window is centred on the lobe and clears the ADC dead time at
        # both ends, so the echo falls between two samples of it.
        divisor = int(system.adc_samples_divisor)
        usable = span - 2 * system.adc_dead_time
        n_samples = int(usable / dwell + 1e-9) // divisor * divisor
        adc = pp.make_adc(
            num_samples=n_samples,
            dwell=dwell,
            delay=pp.round_to_raster(
                0.5 * (span - n_samples * dwell), system.adc_raster_time
            ),
            system=system,
        )

        gx_pre = pp.make_trapezoid("x", area=-0.5 * gx.area, system=system)
        gy_pre = pp.make_trapezoid("y", area=0.5 * matrix / fov, system=system)
        gy_rew = pp.scale_grad(gy_pre, -1.0)
        gx_spoil = pp.make_trapezoid(
            "x", area=-0.5 * gx.area + spoiling_cycles * matrix / fov, system=system
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(*filter(None, (rf, gz)))
        self.seq.add_block(*filter(None, (gx_pre, gy_pre, gz_reph)))
        self.seq.add_block(gx, adc)
        self.seq.add_block(gx_spoil, gy_rew)

        rf_center = float(rf.delay) + float(rf.center)
        self.echo_time = (
            pp.calc_duration(*filter(None, (rf, gz)))
            - rf_center
            + pp.calc_duration(*filter(None, (gx_pre, gy_pre, gz_reph)))
            + 0.5 * span
        )
        self.center = rf_center + self.echo_time
        self.center_sample = round((0.5 * span - adc.delay) / dwell)
        self.bandwidth_hz = 1.0 / dwell


# %%
# Readout duration
# ----------------
#
# Both designs sample the same extent of k-space, so both resolve the same
# matrix over the same field of view. The flat-top design covers that extent
# on its plateau alone, and its ramps add duration without adding samples.
# Here is a 192-point line over 220 mm, at the same gradient amplitude in both
# lobes:

system = pp.Opts(
    max_grad=40.0,
    grad_unit="mT/m",
    max_slew=150.0,
    slew_unit="T/m/s",
    rf_dead_time=100e-6,
    rf_ringdown_time=30e-6,
    adc_dead_time=10e-6,
)

FOV = 220e-3
MATRIX = 192

ramp_sampled = pp.make_trapezoid("x", area=MATRIX / FOV, system=system)
flat_topped = pp.make_trapezoid(
    "x",
    amplitude=ramp_sampled.amplitude,
    flat_time=pp.round_to_raster(
        MATRIX / FOV / ramp_sampled.amplitude, system.grad_raster_time
    ),
    system=system,
)
for name, lobe in (("ramp-sampled", ramp_sampled), ("flat top only", flat_topped)):
    print(
        f"{name:14} lobe {pp.calc_duration(lobe) * 1e6:6.0f} us, "
        f"flat {lobe.flat_time * 1e6:6.0f} us, amplitude "
        f"{lobe.amplitude / system.gamma * 1e3:5.1f} mT/m"
    )

# %%
# One repetition
# --------------
#
# The module follows a slice-selective excitation, from which it takes the
# pulse, the selection gradient and the rephaser. The printout lists the
# events it publishes, its sample count and bandwidth, and its timing:

excitation = sequences.SpatialSelectiveExcitation(
    system, flip_angle_deg=12.0, thickness_m=5e-3, duration_s=3e-3
)
readout = RampSampledLineReadout(
    system,
    excitation.rf,
    excitation.gz,
    excitation.gz_reph,
    fov=FOV,
    matrix=MATRIX,
)

print("events:", ", ".join(sorted(vars(readout.events))))
print(
    f"{int(readout.adc.num_samples)} samples at {readout.bandwidth_hz * 1e-3:.0f} kHz, "
    f"echo at sample {readout.center_sample}"
)
print(
    f"TE {readout.echo_time * 1e3:.3f} ms over a {readout.duration * 1e3:.3f} ms "
    "repetition"
)

# %%
# Sample spacing along the line
# -----------------------------
#
# ``calculate_kspace`` is one of the analyses a module forwards to the sequence
# it built, so the sample positions come from the events themselves rather than
# from the design arithmetic. The spacing is finest on the ramps, where the
# gradient is weakest, and largest on the plateau, where it remains below the
# Nyquist spacing :math:`1/\mathrm{FOV}`. Consecutive samples at the start and
# at the end of the window, taken while the gradient is near zero, are almost
# coincident in k, so these edge samples are redundant. The nonuniform sample
# positions require regridding during reconstruction.

k_adc = readout.calculate_kspace()[0]
k_read = k_adc[0]
spacing = np.diff(k_read)
print(
    f"k spacing from {spacing.min():.2f} to {spacing.max():.2f} 1/m, "
    f"Nyquist {1 / FOV:.2f} 1/m"
)

# sphinx_gallery_start_ignore
sampling_figure(k_read, spacing, 1 / FOV)
# sphinx_gallery_end_ignore

# %%
# Scan loop
# ---------
#
# The loop scales the published phase encode per line and labels the
# acquisition; the rest of the layout is played as the module laid it out.

seq = pp.Sequence(system=system)
for line in range(MATRIX):
    ky = (line - MATRIX // 2) / (MATRIX / 2)
    seq.add_block(excitation.rf, excitation.gz, pp.make_label("LIN", "SET", line))
    seq.add_block(readout.gx_pre, pp.scale_grad(readout.gy_pre, ky), excitation.gz_reph)
    seq.add_block(readout.gx, readout.adc)
    seq.add_block(readout.gx_spoil, pp.scale_grad(readout.gy_rew, ky))

print(
    f"{seq.num_blocks} blocks, {seq.duration()[0]:.2f} s, timing {seq.check_timing()[0]}"
)

# %%
# The diagram shows one repetition.

seq.paper_plot(tr=1)

# %%
# As a spec
# ---------
#
# What this Tour built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Write a pypulseqpp SequenceModule RampSampledLineReadout(system, rf,
#    gz, gz_reph, fov, matrix, readout_bandwidth_hz=500e3,
#    spoiling_cycles=4): a Cartesian line readout whose readout lobe has
#    area matrix/fov and whose ADC window is centred on the whole lobe,
#    ramps included, at a dwell on the ADC raster and clear of the ADC
#    dead time. The prephaser has half the lobe's area; publish gx_pre,
#    gy_pre, gx, adc, gx_spoil and gy_rew, and set center to the echo.
#    For 40 mT/m, 150 T/m/s, 220 mm and 192 points, compare the lobe
#    with a flat-top lobe of the same amplitude, read the sample
#    spacing from calculate_kspace against 1/FOV, and play it over 192
#    phase encodes with a LIN label and check the timing.
