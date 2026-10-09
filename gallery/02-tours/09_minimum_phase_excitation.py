r"""
========================
Minimum-phase excitation
========================

The shipped excitation modules design linear-phase SLR pulses, whose energy is
symmetric about the middle of the pulse. A minimum-phase design concentrates
RF energy near the end of the waveform, so that at a fixed duration and
time-bandwidth product the interval from the pulse to the echo is shorter. The
peak :math:`B_1` is larger, and the phase of the slice profile is not linear.

In this Tour you write an excitation module for the minimum-phase design, and
compare its envelope, slice profile and peak :math:`B_1` with the shipped
linear-phase module. You then play both in front of the same readout and read
the shortest echo time each reaches.

**Prerequisites:** lessons 1 and 10 of the :doc:`course </examples/course>`.
Lesson 10 writes a custom module, a T2 preparation, by subclassing
:class:`~pypulseqpp.sequences.RfModule`; this Tour uses the same interface.

The module concept, and the events a module publishes, are described in
:doc:`/explanations/sequence-modules`.
"""

# sphinx_gallery_start_ignore
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE_WIDTH = 7.8  # inches, the width of the documentation column
warnings.filterwarnings("ignore", message="Specified RF delay")


def design_figure(designs, thickness_m):
    """Pulse envelopes beside the slice profiles they produce."""
    figure, (envelope_axis, profile_axis) = plt.subplots(
        1, 2, figsize=(PAGE_WIDTH, 3.4), layout="constrained"
    )
    for label, entry in designs.items():
        envelope_axis.plot(1e3 * entry["time"], entry["envelope"], lw=1.2, label=label)
        profile_axis.plot(1e3 * entry["position"], entry["profile"], lw=1.2)
    envelope_axis.set_xlabel("time (ms)")
    envelope_axis.set_ylabel("$|B_1|$ (Hz)")
    figure.legend(*envelope_axis.get_legend_handles_labels(), loc="outside right upper")
    profile_axis.axvspan(
        -0.5e3 * thickness_m,
        0.5e3 * thickness_m,
        color="0.5",
        alpha=0.15,
        lw=0,
        zorder=0,
    )
    profile_axis.set_xlim(-10, 10)
    profile_axis.set_xlabel("position (mm)")
    profile_axis.set_ylabel(r"$|M_{xy}|$, normalised")
    profile_axis.set_title("nominal slice in grey")
    return figure


# sphinx_gallery_end_ignore

# %%
# The excitation module
# ---------------------
#
# The module follows the structure you wrote in lesson 10: ``init_module``
# builds ``self.seq`` and sets ``self.center``, and the events bound to its
# local variables are published under those names. Subclassing
# :class:`~pypulseqpp.sequences.RfModule` also gives the module
# :meth:`~pypulseqpp.sequences.RfModule.sim_rf`, which simulates its pulse
# against off-resonance.
#
# What differs is the pulse. :func:`~pypulseqpp.make_slr_pulse` designs it with
# ``filter_type="min"``. Its ``center_pos`` places the effective RF centre as a
# fraction of the duration, and that sets both the area of the rephaser the
# function returns and the instant a readout module measures its echo time
# from. A minimum-phase pulse is used at ``center_pos=1.0``, its own end.

import numpy as np

import pypulseqpp as pp
from pypulseqpp import sequences


class MinimumPhaseExcitation(sequences.RfModule):
    """Slice-selective excitation whose energy is concentrated at its end.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    flip_angle_deg : float
        Nominal flip angle (degrees).
    thickness_m : float
        Slice thickness (m).
    duration_s : float, optional
        Pulse duration (s).
    time_bw_product : float, optional
        Time-bandwidth product of the SLR sequences.
    center_pos : float, optional
        Effective centre of the pulse, as a fraction of its duration.
    axis : {'z', 'x', 'y'}, optional
        Selection axis.

    Attributes
    ----------
    rf : RfEvent
        The pulse.
    gz : GradEvent
        Its selection gradient.
    gz_reph : GradEvent
        The rephaser that unwinds the selection played after the effective
        centre.
    selection_amplitude : float
        Selection gradient amplitude (Hz/m).
    slice_thickness : float
        Thickness the pulse and its selection gradient produce (m): the
        pulse's measured bandwidth over the gradient amplitude.
    """

    def init_module(
        self,
        system: pp.Opts,
        flip_angle_deg: float,
        thickness_m: float,
        duration_s: float = 3e-3,
        *,
        time_bw_product: float = 4.0,
        center_pos: float = 1.0,
        axis: str = "z",
    ) -> None:
        rf, gz, gz_reph = pp.make_slr_pulse(
            np.deg2rad(flip_angle_deg),
            duration=duration_s,
            slice_thickness=thickness_m,
            time_bw_product=time_bw_product,
            filter_type="min",
            center_pos=center_pos,
            return_gz=True,
            use="excitation",
            system=system,
        )
        gz.channel = axis
        gz_reph.channel = axis

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf, gz)
        self.seq.add_block(gz_reph)

        self.center = float(rf.delay) + float(rf.center)
        self.selection_amplitude = float(gz.amplitude)
        self.slice_thickness = float(pp.calc_rf_bandwidth(rf) / abs(gz.amplitude))


# %%
# Linear and minimum phase
# ------------------------
#
# The two modules take the same flip angle, slice thickness and duration. The
# printout lists the events a module publishes and, for each design, where the
# effective RF centre falls and how large the rephaser is:

system = pp.Opts(
    max_grad=40.0,
    grad_unit="mT/m",
    max_slew=150.0,
    slew_unit="T/m/s",
    rf_dead_time=100e-6,
    rf_ringdown_time=30e-6,
    adc_dead_time=10e-6,
)

FLIP_ANGLE_DEG = 20.0
THICKNESS_M = 5e-3
DURATION_S = 3e-3

minimum_phase = MinimumPhaseExcitation(system, FLIP_ANGLE_DEG, THICKNESS_M, DURATION_S)
linear_phase = sequences.SpatialSelectiveExcitation(
    system, FLIP_ANGLE_DEG, THICKNESS_M, DURATION_S
)

print("events:", ", ".join(sorted(vars(minimum_phase.events))))
for name, module in (("linear", linear_phase), ("minimum", minimum_phase)):
    print(
        f"{name}-phase: center at {module.center * 1e3:.3f} ms of "
        f"{module.duration * 1e3:.3f} ms, rephaser area "
        f"{module.gz_reph.area:.1f} 1/m"
    )

# %%
# The rephaser compensates the slice-selection moment accumulated after the
# effective RF centre. At ``center_pos=1.0`` that moment is the fall ramp
# alone, so the rephaser block has its minimum duration and the pulse ends a
# gradient raster period or two before the encoding starts.
#
# Pulse envelope and slice profile
# --------------------------------
#
# ``sim_rf`` simulates the pulse across off-resonance; dividing the frequency
# by the selection amplitude reads the result as a position along the slice
# axis. The minimum-phase envelope rises towards the end of the pulse and its
# peak is several times that of the linear-phase pulse:

designs = {}
for name, module in (("linear phase", linear_phase), ("minimum phase", minimum_phase)):
    _, mz_xy, frequency = module.sim_rf()[:3]
    designs[name] = {
        "time": module.rf.t,
        "envelope": np.abs(module.rf.signal),
        "position": frequency / module.selection_amplitude,
        "profile": np.abs(mz_xy) / np.abs(mz_xy).max(),
    }
    print(
        f"{name}: peak B1 {np.abs(module.rf.signal).max():.0f} Hz, "
        f"slice {module.slice_thickness * 1e3:.2f} mm"
    )

# sphinx_gallery_start_ignore
design_figure(designs, THICKNESS_M)
# sphinx_gallery_end_ignore

# %%
# Echo time
# ---------
#
# A readout module takes the pulse, its selection gradient and its rephaser,
# and measures the echo time from the pulse's effective centre. Applying the
# same readout to each excitation isolates the difference in echo time that
# the excitation causes. ``te=None`` asks for the shortest echo time the
# readout reaches, and ``tr=None`` for the shortest repetition.

readouts = {
    name: sequences.LineReadout2D(
        system,
        module.rf,
        module.gz,
        module.gz_reph,
        fov=(220e-3, 220e-3),
        matrix=(192, 192),
        te=None,
        tr=None,
    )
    for name, module in (
        ("linear phase", linear_phase),
        ("minimum phase", minimum_phase),
    )
}
for name, readout in readouts.items():
    print(
        f"{name}: shortest TE {readout.echo_time * 1e3:.3f} ms, "
        f"repetition {readout.duration * 1e3:.3f} ms"
    )

# %%
# One repetition of the short-TE sequences.

seq = pp.Sequence(system=system)
for block in readouts["minimum phase"].blocks:
    seq.add_block(*block)
seq.paper_plot(tr=1)

# %%
# As a spec
# ---------
#
# What this Tour built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Write a pypulseqpp RfModule MinimumPhaseExcitation(system,
#    flip_angle_deg, thickness_m, duration_s=3e-3, time_bw_product=4,
#    center_pos=1.0, axis="z") that designs its pulse with
#    make_slr_pulse(filter_type="min", center_pos=center_pos,
#    return_gz=True, use="excitation"), plays the pulse with its
#    selection gradient and then the rephaser, sets center to the
#    effective RF centre, and reports the slice thickness from the
#    pulse bandwidth over the selection amplitude. For 40 mT/m and
#    150 T/m/s, 20 degrees, 5 mm and 3 ms, compare it with
#    sequences.SpatialSelectiveExcitation: the envelope, the simulated
#    slice profile against position, the peak B1, and the shortest echo
#    time of a 192x192 LineReadout2D (te=None, tr=None) behind each.
