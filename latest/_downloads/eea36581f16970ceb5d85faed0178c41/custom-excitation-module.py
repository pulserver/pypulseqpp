"""
=================================
A minimum-phase excitation module
=================================

The shipped excitation modules design linear-phase SLR pulses, whose energy is
symmetric about the middle of the pulse. A minimum-phase design concentrates
the energy at the end instead, which shortens the interval between the pulse
and the echo at the same duration and time-bandwidth product, at the cost of a
higher peak :math:`B_1` and a slice-profile phase that is no longer linear.

This example implements that design as a module against the
:class:`~pypulseqpp.sequences.RfModule` contract, and measures what it buys.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE_WIDTH = 8.6  # inches, the width of the documentation column

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "figure.figsize": (PAGE_WIDTH, 3.4),
        "savefig.dpi": 110,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)


def design_figure(designs, thickness_m):
    """Pulse envelopes beside the slice profiles they produce."""
    figure, (envelope_axis, profile_axis) = plt.subplots(
        1, 2, figsize=(PAGE_WIDTH, 3.4)
    )
    for label, entry in designs.items():
        envelope_axis.plot(1e3 * entry["time"], entry["envelope"], lw=1.2, label=label)
        profile_axis.plot(1e3 * entry["position"], entry["profile"], lw=1.2)
    envelope_axis.set_xlabel("time (ms)")
    envelope_axis.set_ylabel("$|B_1|$ (Hz)")
    envelope_axis.legend(frameon=False, fontsize=9)
    profile_axis.axvspan(
        -0.5e3 * thickness_m, 0.5e3 * thickness_m, color="0.9", lw=0, zorder=0
    )
    profile_axis.set_xlim(-10, 10)
    profile_axis.set_xlabel("position (mm)")
    profile_axis.set_ylabel(r"$|M_{xy}|$, normalised")
    profile_axis.set_title("nominal slice in grey")
    figure.tight_layout()
    return figure


# sphinx_gallery_end_ignore

# %%
# Required interface
# -------------------
#
# A module implements ``init_module``: it assigns ``self.seq``, adds the blocks
# of its layout to it, and sets :attr:`~pypulseqpp.sequences.SequenceModule.center`,
# the timing reference the rest of the sequence is placed against. Events
# bound to local variables of ``init_module`` are published under those names,
# so nothing is returned.
#
# Subclassing :class:`~pypulseqpp.sequences.RfModule` rather than
# :class:`~pypulseqpp.sequences.SequenceModule` adds
# :meth:`~pypulseqpp.sequences.RfModule.sim_rf`, which simulates the module's
# pulse against off-resonance.
#
# ``center_pos`` is what makes the design a short-TE one: it places the
# effective centre of the pulse, which sets both the rephasing area
# :func:`~pypulseqpp.make_slr_pulse` returns and the instant a readout module
# measures its echo time from. A minimum-phase pulse is used at
# ``center_pos=1.0``, its own end.

import numpy as np

import pypulseqpp as pp
import pypulseqpp.sequences as design


class MinimumPhaseExcitation(design.RfModule):
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
        Time-bandwidth product of the SLR design.
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
# Published events
# ----------------

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
linear_phase = design.SpatialSelectiveExcitation(
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
# The rephaser carries the selection area played after the effective centre.
# At ``center_pos=1.0`` that is the fall ramp alone, so the rephaser block
# collapses to its shortest and the pulse ends a gradient raster or two before
# the encoding starts.
#
# Pulse envelope and slice profile
# --------------------------------
#
# ``sim_rf`` simulates the pulse across off-resonance; dividing by the
# selection amplitude reads the result as a position.

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

design_figure(designs, THICKNESS_M)

# %%
# Echo time
# ---------
#
# A readout module takes the pulse, its selection gradient and its rephaser,
# and measures the echo time from the pulse's effective centre. Handing it each
# excitation in turn prices the design in echo time.

readouts = {
    name: design.LineReadout2D(
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
# One repetition of the short-TE design.

seq = pp.Sequence(system=system)
for block in readouts["minimum phase"].blocks:
    seq.add_block(*block)
seq.paper_plot(tr=1)
