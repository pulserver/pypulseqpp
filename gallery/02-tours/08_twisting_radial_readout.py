r"""
=======================
Twisting radial readout
=======================

A radial spoke samples the centre of k-space far more densely than the
periphery: at radius :math:`k`, adjacent spokes of an :math:`N`-interleaf set
are :math:`2\pi k / N` apart, which exceeds the Nyquist spacing
:math:`1/\mathrm{FOV}` beyond a transition radius

.. math::

    k_t = \frac{N}{2\pi\,\mathrm{FOV}}.

A twisting radial line [JNM92]_ departs from the spoke beyond that radius and
accumulates azimuth with radius, so that the perpendicular distance between
neighbouring interleaves stays at the Nyquist spacing.

In this Tour you state one interleaf as a path in k-space, let the
time-optimal solver turn it into a gradient waveform under the amplitude and
slew limits, and wrap it in a readout module. You compare its readout
duration with that of a constant-density spiral of the same coverage, and
rotate the solved interleaf once per shot in a scan loop.

**Prerequisites:** lessons 5 and 10 of the :doc:`course </examples/course>`.

The module interface, and the events a module publishes, are described in
:doc:`/explanations/sequence-modules`.
"""

# %%
# The k-space path
# ----------------
#
# The interleaf is a polyline in k-space: its samples set the geometry and
# nothing else. :class:`~pypulseqpp.sequences.Arbitrary` passes it to the
# time-optimal solver, which assigns the timing under the amplitude and slew
# limits and builds the gradient events, the acquisition window and the
# rewinder back to k = 0. Along the radius, the azimuth advances by the amount
# that keeps neighbouring interleaves one Nyquist spacing apart.

import numpy as np
from scipy.integrate import cumulative_trapezoid

import pypulseqpp as pp
from pypulseqpp import sequences


def twirl_path(fov: float, matrix: int, interleaves: int, samples: int = 2048):
    """Return the ``(samples, 2)`` k-space path of one interleaf, in 1/m."""
    radius = np.linspace(0.0, matrix / (2 * fov), samples)
    transition = interleaves / (2 * np.pi * fov)
    twisting = radius > transition
    slope = np.zeros_like(radius)
    slope[twisting] = (
        np.sqrt((2 * np.pi * fov * radius[twisting] / interleaves) ** 2 - 1.0)
        / radius[twisting]
    )
    angle = cumulative_trapezoid(slope, radius, initial=0.0)
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])


# %%
# The interleaf
# -------------
#
# ``Arbitrary`` stretches the readout to hold the samples it is given, so the
# sample count has to match the interleaf rather than the matrix. The
# time-optimal duration is not known until the path is solved, so it is solved
# once with two samples to measure that duration, and once more with the number
# of samples the duration holds at the requested rate.


def twirl_interleaf(
    system: pp.Opts,
    fov: float,
    matrix: int,
    interleaves: int,
    *,
    readout_bandwidth_hz: float = 250e3,
):
    """Solve one twisting radial interleaf and fill it with samples."""
    path = twirl_path(fov, matrix, interleaves)
    probe = sequences.Arbitrary(
        system, path, matrix=2, bandwidth_hz_px=readout_bandwidth_hz
    )
    samples = int(probe.read_duration * readout_bandwidth_hz)
    return sequences.Arbitrary(
        system, path, matrix=samples, bandwidth_hz_px=readout_bandwidth_hz
    )


# %%
# The readout module
# ------------------
#
# :class:`~pypulseqpp.sequences.NonCartesianReadout` plays any two-channel
# interleaf: it places the prewinder, the acquisition and the rewinder against
# the pulse it is given, solves the echo time and the repetition time, and adds
# the spoiler. A family that designs its own interleaf subclasses it, builds
# the trajectory in ``init_module`` and forwards the rest, which is how the
# shipped spiral and rosette readouts are written.


class TwirlReadout2D(sequences.NonCartesianReadout):
    """One twisting radial interleaf in a plane.

    Parameters
    ----------
    fov : float
        Isotropic in-plane field of view (m).
    matrix : int
        In-plane matrix size.
    interleaves : int
        Number of interleaves that sets the pitch and the transition
        radius. The loop may acquire any number of rotated copies.
    readout_bandwidth_hz : float, optional
        Requested ADC sampling rate (Hz).
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
        interleaves: int,
        readout_bandwidth_hz: float = 250e3,
        **kwargs,
    ) -> None:
        trajectory = twirl_interleaf(
            system,
            fov,
            matrix,
            interleaves,
            readout_bandwidth_hz=readout_bandwidth_hz,
        )
        super().init_module(system, rf, gz, gz_reph, trajectory=trajectory, **kwargs)


# %%
# Readout duration against a spiral
# ---------------------------------
#
# A constant-density spiral designed for the same interleaf count samples the
# same field of view at the same resolution with a longer readout: the
# twisting interleaf crosses the centre of k-space radially, where the spiral
# has to wind through it at the Nyquist pitch.

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
MATRIX = 128
INTERLEAVES = 16

interleaf = twirl_interleaf(system, FOV, MATRIX, INTERLEAVES)
spiral = sequences.Spiral(system, FOV, MATRIX, design_interleaves=INTERLEAVES)
for name, shape in (("twisting radial", interleaf), ("spiral", spiral)):
    print(
        f"{name:16} {shape.n_samples:5d} samples, readout "
        f"{shape.read_duration * 1e3:5.2f} ms, interleaf "
        f"{shape.duration * 1e3:5.2f} ms"
    )
print(
    f"transition radius {INTERLEAVES / (2 * np.pi * FOV):.1f} of "
    f"{MATRIX / (2 * FOV):.1f} 1/m"
)

# %%
# One repetition
# --------------

excitation = sequences.SpatialSelectiveExcitation(
    system, flip_angle_deg=15.0, thickness_m=5e-3, duration_s=3e-3
)
readout = TwirlReadout2D(
    system,
    excitation.rf,
    excitation.gz,
    excitation.gz_reph,
    fov=FOV,
    matrix=MATRIX,
    interleaves=INTERLEAVES,
    te=None,
    tr=None,
    spoiling_cycles=4.0,
)
print("events:", ", ".join(sorted(vars(readout.events))))
print(
    f"TE {readout.echo_time * 1e3:.3f} ms over a {readout.duration * 1e3:.3f} ms "
    "repetition"
)

seq = pp.Sequence(system=system)
for block in readout.blocks:
    seq.add_block(*block)
seq.paper_plot(tr=1)

# %%
# Scan loop
# ---------
#
# One solved interleaf is turned per shot by a rotation extension, which the
# loop adds to every block that drives an in-plane gradient. The interleaf is
# designed once and stored once; only the rotation changes from shot to shot.

angles = 2 * np.pi * np.arange(INTERLEAVES) / INTERLEAVES
rotations = [pp.make_rotation(float(angle)) for angle in angles]

scan = pp.Sequence(system=system)
for shot, rotation in enumerate(rotations):
    scan.add_block(excitation.rf, excitation.gz, pp.make_label("LIN", "SET", shot))
    for block in readout.blocks[1:]:
        events = list(block)
        if any(getattr(event, "channel", None) in ("x", "y") for event in events):
            events.append(rotation)
        scan.add_block(*events)

print(
    f"{scan.num_blocks} blocks, {scan.duration()[0] * 1e3:.1f} ms, "
    f"timing {scan.check_timing()[0]}"
)

# %%
# The interleaves the loop acquired.

pp.plot.plot_kspace(scan, plane="xy")

# %%
# References
# ----------
#
# .. [JNM92] Jackson JI, Nishimura DG, Macovski A. Twisting radial lines with
#    application to robust magnetic resonance imaging of irregular flow.
#    *Magnetic Resonance in Medicine*. 1992;28(2):251-263.
#    https://doi.org/10.1002/mrm.1910280209

# %%
# As a spec
# ---------
#
# What this Tour built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Write a pypulseqpp readout module TwirlReadout2D, a subclass of
#    sequences.NonCartesianReadout, for a 220 mm FOV, a 128 matrix and
#    16 interleaves. The interleaf is a twisting radial line: radial up
#    to the transition radius N/(2 pi FOV), then azimuth advancing so
#    that neighbouring interleaves stay 1/FOV apart. Solve the path
#    with sequences.Arbitrary at 250 kHz, once to measure the duration
#    and once with the sample count it holds. Compare its readout
#    duration with sequences.Spiral for the same interleaf count on
#    40 mT/m and 150 T/m/s. Play 16 shots after a 15 degree, 5 mm
#    slice excitation, turning the interleaf with make_rotation on
#    every block that drives x or y, label LIN, check the timing and
#    plot the k-space trajectory.
