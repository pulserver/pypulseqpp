r"""
==========================
10. Custom sequence module
==========================

The shipped modules cover the common parts of a sequence, but sooner or later
you need one that is not there. A module is a small class, and writing one
is the same work as designing its events by hand, done once and given a
name.

In this lesson you write a *T2 preparation*: a train of pulses that tips the
magnetization into the transverse plane, lets it decay with T2 for a set
echo time, and stores what is left back along z. Played in place of the
inversion, it turns your MPRAGE from T1- into T2-weighted. Then you compare
your module with the shipped
:class:`~pypulseqpp.sequences.T2Preparation`, and see why that one is built
from adiabatic pulses.

**Learning objectives**

- Subclass :class:`~pypulseqpp.sequences.RfModule` and implement
  ``init_module``: build ``self.seq`` and set ``self.center``.
- Time a train of pulses by their centres, on the block raster.
- Swap a preparation in your sequence function without touching its loop.
- Simulate how a preparation responds to errors in the transmit field and
  the resonance frequency.

Previous: :doc:`09_sequence_modules`. This is the last lesson of the course;
the Tours and the :doc:`/sequences` catalogue go further from here.
"""

# sphinx_gallery_start_ignore
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pypulseqpp.plot._style import SIGNED

PAGE_WIDTH = 7.8  # inches, the width of the documentation column
warnings.filterwarnings("ignore", message="Specified RF delay")
# sphinx_gallery_end_ignore

# %%
# The preparation
# ---------------
#
# The simplest robust T2 preparation is four hard pulses:
#
# 1. a 90° pulse about x tips the magnetization into the transverse plane;
# 2. two 180° pulses about y, a quarter and three quarters of the way
#    through the echo time, refocus it twice, so off-resonance during the
#    train cancels and the errors of the two pulses largely cancel too;
# 3. a 90° pulse about -x, at the echo time, stores it back along +z;
# 4. a crusher dephases whatever the pulses left in the transverse plane.
#
# The echo time is measured from the centre of the first pulse to the centre
# of the last. The magnetization comes back along z scaled by
# :math:`e^{-\mathrm{TE}/T_2}`, and the next excitation reads that.
#
# The module
# ----------
#
# A module subclasses :class:`~pypulseqpp.sequences.SequenceModule`, or
# :class:`~pypulseqpp.sequences.RfModule` when it plays RF, and implements
# one method, ``init_module``. Its parameters are the module's parameters,
# after ``system``. It assigns ``self.seq``, adds its blocks to it, and sets
# ``self.center``, the instant the rest of the sequence is timed against:
# here the centre of the storing pulse, when the preparation is complete. The
# local variables that hold events, the pulses, the waits and the crusher,
# are published on the module under their own names.
import numpy as np

import pypulseqpp as pp
from pypulseqpp import sequences


class HardT2Preparation(sequences.RfModule):
    """T2 preparation of hard pulses: 90 x, 180 y, 180 y, 90 -x, crusher.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    echo_time_s : float
        Preparation echo time (s), from the first pulse centre to the last.
    pulse_duration_s : float, default=0.5e-3
        Duration of a 90 degree pulse (s); a 180 is twice as long.
    voxel_size_m : float, default=1e-3
        Length the crusher dephases four cycles over (m).
    """

    def init_module(
        self, system, echo_time_s, *, pulse_duration_s=0.5e-3, voxel_size_m=1e-3
    ):
        tip_down = pp.make_block_pulse(
            np.pi / 2,
            duration=pulse_duration_s,
            delay=system.rf_dead_time,
            use="preparation",
            system=system,
        )
        refocus = pp.make_block_pulse(
            np.pi,
            duration=2 * pulse_duration_s,
            phase_offset=np.pi / 2,
            delay=system.rf_dead_time,
            use="refocusing",
            system=system,
        )
        tip_up = pp.make_block_pulse(
            np.pi / 2,
            duration=pulse_duration_s,
            phase_offset=np.pi,
            delay=system.rf_dead_time,
            use="preparation",
            system=system,
        )
        gz_spoil = pp.make_crusher(4.0, voxel_size_m, channel="z", system=system)[0]

        # Each pulse's centre from the start of its block, and the time from
        # its centre to the end of the block.
        def centre(rf):
            return rf.delay + pp.calc_rf_center(rf)[0]

        def tail(rf):
            return pp.calc_duration(rf) - centre(rf)

        raster = system.block_duration_raster
        quarter = echo_time_s / 4
        wait_first = pp.make_delay(
            pp.round_to_raster(quarter - tail(tip_down) - centre(refocus), raster)
        )
        wait_inner = pp.make_delay(
            pp.round_to_raster(2 * quarter - tail(refocus) - centre(refocus), raster)
        )
        wait_last = pp.make_delay(
            pp.round_to_raster(quarter - tail(refocus) - centre(tip_up), raster)
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(tip_down)
        self.seq.add_block(wait_first)
        self.seq.add_block(refocus)
        self.seq.add_block(wait_inner)
        self.seq.add_block(refocus)
        self.seq.add_block(wait_last)
        self.seq.add_block(tip_up)
        self.seq.add_block(gz_spoil)

        storing = self.seq.duration()[0] - pp.calc_duration(gz_spoil) - tail(tip_up)
        self.center = storing
        self.echo_time = storing - centre(tip_down)


# %%
# Your module is constructed like a shipped one, and reports the same things:
preparation = HardT2Preparation(
    pp.Opts(rf_dead_time=100e-6, rf_ringdown_time=20e-6), 50e-3, voxel_size_m=5e-3
)
print(
    f"echo time {1e3 * preparation.echo_time:.2f} ms, "
    f"duration {1e3 * preparation.duration:.2f} ms, "
    f"events: {', '.join(sorted(vars(preparation.events)))}"
)
preparation.paper_plot()

# %%
# Swapping the preparation
# ------------------------
#
# Your sequence function of lesson 9 played the inversion as
# ``for block in inversion.blocks``, and placed the first excitation from
# the inversion's ``center``. Neither cares what the preparation is. Add a
# ``preparation`` parameter that chooses the module, and the loop stays as
# it was. ``ti`` becomes the delay from the preparation's centre to the first
# excitation: the inversion time for the inversion, and a short gap for the
# T2 preparation.
GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5)) / 2


def prepared_radial(
    system: pp.Opts,
    *,
    preparation: str = "inversion",
    t2_prep_te: float = 50e-3,
    fov: float = 220e-3,
    matrix: int = 128,
    partitions: int = 16,
    partition_thickness: float = 5e-3,
    spokes_per_train: int = 64,
    flip_angle_deg: float = 8.0,
    ti: float = 900e-3,
    tr: float = 2.5,
) -> pp.Sequence:
    """3D radial gradient echo on a stack of stars, one partition per preparation.

    Parameters
    ----------
    preparation : {'inversion', 't2'}, default='inversion'
        An adiabatic inversion, or a hard-pulse T2 preparation.
    t2_prep_te : float, default=0.05
        Echo time of the T2 preparation (s).
    fov : float, default=0.22
        In-plane field of view (m).
    matrix : int, default=128
        In-plane matrix size.
    partitions : int, default=16
        Number of partitions.
    partition_thickness : float, default=0.005
        Thickness of one partition (m).
    spokes_per_train : int, default=64
        Golden-angle spokes played after each preparation.
    flip_angle_deg : float, default=8.0
        Readout flip angle (degrees).
    ti : float, default=0.9
        Time from the preparation's centre to the first excitation (s).
    tr : float, default=2.5
        Time between two preparations (s).

    Raises
    ------
    ValueError
        If ``preparation`` is unknown, or the train does not fit between
        ``ti`` and the next preparation.
    """
    slab = partitions * partition_thickness
    if preparation == "inversion":
        prep = sequences.InversionPreparation(system, voxel_size_m=partition_thickness)
    elif preparation == "t2":
        prep = HardT2Preparation(system, t2_prep_te, voxel_size_m=partition_thickness)
    else:
        raise ValueError(
            f"preparation must be 'inversion' or 't2', got {preparation!r}"
        )
    excitation = sequences.SpatialSelectiveExcitation(
        system, flip_angle_deg, slab, duration_s=1e-3, is_slab=True
    )
    readout = sequences.RadialStackReadout(
        system,
        excitation.rf,
        excitation.gz,
        fov=fov,
        matrix=matrix,
        fov_z=slab,
        matrix_z=partitions,
        readout_bandwidth_hz=100e3,
        spoiling_cycles=4.0,
    )
    pulse, _, read, _ = readout.blocks

    steps = (np.arange(partitions) - partitions // 2) / (partitions / 2)
    gz_encode = [pp.scale_grad(readout.gz_pre, step) for step in steps]
    gz_rewind = [pp.scale_grad(readout.gz_rew, step) for step in steps]

    raster = system.block_duration_raster
    ti_wait = pp.round_to_raster(
        ti - (prep.duration - prep.center) - excitation.center, raster
    )
    if ti_wait < raster:
        raise ValueError(f"TI of {1e3 * ti:.1f} ms is shorter than the preparation")
    played = prep.duration + ti_wait + spokes_per_train * readout.duration
    recovery = pp.round_to_raster(tr - played, raster)
    if recovery < raster:
        raise ValueError(
            f"TR of {1e3 * tr:.0f} ms is shorter than the "
            f"{1e3 * (played + raster):.0f} ms one preparation takes"
        )
    wait_ti, wait_recovery = pp.make_delay(ti_wait), pp.make_delay(recovery)

    angles = (np.arange(partitions * spokes_per_train) * GOLDEN_ANGLE) % np.pi
    rotations = [pp.make_rotation(angle) for angle in angles]
    phases = pp.make_rf_spoiling_schedule(
        partitions * spokes_per_train, increment=np.deg2rad(117.0)
    )

    seq = pp.Sequence(system)
    labels = sequences.Labels()
    n = 0
    for partition in range(partitions):
        for block in prep.blocks:
            seq.add_block(*block)
        seq.add_block(wait_ti)
        for spoke in range(spokes_per_train):
            readout.rf.phase_offset = readout.adc.phase_offset = phases[n]
            seq.add_block(*pulse, *labels(LIN=spoke, PAR=partition))
            seq.add_block(gz_encode[partition])
            seq.add_block(*read, rotations[n])
            seq.add_block(gz_rewind[partition], readout.gz_spoil)
            n += 1
        seq.add_block(wait_recovery)

    seq.set_definition("Name", f"radial_{preparation}")
    seq.set_definition("FOV", [fov, fov, slab])
    seq.set_definition("Matrix", [matrix, matrix, partitions])
    seq.set_definition("TR", tr)
    return seq


# %%
# The same function now plays either scan:
system = pp.Opts(
    max_grad=32.0,
    grad_unit="mT/m",
    max_slew=130.0,
    slew_unit="T/m/s",
    rf_dead_time=100e-6,
    rf_ringdown_time=20e-6,
    adc_dead_time=10e-6,
)
mprage = prepared_radial(system)
t2_prepared = prepared_radial(system, preparation="t2", ti=20e-3)
for name, seq in [("inversion", mprage), ("T2 preparation", t2_prepared)]:
    ok, errors = seq.check_timing()
    print(f"{name:15} timing ok: {ok}, scan time {seq.duration()[0]:.1f} s")

t2_prepared.paper_plot(time_range=(0, 0.12))

# %%
# After the preparation, each tissue starts the train with what survived the
# echo time, :math:`e^{-\mathrm{TE}/T_2}`, and the contrast is the reverse of
# the MPRAGE's: CSF, with its long T2, is the brightest.

# sphinx_gallery_start_ignore
for name, t2 in {"white matter": 70e-3, "grey matter": 100e-3, "CSF": 2.0}.items():
    print(
        f"{name:13} T2 {1e3 * t2:6.0f} ms, Mz after preparation {np.exp(-50e-3 / t2):.2f}"
    )
# sphinx_gallery_end_ignore

# %%
# Sensitivity to B1 and off-resonance
# ------------------------------------
#
# A preparation is played before every train, across the whole head, where
# the transmit field :math:`B_1` varies by tens of per cent and the resonance
# frequency by tens to hundreds of hertz. Without relaxation, an ideal
# preparation returns the magnetization to :math:`M_z = 1` whatever the field.
# The figure below simulates the RF of your module, and of the shipped
# :class:`~pypulseqpp.sequences.T2Preparation`, over a range of both errors
# with :func:`~pypulseqpp.sim_bloch`:
shipped = sequences.T2Preparation(system, 50e-3, voxel_size_m=5e-3)

# sphinx_gallery_start_ignore
B1_SCALES = np.linspace(0.6, 1.4, 41)
OFFSETS = np.linspace(-250.0, 250.0, 41)
DT = 2e-6


def rf_waveform(module):
    """The module's RF as one complex waveform on a DT grid, in Hz."""
    total = sum(pp.calc_duration(*block) for block in module.blocks)
    waveform = np.zeros(round(total / DT), complex)
    start = 0.0
    for block in module.blocks:
        for event in block:
            if type(event).__name__ != "RfEvent":
                continue
            t = (np.arange(round(event.shape_dur / DT)) + 0.5) * DT
            samples = np.interp(t, event.t, event.signal.real) + 1j * np.interp(
                t, event.t, event.signal.imag
            )
            first = round((start + event.delay) / DT)
            waveform[first : first + len(t)] = samples * np.exp(1j * event.phase_offset)
        start += pp.calc_duration(*block)
    return waveform


def stored_mz(module):
    """Mz after the module, over B1 scale (rows) and off-resonance (columns)."""
    b1 = rf_waveform(module)
    scale, offset = np.meshgrid(B1_SCALES, OFFSETS, indexing="ij")
    final = pp.sim_bloch(scale.reshape(-1, 1) * b1[None, :], offset.reshape(-1, 1), DT)
    return final[:, 2].reshape(scale.shape)


figure, axes = plt.subplots(
    1, 2, figsize=(PAGE_WIDTH, 3.0), layout="constrained", sharey=True
)
for axis, module, title in zip(
    axes,
    (preparation, shipped),
    ("hard pulses (yours)", "adiabatic (shipped)"),
    strict=True,
):
    image = axis.imshow(
        stored_mz(module),
        origin="lower",
        aspect="auto",
        extent=(OFFSETS[0], OFFSETS[-1], B1_SCALES[0], B1_SCALES[-1]),
        vmin=-1,
        vmax=1,
        cmap=SIGNED,
    )
    axis.set_title(title)
    axis.set_xlabel("off-resonance (Hz)")
axes[0].set_ylabel("$B_1$ scale")
figure.colorbar(image, ax=axes, label="$M_z$ after preparation")
plt.show()
# sphinx_gallery_end_ignore

# %%
# On resonance your hard-pulse train is robust to :math:`B_1`: the second
# 180° pulse undoes most of the error of the first. Off resonance, the
# refocusing is imperfect, the magnetization it leaves behind precesses during
# the long waits, and the stored :math:`M_z` oscillates with the frequency,
# with a period of :math:`2/\mathrm{TE} = 40` Hz. The shipped preparation
# replaces the 90° pulses by adiabatic half passages and the 180° pulses by
# adiabatic full passages, whose rotation depends little on the field above a
# threshold, and it stays close to one over most of the range. The two
# modules share their structure and differ only in the pulses.
#
# Further reading
# ---------------
#
# The :doc:`/sequences` catalogue is a set of sequence functions written as in
# lessons 8 and 9, and the Tours treat individual topics in more depth.
#
# As a spec
# ---------
#
# What this lesson built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Write a pypulseqpp RfModule HardT2Preparation(system, echo_time_s,
#    pulse_duration_s=0.5e-3, voxel_size_m=1e-3): block pulses 90 x,
#    180 y, 180 y, 90 -x with the 180s at TE/4 and 3TE/4 (centre to
#    centre, waits on the block raster), then a 4-cycle z crusher; center
#    is the storing pulse's centre. Add a preparation parameter
#    {'inversion', 't2'} to the lesson-9 radial MPRAGE function that picks
#    InversionPreparation or this module, and compare the stored Mz of
#    this module and the shipped T2Preparation over B1 0.6-1.4 and
#    +-250 Hz with sim_bloch.
