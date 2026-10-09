r"""
=====================
8. Sequence functions
=====================

Your MPRAGE of lesson 6 is a script: the protocol sits in constants at the
top, and changing the inversion time means editing the file. In this lesson
you turn it into a *sequence function*, a function of the scanner and the
protocol that returns the sequence. That is the form every sequence shipped
with pypulseqpp takes, and the form a scanner console or a command line can
drive: they read the protocol from the function's signature and docstring,
and call it with the values the user prescribes.

**Learning objectives**

- Write a sequence as ``sequence(system, **protocol)``, with the protocol as
  keyword parameters documented in a NumPy ``Parameters`` section.
- Read the protocol back with :func:`~pypulseqpp.sequences.parameters`.
- Raise an error that names the limit when a protocol cannot be played.
- Run the function from the command line with :func:`pypulseqpp.cli.run`, and
  write what it returns with :func:`~pypulseqpp.sequences.write`.

Previous: :doc:`07_hardware_and_safety_checks`. Next:
:doc:`09_sequence_modules`, where the same function is built from sequence
modules.
"""

# %%
# The function
# ------------
#
# The scanner comes first, as ``system``; everything after it is the
# protocol, as keyword parameters with defaults. The docstring's
# ``Parameters`` section gives each one a description a user can read in a
# protocol editor, with its unit in parentheses.
#
# The body is lesson 6, with the constants replaced by the parameters. It
# designs every event once, before the loop, as before, and it checks what
# only it can check: whether the train fits in the TR the user asked for.
import numpy as np

import pypulseqpp as pp
from pypulseqpp import sequences

GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5)) / 2


def radial_mprage(
    system: pp.Opts,
    *,
    fov: float = 220e-3,
    matrix: int = 128,
    partitions: int = 16,
    partition_thickness: float = 5e-3,
    spokes_per_train: int = 64,
    flip_angle_deg: float = 8.0,
    ti: float = 900e-3,
    tr: float = 2.5,
) -> pp.Sequence:
    """3D radial MPRAGE on a stack of stars, one partition per inversion.

    Parameters
    ----------
    fov : float, default=0.22
        In-plane field of view (m).
    matrix : int, default=128
        In-plane matrix size.
    partitions : int, default=16
        Number of partitions.
    partition_thickness : float, default=0.005
        Thickness of one partition (m).
    spokes_per_train : int, default=64
        Golden-angle spokes played after each inversion.
    flip_angle_deg : float, default=8.0
        Readout flip angle (degrees).
    ti : float, default=0.9
        Inversion time (s), from the inversion centre to the first excitation.
    tr : float, default=2.5
        Time between two inversions (s).

    Raises
    ------
    ValueError
        If the train does not fit between TI and the next inversion.
    """
    slab = partitions * partition_thickness
    delta_k = 1 / fov

    rf_inv = pp.make_adiabatic_pulse(
        pulse_type="hypsec",
        duration=10e-3,
        bandwidth=40e3,
        adiabaticity=4,
        delay=system.rf_dead_time,
        use="inversion",
        system=system,
    )
    gz_crush = pp.make_crusher(4.0, partition_thickness, channel="z", system=system)[0]
    rf, gz, gz_reph = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=1e-3,
        slice_thickness=slab,
        apodization=0.5,
        time_bw_product=4.0,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
        return_gz=True,
    )

    # The merged prewinder and readout of lesson 5.
    dwell, readout_time = pp.calc_adc_timing(
        matrix,
        10e-6,
        grad_raster_time=system.grad_raster_time,
        adc_raster_time=system.adc_raster_time,
    )
    raster = system.grad_raster_time
    plateau = readout_time + raster * np.ceil(2 * system.adc_dead_time / raster)
    G = matrix * delta_k / readout_time
    _, times, amplitudes = pp.make_extended_trapezoid_area(
        channel="x", grad_start=0.0, grad_end=G, area=-G * plateau / 2, system=system
    )
    plateau_start = times[-1]
    g_read = pp.make_extended_trapezoid(
        "x",
        times=np.append(times, plateau_start + plateau),
        amplitudes=np.append(amplitudes, G),
        system=system,
    )
    adc = pp.make_adc(
        matrix,
        dwell=dwell,
        delay=plateau_start + (plateau - readout_time) / 2,
        system=system,
    )
    g_spoil, _, _ = pp.make_extended_trapezoid_area(
        channel="x",
        grad_start=G,
        grad_end=0.0,
        area=2 * matrix * delta_k,
        system=system,
    )

    # One partition encoding and one rewinder per partition.
    kz = (np.arange(partitions) - partitions // 2) / slab
    gz_encode = [
        pp.make_trapezoid(
            "z", area=gz_reph.area + k, duration=plateau_start, system=system
        )
        for k in kz
    ]
    gz_rewind = [
        pp.make_trapezoid(
            "z", area=-k, duration=pp.calc_duration(g_spoil), system=system
        )
        for k in kz
    ]

    # The waits that set TI and TR, and the check that they are possible.
    block_raster = system.block_duration_raster
    inversion_to_end = pp.calc_duration(rf_inv) - (
        rf_inv.delay + pp.calc_rf_center(rf_inv)[0]
    )
    ti_wait = pp.round_to_raster(
        ti
        - inversion_to_end
        - pp.calc_duration(gz_crush)
        - (rf.delay + pp.calc_rf_center(rf)[0]),
        block_raster,
    )
    if ti_wait < block_raster:
        raise ValueError(f"TI of {1e3 * ti:.1f} ms is shorter than the inversion")
    spoke_tr = (
        pp.calc_duration(rf, gz)
        + pp.calc_duration(g_read, adc, gz_encode[0])
        + pp.calc_duration(g_spoil, gz_rewind[0])
    )
    played = (
        pp.calc_duration(rf_inv)
        + pp.calc_duration(gz_crush)
        + ti_wait
        + spokes_per_train * spoke_tr
    )
    recovery = pp.round_to_raster(tr - played, block_raster)
    if recovery < block_raster:
        raise ValueError(
            f"TR of {1e3 * tr:.0f} ms is shorter than the "
            f"{1e3 * (played + block_raster):.0f} ms one inversion takes"
        )
    wait_ti, wait_recovery = pp.make_delay(ti_wait), pp.make_delay(recovery)

    seq = pp.Sequence(system)
    labels = sequences.Labels()
    n = 0
    for partition in range(partitions):
        seq.add_block(rf_inv)
        seq.add_block(gz_crush)
        seq.add_block(wait_ti)
        for spoke in range(spokes_per_train):
            angle = (n * GOLDEN_ANGLE) % np.pi
            phase = (np.deg2rad(117.0) * spoke * (spoke + 1) / 2) % (2 * np.pi)
            rf.phase_offset = adc.phase_offset = phase
            seq.add_block(rf, gz, *labels(LIN=spoke, PAR=partition))
            seq.add_block(
                *pp.rotate(g_read, angle=angle, axis="z"), adc, gz_encode[partition]
            )
            seq.add_block(
                *pp.rotate(g_spoil, angle=angle, axis="z"), gz_rewind[partition]
            )
            n += 1
        seq.add_block(wait_recovery)

    seq.set_definition("Name", "radial_mprage")
    seq.set_definition("FOV", [fov, fov, slab])
    seq.set_definition("Matrix", [matrix, matrix, partitions])
    seq.set_definition("TI", ti)
    seq.set_definition("TR", tr)
    return seq


# %%
# Calling it
# ----------
#
# The scanner is a separate argument because the same protocol runs on
# different scanners, and the same scanner runs many protocols:
system = pp.Opts(
    max_grad=32.0,
    grad_unit="mT/m",
    max_slew=130.0,
    slew_unit="T/m/s",
    rf_dead_time=100e-6,
    rf_ringdown_time=20e-6,
    adc_dead_time=10e-6,
)
seq = radial_mprage(system)
print(f"timing ok: {seq.check_timing()[0]}, scan time {seq.duration()[0]:.1f} s")

seq = radial_mprage(system, ti=1.1, tr=3.0, partitions=8)
print(f"timing ok: {seq.check_timing()[0]}, scan time {seq.duration()[0]:.1f} s")

# %%
# A protocol the function cannot play is an error, and the message says why,
# in the units the user wrote:
try:
    radial_mprage(system, spokes_per_train=400, tr=1.5)
except ValueError as error:
    print(error)

# %%
# The protocol
# ------------
#
# :func:`~pypulseqpp.sequences.parameters` reads the protocol of a sequence
# function: the type and default of each parameter from the signature, its
# unit and description from the docstring. This is all a protocol editor
# needs to show your sequence, with no list of parameters written anywhere
# else.
for name, parameter in sequences.parameters(radial_mprage).items():
    print(
        f"{name:20} {parameter.type.__name__:6} {parameter.unit:8} {parameter.default}"
    )

# %%
# The command line
# ----------------
#
# :func:`pypulseqpp.cli.run` builds a command line from the same signature:
# one option per protocol parameter, with the docstring as its help, and
# options for the gradient limits from which it builds ``system``. A script
# that ends with
#
# .. code-block:: python
#
#    if __name__ == "__main__":
#        raise SystemExit(cli.run(radial_mprage, sys.argv[1:]))
#
# runs as ``python radial_mprage.py --ti 1.1 -o mprage.seq``. Its help
# text:
import contextlib

from pypulseqpp import cli

with contextlib.suppress(SystemExit):
    cli.run(radial_mprage, ["--help"])

# %%
# Writing the files
# -----------------
#
# :func:`~pypulseqpp.sequences.write` writes what the function returns, and
# returns the paths it wrote. A function may also return a list of
# sequences, prescans first and the main sequence last; ``write`` then
# writes one file each, linked in order through the ``NextSequence``
# definition.
from pathlib import Path
from tempfile import mkdtemp

folder = Path(mkdtemp())
written = sequences.write(folder / "radial_mprage.seq", seq)
print([Path(path).name for path in written])

# %%
# The shipped sequences
# ---------------------
#
# Every sequence in :doc:`/sequences` is a function of this kind, in its own
# module of ``pypulseqpp.sequences``, and the module itself is callable as the
# function. The shipped radial MPRAGE, which you checked in lesson 7, has many
# more parameters than yours, from partial Fourier to the excitation pulse,
# but they are read the same way:
protocol = sequences.parameters(sequences.mprage_stack_of_stars3D_sequence)
print(f"{len(protocol)} parameters:", ", ".join(protocol))

# %%
# Pulserver, which runs these sequences on a scanner, builds its protocol
# editor from the same record, and calls the function with what the user
# prescribed.
#
# As a spec
# ---------
#
# What this lesson built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Turn the lesson-6 radial MPRAGE script into a pypulseqpp sequence
#    function radial_mprage(system, *, fov, matrix, partitions,
#    partition_thickness, spokes_per_train, flip_angle_deg, ti, tr) that
#    returns the Sequence. Document every parameter in a NumPy Parameters
#    section with its unit in parentheses, raise ValueError naming the
#    minimum when TI or TR is too short, set the Name, FOV, Matrix, TI and
#    TR definitions, and expose it on the command line with
#    pypulseqpp.cli.run.
