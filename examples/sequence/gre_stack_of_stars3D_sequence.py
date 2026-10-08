"""RF-spoiled 3D stack-of-stars gradient echo."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: How far each partition turns its tilts, as a fraction of the half turn a
#: spoke covers: not at all, by the golden ratio ``1 / phi``, or by the tiny
#: golden angle ``1 / (phi + 1)``.
PARTITION_SHIFTS = {
    "none": 0.0,
    "golden": 2 / (1 + math.sqrt(5)),
    "tiny_golden": 2 / (3 + math.sqrt(5)),
}

NAME = "gre_stack_of_stars_3d"
MAX_GRAD = 80.0
MAX_SLEW = 200.0
#: SLR design of the slab-selective pulse.
PULSE_DURATION = 3e-3
TIME_BW_PRODUCT = 4.0
#: Duration of the nonselective hard pulse (s).
HARD_PULSE_DURATION = 0.5e-3
#: Fat methylene shift from water (ppm), converted against ``system.B0``
#: when the spectral-spatial pulse is built.
FAT_SHIFT_PPM = -3.4
#: Quadratic RF spoiling phase increment (degrees).
RF_SPOILING_INCREMENT_DEG = 117.0
#: Dephasing left on the partition axis at the end of each repetition, in
#: cycles across one voxel.
SPOILING_CYCLES = 4.0


def gre_stack_of_stars3d(
    system: pp.Opts | None = None,
    *,
    fov: float = 220e-3,
    n: int = 128,
    fov_z: float = 128e-3,
    n_z: int = 64,
    flip_angle_deg: float = 12.0,
    te: float | None = None,
    tr: float | None = None,
    readout_bandwidth_hz: float = 250e3,
    ry: int = 1,
    rz: int = 1,
    partial_fourier_z: float = 1.0,
    n_dummy: int | None = None,
    excitation: str = "slab",
    partition_angle_shift: str = "none",
    n_acs_z: int = 16,
    readout_oversampling: float = 2.0,
) -> pp.Sequence:
    """RF-spoiled 3D stack of stars: radial spokes in-plane, Cartesian partitions along z.

    One spoke waveform serves every shot: its angle is a rotation extension
    and its partition an amplitude on the encode pair. The Nyquist set is
    ``ceil(pi / 2 * n)`` spokes spread evenly over half a turn; every
    ``ry``-th of them is played, in order, each at every acquired partition
    before the next. Acquisitions carry the spoke as ``LIN`` and the partition
    as ``PAR``.

    Under partition undersampling the central ``n_acs_z`` partitions are
    acquired in full at every tilt, ahead of the rest, and marked ``IMA``.

    Parameters
    ----------
    system : pypulseqpp.Opts, default=None
        System limits, held under the module's ``MAX_GRAD`` (80 mT/m) and
        ``MAX_SLEW`` (200 T/m/s). ``None`` is ``pypulseqpp.Opts()``.
    fov : float, default=0.22
        Isotropic in-plane field of view (m).
    n : int, default=128
        In-plane matrix size; a spoke reads it edge to edge.
    fov_z : float, default=0.128
        Field of view along the partitions (m). The slab excited is
        ``fov_z`` thick.
    n_z : int, default=64
        Number of partitions.
    flip_angle_deg : float, default=12.0
        Excitation flip angle (degrees).
    te : float | None, default=None
        Echo time to the spoke's centre crossing (s). ``None`` is as short
        as possible.
    tr : float | None, default=None
        Repetition time (s), one per excitation. ``None`` is as short as
        possible.
    readout_bandwidth_hz : float, default=250000.0
        Requested receiver bandwidth (Hz).
    ry : int, default=1
        Angular undersampling: one spoke in every ``ry`` of the Nyquist set
        is played.
    rz : int, default=1
        Partition undersampling: one partition in every ``rz`` is
        acquired, the centre one among them.
    partial_fourier_z : float, default=1.0
        Fraction of the partition extent acquired, in ``[0.75, 1]``.
    n_dummy : int or None, default=None
        Non-acquiring repetitions, at the first spoke's angle and the
        centre partition, before the scan.
        ``None`` plays as many as bring the steady state within 1 % (:func:`~pypulseqpp.sequences.steady_state_dummies`).
    excitation : {'slab', 'nonselective', 'spsp'}, default='slab'
        A slab-selective SLR pulse, a hard pulse, or a slab- and
        water-selective spectral-spatial pulse.
    partition_angle_shift : {'none', 'golden', 'tiny_golden'}, default='none'
        How far each partition turns the spokes past the previous one, as
        :data:`PARTITION_SHIFTS` names the fractions of a half turn.
    n_acs_z : int, default=16
        Fully sampled calibration partitions at the centre, acquired at
        every tilt ahead of the rest when ``rz > 1``.
    readout_oversampling : float, default=2.0
        Readout oversampling factor, at least one.

    Returns
    -------
    pypulseqpp.Sequence
        The designed sequence.

    Raises
    ------
    ValueError
        If ``excitation`` or ``partition_angle_shift`` is unknown, an
        undersampling factor is below one, ``partial_fourier_z`` is outside
        ``[0.75, 1]``, or the TE or TR is shorter than the readout takes.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre_stack_of_stars3D_sequence(n=32, n_z=4)
    >>> seq.check_timing()[0]
    True
    """
    system = pp.cap_system(
        pp.Opts() if system is None else system,
        max_grad=MAX_GRAD,
        max_slew=MAX_SLEW,
    )
    if excitation not in sequences.EXCITATIONS:
        raise ValueError(
            f"excitation must be one of {sequences.EXCITATIONS}, got {excitation!r}"
        )
    if partition_angle_shift not in PARTITION_SHIFTS:
        raise ValueError(
            f"partition_angle_shift must be one of {tuple(PARTITION_SHIFTS)}, "
            f"got {partition_angle_shift!r}"
        )
    if ry < 1 or rz < 1:
        raise ValueError(f"ry and rz must be at least 1, got {ry} and {rz}")
    if not 0.75 <= partial_fourier_z <= 1.0:
        raise ValueError(
            f"partial_fourier_z must lie in [0.75, 1], got {partial_fourier_z}"
        )

    exc = sequences.make_excitation(
        system,
        excitation,
        flip_angle_deg,
        fov_z,
        duration_s=PULSE_DURATION,
        time_bw_product=TIME_BW_PRODUCT,
        hard_duration_s=HARD_PULSE_DURATION,
        fat_shift_ppm=FAT_SHIFT_PPM,
    )
    gz = getattr(exc, "gz", None)
    ro = sequences.RadialStackReadout(
        system,
        exc.rf,
        gz,
        fov=fov,
        matrix=n,
        fov_z=fov_z,
        matrix_z=n_z,
        te=te,
        oversampling=readout_oversampling,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=SPOILING_CYCLES,
    )

    # A full spoke covers half a turn, which the Nyquist set divides evenly.
    n_nyquist = math.ceil(np.pi / 2 * n)
    span = np.pi
    angles = span * np.arange(0, n_nyquist, ry) / n_nyquist
    shift = PARTITION_SHIFTS[partition_angle_shift] * span
    calibrating, imaging = pp.make_cartesian_axis_sampling(
        n_z, rz, n_acs_z, partial_fourier=partial_fourier_z
    )
    calibration = set(calibrating)
    # The calibration partitions lead, at every tilt, then the rest.
    views = [
        (spoke, z)
        for partitions in (calibrating, imaging)
        for spoke in range(len(angles))
        for z in partitions
    ]
    rotations: dict[float, object] = {}

    raster = system.block_duration_raster
    wait_tr = None
    repetition_time = ro.duration
    if tr is not None:
        if tr < ro.duration - 1e-9:
            raise ValueError(
                f"the requested TR of {tr * 1e3:.3f} ms is shorter than the "
                f"{ro.duration * 1e3:.3f} ms one repetition takes"
            )
        pad = pp.round_to_raster(tr - ro.duration, raster)
        if pad > 0:
            wait_tr = pp.make_delay(pad)
            repetition_time += pad
    if n_dummy is None:
        n_dummy = sequences.steady_state_dummies(repetition_time, flip_angle_deg)

    seq, labels = pp.Sequence(system), sequences.Labels()

    def rotation_at(spoke: int, partition: int):
        """Return the rotation extension turning ``spoke`` at ``partition``.

        The extension is shared between shots at the same angle.
        """
        angle = float((angles[spoke] + partition * shift) % span)
        key = round(angle, 12)
        if key not in rotations:
            rotations[key] = pp.make_rotation(angle)
        return rotations[key]

    def kernel(view: tuple[int, int] | None, phase: float) -> None:
        """Add one excitation, one spoke at one partition; ``None`` is a dummy.

        The readout's blocks after the pulse are played as it laid them out,
        with the partition encode scaled and every block that drives an
        in-plane gradient turned to the shot's angle. ``phase`` is the RF and
        ADC phase of the repetition, in radians.
        """
        exc.rf.phase_offset = phase
        ro.adc.phase_offset = phase

        if view is None:
            spoke, partition = 0, n_z // 2
            label_events = labels(ONCE=1)
        else:
            spoke, partition = view
            label_events = labels(
                LIN=spoke, PAR=partition, IMA=partition in calibration, ONCE=0
            )
        kz = (partition - n_z // 2) / (n_z / 2)
        encoded = {
            id(ro.gz_pre): pp.scale_grad(ro.gz_pre, kz),
            id(ro.gz_rew): pp.scale_grad(ro.gz_rew, kz),
        }
        rotation = rotation_at(spoke, partition)

        seq.add_block(exc.rf, *([] if gz is None else [gz]), *label_events)
        for block in ro.blocks[1:]:
            events = [
                encoded.get(id(event), event)
                for event in block
                if view is not None or event is not ro.adc
            ]
            if any(getattr(event, "channel", None) in ("x", "y") for event in events):
                events.append(rotation)
            seq.add_block(*events)
        if wait_tr is not None:
            seq.add_block(wait_tr)

    # The dummies play first, then every acquired partition of each spoke in
    # turn.
    played = [None] * n_dummy + views
    phases = pp.make_rf_spoiling_schedule(
        len(played), increment=np.deg2rad(RF_SPOILING_INCREMENT_DEG)
    )
    for view, phase in zip(played, phases, strict=True):
        kernel(view, phase)

    # The volume's offset is applied to the finished sequence with
    # pp.TransformFOV.
    definitions = {
        "FOV": [fov, fov, fov_z],
        "Matrix": [n, n, n_z],
        "Name": NAME,
        "TE": ro.echo_time,
        "TR": repetition_time,
        "Trajectory": "stack_of_stars",
        "Excitation": excitation,
        "NumSpokes": len(angles),
        "PartitionAngleShift": partition_angle_shift,
        "kSpaceCenterPartition": n_z // 2,
        "kSpaceCenterSample": ro.center_sample,
        "SliceThickness": fov_z,
    }
    for key, value in definitions.items():
        seq.set_definition(key=key, value=value)
    return seq


main = gre_stack_of_stars3d

if __name__ == "__main__":
    raise SystemExit(
        cli.run(main, sys.argv[1:], default_output="gre_stack_of_stars_3d.seq")
    )
