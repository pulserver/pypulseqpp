"""Excitation, refocusing and inversion acting on everything in the transmit coil."""

from __future__ import annotations

__all__ = ["Inversion", "NonSelectiveExcitation", "NonSelectiveRefocusing"]

import numpy as np

import pypulseqpp as pp

from ._base import RfModule, rf_reference

_AXES = ("x", "y", "z")


class NonSelectiveExcitation(RfModule):
    """Rectangular RF excitation without a selection gradient.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    flip_angle_deg : float, optional
        Nominal flip angle (degrees).
    duration_s : float, optional
        Pulse duration (s); spectral width scales inversely with duration.
    use : str, optional
        Pulseq use tag. Undefined pulses are treated as excitations by
        k-space analysis.
    freq_offset_hz, phase_offset_rad : float, optional
        Transmit offsets designed into the pulse. A scan loop moves
        ``module.rf.freq_offset`` and ``module.rf.phase_offset`` per shot
        instead of rebuilding the module.

    Attributes
    ----------
    rf : RfEvent
        The pulse.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> excitation = design.NonSelectiveExcitation(pp.Opts(), flip_angle_deg=10.0)
    >>> round(excitation.center * 1e6)
    500
    """

    def init_module(
        self,
        system: pp.Opts,
        flip_angle_deg: float = 10.0,
        duration_s: float = 1e-3,
        *,
        use: str = "excitation",
        freq_offset_hz: float = 0.0,
        phase_offset_rad: float = 0.0,
    ) -> None:
        rf = pp.make_block_pulse(
            flip_angle=np.deg2rad(flip_angle_deg),
            duration=duration_s,
            freq_offset=freq_offset_hz,
            phase_offset=phase_offset_rad,
            use=use,
            system=system,
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf)

        self.center = rf_reference(rf)


class NonSelectiveRefocusing(RfModule):
    """Rectangular refocusing pulse between two identical crushers.

    The default phase is pi/2 radians for CPMG with a zero-phase excitation.
    Set spoiling_cycles=0 to omit the crushers.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    flip_angle_deg : float, optional
        Nominal flip angle (degrees).
    duration_s : float, optional
        Pulse duration (s).
    spoiling_cycles : float, optional
        Cycles of dephasing each crusher winds across ``voxel_size_m``. Zero
        leaves the pulse bare, which is what an echo train that crushes
        elsewhere wants.
    voxel_size_m : float, optional
        Length the dephasing is counted over (m).
    axis : {'z', 'x', 'y'}, optional
        Crusher axis.
    phase_offset_rad : float, optional
        RF phase. The CPMG quarter turn by default.
    use : str, optional
        What the pulse is for; the trajectory core negates accumulated k at a
        refocusing pulse, so this is not cosmetic.

    Attributes
    ----------
    rf_ref : RfEvent
        The refocusing pulse.
    gz_spoil : GradEvent
        The crusher, published once because both sides play the same event.

    Raises
    ------
    ValueError
        If ``spoiling_cycles`` is negative, ``voxel_size_m`` is not positive,
        or ``axis`` is not a gradient channel.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> refocusing = design.NonSelectiveRefocusing(pp.Opts())
    >>> len(refocusing.blocks), refocusing.blocks[0] == refocusing.blocks[2]
    (3, True)

    >>> len(design.NonSelectiveRefocusing(pp.Opts(), spoiling_cycles=0.0).blocks)
    1
    """

    def init_module(
        self,
        system: pp.Opts,
        flip_angle_deg: float = 180.0,
        duration_s: float = 1e-3,
        *,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
        axis: str = "z",
        phase_offset_rad: float = np.pi / 2,
        use: str = "refocusing",
    ) -> None:
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")

        rf_ref = pp.make_block_pulse(
            flip_angle=np.deg2rad(flip_angle_deg),
            duration=duration_s,
            phase_offset=phase_offset_rad,
            use=use,
            system=system,
        )

        self.seq = pp.Sequence(system)

        if spoiling_cycles:
            gz_spoil, _, _ = pp.make_crusher(
                spoiling_cycles, voxel_size_m, axis, system=system
            )
            self.seq.add_block(gz_spoil)

        lead = self.seq.duration()[0]
        self.seq.add_block(rf_ref)
        if spoiling_cycles:
            self.seq.add_block(gz_spoil)

        self.center = lead + rf_reference(rf_ref)


class Inversion(RfModule):
    """Non-selective adiabatic inversion without a crusher or inversion-time delay.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    duration_s : float, optional
        Pulse duration (s). Adiabaticity is a condition on sweeping slowly
        enough, so this is not free to shorten.
    pulse_type : str, optional
        Sweep family, as :func:`pypulseq.make_adiabatic_pulse` names them
        (``"hypsec"``, ``"wurst"``).
    bandwidth_hz : float, optional
        Frequency width of the sweep (Hz).
    adiabaticity : int, optional
        Sweep-rate margin over the adiabatic condition.
    use : str, optional
        What the pulse is for.

    Attributes
    ----------
    rf_prep : RfEvent
        The pulse.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> inversion = design.Inversion(pp.Opts(), duration_s=8e-3)
    >>> round(inversion.duration * 1e3, 1)
    8.0
    """

    def init_module(
        self,
        system: pp.Opts,
        duration_s: float = 10e-3,
        *,
        pulse_type: str = "hypsec",
        bandwidth_hz: float = 40e3,
        adiabaticity: int = 4,
        use: str = "inversion",
    ) -> None:
        rf_prep = pp.make_adiabatic_pulse(
            pulse_type=pulse_type,
            duration=duration_s,
            bandwidth=bandwidth_hz,
            adiabaticity=adiabaticity,
            use=use,
            system=system,
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf_prep)

        self.center = rf_reference(rf_prep)
