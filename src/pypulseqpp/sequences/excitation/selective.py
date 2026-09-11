"""Excitation and refocusing of one slice or one slab."""

from __future__ import annotations

__all__ = ["SpatialSelectiveExcitation", "SpatialSelectiveRefocusing"]

import numpy as np

import pypulseqpp as pp

from ._base import RfModule, rf_reference

_AXES = ("x", "y", "z")


class SpatialSelectiveExcitation(RfModule):
    """SLR excitation with a selection gradient and optional rephasing.

    A slice publishes separate gz and gz_reph events in two blocks. With
    is_slab=True, rephasing is merged into gz in one block. Set rephase=False
    to omit it; a readout may instead include the rephaser's moment.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    flip_angle_deg : float
        Nominal flip angle (degrees).
    thickness_m : float
        Slice or slab thickness (m).
    duration_s : float, optional
        Pulse duration (s).
    is_slab : bool, optional
        Merge the rephaser into the selection gradient, as above.
    rephase : bool, optional
        Include a slice rephaser.
    time_bw_product : float, optional
        Time-bandwidth product. Higher is a squarer profile and a longer pulse
        at the same bandwidth.
    axis : {'z', 'x', 'y'}, optional
        Selection axis.
    pulse_type : {'st', 'ex', 'se', 'inv', 'sat'}, optional
        SLR design family. ``"ex"`` for a large-tip excitation, ``"se"`` for a
        refocusing pulse.
    use : str, optional
        Pulseq RF-use tag, used by trajectory integration.
    passband_ripple, stopband_ripple : float, optional
        Ripple allowed in each band of the slice profile.

    Attributes
    ----------
    rf : RfEvent
        The pulse. Its ``delay`` already places it on the gradient's flat top.
    gz : TrapEvent or GradEvent
        The selection gradient; under ``is_slab`` the rephaser is part of it.
    gz_reph : TrapEvent
        The rephaser. Only when not ``is_slab`` and ``rephase``.
    selection_amplitude : float
        Plateau amplitude of the selection lobe (Hz/m), which a slice offset
        is converted against: ``freq_offset = selection_amplitude * position``.

    Raises
    ------
    ValueError
        If ``thickness_m`` or ``duration_s`` is not positive, or ``axis`` is
        not a gradient channel.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> slice_ = design.SpatialSelectiveExcitation(pp.Opts(), 15.0, 5e-3)
    >>> len(slice_.blocks), slice_.gz_reph.channel
    (2, 'z')

    >>> slab = design.SpatialSelectiveExcitation(pp.Opts(), 8.0, 0.12, is_slab=True)
    >>> len(slab.blocks), slab.gz.type
    (1, 'grad')
    """

    def init_module(
        self,
        system: pp.Opts,
        flip_angle_deg: float,
        thickness_m: float,
        duration_s: float = 3e-3,
        *,
        is_slab: bool = False,
        rephase: bool = True,
        time_bw_product: float = 4.0,
        axis: str = "z",
        pulse_type: str = "st",
        use: str = "excitation",
        passband_ripple: float = 0.01,
        stopband_ripple: float = 0.01,
    ) -> None:
        if thickness_m <= 0:
            raise ValueError("thickness_m must be positive")
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")

        rf, gz, gz_reph = pp.make_slr_pulse(
            np.deg2rad(flip_angle_deg),
            duration=duration_s,
            slice_thickness=thickness_m,
            time_bw_product=time_bw_product,
            pulse_type=pulse_type,
            passband_ripple=passband_ripple,
            stopband_ripple=stopband_ripple,
            return_gz=True,
            use=use,
            system=system,
        )
        gz.channel = axis
        gz_reph.channel = axis

        # The thickness the pulse and its selection gradient actually produce,
        # which is the pulse's measured bandwidth over the gradient it is
        # played on. It differs from `thickness_m` by however much the
        # designed envelope's spectrum differs from its nominal time-bandwidth
        # product, so it is what a reconstruction should be told. Taken before
        # a slab's rephaser is concatenated onto the lobe.
        self.slice_thickness = float(pp.calc_rf_bandwidth(rf) / abs(gz.amplitude))
        self.selection_amplitude = float(gz.amplitude)

        self.seq = pp.Sequence(system)

        if is_slab:
            if rephase:
                # Concatenated, not summed over one interval: the rephaser
                # starts where the selection lobe ends.
                gz_reph.delay = pp.calc_duration(gz)
                gz = pp.add_gradients(grads=[gz, gz_reph], system=system)
            self.seq.add_block(rf, gz)
        else:
            self.seq.add_block(rf, gz)
            if rephase:
                self.seq.add_block(gz_reph)

        self.center = rf_reference(rf)


class SpatialSelectiveRefocusing(RfModule):
    """SLR refocusing with matched crushers joined to the selection plateau.

    Uses the spin-echo SLR profile. The default RF phase is pi/2 radians
    for a zero-phase excitation (CPMG).

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    thickness_m : float
        Slice thickness (m).
    flip_angle_deg : float, optional
        Nominal flip angle (degrees).
    duration_s : float, optional
        Pulse duration (s).
    spoiling_cycles : float, optional
        Cycles of dephasing each crusher winds across ``voxel_size_m``. Zero
        leaves the bare selection lobe.
    voxel_size_m : float, optional
        Length the dephasing is counted over (m).
    time_bw_product : float, optional
        Time-bandwidth product.
    axis : {'z', 'x', 'y'}, optional
        Selection axis; the crushers share it.
    phase_offset_rad : float, optional
        RF phase. The CPMG quarter turn by default.
    use : str, optional
        What the pulse is for; the trajectory core negates accumulated k at a
        refocusing pulse.
    passband_ripple, stopband_ripple : float, optional
        Ripple allowed in each band of the slice profile.

    Attributes
    ----------
    rf_ref : RfEvent
        The refocusing pulse, delayed onto the plateau between the crushers.
    gz : GradEvent
        Crusher, selection plateau and crusher, as one gradient.
    selection_amplitude : float
        Plateau amplitude of the selection lobe (Hz/m), which a slice offset
        is converted against; ``gz.amplitude`` is the crushers' peak.

    Raises
    ------
    ValueError
        If a thickness, duration or voxel size is not positive, ``axis`` is not
        a gradient channel, or ``spoiling_cycles`` is negative.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> refocusing = design.SpatialSelectiveRefocusing(pp.Opts(), 5e-3)
    >>> len(refocusing.blocks), refocusing.gz.type
    (1, 'grad')
    """

    def init_module(
        self,
        system: pp.Opts,
        thickness_m: float,
        flip_angle_deg: float = 180.0,
        duration_s: float = 3e-3,
        *,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
        time_bw_product: float = 4.0,
        axis: str = "z",
        phase_offset_rad: float = np.pi / 2,
        use: str = "refocusing",
        passband_ripple: float = 0.01,
        stopband_ripple: float = 0.01,
    ) -> None:
        if thickness_m <= 0:
            raise ValueError("thickness_m must be positive")
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")

        rf_ref, gz, _ = pp.make_slr_pulse(
            np.deg2rad(flip_angle_deg),
            duration=duration_s,
            slice_thickness=thickness_m,
            time_bw_product=time_bw_product,
            pulse_type="se",
            phase_offset=phase_offset_rad,
            passband_ripple=passband_ripple,
            stopband_ripple=stopband_ripple,
            return_gz=True,
            use=use,
            system=system,
        )
        gz.channel = axis
        self.selection_amplitude = float(gz.amplitude)

        if spoiling_cycles:
            # Each crusher is solved to arrive at, or leave from, the plateau,
            # so the three pieces share their vertices and become one waveform:
            # crusher up, selection flat top, crusher down.
            _, times_pre, amplitudes_pre = pp.make_crusher(
                spoiling_cycles,
                voxel_size_m,
                axis,
                grad_end=gz.amplitude,
                system=system,
            )
            _, times_post, amplitudes_post = pp.make_crusher(
                spoiling_cycles,
                voxel_size_m,
                axis,
                grad_start=gz.amplitude,
                system=system,
            )
            flat_start = float(times_pre[-1])
            flat_end = flat_start + float(gz.flat_time)
            rf_ref.delay += flat_start - float(gz.rise_time)
            gz = pp.make_extended_trapezoid(
                channel=axis,
                amplitudes=np.concatenate(
                    [amplitudes_pre, [gz.amplitude], amplitudes_post[1:]]
                ),
                times=np.concatenate(
                    [times_pre, [flat_end], flat_end + np.asarray(times_post[1:])]
                ),
                system=system,
            )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf_ref, gz)

        self.center = rf_reference(rf_ref)
