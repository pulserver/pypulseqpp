"""Fat saturation, optionally confined to a band and pointed where asked."""

from __future__ import annotations

__all__ = ["FatSaturation"]

from typing import Any

import numpy as np

import pypulseqpp as pp

from ..excitation._base import RfModule, rf_reference
from ._common import AXES, spoiler_gradients

#: Where the main fat resonance sits relative to water. Negative: fat precesses
#: more slowly, so it is downfield of water in frequency.
FAT_SHIFT_PPM = -3.45

#: The transform exemptions a placed module carries. Its pulse already points
#: where it was told to, so nothing downstream may point it again.
FOV_EXEMPT_FLAGS = ("NOPOS", "NOROT")


class FatSaturation(RfModule):
    """SLR fat saturation with a three-axis spoiler and optional spatial selection.

    Position and orientation place the selection band at construction.
    NOPOS and NOROT are set on entry and cleared on exit so subsequent FOV
    placement leaves the band unchanged. Position is specified in logical
    millimetres; the RF frequency offset remains in ppm.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    freq_offset_ppm : float, optional
        Fat offset from water (ppm). Carried on the pulse as a ppm offset
        rather than as hertz, so the interpreter resolves it against the
        field the scan actually runs at.
    flip_angle_deg : float, optional
        Saturation flip angle (degrees).
    bandwidth_hz : float, optional
        Spectral passband (Hz). With ``time_bw_product`` it fixes the pulse
        duration, which is ``time_bw_product / bandwidth_hz``.
    time_bw_product : float, optional
        Time-bandwidth product.
    thickness_m : float, optional
        Band thickness (m). ``None`` saturates the whole transmit volume,
        which is the usual fat saturation and the only form that needs no
        gradient.
    axis : {'z', 'x', 'y'}, optional
        Band normal, before ``orientation`` turns it.
    position_mm : sequence of float, optional
        ``(dx, dy, dz)`` offset of the band from the isocentre, in the logical
        frame, in millimetres. Needs ``thickness_m``.
    orientation : array_like or scipy.spatial.transform.Rotation, optional
        A ``(3, 3)`` matrix, or a rotation, taking the logical frame to the one
        the band should sit in. Needs ``thickness_m``.
    use_rotation_extension : bool, optional
        Carry the orientation as a ``ROTATIONS`` extension rather than baking
        it into new waveforms. Only ``True`` is implemented; see
        :class:`~pypulseqpp.TransformFOV`.
    spoiling_cycles : float, optional
        Cycles of dephasing each spoiler axis winds across ``voxel_size_m``.
    voxel_size_m : float, optional
        Length the dephasing is counted over (m).

    Attributes
    ----------
    rf_prep : RfEvent
        The saturation pulse, as placed.
    gz : GradEvent
        Its selection gradient. Only when ``thickness_m`` was given, and named
        for the logical axis whatever ``axis`` was.
    gx_spoil, gy_spoil, gz_spoil : GradEvent
        Closing spoiler on all three axes.
    prep_labels : list of LabelSetEvent
        ``NOPOS`` and ``NOROT``, set on the first block.
    reset_labels : list of LabelSetEvent
        The same two cleared, on the last.
    freq_offset_ppm : float
        The fat offset the pulse carries (ppm).
    freq_offset_hz : float
        What that comes to at ``system.B0`` (Hz), for reference; the pulse
        itself is field-independent.

    Raises
    ------
    ValueError
        If a bandwidth, thickness or voxel size is not positive, ``axis`` is
        not a gradient channel, the spoiler was asked for zero cycles, or a
        position or orientation was given for a band that has no thickness.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> fatsat = design.FatSaturation(pp.Opts(B0=3.0))
    >>> len(fatsat.blocks), round(fatsat.freq_offset_hz)
    (2, -441)

    >>> [event.type for event in fatsat.blocks[0]]
    ['rf', 'labelset', 'labelset']
    """

    def init_module(
        self,
        system: pp.Opts,
        *,
        freq_offset_ppm: float = FAT_SHIFT_PPM,
        flip_angle_deg: float = 110.0,
        bandwidth_hz: float = 250.0,
        time_bw_product: float = 2.0,
        thickness_m: float | None = None,
        axis: str = "z",
        position_mm: Any = None,
        orientation: Any = None,
        use_rotation_extension: bool = True,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
    ) -> None:
        if bandwidth_hz <= 0 or time_bw_product <= 0:
            raise ValueError("bandwidth_hz and time_bw_product must be positive")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if spoiling_cycles <= 0:
            raise ValueError(
                "a fat saturation has to be spoiled: without it the fat the pulse just "
                "tipped over stays in the transverse plane and is read with the water"
            )
        if axis not in AXES:
            raise ValueError(f"axis must be one of {AXES}, got {axis!r}")
        if thickness_m is not None and thickness_m <= 0:
            raise ValueError("thickness_m must be positive")
        if thickness_m is None and (position_mm is not None or orientation is not None):
            raise ValueError(
                "a saturation without a thickness covers the whole transmit volume, so it "
                "has nowhere to be put; give thickness_m to make it a band"
            )

        designed = pp.make_slr_pulse(
            np.deg2rad(flip_angle_deg),
            duration=time_bw_product / bandwidth_hz,
            slice_thickness=0.0 if thickness_m is None else thickness_m,
            time_bw_product=time_bw_product,
            pulse_type="sat",
            freq_ppm=freq_offset_ppm,
            return_gz=thickness_m is not None,
            use="saturation",
            system=system,
        )
        rf_prep, gz = (
            (designed[0], designed[1]) if thickness_m is not None else (designed, None)
        )
        if gz is not None:
            gz.channel = axis

        gx_spoil, gy_spoil, gz_spoil = spoiler_gradients(
            system, spoiling_cycles, voxel_size_m
        )

        designed_seq = pp.Sequence(system)
        if gz is not None:
            designed_seq.add_block(rf_prep, gz)
        else:
            designed_seq.add_block(rf_prep)
        designed_seq.add_block(gx_spoil, gy_spoil, gz_spoil)

        if position_mm is not None or orientation is not None:
            # Over the pulse block only. The spoiler has no orientation to get
            # right -- it dephases the same however it is turned -- and turning
            # it would mix three lobes each already solved against the full
            # slew limit, so the rotated sum would exceed it.
            designed_seq = pp.TransformFOV(
                rotation=_as_matrix(orientation),
                # A shift is written in metres; the argument is millimetres
                # because a prescription is.
                translation=None
                if position_mm is None
                else tuple(float(v) * 1e-3 for v in position_mm),
                use_rotation_extension=use_rotation_extension,
                system=system,
            ).apply_to_sequence(
                designed_seq,
                time_range=[0.0, 0.5 * designed_seq.block_durations[1]],
            )

        # Rebuilt rather than edited in place: the placement rewrote the pulse,
        # so what this module plays -- and therefore publishes -- is the
        # transformed event, and the flags go on with it. The exemption belongs
        # on the block that carries the pulse, and a block of its own would be
        # a block of dead time.
        placed = [designed_seq.get_block(index) for index in designed_seq.block_events]
        rf_prep, gz = placed[0].rf, placed[0].gz
        gx_spoil, gy_spoil, gz_spoil = placed[-1].gx, placed[-1].gy, placed[-1].gz

        prep_labels = [
            pp.make_label(type="SET", label=name, value=1) for name in FOV_EXEMPT_FLAGS
        ]
        # Cleared on the way out: Pulseq labels are sticky, so an exemption left
        # set would go on exempting every block after this module.
        reset_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in FOV_EXEMPT_FLAGS
        ]

        self.seq = pp.Sequence(system)
        for index, block in enumerate(placed):
            self.seq.add_block(
                *(
                    event
                    for event in pp.block_to_events(block)
                    if hasattr(event, "type")
                ),
                *(prep_labels if index == 0 else ()),
                *(reset_labels if index == len(placed) - 1 else ()),
            )

        self.center = rf_reference(rf_prep)
        self.freq_offset_ppm = float(freq_offset_ppm)
        self.freq_offset_hz = float(1e-6 * system.gamma * system.B0 * freq_offset_ppm)


def _as_matrix(orientation: Any) -> np.ndarray | None:
    if orientation is None:
        return None
    if hasattr(orientation, "as_matrix"):
        return orientation.as_matrix()
    return np.asarray(orientation, dtype=float)
