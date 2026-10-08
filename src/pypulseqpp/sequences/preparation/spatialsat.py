"""Spatial saturation band of any orientation, placed in the physical frame."""

from __future__ import annotations

__all__ = ["SpatialSaturation", "spatial_saturations"]

from collections.abc import Iterable
from typing import Any

import numpy as np

import pypulseqpp as pp

from ..excitation._base import RfModule, rf_reference
from ._common import AXES, spoiler_gradients

#: The transform exemptions the pulse block carries: the band is placed in the
#: physical frame, so neither the prescription's offset nor its rotation may
#: move it.
FOV_EXEMPT_FLAGS = ("NOPOS", "NOROT")


class SpatialSaturation(RfModule):
    """Slab-selective saturation pulse in the physical frame, followed by a three-axis spoiler.

    The selection gradient points along ``normal`` and is played on the
    gradient channels in proportion to its components. The slab is centred
    ``position`` from the isocentre along ``normal``. ``NOPOS`` and ``NOROT``
    are set on the pulse block and cleared on the spoiler block, so neither a
    :class:`~pypulseqpp.TransformFOV` offset nor the interpreter's prescription
    rotation moves the band: it stays where it was placed in the physical
    frame, whatever the field of view.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    normal : array_like of float
        Slab normal as ``(x, y, z)`` along the physical gradient axes.
        Normalised by the module; not the zero vector.
    position : float
        Centre of the slab along ``normal`` from the isocentre (m).
    thickness : float
        Slab thickness along ``normal`` (m).
    flip_angle_deg : float, default=90.0
        Saturation flip angle (degrees).
    duration_s : float, default=0.003
        Pulse duration (s).
    time_bw_product : float, default=4.0
        Time-bandwidth product. With the duration it fixes the selection
        amplitude ``time_bw_product / (duration_s * thickness)`` (Hz/m).
    spoiling_cycles : float, default=4.0
        Cycles of dephasing each spoiler axis winds across ``voxel_size_m``.
    voxel_size_m : float, default=0.001
        Length the dephasing is counted over (m).
    labels : sequence of str, default=None
        Counters set to zero on the pulse block.

    Attributes
    ----------
    rf_prep : RfEvent
        The saturation pulse. Its frequency offset is the selection amplitude
        times ``position`` (Hz) and its phase offset references the phase to
        the pulse centre.
    gx_sel, gy_sel, gz_sel : GradEvent
        Selection gradient on each channel, ``gz.amplitude * normal`` as
        stored (Hz/m); absent for a channel whose component is zero.
    gx_spoil, gy_spoil, gz_spoil : GradEvent
        Closing spoiler on all three axes.
    prep_labels : list of LabelSetEvent
        ``NOPOS`` and ``NOROT`` set, then one per name in ``labels``, on the
        pulse block.
    reset_labels : list of LabelSetEvent
        ``NOPOS`` and ``NOROT`` cleared, on the spoiler block.

    Raises
    ------
    ValueError
        If ``normal`` is the zero vector or not three components, or a
        thickness, duration, time-bandwidth product, spoiling count or voxel
        size is not positive.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> band = design.SpatialSaturation(pp.Opts(), (0, 1, 0), 0.05, 0.02)
    >>> len(band.blocks)
    2
    >>> [(event.label, int(event.value)) for event in band.prep_labels]
    [('NOPOS', 1), ('NOROT', 1)]
    """

    def init_module(
        self,
        system: pp.Opts,
        normal: Any,
        position: float,
        thickness: float,
        *,
        flip_angle_deg: float = 90.0,
        duration_s: float = 3e-3,
        time_bw_product: float = 4.0,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
        labels: tuple[str, ...] | None = None,
    ) -> None:
        normal = np.asarray(normal, dtype=float)
        if normal.shape != (3,) or not np.linalg.norm(normal):
            raise ValueError("normal must be a nonzero 3-vector")
        if (
            min(thickness, duration_s, time_bw_product, spoiling_cycles, voxel_size_m)
            <= 0
        ):
            raise ValueError(
                "thickness, duration_s, time_bw_product, spoiling_cycles and "
                "voxel_size_m must be positive"
            )
        normal = normal / np.linalg.norm(normal)

        rf_prep, gz = pp.make_sinc_pulse(
            np.deg2rad(flip_angle_deg),
            duration=duration_s,
            slice_thickness=thickness,
            time_bw_product=time_bw_product,
            return_gz=True,
            use="saturation",
            system=system,
        )[:2]
        # Same convention as a selective excitation: the offset places the
        # slab, and the phase is referenced to the pulse centre.
        rf_prep.freq_offset = float(gz.amplitude) * float(position)
        rf_prep.phase_offset = -2 * np.pi * rf_prep.freq_offset * rf_prep.center

        selection = []
        for axis, component in zip(AXES, normal, strict=True):
            if component == 0.0:
                continue
            gradient = pp.scale_grad(gz, float(component))
            gradient.channel = axis
            setattr(self, f"g{axis}_sel", gradient)
            selection.append(gradient)

        gx_spoil, gy_spoil, gz_spoil = spoiler_gradients(
            system, spoiling_cycles, voxel_size_m
        )
        prep_labels = [
            pp.make_label(type="SET", label=name, value=1) for name in FOV_EXEMPT_FLAGS
        ] + [pp.make_label(type="SET", label=name, value=0) for name in labels or ()]
        # Cleared on the way out: Pulseq labels are sticky, so an exemption left
        # set would go on exempting every block after this module.
        reset_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in FOV_EXEMPT_FLAGS
        ]

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf_prep, *selection, *prep_labels)
        self.seq.add_block(gx_spoil, gy_spoil, gz_spoil, *reset_labels)
        self.center = rf_reference(rf_prep)


def spatial_saturations(
    system: pp.Opts,
    bands: Iterable[tuple[Any, float, float]],
    **kwargs: Any,
) -> list[SpatialSaturation]:
    """Return the :class:`SpatialSaturation` of each ``(normal, position, thickness)`` band with a thickness.

    A band of zero thickness is not played, which is how a sequence function
    leaves one of its bands off. ``kwargs`` go to every module.

    Raises
    ------
    ValueError
        If a thickness is negative, or a band with a thickness has a zero
        normal.
    """
    played = []
    for normal, position, thickness in bands:
        if thickness < 0:
            raise ValueError(f"a band thickness must not be negative, got {thickness}")
        if not thickness:
            continue
        if not any(normal):
            raise ValueError(
                "a saturation band with a thickness needs a nonzero normal"
            )
        played.append(SpatialSaturation(system, normal, position, thickness, **kwargs))
    return played
