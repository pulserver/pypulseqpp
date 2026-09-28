"""RF-module simulation and pulse-centre timing."""

from __future__ import annotations

__all__ = ["RfModule", "rf_reference", "selective_slr"]

from typing import Any

import numpy as np

import pypulseqpp as pp

from .._module import SequenceModule


def rf_reference(rf: Any) -> float:
    """Return the RF pulse centre relative to its block start, in seconds: delay + center.

    Parameters
    ----------
    rf : object
        An RF event.

    Returns
    -------
    float
        Its centre from the start of the block that plays it, in seconds.
    """
    return float(rf.delay) + float(rf.center)


def selective_slr(flip_angle_deg: float, thickness_m: float, **design: Any) -> tuple:
    """Design an SLR pulse with the selection gradient and rephaser that select ``thickness_m``.

    The thickness a pulse selects is its measured bandwidth,
    :func:`pypulseqpp.calc_rf_bandwidth`, over the selection plateau.
    :func:`pypulseqpp.make_slr_pulse` sizes the plateau for the nominal
    bandwidth, the time-bandwidth product over the duration, from which the
    designed envelope's spectrum departs; the plateau is sized again for the
    measured bandwidth.

    Parameters
    ----------
    flip_angle_deg : float
        Nominal flip angle (degrees).
    thickness_m : float
        Thickness to select (m).
    **design
        The rest of :func:`pypulseqpp.make_slr_pulse`'s arguments, without
        ``slice_thickness`` and ``return_gz``.

    Returns
    -------
    tuple
        ``(rf, gz, gz_reph)``, as :func:`pypulseqpp.make_slr_pulse` returns
        them under ``return_gz``.
    """
    flip = np.deg2rad(flip_angle_deg)
    rf, gz, _ = pp.make_slr_pulse(
        flip, slice_thickness=thickness_m, return_gz=True, **design
    )
    nominal = abs(gz.amplitude) * thickness_m
    return pp.make_slr_pulse(
        flip,
        slice_thickness=thickness_m * nominal / pp.calc_rf_bandwidth(rf),
        return_gz=True,
        **design,
    )


class RfModule(SequenceModule):
    """Sequence module with off-resonance simulation of an individual RF pulse.

    Every excitation and preparation module the package ships is an
    ``RfModule``, so the off-resonance response of its pulse is available from
    the module itself.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> module = design.SpatialSelectiveExcitation(pp.Opts(), 15.0, 5e-3)
    >>> isinstance(module, design.RfModule)
    True
    >>> mz_z, mz_xy, frequency = module.sim_rf()[:3]
    >>> mz_xy.shape == frequency.shape
    True
    """

    def sim_rf(self, pulse=None, **kwargs):
        """Simulate this module's pulse across off-resonance.

        Parameters
        ----------
        pulse : RfEvent, default=None
            Pulse to simulate; defaults to the first RF event in block order.
        **kwargs
            Forwarded to :func:`pypulseqpp.sim_rf` (``rephase_factor``,
            ``df``, ``bandwidth_multiplier``, ``dt``).

        Returns
        -------
        tuple
            MATLAB's six answers: ``mz_z``, ``mz_xy``, the frequency axis,
            ``ref_eff``, ``mx_xy`` and ``my_xy``.

        Examples
        --------
        >>> import pypulseqpp.sequences as design
        >>> import pypulseqpp as pp
        >>> pulse = design.SpatialSelectiveExcitation(pp.Opts(), 15.0, 5e-3)
        >>> mz_z, mz_xy, frequency = pulse.sim_rf()[:3]
        >>> mz_z.shape == frequency.shape
        True
        """
        import pypulseqpp as pp

        if pulse is None:
            pulse = next(
                event
                for block in self.blocks
                for event in block
                if getattr(event, "type", None) == "rf"
            )
        return pp.sim_rf(pulse, **kwargs)
