"""RF-module simulation and pulse-centre timing."""

from __future__ import annotations

__all__ = ["RfModule", "rf_reference"]

from typing import Any

from .._module import SequenceModule


def rf_reference(rf: Any) -> float:
    """Return the RF centre relative to its block start, in seconds: delay + center."""
    return float(rf.delay) + float(rf.center)


class RfModule(SequenceModule):
    """Sequence module with off-resonance simulation of an individual RF pulse."""

    def sim_rf(self, pulse=None, **kwargs):
        """Simulate this module's pulse across off-resonance.

        Parameters
        ----------
        pulse : RfEvent, optional
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
