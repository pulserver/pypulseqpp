"""Diffusion preparation: weight the magnetisation, then store it on z."""

from __future__ import annotations

__all__ = ["DiffusionPreparation"]

import math

import numpy as np

import pypulseqpp as pp

from ..excitation._base import RfModule, rf_reference
from ._common import spoiler_gradients


class DiffusionPreparation(RfModule):
    """Non-selective diffusion preparation with hard tip-down and storage pulses.

    A pair of identical diffusion lobes surrounds one refocusing pulse. The
    pair is designed on one axis at ``b_value``; scale it for lower b-values
    (:meth:`scale_for`) and rotate it per direction. The closing spoiler is
    separate from the pair. The lobes turn with the prescription, so a
    direction is in its logical frame, as
    :func:`~pypulseqpp.io.read_diffusion_table` returns it.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    b_value : float
        Design b-value (s/mm^2), the largest the module plays; lower values are
        reached by scaling the lobes.
    gradient_duration_s : float, default=0.02
        Duration of each diffusion lobe (s).
    gradient_separation_s : float, default=0.03
        Onset-to-onset separation of the two lobes (s); the gap between them
        holds the refocusing pulse.
    pulse_duration_s : float, default=0.0005
        Duration of each 90 degree pulse (s).
    refocusing_duration_s : float, default=0.001
        Duration of the 180 degree pulse (s).
    axis : {'z', 'x', 'y'}, default='z'
        Gradient channel the pair is designed on.
    spoiling_cycles : float, default=4.0
        Cycles of dephasing of the closing spoiler across ``voxel_size_m``.
        Zero omits the spoiler.
    voxel_size_m : float, default=0.001
        Length the dephasing is counted over (m).
    labels : sequence of str, default=None
        Labels set to 0 on the first pulse's block, to be changed per shot by
        the caller.

    Attributes
    ----------
    rf_prep : RfEvent
        The 90 degree tip-down pulse.
    rf_ref : RfEvent
        The 180 degree refocusing pulse between the lobes.
    rf_store : RfEvent
        The 90 degree storage pulse.
    g_diff : TrapEvent
        The diffusion lobe at the design b-value, played identically on both
        sides of the refocusing pulse.
    wait_first, wait_last : DelayEvent
        The gaps before and after the refocusing pulse; each is absent when
        zero.
    gx_spoil, gy_spoil, gz_spoil : GradEvent
        Spoiler gradient of each axis; absent when ``spoiling_cycles`` is zero.
    prep_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``.
    b_value : float
        Design b-value of the lobes as played (s/mm^2).
    max_b_value : float
        b-value this timing reaches at ``system.max_grad`` (s/mm^2).

    Raises
    ------
    ValueError
        If a duration is out of order, the separation cannot hold the
        refocusing pulse, or the b-value exceeds what the system can reach --
        the message says what it can.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> prep = design.DiffusionPreparation(system, 500.0)
    >>> round(prep.b_value)
    500
    >>> round(prep.scale_for(125.0), 3)
    0.5

    The whole module, pulses and diffusion lobes together. Both lobes have
    the same sign; the refocusing pulse between them inverts the phase
    accrued during the first, so the pair produces diffusion weighting
    rather than cancelling:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp.sequences import DiffusionPreparation

       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       DiffusionPreparation(system, 500.0).seq.paper_plot()
       plt.show()
    """

    def init_module(
        self,
        system: pp.Opts,
        b_value: float,
        *,
        gradient_duration_s: float = 20e-3,
        gradient_separation_s: float = 30e-3,
        pulse_duration_s: float = 0.5e-3,
        refocusing_duration_s: float = 1e-3,
        axis: str = "z",
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
        labels: tuple[str, ...] | None = None,
    ) -> None:
        if b_value <= 0:
            raise ValueError("b_value must be positive")
        if gradient_duration_s <= 0 or gradient_separation_s <= gradient_duration_s:
            raise ValueError(
                "require gradient_separation_s > gradient_duration_s > 0: the lobes have to "
                "be separated by more than their own length for anything to sit between them"
            )
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")

        rf_prep = pp.make_block_pulse(
            flip_angle=np.pi / 2.0,
            duration=pulse_duration_s,
            use="preparation",
            system=system,
        )
        rf_ref = pp.make_slr_pulse(
            np.pi,
            duration=refocusing_duration_s,
            slice_thickness=0.0,
            pulse_type="se",
            phase_offset=np.pi / 2.0,
            use="refocusing",
            system=system,
        )
        # Half a turn from the tip-down: the refocusing pulse is about the
        # perpendicular axis, which leaves the tipped magnetisation where it is,
        # so undoing the first rotation is what carries it back to +z.
        rf_store = pp.make_block_pulse(
            flip_angle=np.pi / 2.0,
            duration=pulse_duration_s,
            phase_offset=np.pi,
            use="preparation",
            system=system,
        )

        g_full = pp.make_trapezoid(
            channel=axis,
            amplitude=system.max_grad,
            duration=gradient_duration_s,
            system=system,
        )
        remaining = (
            gradient_separation_s - pp.calc_duration(g_full) - pp.calc_duration(rf_ref)
        )
        if remaining < -1e-12:
            raise ValueError(
                f"a separation of {gradient_separation_s * 1e3:.3f} ms cannot hold a "
                f"{pp.calc_duration(g_full) * 1e3:.3f} ms lobe and a "
                f"{pp.calc_duration(rf_ref) * 1e3:.3f} ms refocusing pulse"
            )
        raster = system.block_duration_raster
        first_gap = pp.round_to_raster(0.5 * remaining, raster)
        second_gap = pp.round_to_raster(remaining - first_gap, raster)

        max_b_value = _pair_b_value(
            g_full,
            first_gap + pp.calc_duration(rf_ref) + second_gap,
            system.grad_raster_time,
        )
        if b_value > max_b_value * (1.0 + 1e-9):
            raise ValueError(
                f"a b-value of {b_value:.1f} s/mm^2 is beyond this timing and gradient "
                f"strength, which reach {max_b_value:.1f} s/mm^2; lengthen the lobes or "
                f"their separation"
            )
        g_diff = pp.scale_grad(g_full, math.sqrt(b_value / max_b_value))

        gx_spoil, gy_spoil, gz_spoil = (
            spoiler_gradients(system, spoiling_cycles, voxel_size_m)
            if spoiling_cycles
            else (None, None, None)
        )
        prep_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf_prep, *prep_labels)
        self.seq.add_block(g_diff)
        if first_gap:
            wait_first = pp.make_delay(first_gap)
            self.seq.add_block(wait_first)
        self.seq.add_block(rf_ref)
        if second_gap:
            wait_last = pp.make_delay(second_gap)
            self.seq.add_block(wait_last)
        self.seq.add_block(g_diff)
        self.seq.add_block(rf_store)
        if spoiling_cycles:
            self.seq.add_block(gx_spoil, gy_spoil, gz_spoil)

        self.center = rf_reference(rf_prep)
        self.b_value = float(b_value)
        self.max_b_value = float(max_b_value)

    def scale_for(self, b_value: float) -> float:
        """Return the gradient-amplitude factor for a requested b-value.

        Parameters
        ----------
        b_value : float
            Wanted b-value (s/mm^2), from zero to the design value.

        Returns
        -------
        float
            Amplitude factor for :func:`pypulseqpp.scale_grad` applied to
            ``g_diff``.

        Raises
        ------
        ValueError
            If ``b_value`` is negative or above the design value.
        """
        if not 0.0 <= b_value <= self.b_value:
            raise ValueError(
                f"b_value must lie between 0 and the design value of {self.b_value:.1f} "
                f"s/mm^2, got {b_value:.1f}"
            )
        return math.sqrt(b_value / self.b_value) if self.b_value else 0.0


def _pair_b_value(lobe, middle_duration: float, raster: float) -> float:
    """Return b (s/mm^2) of two identical lobes around a refocusing pulse.

    The second lobe is negated in the effective gradient.
    """
    count = round(pp.calc_duration(lobe) / raster)
    times = (np.arange(count) + 0.5) * raster
    knots = np.cumsum([0.0, lobe.rise_time, lobe.flat_time, lobe.fall_time])
    samples = np.interp(times, knots, [0.0, lobe.amplitude, lobe.amplitude, 0.0])

    gap = np.zeros(round(middle_duration / raster))
    gradient = np.concatenate((samples, gap, -samples))
    # Amplitudes are Hz/m, so 2 pi turns the integral into rad/m; the result is
    # s/m^2 and s/mm^2 is a million times smaller.
    q_rad_per_m = 2.0 * np.pi * np.cumsum(gradient) * raster
    return float(np.sum(q_rad_per_m**2) * raster / 1e6)
