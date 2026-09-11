"""T2 and joint T1/T2 preparation: weight the magnetization, then store it."""

from __future__ import annotations

__all__ = ["T1T2Preparation", "T2Preparation"]

import numpy as np

import pypulseqpp as pp

from ..excitation._base import RfModule, rf_reference
from ._common import spoiler_gradients

_FINAL_TIPS = ("up", "down")


class T2Preparation(RfModule):
    """Adiabatic T2 preparation with tip-down, refocusing and storage pulses.

    Refocusing pulses must occur in pairs to cancel their B1-dependent phase.
    The final half passage is time-reversed and conjugated relative to the
    first; final_tip selects storage on +z or -z.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    echo_time_s : float
        Preparation echo time (s), first pulse centre to last.
    final_tip : {'up', 'down'}, optional
        Store on ``+z`` (pure T2 weighting) or ``-z``, which starts a T1
        recovery as well -- see :class:`T1T2Preparation`.
    n_refocus : int, optional
        Adiabatic refocusing pulses, spread evenly through the echo time. Must
        be even; see above.
    half_passage_duration_s : float, optional
        Duration of each half passage (s).
    refocusing_duration_s : float, optional
        Duration of each refocusing pulse (s).
    adiabaticity : int, optional
        Sweep-rate margin over the adiabatic condition, for every pulse.
    dwell_s : float, optional
        RF raster (s).
    spoiling_cycles : float, optional
        Cycles of dephasing the closing spoiler winds across ``voxel_size_m``
        on each axis. Zero omits the spoiler block.
    voxel_size_m : float, optional
        Length the dephasing is counted over (m).
    labels : sequence of str, optional
        Counters set to 0 on the first pulse's block.

    Attributes
    ----------
    rf_prep : RfEvent
        The half passage that tips down.
    rf_ref : RfEvent
        The refocusing pulse; every repeat plays this one event.
    rf_store : RfEvent
        The reverse half passage that stores what is left.
    wait_first, wait_inner, wait_last : DelayEvent
        The gaps before the first refocusing pulse, between refocusing
        pulses, and before ``rf_store``. Each is absent when its gap is
        zero, and ``wait_inner`` when there is one refocusing pulse.
    gx_spoil, gy_spoil, gz_spoil : GradEvent
        The closing spoiler.
    prep_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``.
    echo_time : float
        The preparation echo time achieved on the block raster (s).

    Raises
    ------
    ValueError
        If ``final_tip`` is not ``'up'`` or ``'down'``, ``n_refocus`` is not a
        positive even number, a size is out of range, or the echo time is
        shorter than the pulses themselves.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> prep = design.T2Preparation(pp.Opts(), 50e-3)
    >>> round(prep.echo_time * 1e3, 2)
    49.99
    """

    def init_module(
        self,
        system: pp.Opts,
        echo_time_s: float,
        *,
        final_tip: str = "up",
        n_refocus: int = 2,
        half_passage_duration_s: float = 4e-3,
        refocusing_duration_s: float = 10.24e-3,
        adiabaticity: int = 8,
        dwell_s: float = 10e-6,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
        labels: tuple[str, ...] | None = None,
    ) -> None:
        if final_tip not in _FINAL_TIPS:
            raise ValueError(
                f"final_tip must be one of {_FINAL_TIPS}, got {final_tip!r}"
            )
        n_refocus = int(n_refocus)
        if n_refocus < 2 or n_refocus % 2:
            raise ValueError(
                "n_refocus must be a positive even number: an adiabatic full passage leaves "
                "a transmit-dependent phase that only a second one undoes"
            )
        if echo_time_s <= 0:
            raise ValueError("echo_time_s must be positive")
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")

        rf_prep, rf_store = pp.make_half_passages(
            half_passage_duration_s,
            adiabaticity=adiabaticity,
            dwell=dwell_s,
            system=system,
        )
        rf_ref = pp.make_adiabatic_pulse(
            pulse_type="hypsec",
            duration=refocusing_duration_s,
            dwell=dwell_s,
            adiabaticity=adiabaticity,
            system=system,
            use="refocusing",
        )

        # With the refocusing pulses paired, the transverse phase arrives back
        # where the tip-down left it, so the storing pulse needs no phase of its
        # own -- except the half turn that stores on -z instead of +z.
        if final_tip == "down":
            rf_store.phase_offset += np.pi

        # The echo time runs centre to centre, and the refocusing pulses sit at
        # the centre of each of the n equal intervals it is cut into.
        interval = echo_time_s / n_refocus
        lead = pp.calc_duration(rf_prep) - pp.calc_rf_center(rf_prep)[0]
        centre_ref = pp.calc_rf_center(rf_ref)[0]
        span_ref = pp.calc_duration(rf_ref)
        trail = pp.calc_rf_center(rf_store)[0]

        first_gap = pp.round_to_raster(
            0.5 * interval - lead - centre_ref, system.block_duration_raster
        )
        inner_gap = pp.round_to_raster(
            interval - span_ref, system.block_duration_raster
        )
        last_gap = pp.round_to_raster(
            0.5 * interval - (span_ref - centre_ref) - trail,
            system.block_duration_raster,
        )
        if min(first_gap, inner_gap, last_gap) < -1e-12:
            shortest = lead + n_refocus * span_ref + trail
            raise ValueError(
                f"the requested preparation echo time of {echo_time_s * 1e3:.3f} ms is "
                f"shorter than the {shortest * 1e3:.3f} ms its pulses already occupy"
            )

        gx_spoil, gy_spoil, gz_spoil = (
            spoiler_gradients(system, spoiling_cycles, voxel_size_m)
            if spoiling_cycles
            else (None, None, None)
        )
        prep_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]

        wait_first = pp.make_delay(first_gap) if first_gap else None
        wait_inner = pp.make_delay(inner_gap) if inner_gap else None
        wait_last = pp.make_delay(last_gap) if last_gap else None

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf_prep, *prep_labels)
        for index in range(n_refocus):
            _wait = wait_first if index == 0 else wait_inner
            if _wait is not None:
                self.seq.add_block(_wait)
            self.seq.add_block(rf_ref)
        if wait_last is not None:
            self.seq.add_block(wait_last)

        # Read off the sequence rather than re-summed from the parts: the
        # blocks have been rounded onto the block raster by now, and the echo
        # time is where the pulses ended up, not where they were asked for.
        store_start = self.seq.duration()[0]
        self.seq.add_block(rf_store)
        if spoiling_cycles:
            self.seq.add_block(gx_spoil, gy_spoil, gz_spoil)

        self.center = rf_reference(rf_prep)
        self.echo_time = store_start + rf_reference(rf_store) - self.center
        self.final_tip = final_tip


class T1T2Preparation(T2Preparation):
    """T2 preparation with storage on -z to initiate T1 recovery.

    Accepts the :class:`T2Preparation` parameters except ``final_tip``, which
    is fixed to ``'down'``; passing it raises ``ValueError``. The acquisition
    loop supplies the recovery interval.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> prep = design.T1T2Preparation(pp.Opts(), 50e-3)
    >>> prep.final_tip
    'down'
    """

    def init_module(self, system: pp.Opts, echo_time_s: float, **kwargs) -> None:
        if "final_tip" in kwargs:
            raise ValueError(
                "T1T2Preparation is the final_tip='down' case; use T2Preparation to choose"
            )
        super().init_module(system, echo_time_s, final_tip="down", **kwargs)
