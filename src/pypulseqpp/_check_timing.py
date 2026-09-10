"""Raster, dead-time, gradient-continuity and stored-duration checks."""

from __future__ import annotations

from types import SimpleNamespace

from . import _ext as _cxx

__all__ = ["check_timing", "describe", "print_error_report"]


#: One message per kind of problem, in f-string syntax over the finding's own
#: fields plus the unit it is read in.
error_messages = {
    "RASTER": "{value*multiplier:.2f} {unit} does not align to {raster} (Nearest valid value: {value_rounded*multiplier:.0f} {unit}, error: {error*multiplier:.2f} {unit})",
    "ADC_DEAD_TIME": "ADC delay is smaller than ADC dead time ({value*multiplier:.2f} {unit} < {dead_time*multiplier:.0f} {unit})",
    "POST_ADC_DEAD_TIME": "Post-ADC dead time exceeds block duration ({value*multiplier:.2f} {unit} + {dead_time*multiplier:.0f} {unit} > {duration*multiplier} {unit})",
    "BLOCK_DURATION_MISMATCH": "Inconsistency between the stored block duration ({duration*multiplier:.2f} {unit}) and the content of the block ({value*multiplier:.2f} {unit})",
    "RF_DEAD_TIME": "Delay of {value*multiplier:.2f} {unit} is smaller than the RF dead time {dead_time*multiplier:.0f} {unit}",
    "RF_RINGDOWN_TIME": "Time between the end of the RF pulse at {value*multiplier:.2f} {unit} and the end of the block at {duration * multiplier:.2f} {unit} is shorter than rf_ringdown_time ({ringdown_time*multiplier:.0f} {unit})",
    "NEGATIVE_DELAY": "Delay is negative {value*multiplier:.2f} {unit}",
    "SOFT_DELAY_FACTOR": "Soft delay {hint}/{numID} has factor parameter as zero, which makes duration calculation undefined.",
    "SOFT_DELAY_DUR_INCONSISTENCY": "Soft delay {hint}/{numID} default duration derived from this block ({value*1e6} us) is inconsistent with the previous default.",
    "SOFT_DELAY_HINT_INCONSISTENCY": "Soft delay {hint}/{numID}: Soft delays with the same numeric ID are expected to share the same text hint but previous hint recorded is {prev_hint}.",
    "SOFT_DELAY_INVALID_NUMID": "Soft delay {hint}/{numID} has an invalid numeric ID {numID}. Numeric IDs must be non-negative integers.",
    "ADC_SAMPLES_DIVISOR": "ADC num_samples is not an integer multiple of adc_samples_divisor ({value} / {divisor}).",
    "GRADIENT_START_DELAY": "Gradient starts at {amplitude:.0f} Hz/m but is delayed by {value*multiplier:.2f} {unit}, which leaves the axis at zero until then",
    "GRADIENT_END_NONZERO": "Gradient is left at {amplitude:.0f} Hz/m after {value*multiplier:.2f} {unit}, before the end of the block at {duration*multiplier:.2f} {unit}",
    "GRADIENT_DISCONTINUITY": "Gradient starts at {value:.0f} Hz/m where the previous block left the axis at {before:.0f} Hz/m, a step of {slew:.0f} Hz/m/s against a limit of {limit:.0f} Hz/m/s",
    "GRADIENT_NOT_RAMPED_DOWN": "The sequence ends with a gradient still on: the axes are not ramped down",
    "FREQ_OFFSET": "Frequency offset of {offset:.0f} Hz is outside what the scanner will play ({limit:.0f} Hz)",
    "TOTAL_DURATION_MISMATCH": "TotalDuration is recorded as {value:.9g} s, but the blocks add up to {duration:.9g} s",
}


def _limit(system, name: str, fallback: float) -> float:
    value = getattr(system, name, None)
    return fallback if value is None else float(value)


def check_timing(seq) -> tuple[bool, list[SimpleNamespace]]:
    """Check timing, gradient continuity and the stored total duration.

    Parameters
    ----------
    seq : Sequence
        The sequence to judge.

    Returns
    -------
    is_ok : bool
        True when nothing was found.
    error_report : list of SimpleNamespace
        One entry per problem, in block order, with anything about the
        sequence as a whole last and carrying block 0. Each carries ``block``,
        ``event``, ``field`` and ``error_type``, plus the values that kind of
        problem reports.

    Raises
    ------
    ValueError
        If ``seq`` was built without a system to judge against.

    Notes
    -----
    Records TotalDuration when absent or invalidated by a sequence edit.
    An existing duration loaded from a file is checked, not overwritten.
    """
    system = seq.system
    if system is None:
        raise ValueError(
            "check_timing() needs the system the sequence was designed "
            "against; build the Sequence with a system= argument"
        )

    findings = _cxx.check_timing(
        seq._native,
        rf_raster_time=_limit(system, "rf_raster_time", 1e-6),
        grad_raster_time=_limit(system, "grad_raster_time", 10e-6),
        adc_raster_time=_limit(system, "adc_raster_time", 100e-9),
        block_duration_raster=_limit(system, "block_duration_raster", 10e-6),
        rf_dead_time=_limit(system, "rf_dead_time", 0.0),
        rf_ringdown_time=_limit(system, "rf_ringdown_time", 0.0),
        adc_dead_time=_limit(system, "adc_dead_time", 0.0),
        adc_samples_divisor=_limit(system, "adc_samples_divisor", 1.0),
        max_freq_offset=_limit(system, "max_freq_offset", 0.0),
        larmor=_limit(system, "gamma", 42576000.0) * _limit(system, "B0", 1.5),
    )
    error_report = [SimpleNamespace(**finding) for finding in findings]
    error_report.extend(_continuity(seq))
    error_report.sort(key=lambda finding: finding.block)

    disagreement = _total_duration(seq)
    if disagreement is not None:
        error_report.append(disagreement)

    return len(error_report) == 0, error_report


def _continuity(seq) -> list[SimpleNamespace]:
    """Report inter-block gradient jumps and a nonzero final gradient."""
    from .safety import check_grad_continuity

    report = check_grad_continuity(seq)[1]
    found = [
        SimpleNamespace(
            block=jump.block,
            event=f"g{'xyz'[jump.axis]}",
            field="first",
            error_type="GRADIENT_DISCONTINUITY",
            value=jump.after,
            before=jump.before,
            slew=jump.slew,
            limit=jump.limit,
        )
        for jump in report.discontinuities
    ]
    if not report.ends_at_zero:
        found.append(
            SimpleNamespace(
                block=len(seq),
                event="grad",
                field="last",
                error_type="GRADIENT_NOT_RAMPED_DOWN",
            )
        )
    return found


def _total_duration(seq) -> SimpleNamespace | None:
    """Record TotalDuration if absent; otherwise check it within 1 ns.

    Return None after recording or on agreement, and a finding on mismatch.
    """
    played = seq.duration()[0]
    recorded = seq.get_definition("TotalDuration") if seq._duration else ""
    if recorded == "":
        seq.set_definition("TotalDuration", played)
        seq._duration = 1
        return None

    stated = float(recorded[0] if isinstance(recorded, list) else recorded)
    if abs(stated - played) <= 1e-9:
        return None
    return SimpleNamespace(
        block=0,
        event="sequence",
        field="TotalDuration",
        error_type="TOTAL_DURATION_MISMATCH",
        value=stated,
        duration=played,
    )


def _format_message(template: str, **fields) -> str:
    """Evaluate a trusted f-string template against the supplied finding fields."""
    return eval(f'f"""{template}"""', fields)  # noqa: S307


def _message(finding: SimpleNamespace) -> str:
    unit, multiplier = ("ns", 1e9) if finding.field == "dwell" else ("us", 1e6)
    return _format_message(
        error_messages[finding.error_type],
        **finding.__dict__,
        unit=unit,
        multiplier=multiplier,
    )


def describe(finding: SimpleNamespace) -> str:
    """Return one finding as a line naming where in the sequence it is."""
    where = f"Block:{finding.block} " if finding.block else ""
    return f"   {where}{finding.event}.{finding.field}: {_message(finding)}"


def print_error_report(
    seq,  # noqa: ARG001 -- the toolbox's signature; nothing here needs it
    error_report: list[SimpleNamespace],
    full_report: bool = False,
    max_errors: int = 10,
    colored: bool = True,
) -> None:
    """Print ``error_report`` grouped by block.

    Parameters
    ----------
    seq : Sequence
        The sequence the report is about. Accepted so the signature is the
        toolbox's; nothing here reads it.
    error_report : list of SimpleNamespace
        What :func:`check_timing` returned.
    full_report : bool, default False
        Print every problem rather than the first ``max_errors``.
    max_errors : int, default 10
        How many to print before summarising the rest.
    colored : bool, default True
        Wrap each message in an ANSI colour.
    """
    if full_report:
        max_errors = len(error_report)

    current_block = None
    for e in error_report[:max_errors]:
        if e.block != current_block:
            print(f"Block {e.block}:" if e.block else "The sequence as a whole:")
            current_block = e.block

        print(
            f"- {e.event}.{e.field}: "
            + ("\x1b[38;5;9m" if colored else "")
            + _message(e)
            + ("\x1b[0m" if colored else "")
        )

    if len(error_report) > max_errors:
        blocks = [e.block for e in error_report[max_errors:]]
        print(
            f"--- {len(error_report) - max_errors} more errors in blocks "
            f"{min(blocks)} to {max(blocks)} hidden ---"
        )
