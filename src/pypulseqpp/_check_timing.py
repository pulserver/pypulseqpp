"""Whether every event time in a sequence lands where a scanner can put it.

A sequencer starts an event on one of its clock ticks and nowhere else, and it
needs a settling window either side of RF and of digitisation. A pulse asked
for 3.7 microseconds into a block is not played 3.7 microseconds in; it is
played wherever the interpreter rounds it to. This says which times cannot be
honoured, and by how far each one misses, before the file leaves the bench.

The judging is a compiled pass over the block table. What comes back from it
is one namespace per problem, carrying the fields its kind reports, which is
what a message template naming them formats against.
"""

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
}


def _limit(system, name: str, fallback: float) -> float:
    """Return what ``system`` says ``name`` is, or ``fallback`` if it is silent."""
    value = getattr(system, name, None)
    return fallback if value is None else float(value)


def check_timing(seq) -> tuple[bool, list[SimpleNamespace]]:
    """Return whether ``seq`` is playable, and every problem found.

    Parameters
    ----------
    seq : Sequence
        The sequence to judge.

    Returns
    -------
    is_ok : bool
        True when nothing was found.
    error_report : list of SimpleNamespace
        One entry per problem, in block order. Each carries ``block``,
        ``event``, ``field`` and ``error_type``, plus the values that kind of
        problem reports.
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
    )
    error_report = [SimpleNamespace(**finding) for finding in findings]
    return len(error_report) == 0, error_report


def _format_message(template: str, **fields) -> str:
    """Evaluate ``template`` as an f-string over ``fields``.

    The templates compute in place -- a time in seconds is printed in
    microseconds by the template rather than by the caller -- so formatting one
    is evaluating it, not substituting into it.
    """
    return eval(f'f"""{template}"""', fields)  # noqa: S307


def _message(finding: SimpleNamespace) -> str:
    """Return what one finding says, in the unit its field is read in."""
    unit, multiplier = ("ns", 1e9) if finding.field == "dwell" else ("us", 1e6)
    return _format_message(
        error_messages[finding.error_type],
        **finding.__dict__,
        unit=unit,
        multiplier=multiplier,
    )


def describe(finding: SimpleNamespace) -> str:
    """Return one finding as a line naming the block and event it is about."""
    return (
        f"   Block:{finding.block} {finding.event}.{finding.field}: {_message(finding)}"
    )


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
            print(f"Block {e.block}:")
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
