"""Pulseq sequence construction, I/O and analysis over the compiled core."""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from pathlib import Path
from types import SimpleNamespace
from warnings import warn

import numpy as np
import pypulseq as _upstream

from . import _ext as _cxx
from ._check_timing import _limit, print_error_report
from ._check_timing import check_timing as _check_timing
from ._kspace import calculate_kspace as _calculate_kspace
from ._kspace import detail as _kspace_detail
from ._report import report_data as _report_data
from ._report import report_text as _report_text
from ._waveforms import adc_times as _adc_times
from ._waveforms import get_gradients as _get_gradients
from ._waveforms import rf_times as _rf_times
from ._waveforms import waveforms as _waveforms
from ._waveforms import waveforms_and_times as _waveforms_and_times
from .plot import _seqeyes as _plot

__all__ = ["Sequence"]

#: Where a written file records its own hash.
_SIGNATURE = re.compile(r"^Hash (\w+)$", re.MULTILINE)

#: The event columns upstream's block table carries. The core's rows do not
#: hold the leading delay, because a pure delay in Pulseq 1.5 is a block
#: duration rather than an event, so a count reported per column pads it back
#: on and a caller's column indices are upstream's.
_UPSTREAM_BLOCK_WIDTH = 7

#: Upstream's pure-Python storage: its libraries, caches, name maps and block
#: counter, all of which the compiled core holds instead.
_UPSTREAM_STORAGE = frozenset(
    {
        "adc_id_to_name_map",
        "adc_library",
        "block_cache",
        "block_trace",
        "delay_library",
        "extension_numeric_idx",
        "extension_string_idx",
        "extensions_library",
        "grad_id_to_name_map",
        "grad_library",
        "label_inc_library",
        "label_set_library",
        "next_free_block_ID",
        "rf_id_to_name_map",
        "rf_library",
        "shape_library",
        "soft_delay_hints",
        "soft_delay_library",
        "trigger_library",
    }
)


class _BlockDurations(MutableMapping):
    """Mutable duration mapping in seconds, keyed by 1-based block index."""

    __slots__ = ("_native",)

    def __init__(self, native) -> None:
        self._native = native

    def __getitem__(self, index: int) -> float:
        if not 1 <= index <= self._native.num_blocks():
            raise KeyError(index)
        return float(self._native.block_durations()[index - 1])

    def __setitem__(self, index: int, seconds: float) -> None:
        if not 1 <= index <= self._native.num_blocks():
            raise KeyError(index)
        self._native.set_block_duration(index, float(seconds))

    def __delitem__(self, index: int) -> None:
        raise TypeError("a block's duration cannot be removed, only changed")

    def __iter__(self):
        return iter(range(1, self._native.num_blocks() + 1))

    def __len__(self) -> int:
        return self._native.num_blocks()

    def __repr__(self) -> str:
        return repr(dict(self))


class Sequence:
    """A Pulseq sequence containing events, blocks and definitions.

    Parameters
    ----------
    system : pypulseq.Opts, optional
        System limits and rasters. Defaults to the shared system; rasters
        are recorded in the native sequence.
    use_block_cache : bool, default True
        Compatibility flag, retained but not used to cache decoded blocks.
    """

    def __init__(self, system=None, use_block_cache: bool = True) -> None:
        self._native = _cxx.Sequence()
        self.system = system
        self._use_block_cache = use_block_cache
        self._use_event_cache = True
        self._trid_names: list[str] = []
        self._analysed_at = self._native.edits()
        self._duration_recorded = 0
        self.signature_type: str | None = None
        self.signature_file: str | None = None
        self.signature_value: str | None = None
        if system is not None:
            self._native.set_rasters(
                system.rf_raster_time,
                system.grad_raster_time,
                system.adc_raster_time,
                system.block_duration_raster,
            )

    def __str__(self) -> str:
        native = self._native
        return "\n".join(
            [
                "Sequence:",
                f"blocks: {native.num_blocks()}",
                f"rf_library: {native.num_rf()}",
                f"grad_library: {native.num_gradients()}",
                f"adc_library: {native.num_adc()}",
                f"extensions_library: {native.num_extensions()}",
                f"shape_library: {native.num_shapes()}",
            ]
        )

    def __len__(self) -> int:
        return self._native.num_blocks()

    def __enter__(self) -> Sequence:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> bool:
        self.clear_caches()
        return False

    def __getattr__(self, name: str):
        if name in _UPSTREAM_STORAGE:
            raise AttributeError(
                f"pypulseqpp.Sequence has no {name!r}: upstream keeps its libraries "
                "and caches as Python objects, and here they are compiled storage, "
                "read through get_block, block_events, block_durations and "
                "definitions"
            )
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    # -- rasters -------------------------------------------------------
    #
    # The rasters the sequence was designed on, as its definitions record
    # them: set from the system at construction, or read from a file.

    @property
    def grad_raster_time(self) -> float:
        """Gradient raster in seconds."""
        return self._native.grad_raster_time()

    @property
    def rf_raster_time(self) -> float:
        """RF raster in seconds."""
        return self._native.rf_raster_time()

    @property
    def adc_raster_time(self) -> float:
        """ADC dwell raster in seconds."""
        return self._native.adc_raster_time()

    @property
    def block_duration_raster(self) -> float:
        """Raster a block duration is a whole number of, in seconds."""
        return self._native.block_duration_raster()

    # -- what has been worked out about the sequence -------------------
    #
    # Whether how long the sequence lasts has been recorded is kept here: it
    # reads 0 until `TotalDuration` is a record of these blocks, and 0 again
    # the moment they change -- a block added or rewritten, a duration set, an
    # axis scaled, a soft delay applied, duplicates collapsed. The core counts
    # its own changes, so that is one comparison here rather than a write per
    # block on the design loop's hot path.

    def _forget_if_changed(self) -> None:
        edits = self._native.edits()
        if edits != self._analysed_at:
            self._analysed_at = edits
            self._duration_recorded = 0

    @property
    def _duration(self) -> int:
        """1 once `TotalDuration` is a record of these blocks; 0 otherwise."""
        self._forget_if_changed()
        return self._duration_recorded

    @_duration.setter
    def _duration(self, recorded: int) -> None:
        self._forget_if_changed()
        self._duration_recorded = recorded

    # -- blocks --------------------------------------------------------

    def add_block(self, *events) -> int:
        """Append one block playing ``events``.

        Parameters
        ----------
        *events
            Any mixture of RF, gradient, ADC, delay, label, trigger,
            rotation, RF shim and soft delay events. A bare delay sets a
            floor on the block's duration.

        Returns
        -------
        int
            The new block's 1-based index.

        Notes
        -----
        The block lasts as long as the longest thing in it, rounded up onto
        the block duration raster, which is what a caller passing a bare
        delay is asking for directly.
        """
        return self._native.add_block_events(*events)

    def set_block(self, index: int, *events) -> None:
        """Write ``events`` over the block at ``index``, 1-based."""
        self._native.set_block_events(index, *events)

    def get_block(self, index: int) -> SimpleNamespace:
        """Decode a block by its 1-based index.

        Returns
        -------
        SimpleNamespace
            Stored block duration and compiled events. Missing scalar events
            are None; trigger and label chains are lists.

        Notes
        -----
        The returned block can be passed to add_block or set_block, including
        its stored duration. Editing decoded events does not edit the stored
        block; use set_block to replace it.
        """
        return SimpleNamespace(**self._native.decode_block(index))

    def get_raw_block_content_IDs(self, index: int) -> SimpleNamespace:
        """Return stored event IDs for a 1-based block index without decoding shapes.

        Missing events have ID zero. The extension chain is a ``(2, n)``
        array of type IDs and reference IDs; block duration is in seconds.
        """
        row = self._native.get_block(index)
        return SimpleNamespace(
            block_duration=row.duration,
            rf=row.rf,
            gx=row.gx,
            gy=row.gy,
            gz=row.gz,
            adc=row.adc,
            ext=self._native.extension_chain(row.ext),
        )

    @property
    def block_durations(self) -> _BlockDurations:
        """Mutable duration mapping in seconds, keyed by 1-based block index.

        Assigning an entry changes the stored block duration.
        """
        return _BlockDurations(self._native)

    def duration(self) -> tuple[float, int, np.ndarray]:
        """Return total duration, block count and per-event block counts.

        Returns
        -------
        duration : float
            Total duration in seconds.
        num_blocks : int
            How many blocks there are.
        event_count : np.ndarray
            How many blocks carry an event in each column of the block
            table, in upstream's column order: delay, RF, the three
            gradient axes, ADC, extension.
        """
        native = self._native
        counts = np.zeros(_UPSTREAM_BLOCK_WIDTH)
        # Column 0 is upstream's delay library, which Pulseq 1.5 does not
        # have: a block with nothing in it lasts for its stored duration.
        counts[1:] = native.event_counts()
        return native.duration(), native.num_blocks(), counts

    @property
    def block_events(self) -> dict:
        """Return event IDs keyed by 1-based block index.

        Rows contain delay, RF, gx, gy, gz, ADC and extension IDs. The delay
        column is zero for Pulseq 1.5. This allocates a dictionary for all blocks.
        """
        rows = np.zeros(
            (self._native.num_blocks(), _UPSTREAM_BLOCK_WIDTH), dtype=np.int32
        )
        rows[:, 1:] = self._native.block_events()
        return dict(enumerate(rows, start=1))

    def find_block_by_time(self, t: float):
        """Return the 1-based index of the block playing at ``t`` seconds.

        Parameters
        ----------
        t : float
            A time from the start of the sequence, in seconds.

        Returns
        -------
        int or None
            The block playing then, or None if ``t`` is past the end.
        """
        durations = self._native.block_durations()
        index = int(np.searchsorted(np.cumsum(durations), t, side="right"))
        if index >= len(durations):
            return None
        if durations[index] <= 0:
            raise ValueError("Block duration cannot be negative")
        return index + 1

    # -- timing --------------------------------------------------------

    def check_timing(self, print_errors: bool = False):
        """Check timing, gradient continuity and the stored total duration.

        Parameters
        ----------
        print_errors : bool, default False
            Print the report as well as returning it.

        Returns
        -------
        is_ok : bool
            True when nothing was found.
        error_report : list of SimpleNamespace
            One entry per problem, in block order.

        Notes
        -----
        May record TotalDuration. Does not check all scanner safety constraints.
        """
        is_ok, error_report = _check_timing(self)
        if not is_ok and print_errors:
            print_error_report(self, error_report)
        return is_ok, error_report

    # -- definitions ---------------------------------------------------

    def set_definition(self, key: str, value) -> None:
        """Record ``key`` in `[DEFINITIONS]`."""
        self._native.set_definition(key, value)

    def get_definition(self, key: str):
        """Return what ``key`` says, or ``''`` if it is not defined."""
        return self._native.definitions().get(key, "")

    @property
    def definitions(self) -> dict:
        """Everything `[DEFINITIONS]` will carry."""
        return self._native.definitions()

    def copy_definitions(self, other_seq: Sequence) -> None:
        """Take every definition ``other_seq`` carries."""
        for key, value in other_seq.definitions.items():
            self.set_definition(key, value)

    # -- extensions ----------------------------------------------------

    def get_extension_type_ID(self, extension_string: str) -> int:
        """Return the numeric id for ``extension_string``, minting one if new."""
        return self._native.extension_type_id(extension_string)

    def get_extension_type_string(self, extension_id: int) -> str:
        """Return the name ``extension_id`` stands for.

        Raises
        ------
        ValueError
            If no extension has been given that id.
        """
        name = self._native.extension_type_name(extension_id)
        if not name:
            raise ValueError(
                f"Extension for the given ID - {extension_id} - is unknown."
            )
        return name

    def set_extension_string_ID(self, extension_str: str, extension_id: int) -> None:
        """Pin ``extension_str`` to ``extension_id``."""
        self._native.set_extension_type_id(extension_str, extension_id)

    # -- TR ids --------------------------------------------------------

    def get_or_create_trid_id(self, label_name: str) -> int:
        """Return the TRID number ``label_name`` is known by, naming it if new.

        Parameters
        ----------
        label_name : str
            What this repetition is called.

        Returns
        -------
        int
            Its number, counting from 1 in the order names were first seen.
        """
        label_name = str(label_name)
        if not label_name:
            raise ValueError("TRID label_name must be a non-empty char/string.")
        if label_name not in self._trid_names:
            self._trid_names.append(label_name)
        return self._trid_names.index(label_name) + 1

    def add_trid(self, label_name: str) -> None:
        """Append a block setting TRID to the number ``label_name`` is known by."""
        if not getattr(self.system, "flag_trid", True):
            return
        from ._make_label import make_label

        self.add_block(
            make_label("TRID", "SET", float(self.get_or_create_trid_id(label_name)))
        )

    # -- gradients -----------------------------------------------------

    def mod_grad_axis(self, axis: str, modifier: float) -> None:
        """Scale every gradient played on ``axis`` by ``modifier``.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            Which axis to act on.
        modifier : float
            What to multiply by. -1 inverts the axis, 0 silences it.

        Raises
        ------
        ValueError
            If ``axis`` is not one of 'x', 'y' or 'z'.
        RuntimeError
            If a gradient is played both on ``axis`` and on another, where
            there is no one answer.

        Notes
        -----
        Only the amplitude moves. The ramps, the delay and the shape stay as
        they are, so the areas scale with the amplitude and the timing does
        not change.

        Silencing the phase encoding is ``mod_grad_axis('y', 0.0)``;
        inverting the readout is ``mod_grad_axis('x', -1.0)``.
        """
        if axis not in ("x", "y", "z"):
            raise ValueError(
                f"Invalid axis. Must be one of 'x', 'y','z'. Passed: {axis}"
            )
        self._native.scale_gradient_axis("xyz".index(axis), float(modifier))

    def flip_grad_axis(self, axis: str) -> None:
        """Invert every gradient played on ``axis``."""
        self.mod_grad_axis(axis, modifier=-1)

    @property
    def num_blocks(self) -> int:
        """Number of blocks; equivalent to ``len(seq)``."""
        return self._native.num_blocks()

    def evaluate_labels(
        self,
        init: dict | None = None,
        evolution: str = "none",
        time_range=None,
        block_range=None,
    ) -> dict:
        """Evaluate running label values in block order.

        Parameters
        ----------
        init : dict, optional
            What a label is before the walk begins. A label named here is
            reported whether or not the blocks touch it, which is what makes
            evaluating a sequence a piece at a time work.
        evolution : {'none', 'blocks', 'adc', 'label'}, default 'none'
            Where to record a value: at the end, at every block, at every
            block that acquires, or at every block that sets or increments
            one.
        time_range : list of float, optional
            Two times in seconds; only the blocks they touch are walked.
        block_range : sequence of int, optional
            Two 1-based block indices. Not with ``time_range``.

        Returns
        -------
        dict
            Label names mapped to their recorded values. Values are arrays if
            any label has more than one recorded value; otherwise they are
            scalars, with zero for an empty record.

        Notes
        -----
        Labels retain their values until set or incremented. For a partial range,
        ``init`` supplies the incoming state; preceding blocks are not evaluated.
        """
        first, last = self._range_for(time_range, block_range)
        found = _cxx.evaluate_labels(
            self._native,
            evolution=evolution,
            first_block=first,
            last_block=last,
            start={} if init is None else {str(k): int(v) for k, v in init.items()},
        )
        # One value is one value, not an array of one -- which is what the
        # reference toolbox hands back, and what `labels['LIN'] == 4` needs.
        if any(len(values) > 1 for values in found.values()):
            return found
        return {name: values[0] if len(values) else 0 for name, values in found.items()}

    def _range_for(self, time_range, block_range) -> tuple[int, int]:
        """Return the blocks a time range or a block range names, 1-based."""
        if block_range is not None and time_range is not None:
            raise ValueError("Specify either block_range or time_range, not both")
        if block_range is not None:
            if len(block_range) != 2:
                raise ValueError(
                    "parameter 'block_range' must contain exactly two numbers"
                )
            import math

            first = max(int(block_range[0]), 1)
            last = 0 if not math.isfinite(block_range[1]) else int(block_range[1])
            return first, last
        if time_range is not None:
            from ._waveforms import _blocks_within

            first, last, _ = _blocks_within(self, time_range)
            return first, last
        return 1, 0

    # -- what the sequence plays ---------------------------------------

    def waveforms_and_times(
        self,
        append_RF: bool = False,
        time_range=None,
        block_range=None,
        *,
        compat: bool = True,
    ):
        """Return the gradient waveforms, the RF moments and the ADC sampling.

        ``compat`` gives upstream's five values, which is what a drop-in
        caller unpacks; False gives everything the pass worked out, including
        the five RF uses those five values cannot carry.

        See :func:`pypulseqpp._waveforms.waveforms_and_times`.
        """
        return _waveforms_and_times(
            self, append_RF, time_range, block_range, compat=compat
        )

    def waveforms(self, append_RF: bool = False, time_range=None, block_range=None):
        """Return gradient corners as time (s) over amplitude (Hz/m), per axis.

        ``append_RF=True`` appends a complex RF channel in Hz.
        """
        return _waveforms(self, append_RF, time_range, block_range)

    def adc_times(self, time_range=None):
        """Return ADC sample times (s) and per-window frequency (Hz) and phase (rad)."""
        return _adc_times(self, time_range)

    def rf_times(self, time_range=None, *, compat: bool = True):
        """Return when the pulses act, and at what frequency and phase.

        ``compat`` gives upstream's four values, which describe two of
        Pulseq's seven RF uses; False gives all seven.
        """
        return _rf_times(self, time_range, compat=compat)

    def calculate_kspace(
        self, trajectory_delay=0.0, gradient_offset=0.0, block_range=None
    ):
        """Integrate physical-axis gradients with excitation resets and refocusing.

        Block rotations are applied. K-space coordinates are in 1/m.

        Parameters
        ----------
        trajectory_delay : float or sequence of float, default 0
            Per-axis timing correction (s); positive values advance the gradient.
        gradient_offset : float or sequence of float, default 0
            A background gradient per axis, in Hz/m.
        block_range : sequence of int, optional
            Two 1-based block indices; only those blocks are followed.

        Returns
        -------
        k_traj_adc : np.ndarray
            3-by-n: where each ADC sample sits in k-space, in 1/m.
        k_traj : np.ndarray
            Full trajectory in 1/m, sampled through ramps and at event times.
        t_excitation : np.ndarray
        t_refocusing : np.ndarray
        t_adc : np.ndarray
            RF-centre and ADC times in seconds relative to the selected range.
        """
        return _calculate_kspace(self, trajectory_delay, gradient_offset, block_range)

    #: Upstream carries this name for the same calculation, and so does this.
    calculate_kspacePP = calculate_kspace

    def _kspace(
        self,
        trajectory_delay=0.0,
        gradient_offset=0.0,
        block_range=None,
        samples_only: bool = False,
    ):
        """Return detailed trajectory results, including times and slice positions.

        With ``samples_only=True``, omit the intermediate trajectory arrays
        without changing ADC sample positions.
        """
        return _kspace_detail(
            self, trajectory_delay, gradient_offset, block_range, samples_only
        )

    def get_gradients(
        self,
        trajectory_delay=0,
        gradient_offset=0,
        time_range=None,
        block_range=None,
    ):
        """Return physical-axis gradient splines in Hz/m over seconds.

        Positive trajectory_delay advances the gradients. A scalar delay or
        offset applies to all axes. Inactive axes with no offset return None.
        Time and block ranges are mutually exclusive.
        """
        return _get_gradients(
            self, trajectory_delay, gradient_offset, time_range, block_range
        )

    # -- what the sequence is ------------------------------------------

    def test_report(self) -> str:
        """Return a formatted sequence timing, encoding and gradient report."""
        return _report_text(_report_data(self))

    def test_report_dict(self) -> dict:
        """Return timing, encoding and gradient statistics; see report_data."""
        return _report_data(self)

    # -- the repeating unit --------------------------------------------

    def _detect_tr(self) -> tuple[int, int]:
        """Detect the period of the block-definition stream.

        Returns
        -------
        size : int
            Blocks per repetition, or zero if no repetition is detected.
        start : int
            1-based start of the first full repetition; earlier blocks form
            the prologue.

        Notes
        -----
        Records ``TRsize`` in sequence definitions. Structural edits invalidate
        the native detection cache.
        """
        recorded = self.get_definition("TRsize")
        if recorded != "":
            size = int(recorded[0] if isinstance(recorded, list) else recorded)
            found, start = self._native.locate_repetition(size)
            if found:
                return found, start + 1

        size, start = self._native.repetition()
        self.set_definition("TRsize", size)
        return size, start + 1

    # -- soft delays ---------------------------------------------------

    def apply_soft_delay(self, **kwargs) -> None:
        """Set each named soft delay to the value given.

        A soft delay says how long its block lasts in terms of a value the
        console supplies: ``duration = value / factor + offset``, rounded onto
        the block duration raster. Naming one here writes that duration into
        every block that plays it.

        Parameters
        ----------
        **kwargs
            What each delay, by its hint, is to be set to, in seconds. A
            delay the sequence carries and nobody names is left alone.

        Raises
        ------
        ValueError
            If a name given is not in the sequence, if a hint and a numeric
            id do not agree about which delay they name, or if the value
            asked for would make a block last less than nothing.

        Warns
        -----
        UserWarning
            When a duration had to move more than half a microsecond to
            reach the raster. Once per delay, not once per block.

        Notes
        -----
        Finding the soft delays is a compiled pass over the block table:
        nothing else in a block is decoded to answer whether it heads one.
        """
        report = self._native.apply_soft_delays(kwargs)
        raster = self.system.block_duration_raster

        for note in report["rounded"]:
            warn(
                f"Soft delay '{note['hint']}' in block {note['block']}: "
                f"Duration rounded by {note['error'] * 1e6:.1f} \u03bcs to align "
                f"with raster time ({raster * 1e6:.1f} \u03bcs). "
                f"This warning is shown only once per soft delay ID.",
                stacklevel=2,
            )

        problem = report["problem"]
        if problem is not None:
            hint, block = problem["hint"], problem["block"]
            number = problem["numID"]
            if problem["kind"] == "hint_renumbered":
                raise ValueError(
                    f"Soft delay in block {block} with numeric ID {number} and "
                    f"string hint '{hint}' is inconsistent with the previous "
                    f"occurrences of the same string hint"
                )
            if problem["kind"] == "number_renamed":
                raise ValueError(
                    f"Soft delay in block {block} with numeric ID {number} and "
                    f"string hint '{hint}' is inconsistent with the previous "
                    f"occurrences of the same numeric ID"
                )
            raise ValueError(
                f"Soft delay '{hint}' in block {block}: Calculated duration is "
                f"negative ({problem['duration'] * 1e6:.1f} \u03bcs). Check the "
                f"offset ({problem['offset'] * 1e6:.1f} \u03bcs) and factor "
                f"({problem['factor']}) parameters."
            )

        for name in kwargs:
            if name not in report["hints"]:
                available = report["hints"]
                raise ValueError(
                    f"Soft delay '{name}' not found in sequence. "
                    f"Available soft delays: {available if available else 'none'}"
                )

    def get_default_soft_delay_values(self):
        """Return what each soft delay stands for if nobody sets it.

        A soft delay says how a block's duration follows from a value the
        console supplies: `duration = value / factor + offset`. Read the other
        way, the duration a block was built with says what value that is --
        and every block sharing a numeric id has to agree about it.

        Returns
        -------
        defaults : dict
            The default value for each soft delay, by its hint.
        error_report : list of str
            One line per disagreement found; empty when they all agree.
        limits : list
            Per numeric id, the default, the hint, the block it came from,
            and the range of values that keep the block duration positive.
        """
        error_report: list[str] = []
        state: list[dict | None] = []

        for index in self.block_events:
            delay = getattr(self.get_block(index), "soft_delay", None)
            if delay is None:
                continue

            if delay.factor == 0:
                error_report.append(
                    f"   Block:{index} soft delay {delay.hint}/{delay.numID} "
                    f"has factor parameter of 0 which is invalid\n"
                )

            number = int(delay.numID)
            if number < 0:
                error_report.append(
                    f"   Block:{index} contains a soft delay {delay.hint} "
                    f"with an invalid numeric ID{number}\n"
                )
                continue

            default = (self.block_durations[index] - delay.offset) * delay.factor
            while len(state) < number + 1:
                state.append(None)

            if state[number] is None:
                state[number] = {
                    "def": default,
                    "hint": delay.hint,
                    "blk": index,
                    "min": 0.0,
                    "max": np.inf,
                }
            else:
                seen = state[number]
                if abs(default - seen["def"]) > 1e-7:
                    error_report.append(
                        f"   Block:{index} soft delay {delay.hint}/{number}: "
                        f"default duration derived from this block "
                        f"({default * 1e6}us) is inconsistent with the previous "
                        f"default ({seen['def'] * 1e6}us) that was derived from "
                        f"block {seen['blk']}\n"
                    )
                if delay.hint != seen["hint"]:
                    error_report.append(
                        f"   Block:{index} soft delay {delay.hint}/{number}: soft "
                        f"delays with the same numeric ID are expected to share "
                        f"the same text hint but previous hint recorded in block "
                        f"{seen['blk']} is {seen['hint']}\n"
                    )

            # The block cannot last less than nothing, so the offset and the
            # sign of the factor set which end the value is bounded at.
            limit = (-delay.offset) * delay.factor
            if delay.factor > 0:
                state[number]["min"] = max(state[number]["min"], limit)
            else:
                state[number]["max"] = min(state[number]["max"], limit)

        defaults: dict[str, float] = {}
        for number, seen in enumerate(state):
            if seen is None:
                warn(
                    f"SoftDelay numeric ID {number} is unused, we expect "
                    f"contiguous numbering of soft delays",
                    stacklevel=2,
                )
                continue
            if seen["hint"] in defaults:
                raise ValueError(
                    f"SoftDelay with numeric ID {number} uses the same hint "
                    f"'{seen['hint']}' as some previous SoftDelay"
                )
            defaults[seen["hint"]] = seen["def"]

        return defaults, error_report, state

    # -- registering an event on its own -------------------------------

    def _register(
        self, event, kinds: tuple[str, ...], called: str, wanted: str
    ) -> dict:
        stored = _cxx.register_event(self._native, event)
        if stored["kind"] not in kinds:
            raise ValueError(f"{called}() takes {wanted}, not a {stored['kind']} event")
        return stored

    def register_rf_event(self, event) -> tuple[int, list[int]]:
        """Store a pulse and return its ids.

        Returns
        -------
        rf_id : int
            The row it was stored as.
        shape_ids : list of int
            Its magnitude, phase and time shapes; the time shape is 0 when
            the pulse sits on the RF raster.
        """
        stored = self._register(event, ("rf",), "register_rf_event", "an RF pulse")
        return stored["id"], stored["shapes"]

    def register_grad_event(self, event):
        """Store a gradient and return its ids.

        Returns
        -------
        int or tuple
            A trapezoid's row id on its own; an arbitrary waveform's row id
            with its waveform and time shape ids, the way the toolboxes
            report them.
        """
        stored = self._register(event, ("grad",), "register_grad_event", "a gradient")
        return (stored["id"], stored["shapes"]) if stored["shapes"] else stored["id"]

    def register_adc_event(self, event) -> tuple[int, int]:
        """Store an ADC and return its ids.

        Returns
        -------
        adc_id : int
            The row it was stored as.
        shape_id : int
            Its phase-modulation shape, 0 when it has none.
        """
        stored = self._register(event, ("adc",), "register_adc_event", "an ADC")
        return stored["id"], stored["shapes"][0]

    def register_label_event(self, event) -> int:
        """Store a label and return its row id."""
        return self._register(
            event, ("LABELSET", "LABELINC"), "register_label_event", "a label"
        )["id"]

    def register_control_event(self, event) -> int:
        """Store a trigger or digital output and return its row id."""
        return self._register(
            event, ("TRIGGERS",), "register_control_event", "a trigger"
        )["id"]

    def register_rotation_event(self, event) -> int:
        """Store a rotation and return its row id."""
        return self._register(
            event, ("ROTATIONS",), "register_rotation_event", "a rotation"
        )["id"]

    def register_rf_shim_event(self, event) -> int:
        """Store an RF shim vector and return its row id."""
        return self._register(
            event, ("RF_SHIMS",), "register_rf_shim_event", "an RF shim"
        )["id"]

    def register_soft_delay_event(self, event) -> int:
        """Store a soft delay and return its row id."""
        return self._register(
            event, ("DELAYS",), "register_soft_delay_event", "a soft delay"
        )["id"]

    # -- files ---------------------------------------------------------

    def write(
        self,
        name,
        create_signature: bool = True,
        remove_duplicates: bool = True,
        check_timing: bool = False,
        v141_compat: bool = False,
    ) -> str | None:
        """Write Pulseq text and record its signature metadata.

        With ``v141_compat=True``, metadata is recorded on the written source
        (the deduplicated copy when requested).

        Parameters
        ----------
        name : str or Path
            Where to write it.
        create_signature : bool, default True
            Sign the file, so a reader can tell it has not been edited.
        remove_duplicates : bool, default True
            Collapse identical library rows first. Write a collapsed
            copy, leaving this sequence's event libraries unchanged.
        check_timing : bool, default False
            Judge the timing first, and warn if anything is wrong.
        v141_compat : bool, default False
            Write 1.4.1 instead, for an interpreter that predates 1.5.

        Returns
        -------
        str or None
            The signature written, or None if the file is unsigned. It is
            the signature of what was written, so with ``remove_duplicates``
            it belongs to the collapsed copy rather than to this sequence.
        """
        if check_timing:
            is_ok, error_report = self.check_timing()
            if not is_ok:
                warn(
                    f"write(): {len(error_report)} timing errors found in the sequence",
                    stacklevel=2,
                )

        source = self.remove_duplicates() if remove_duplicates else self
        if v141_compat:
            return source.write_v141(name, create_signature)

        written = _cxx.write_text(source._native, create_signature)
        Path(name).write_bytes(written)
        return self._note_signature(written, "text")

    def write_binary(self, name, create_signature: bool = True) -> str | None:
        """Write Pulseq binary, with float32 shape samples.

        Parameters
        ----------
        name : str or Path
            Where to write it.
        create_signature : bool, default True
            Append the signature section: an MD5 of everything above it, so a
            file says whether it is the file that was written.

        Returns
        -------
        str or None
            The signature written, or None when none was asked for.
        """
        written = _cxx.write_binary(self._native, create_signature)
        Path(name).write_bytes(written)
        return self._note_binary_signature(written)

    def write_v141(
        self, name, create_signature: bool = True, gamma=42576000.0, field=1.5
    ) -> str | None:
        """Write Pulseq 1.4.1 text, returning its MD5 signature or None.

        ``gamma`` is in Hz/T and ``field`` in T; together they convert ppm
        offsets to absolute offsets. Soft delays are omitted with a warning.
        Rotation and RF-shim extensions raise RuntimeError.
        """
        written = _cxx.write_text_v141(self._native, create_signature, gamma, field)
        Path(name).write_bytes(written)
        return self._note_signature(written, "text")

    def _note_binary_signature(self, written: bytes) -> str | None:
        """Record and return the signature a binary file carries, if any."""
        found = _cxx.binary_signature(written)
        if not found["type"]:
            self.signature_type = self.signature_file = self.signature_value = None
            return None
        self.signature_type = found["type"]
        self.signature_file = "bin"
        self.signature_value = found["value"]
        return self.signature_value

    def _note_signature(self, written: bytes, kind: str) -> str | None:
        """Record and return the signature ``written`` carries, if any."""
        found = _SIGNATURE.search(written.decode("ascii", "replace"))
        if found is None:
            self.signature_type = self.signature_file = self.signature_value = None
            return None
        self.signature_type = "md5"
        self.signature_file = kind
        self.signature_value = found.group(1)
        return self.signature_value

    def read(
        self,
        file_path,
        detect_rf_use: bool = False,
        remove_duplicates: bool = True,
        verify: bool = False,
    ) -> None:
        """Replace this sequence with the one in ``file_path``.

        Text or binary, told apart by what is in the bytes. Every revision
        from 1.2.0 is read, older ones by converting them.

        Parameters
        ----------
        file_path : str or Path
            The file to read.
        detect_rf_use : bool, default False
            Work out what each unlabelled pulse is for, from what it does.
            Before revision 1.5.0 the format had nowhere to record it, so a
            file older than that arrives with its pulses unlabelled. Pulses
            the file does label are left alone.
        remove_duplicates : bool, default True
            Collapse identical library rows after reading.
        verify : bool, default False
            Check the file against the signature it carries.
        """
        contents = Path(file_path).read_bytes()
        binary = _cxx.is_binary(contents)
        if binary:
            # A binary file carries its signature at the end, over the bytes
            # before it, so it is checked before they are parsed: what a
            # signature is for is saying the bytes are wrong rather than
            # letting the parser say something else about them. The text
            # reader checks its own as it parses.
            found = _cxx.binary_signature(contents)
            if verify and found["type"] and not found["valid"]:
                raise RuntimeError(
                    f"read(): the file's {found['type']} signature is not the "
                    "signature of its contents"
                )
        self._native = _cxx.read(contents, verify)
        if binary:
            self._note_binary_signature(contents)
        if detect_rf_use:
            system = self.system
            labelled = self._native.detect_rf_uses(
                _limit(system, "B0", 1.5), _limit(system, "gamma", 42576000.0)
            )
            if labelled == 0:
                warn(
                    "read(): detect_rf_use had nothing to do; every pulse in "
                    "this file already records what it is for",
                    stacklevel=2,
                )
        if remove_duplicates:
            self._native.remove_duplicates()

        # A file that declares how long it lasts is held to it: the first
        # timing check compares rather than records.
        self._analysed_at = self._native.edits()
        self._duration_recorded = 1 if self.get_definition("TotalDuration") != "" else 0

    # -- the format this sequence is held as ---------------------------
    #
    # What the sequence *is*, not what the file it came from said. A file
    # older than 1.5 is converted as it is read -- an RF pulse's centre and a
    # gradient's first and last sample are derived, because 1.4 has no column
    # for them -- so what is held afterwards is a 1.5 sequence and says so.
    # Nor is it what a file written from it will declare: this package writes
    # 1.5.1, so a sequence read as 1.5.0 is written as 1.5.1.

    @property
    def version_major(self) -> int:
        """The major version of the Pulseq format this sequence is held as."""
        return self._native.version_major()

    @property
    def version_minor(self) -> int:
        """Its minor version."""
        return self._native.version_minor()

    @property
    def version_revision(self) -> int:
        """Its revision."""
        return self._native.version_revision()

    # -- drawing -------------------------------------------------------

    def plot(
        self,
        label: str = "",
        show_blocks: bool = False,
        save: bool = False,
        time_range=(0, np.inf),
        time_disp: str = "s",
        grad_disp: str = "kHz/m",
        plot_now: bool = True,
        clear: bool = True,
        overlay=None,
        stacked: bool = False,
        show_guides: bool = False,
        *,
        block_range=None,
        tr_range=None,
    ) -> _plot.Viewer:
        """Draw the sequence in SeqEyes.

        SeqEyes is an optional dependency, installed with
        ``pip install 'pypulseqpp[plot]'``. It reads the sequence as a file,
        so what it is handed is only what is asked to be drawn.

        Parameters
        ----------
        label, show_blocks, save, time_disp, grad_disp, clear, overlay, stacked, show_guides
            Upstream's, accepted so a script written for it runs. SeqEyes
            draws units, labels and block edges its own way, from its own
            settings; one given a value other than its default is reported
            and ignored.
        time_range : sequence of float, default (0, inf)
            The seconds to draw, measured from the start of the scan.
        plot_now : bool, default True
            Wait for the window to be closed before returning. When False,
            the window is left open and the returned viewer is live.
        block_range : sequence of int, optional
            The first and last block to draw, 1-based and inclusive.
        tr_range : sequence of int, optional
            The first and last repetition to draw, 1-based and inclusive.
            A repetition is the period of the block definition stream, and
            the first is the first full one, so a prologue -- dummy shots, a
            preparation, a noise scan -- is not counted.

        Returns
        -------
        Viewer
            The window: `Viewer.wait` blocks until it is closed and
            `Viewer.close` closes it.

        Raises
        ------
        ModuleNotFoundError
            If SeqEyes is not installed.
        ValueError
            If more than one range is given, a range is outside the sequence,
            or ``tr_range`` is asked of a sequence that does not repeat.
        """
        whole = tuple(time_range) == (0, np.inf)
        return _plot.plot(
            self,
            time_range=None if whole else time_range,
            block_range=block_range,
            tr_range=tr_range,
            plot_now=plot_now,
            label=label,
            show_blocks=show_blocks,
            save=save,
            time_disp=time_disp,
            grad_disp=grad_disp,
            clear=clear,
            overlay=overlay,
            stacked=stacked,
            show_guides=show_guides,
        )

    def paper_plot(
        self,
        time_range=(0, np.inf),
        line_width: float = 1.2,
        axes_color="0.9",
        rf_color="black",
        gx_color="black",
        gy_color="black",
        gz_color="black",
        rf_plot: str = "abs",
        *,
        tr=None,
        max_underlays: int = 16,
        underlay_color="0.8",
        ax=None,
    ) -> SimpleNamespace:
        """Draw a publication-style diagram of one repetition, with mrsd.

        Rows are RF, the physical gradient axes z, y and x, and ADC, each drawn
        from the waveform the sequence plays and scaled to its row: the three
        gradient rows share one scale. The other repetitions are drawn
        underneath in ``underlay_color``, which shows what changes from one to
        the next, and a TR interval is marked below.

        Parameters
        ----------
        time_range : sequence of float, default (0, inf)
            Upstream's window, in seconds. Given, the blocks it touches are
            drawn alone, without repetitions underneath.
        line_width, axes_color, rf_color, gx_color, gy_color, gz_color, rf_plot
            Upstream's styling parameters, with mrsd's defaults: black events
            on light-grey baselines. ``rf_color`` also draws the ADC;
            ``rf_plot`` is ``"abs"``, ``"real"`` or ``"imag"``.
        tr : int, optional
            1-based repetition to draw, counted from the first full one. By
            default, the one in which a physical axis reaches its largest
            magnitude.
        max_underlays : int, default 16
            At most this many repetitions, evenly spaced, are drawn underneath,
            together with those in which each axis reaches its most negative and
            most positive value; 0 draws none.
        underlay_color : color, default "0.8"
        ax : matplotlib.axes.Axes, optional
            Axes to draw in; a new figure by default.

        Returns
        -------
        SimpleNamespace
            ``diagram``, the :class:`mrsd.Diagram`, whose ``annotate`` and
            ``interval`` add labels; ``tr``, the repetition drawn solid, and
            ``underlays``, those drawn underneath, 1-based (``None`` and
            empty without a repetition).

        Notes
        -----
        Repetitions are detected from the block definitions. Choosing them is
        one compiled pass over the block table, and only the repetitions drawn
        are expanded, so the cost does not grow with the length of the scan. A
        sequence without a repetition is drawn whole.
        """
        from .plot._paper import paper_plot

        return paper_plot(
            self,
            time_range=time_range,
            line_width=line_width,
            axes_color=axes_color,
            rf_color=rf_color,
            gx_color=gx_color,
            gy_color=gy_color,
            gz_color=gz_color,
            rf_plot=rf_plot,
            tr=tr,
            max_underlays=max_underlays,
            underlay_color=underlay_color,
            ax=ax,
        )

    # -- the scanner ---------------------------------------------------

    #: Upstream scanner installation, using this sequence's write method.
    install = _upstream.Sequence.install

    # -- collapsing ----------------------------------------------------

    def remove_duplicates(self, in_place: bool = False) -> Sequence:
        """Collapse identical library rows and renumber what points at them.

        Parameters
        ----------
        in_place : bool, default False
            Collapse this sequence. Otherwise a copy is collapsed and this
            one is left as it is.

        Returns
        -------
        Sequence
            The collapsed sequence: this one, or the copy.
        """
        target = self if in_place else self._copy()
        target._native.remove_duplicates()
        return target

    def _copy(self) -> Sequence:
        """Copy through binary serialisation, sharing the system object.

        Waveform samples have binary float32 precision. Cache flags and TRID
        names are copied; Python analysis caches are not.
        """
        other = Sequence(self.system)
        other._native = _cxx.read(_cxx.write_binary(self._native), False)
        other._use_block_cache = self._use_block_cache
        other._use_event_cache = self._use_event_cache
        other._trid_names = list(self._trid_names)
        return other

    # -- aliases -------------------------------------------------------
    #
    # Names the toolboxes carry for what is above, so a design script that
    # spells a call the other way round still runs.

    def write_file(self, filename) -> None:
        """`write(filename, create_signature=False)`."""
        self.write(filename, create_signature=False)

    def read_binary(self, filename) -> None:
        """`read(filename)`, which tells text and binary apart by itself."""
        self.read(filename)

    # -- no-ops --------------------------------------------------------
    #
    # Compatibility flags do not control native shape registrations or the
    # revision-guarded analysis caches. Decoded blocks are not cached.

    @property
    def use_block_cache(self) -> bool:
        """Compatibility flag; decoded blocks are not cached."""
        return self._use_block_cache

    @use_block_cache.setter
    def use_block_cache(self, enabled: bool) -> None:
        self._use_block_cache = enabled

    @property
    def use_event_cache(self) -> bool:
        """Compatibility flag; registration results are not cached in Python."""
        return self._use_event_cache

    @use_event_cache.setter
    def use_event_cache(self, enabled: bool) -> None:
        self._use_event_cache = enabled

    @property
    def block_cache_size(self) -> int:
        """Zero: no block is held decompressed."""
        return 0

    @property
    def event_cache_size(self) -> int:
        """Zero: no registration result is held."""
        return 0

    def clear_block_cache(self) -> None:
        """Compatibility no-op."""

    def clear_event_cache(self) -> None:
        """Compatibility no-op."""

    def clear_caches(self) -> None:
        """Compatibility no-op."""
