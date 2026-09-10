"""The sequence a design script builds, over the compiled core.

What a caller holds is this; what does the work is a `pypulseqpp._ext`
sequence underneath. The methods here are the ones a design loop and a
writer need, and each is a single call into the core rather than a loop in
Python: `add_block` unpacks its events and registers them inside C++, and
reading and writing are compiled passes.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from pathlib import Path
from types import SimpleNamespace
from warnings import warn

import numpy as np
import pypulseq as _upstream

from . import _ext as _cxx
from . import _plot
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

__all__ = ["Sequence"]

#: Where a written file records its own hash.
_SIGNATURE = re.compile(r"^Hash (\w+)$", re.MULTILINE)

#: The event columns upstream's block table carries. The core's rows do not
#: hold the leading delay, because a pure delay in Pulseq 1.5 is a block
#: duration rather than an event, so a count reported per column pads it back
#: on and a caller's column indices are upstream's.
_UPSTREAM_BLOCK_WIDTH = 7


class _BlockDurations(MutableMapping):
    """Every block's duration in seconds, keyed by 1-based block index.

    The shape the toolboxes hand back, over the block table the core holds:
    reading one reads the table, writing one writes it. It is a view rather
    than a copy, so it reflects the sequence as it stands.
    """

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
    """A Pulseq sequence: event libraries, a block table, and definitions."""

    def __init__(self, system=None, use_block_cache: bool = True) -> None:
        """Start an empty sequence.

        Parameters
        ----------
        system : pypulseq.Opts, optional
            The system the sequence is designed against. Its rasters are
            recorded, because a block's duration is stored as a count of
            them and a reader cannot recover the seconds without them.
        use_block_cache : bool, default True
            Accepted and remembered; nothing here caches. See the no-ops at
            the end of this class.
        """
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

    # -- what has been worked out about the sequence -------------------
    #
    # Whether how long the sequence lasts has been recorded is kept here: it
    # reads 0 until `TotalDuration` is a record of these blocks, and 0 again
    # the moment they change -- a block added or rewritten, a duration set, an
    # axis scaled, a soft delay applied, duplicates collapsed. The core counts
    # its own changes, so that is one comparison here rather than a write per
    # block on the design loop's hot path.

    def _forget_if_changed(self) -> None:
        """Drop what was worked out if the sequence has changed since."""
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
        """Return the block at ``index``, 1-based, as the events it plays.

        Parameters
        ----------
        index : int
            Which block, counting from 1.

        Returns
        -------
        SimpleNamespace
            ``block_duration`` and one field per event: ``rf``, ``gx``,
            ``gy``, ``gz``, ``adc``, ``soft_delay`` and ``rotation``, each
            None when the block has none, and ``trig`` and ``label`` as
            lists when it has any.

        Notes
        -----
        The events are the compiled kind, so they read in Python the way a
        factory's do -- ``rf.signal``, ``gx.waveform``, ``gx.area`` -- and go
        back into `add_block` or `set_block` on the fast path. They carry the
        shapes they were stored under, so reading a block out and putting it
        back registers no shape twice.

        The whole block can be passed on as it stands: `add_block` and
        `set_block` take one, which is how a block moves from one sequence to
        another with its duration intact.
        """
        return SimpleNamespace(**self._native.decode_block(index))

    def get_raw_block_content_IDs(self, index: int) -> SimpleNamespace:
        """Return the block at ``index`` as the ids its events are stored under.

        Parameters
        ----------
        index : int
            Which block, counting from 1.

        Returns
        -------
        SimpleNamespace
            ``block_duration`` and the library id of each event, 0 where the
            block has none. ``ext`` is the extension chain, as a 2-by-n array
            of type and reference ids.

        Notes
        -----
        Nothing is decompressed: this is the row of the block table, which is
        what a caller comparing blocks or counting distinct events wants.
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
        """Every block's duration in seconds, keyed by 1-based block index.

        Writing one sets it: ``seq.block_durations[3] = 5e-3`` moves that
        block's duration in the table underneath.

        Notes
        -----
        The whole column, for a caller reading rather than editing, is
        ``seq._native.block_durations()`` -- an array pointing straight into
        the block table, which is what makes a million-row table free to sum.
        """
        return _BlockDurations(self._native)

    def duration(self) -> tuple[float, int, np.ndarray]:
        """Return how long the sequence plays for, and what it is made of.

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

        Notes
        -----
        Both the sum and the counts are one compiled pass over the block
        table, so asking this of a million-block scan reads the table twice
        and allocates nothing the size of it.
        """
        native = self._native
        counts = np.zeros(_UPSTREAM_BLOCK_WIDTH)
        # Column 0 is upstream's delay library, which Pulseq 1.5 does not
        # have: a block with nothing in it lasts for its stored duration.
        counts[1:] = native.event_counts()
        return native.duration(), native.num_blocks(), counts

    @property
    def block_events(self) -> dict:
        """Return every block's event ids, keyed by 1-based block index.

        Each value is a row in upstream's column order: delay, RF, the
        three gradient axes, ADC, extension. Column 0 is upstream's delay
        library, which Pulseq 1.5 does not have.

        Notes
        -----
        This builds a dictionary the length of the sequence, which is what
        the toolboxes hand back and what a script inspecting a sequence
        expects. It is for looking at a sequence, not for walking one: the
        block table itself is `block_durations` and the core's own view,
        which cost nothing to read a column out of.
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
        """Return whether the sequence is playable, and every problem found.

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
        """How many blocks the sequence has.

        ``len(seq)`` answers the same question and is what a Python caller
        reaches for; this is what a script reads when it is saying so.
        """
        return self._native.num_blocks()

    def evaluate_labels(
        self,
        init: dict | None = None,
        evolution: str = "none",
        time_range=None,
        block_range=None,
    ) -> dict:
        """Return what each label the sequence uses is set to.

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
            One entry per label used, name to value. With ``evolution``, the
            values are arrays -- one entry per point recorded; without it,
            each is the single number the label finishes at.

        Notes
        -----
        A label is running state: set or incremented where a block says so,
        and in force until another block says otherwise. So this is a walk
        over the blocks in order -- but over the extension chains rather than
        over decoded blocks, and a block carrying no label costs a column
        read, which is most of them.
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
        """Return the gradient waveforms alone, one 2-by-n array per axis."""
        return _waveforms(self, append_RF, time_range, block_range)

    def adc_times(self, time_range=None):
        """Return when every ADC sample is taken, and each window's offsets."""
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
        """Return where the sequence goes in k-space, and where it samples.

        A gradient moves the spins' phase, and the phase they have
        accumulated is where the sequence has got to in k-space -- so the
        trajectory is the integral of the gradient waveforms. An excitation
        starts the phase over and a refocusing turns it around, which is what
        makes a spin echo come back.

        Parameters
        ----------
        trajectory_delay : float or sequence of float, default 0
            How late each axis plays what it was asked to, in seconds.
        gradient_offset : float or sequence of float, default 0
            A background gradient per axis, in Hz/m.
        block_range : sequence of int, optional
            Two 1-based block indices; only those blocks are followed.

        Returns
        -------
        k_traj_adc : np.ndarray
            3-by-n: where each ADC sample sits in k-space, in 1/m.
        k_traj : np.ndarray
            The whole trajectory, at every time it changes direction.
        t_excitation : np.ndarray
        t_refocusing : np.ndarray
        t_adc : np.ndarray
            When the pulses act and the samples are taken.
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
        """Return everything following the trajectory produces, by name.

        The five values `calculate_kspace` hands back are what upstream
        reports; this is all ten the reference toolbox does -- the
        trajectory's own time base, the slice positions and the gradients as
        splines besides -- for the analysis here that wants them.

        With ``samples_only`` it answers where the samples were taken and
        leaves the trajectory between them unbuilt, which is most of the
        work and no part of the answer.
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
        """Return each gradient axis as a piecewise polynomial."""
        return _get_gradients(
            self, trajectory_delay, gradient_offset, time_range, block_range
        )

    # -- what the sequence is ------------------------------------------

    def test_report(self) -> str:
        """Return what the sequence is, as the report a person reads."""
        return _report_text(_report_data(self))

    def test_report_dict(self) -> dict:
        """Return what the sequence is, as named statistics.

        See :func:`pypulseqpp._report.report_data`.
        """
        return _report_data(self)

    # -- the repeating unit --------------------------------------------

    def _detect_tr(self) -> tuple[int, int]:
        """Return the repeating unit of the scan, in blocks.

        A scan is a handful of things played over and over with different
        numbers in them, and the stream of block definition ids is where that
        shows: a gradient echo reads 1 2 3 4 1 2 3 4 whatever its phase
        encode is doing. This is the period of that stream and where it
        starts, so the blocks before the start are the prologue -- dummy
        shots, preparation, a noise scan -- and everything from there on is
        the scan repeating.

        Returns
        -------
        size : int
            How many blocks one repetition lasts, or 0 if the sequence does
            not repeat.
        start : int
            The 1-based index of the first block of the first full
            repetition.

        Notes
        -----
        Private because neither toolbox has it: this is what the analysis
        here reaches for, not part of the API a design script is written
        against.

        The answer is recorded as the ``TRsize`` definition and read from
        there next time, so a sequence written and read back does not have to
        work it out again. ``TRsize`` is this package's own name, not one the
        Pulseq format defines; nothing writes it unless this is called.

        The core remembers it too, and forgets on anything that could change
        it: adding or rewriting a block, and collapsing duplicates, which
        renumbers the very ids the repeat is read off. Either way the next
        call works it out again. That is the same bargain
        `remove_duplicates` makes -- do the pass once, skip it until
        something invalidates it.
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
        """Register ``event``, insisting it is one of ``kinds``."""
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
        """Write a Pulseq `.seq` file.

        Parameters
        ----------
        name : str or Path
            Where to write it.
        create_signature : bool, default True
            Sign the file, so a reader can tell it has not been edited.
        remove_duplicates : bool, default True
            Collapse identical library rows first. The sequence held here is
            left as it is; what is written is the collapsed copy.
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
        """Write the binary form, which a scanner parses faster.

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
        """Write a Pulseq 1.4.1 file, for an interpreter that predates 1.5."""
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

    # -- the scanner ---------------------------------------------------

    #: Upstream's own, run against this sequence.
    #:
    #: It finds the scanner and hands the sequence to whatever that scanner's
    #: installer wants, and the only thing an installer asks of a sequence is
    #: `write(filename)` -- which means here what it means there. So the
    #: method is taken rather than rewritten, and a scanner PyPulseq learns to
    #: talk to is one this talks to as well.
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
        """Return a sequence holding everything this one holds.

        The core has no copy of its own, so the copy goes through the binary
        form: it carries every library, the block table and the definitions,
        and reading it back is a compiled pass.
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
    # The toolboxes cache decompressed blocks and registration results in
    # Python, and a design script turns that off or clears it to bound its
    # memory. Nothing here caches: a block is decoded in C++ on the way out
    # and an event carries the ids it was registered under. So the flags are
    # remembered and never read, the sizes are zero, and clearing is
    # nothing to do.

    @property
    def use_block_cache(self) -> bool:
        """What was asked for. Nothing here caches decompressed blocks."""
        return self._use_block_cache

    @use_block_cache.setter
    def use_block_cache(self, enabled: bool) -> None:
        self._use_block_cache = enabled

    @property
    def use_event_cache(self) -> bool:
        """What was asked for. Nothing here caches registration results."""
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
        """Nothing to clear."""

    def clear_event_cache(self) -> None:
        """Nothing to clear."""

    def clear_caches(self) -> None:
        """Nothing to clear."""
