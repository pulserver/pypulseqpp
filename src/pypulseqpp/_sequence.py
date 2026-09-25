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
from ._kspace import adc_kspace as _adc_kspace
from ._kspace import calculate_kspace as _calculate_kspace
from ._kspace import detail as _kspace_detail
from ._libraries import SequenceLibraries
from ._libraries import libraries as _libraries
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


def _contents(source) -> bytes:
    """Return the bytes of a file named by a path or held by a binary file object."""
    read = getattr(source, "read", None)
    if read is None:
        return Path(source).read_bytes()
    contents = read()
    if not isinstance(contents, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"read(): the file object returned {type(contents).__name__}; "
            "open the file in binary mode"
        )
    return bytes(contents)


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
    system : Opts, default=None
        System limits and rasters. Defaults to the shared system; rasters
        are recorded in the native sequence.
    use_block_cache : bool, default=True
        Compatibility flag, retained but not used to cache decoded blocks.

    Attributes
    ----------
    system : Opts | None, default=None
        The limits the sequence was constructed with.
    num_blocks : int
        Number of blocks; equivalent to ``len(seq)``.
    block_durations : MutableMapping[int, float]
        Duration of each block in seconds, keyed by 1-based block index.
        Assigning an entry changes the stored block duration.
    block_events : dict[int, NDArray[np.int32]]
        Event IDs of each block, keyed by 1-based block index: delay, RF,
        gx, gy, gz, ADC and extension. The delay column is zero for Pulseq
        1.5. Reading it allocates a dictionary for all blocks.
    definitions : dict[str, Any]
        The definitions written to the `[DEFINITIONS]` section.
    grad_raster_time : float
        Gradient raster in seconds.
    rf_raster_time : float
        RF raster in seconds.
    adc_raster_time : float
        ADC dwell raster in seconds.
    block_duration_raster : float
        Block duration raster in seconds; every block duration is a whole
        number of these.
    version_major : int
        Major version of the Pulseq format the sequence is held as. A file
        older than 1.5 is converted as it is read, so this is the format of
        the sequence in memory, not the version its file declared.
    version_minor : int
        Minor version of the format the sequence is held as.
    version_revision : int
        Revision of the format the sequence is held as.
    signature_type : str | None
        Hash algorithm of the signature carried by the last file written, or
        by the last binary file read: ``'md5'``, or None when it had none.
    signature_file : str | None
        Format that file was written in, ``'text'`` or ``'bin'``.
    signature_value : str | None
        The signature itself.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> seq = pp.Sequence(pp.Opts())
    >>> seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3))
    1
    >>> seq.add_block(pp.make_delay(1e-3))
    2
    >>> seq.num_blocks, round(seq.duration()[0], 6)
    (2, 0.003)
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
        """Block duration raster in seconds; every block duration is a whole number of these."""
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
        The block lasts as long as its longest event, rounded up onto the
        block duration raster.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> gx = pp.make_trapezoid("x", area=1000, duration=2e-3)
        >>> seq.add_block(gx, pp.make_delay(5e-3))
        1
        >>> seq.block_durations[1]
        0.005
        """
        return self._native.add_block_events(*events)

    def set_block(self, index: int, *events) -> None:
        """Write ``events`` over the block at ``index``, 1-based.

        Parameters
        ----------
        index : int
            The block to replace, 1-based.
        *events
            What the block plays from now on, as for `add_block`.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3))
        1
        >>> seq.set_block(1, pp.make_delay(1e-3))
        >>> seq.get_block(1).gx is None, seq.block_durations[1]
        (True, 0.001)
        """
        self._native.set_block_events(index, *events)

    def get_block(self, index: int) -> SimpleNamespace:
        """Decode a block by its 1-based index.

        Parameters
        ----------
        index : int
            The block to decode, 1-based.

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

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_adc(num_samples=64, duration=3.2e-3))
        1
        >>> block = seq.get_block(1)
        >>> int(block.adc.num_samples), block.gx, block.block_duration
        (64, None, 0.0032)
        """
        return SimpleNamespace(**self._native.decode_block(index))

    def get_raw_block_content_IDs(self, index: int) -> SimpleNamespace:
        """Return stored event IDs for a 1-based block index without decoding shapes.

        Missing events have ID zero. The extension chain is a ``(2, n)``
        array of type IDs and reference IDs; block duration is in seconds.

        Parameters
        ----------
        index : int
            The 1-based block index.

        Returns
        -------
        types.SimpleNamespace
            ``block_duration`` and the stored id of each event slot.
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
        event_count : NDArray[np.float64]
            How many blocks carry an event in each column of the block
            table, in upstream's column order: delay, RF, the three
            gradient axes, ADC, extension.

        Examples
        --------
        >>> import numpy as np
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_block_pulse(np.pi / 2, duration=1e-3))
        1
        >>> seq.add_block(pp.make_adc(num_samples=64, duration=3.2e-3))
        2
        >>> seq.duration()
        (0.0042..., 2, array([0., 1., 0., 0., 0., 1., 0.]))
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
        """Return the 1-based index of the block containing time ``t`` in seconds.

        Parameters
        ----------
        t : float
            A time from the start of the sequence, in seconds.

        Returns
        -------
        int | None
            The block playing then, or None if ``t`` is past the end.

        Raises
        ------
        ValueError
            If the block found at ``t`` has a non-positive duration.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_delay(2e-3))
        1
        >>> seq.add_block(pp.make_delay(1e-3))
        2
        >>> seq.find_block_by_time(2.5e-3), seq.find_block_by_time(1.0)
        (2, None)
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
        print_errors : bool, default=False
            Print the report as well as returning it.

        Returns
        -------
        is_ok : bool
            True when nothing was found.
        error_report : list[SimpleNamespace]
            One entry per problem, in block order.

        Notes
        -----
        May record TotalDuration. Does not check all scanner safety constraints.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3))
        1
        >>> seq.check_timing()
        (True, [])
        >>> seq.get_definition("TotalDuration")
        [0.002]
        """
        is_ok, error_report = _check_timing(self)
        if not is_ok and print_errors:
            print_error_report(self, error_report)
        return is_ok, error_report

    # -- definitions ---------------------------------------------------

    def set_definition(self, key: str, value) -> None:
        """Record ``key`` in `[DEFINITIONS]`.

        Parameters
        ----------
        key : str
            The definition's name.
        value : str | float | Sequence[float]
            What it says; a value already recorded under ``key`` is replaced.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence()
        >>> seq.set_definition("FOV", [0.25, 0.25, 0.005])
        >>> seq.definitions
        {'FOV': [0.25, 0.25, 0.005]}
        """
        self._native.set_definition(key, value)

    def get_definition(self, key: str):
        """Return the value recorded for ``key``, or ``''`` when it is not defined.

        Parameters
        ----------
        key : str
            The definition's name.

        Returns
        -------
        str | float | list[float]
            The recorded value, or ``''`` when ``key`` is not defined.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence()
        >>> seq.set_definition("FOV", [0.25, 0.25, 0.005])
        >>> seq.get_definition("FOV"), seq.get_definition("Name")
        ([0.25, 0.25, 0.005], '')
        """
        return self._native.definitions().get(key, "")

    @property
    def definitions(self) -> dict:
        """The definitions written to the `[DEFINITIONS]` section."""
        return self._native.definitions()

    def copy_definitions(self, other_seq: Sequence) -> None:
        """Copy every definition from ``other_seq`` into this sequence.

        Parameters
        ----------
        other_seq : Sequence
            The sequence to copy from. A definition both carry takes its
            value.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> source = pp.Sequence()
        >>> source.set_definition("Name", "gre")
        >>> seq = pp.Sequence()
        >>> seq.copy_definitions(source)
        >>> seq.get_definition("Name")
        'gre'
        """
        for key, value in other_seq.definitions.items():
            self.set_definition(key, value)

    # -- extensions ----------------------------------------------------

    def get_extension_type_ID(self, extension_string: str) -> int:
        """Return the numeric id for ``extension_string``, assigning one if new.

        Parameters
        ----------
        extension_string : str
            The extension's name, as the file spells it.

        Returns
        -------
        int
            Its id in this sequence.
        """
        return self._native.extension_type_id(extension_string)

    def get_extension_type_string(self, extension_id: int) -> str:
        """Return the name ``extension_id`` maps to.

        Parameters
        ----------
        extension_id : int
            The id to look up.

        Returns
        -------
        str
            The extension's name.

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
        """Pin ``extension_str`` to ``extension_id``.

        Parameters
        ----------
        extension_str : str
            The extension's name.
        extension_id : int
            The id it is to be stored under.
        """
        self._native.set_extension_type_id(extension_str, extension_id)

    # -- TR ids --------------------------------------------------------

    def get_or_create_trid_id(self, label_name: str) -> int:
        """Return the TRID number assigned to ``label_name``, assigning one if new.

        Parameters
        ----------
        label_name : str
            What this repetition is called.

        Returns
        -------
        int
            Its number, counting from 1 in the order names were first seen.

        Raises
        ------
        ValueError
            If ``label_name`` is empty.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence()
        >>> [seq.get_or_create_trid_id(name) for name in ("prep", "imaging", "prep")]
        [1, 2, 1]
        """
        label_name = str(label_name)
        if not label_name:
            raise ValueError("TRID label_name must be a non-empty char/string.")
        if label_name not in self._trid_names:
            self._trid_names.append(label_name)
        return self._trid_names.index(label_name) + 1

    def add_trid(self, label_name: str) -> None:
        """Append a block setting TRID to the number assigned to ``label_name``.

        Nothing is appended when the system's ``flag_trid`` is False.

        Parameters
        ----------
        label_name : str
            What this repetition is called; see `get_or_create_trid_id`.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_trid("imaging")
        >>> label = seq.get_block(1).label[0]
        >>> label.label, label.value
        ('TRID', 1)
        """
        if not getattr(self.system, "flag_trid", True):
            return
        from ._make_label import make_label

        self.add_block(
            make_label("TRID", "SET", float(self.get_or_create_trid_id(label_name)))
        )

    # -- gradients -----------------------------------------------------

    def mod_grad_axis(self, axis: str, modifier: float) -> None:
        """Scale the amplitude of every gradient on ``axis`` by ``modifier``.

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

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3))
        1
        >>> seq.mod_grad_axis("x", 0.5)
        >>> seq.get_block(1).gx.area
        500.0
        """
        if axis not in ("x", "y", "z"):
            raise ValueError(
                f"Invalid axis. Must be one of 'x', 'y','z'. Passed: {axis}"
            )
        self._native.scale_gradient_axis("xyz".index(axis), float(modifier))

    def flip_grad_axis(self, axis: str) -> None:
        """Negate the amplitude of every gradient on ``axis``.

        Equivalent to ``mod_grad_axis(axis, -1)``.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            Which axis to act on.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3))
        1
        >>> seq.flip_grad_axis("x")
        >>> seq.get_block(1).gx.area
        -1000.0
        """
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
        init : dict[str, int], default=None
            Initial value of each label before evaluation begins. A label named here is
            reported whether or not the blocks touch it, so that a sequence
            can be evaluated one range at a time.
        evolution : {'none', 'blocks', 'adc', 'label'}, default='none'
            Where to record a value: at the end, at every block, at every
            block that acquires, or at every block that sets or increments
            one.
        time_range : Sequence[float], default=None
            Two times in seconds; only the blocks they touch are walked.
        block_range : Sequence[int], default=None
            Two 1-based block indices. Not with ``time_range``.

        Returns
        -------
        dict[str, int | np.int32 | NDArray[np.int32]]
            Label names mapped to their recorded values. Values are arrays if
            any label has more than one recorded value; otherwise they are
            scalars, with zero for an empty record.

        Notes
        -----
        Labels retain their values until set or incremented. For a partial range,
        ``init`` supplies the incoming state; preceding blocks are not evaluated.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> adc = pp.make_adc(num_samples=64, duration=3.2e-3)
        >>> for line in range(3):
        ...     _ = seq.add_block(pp.make_label("LIN", "SET", line), adc)
        >>> int(seq.evaluate_labels()["LIN"])
        2
        >>> seq.evaluate_labels(evolution="adc")["LIN"]
        array([0, 1, 2], dtype=int32)
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
        """Return the gradient waveforms, the RF pulse timing and the ADC sampling.

        Parameters
        ----------
        append_RF : bool, default=False
            Also return the RF envelope, as a fourth waveform channel.
        time_range : Sequence[float], default=None
            Two times in seconds; only the blocks they touch are expanded.
        block_range : Sequence[int], default=None
            Two 1-based block indices. Not with ``time_range``.
        compat : bool, default=True
            Return upstream's five values, which a drop-in caller unpacks.
            False returns a named result covering all seven Pulseq RF uses,
            which those five values cannot carry.

        Returns
        -------
        tuple | WaveformsAndTimes
            With ``compat``: ``(wave_data, tfp_excitation, tfp_refocusing,
            t_adc, fp_adc)``. Otherwise a named result whose ``waveforms``,
            ``rf`` and ``adc`` carry every RF use tag, per-sample ADC phases
            and the 1-based block each pulse and sample belongs to.

        Notes
        -----
        Gradient channels are ``(2, n)`` arrays of time (s) and amplitude
        (Hz/m); the optional RF channel is complex with amplitude in Hz. Block
        rotations are applied. A time range keeps times measured from the
        start of the scan; a block range restarts them at its first block.

        Examples
        --------
        >>> import numpy as np
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_block_pulse(np.pi / 2, duration=1e-3))
        1
        >>> seq.add_block(pp.make_adc(num_samples=64, duration=3.2e-3))
        2
        >>> wave_data, tfp_excitation, tfp_refocusing, t_adc, fp_adc = seq.waveforms_and_times()
        >>> tfp_excitation[0], t_adc.shape
        (array([0.0005]), (64,))
        >>> seq.waveforms_and_times(compat=False).rf.use
        ('undefined',)
        """
        return _waveforms_and_times(
            self, append_RF, time_range, block_range, compat=compat
        )

    def waveforms(self, append_RF: bool = False, time_range=None, block_range=None):
        """Return gradient corners as time (s) over amplitude (Hz/m), per axis.

        Parameters
        ----------
        append_RF : bool, default=False
            Also return the RF envelope, as a fourth, complex channel in Hz.
        time_range : Sequence[float], default=None
            Two times in seconds; only the blocks they touch are expanded.
        block_range : Sequence[int], default=None
            Two 1-based block indices. Not with ``time_range``.

        Returns
        -------
        list[NDArray]
            One ``(2, n)`` array of time over amplitude per axis, empty where
            an axis plays nothing, then the RF channel when asked for; the
            first value `waveforms_and_times` returns.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_trapezoid("x", flat_area=1000, flat_time=3.2e-3))
        1
        >>> gx, gy, gz = seq.waveforms()
        >>> gx.shape, gy.shape
        ((2, 4), (2, 0))
        """
        return _waveforms(self, append_RF, time_range, block_range)

    def adc_times(self, time_range=None):
        """Return ADC sample times (s) and per-window frequency (Hz) and phase (rad).

        Parameters
        ----------
        time_range : Sequence[float], default=None
            Two times in seconds; only the blocks they touch are expanded.

        Returns
        -------
        t_adc : NDArray[np.float64]
            When every sample is taken, in seconds from the start of the scan.
        fp_adc : NDArray[np.float64]
            ``(n, 2)``, one row per ADC window rather than per sample:
            frequency (Hz) and phase (rad) offsets, without ppm corrections.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_adc(num_samples=64, duration=3.2e-3))
        1
        >>> t_adc, fp_adc = seq.adc_times()
        >>> t_adc.shape, fp_adc.shape
        ((64,), (1, 2))
        """
        return _adc_times(self, time_range)

    def rf_times(self, time_range=None, *, compat: bool = True):
        """Return RF pulse centre times with their frequency and phase offsets.

        Parameters
        ----------
        time_range : Sequence[float], default=None
            Two times in seconds; only the blocks they touch are expanded.
        compat : bool, default=True
            Return upstream's four values, which describe two of Pulseq's
            seven RF uses. False returns a named result covering all seven.

        Returns
        -------
        tuple | RfTimes
            With ``compat``: ``(t_excitation, fp_excitation, t_refocusing,
            fp_refocusing)``. A pulse whose row records no use is counted as
            an excitation, as upstream counts it. Otherwise a named result
            carrying ``t``, ``freq_offset``, ``phase_offset``, ``use`` and
            ``block`` for every pulse.

        Examples
        --------
        >>> import numpy as np
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_block_pulse(np.pi / 2, duration=1e-3))
        1
        >>> t_excitation, fp_excitation, t_refocusing, fp_refocusing = seq.rf_times()
        >>> t_excitation, t_refocusing.size
        (array([0.0005]), 0)
        """
        return _rf_times(self, time_range, compat=compat)

    def calculate_kspace(
        self, trajectory_delay=0.0, gradient_offset=0.0, block_range=None
    ):
        """Integrate the gradients with excitation resets and refocusing.

        Each block's rotation is applied. K-space coordinates are in 1/m.

        Parameters
        ----------
        trajectory_delay : float | ArrayLike, default=0.0
            Per-axis timing correction (s); positive values advance the gradient.
        gradient_offset : float | ArrayLike, default=0.0
            A background gradient per axis, in Hz/m.
        block_range : Sequence[int], default=None
            Two 1-based block indices; only those blocks are followed.

        Returns
        -------
        k_traj_adc : NDArray[np.float64]
            ``(3, n)``: the k-space location of each ADC sample, in 1/m.
        k_traj : NDArray[np.float64]
            Full trajectory in 1/m, sampled through ramps and at event times.
        t_excitation : NDArray[np.float64]
            Centre of each excitation pulse, in seconds from the start of the
            selected range.
        t_refocusing : NDArray[np.float64]
            Centre of each refocusing pulse, in seconds from the same origin.
        t_adc : NDArray[np.float64]
            Time of each ADC sample, in seconds from the same origin.

        Examples
        --------
        >>> import numpy as np
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_block_pulse(np.pi / 2, duration=1e-3))
        1
        >>> seq.add_block(pp.make_adc(num_samples=64, duration=3.2e-3))
        2
        >>> k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
        >>> k_traj_adc.shape, t_excitation
        ((3, 64), array([0.0005]))
        """
        return _calculate_kspace(self, trajectory_delay, gradient_offset, block_range)

    #: Upstream carries this name for the same calculation, and so does this.
    calculate_kspacePP = calculate_kspace

    def adc_kspace(self, trajectory_delay=0.0, gradient_offset=0.0, block_range=None):
        """Return the k-space location of each ADC sample, in 1/m.

        The first result of :meth:`calculate_kspace`, integrated the same way
        with block rotations applied, without sampling the trajectory between
        the samples; excitations reset it and refocusing pulses invert it.

        Parameters
        ----------
        trajectory_delay : float | ArrayLike, default=0.0
            Per-axis timing correction (s); positive values advance the gradient.
        gradient_offset : float | ArrayLike, default=0.0
            A background gradient per axis, in Hz/m.
        block_range : Sequence[int], default=None
            Two 1-based block indices; only those blocks are followed.

        Returns
        -------
        NDArray[np.float64]
            ``(3, n)``: one column per ADC sample, in play order.

        Examples
        --------
        >>> import numpy as np
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_block_pulse(np.pi / 2, duration=1e-3))
        1
        >>> seq.add_block(pp.make_adc(num_samples=64, duration=3.2e-3))
        2
        >>> seq.adc_kspace().shape
        (3, 64)
        """
        return _adc_kspace(self, trajectory_delay, gradient_offset, block_range)

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
        """Return the gradient splines after each block's rotation, in Hz/m over seconds.

        Parameters
        ----------
        trajectory_delay : float | Sequence[float], default=0
            Timing correction in seconds, one value for all axes or one per
            axis; positive values advance the gradients.
        gradient_offset : float | Sequence[float], default=0
            A background gradient in Hz/m, one value for all axes or one per
            axis.
        time_range : Sequence[float], default=None
            Two times in seconds; only the blocks they touch are expanded.
        block_range : Sequence[int], default=None
            Two 1-based block indices. Not with ``time_range``.

        Returns
        -------
        list[scipy.interpolate.PPoly | None]
            One spline per axis, None where an axis plays nothing and has no
            offset.

        Warns
        -----
        UserWarning
            When ``trajectory_delay`` exceeds 100 us on any axis.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_trapezoid("x", flat_area=1000, flat_time=3.2e-3))
        1
        >>> gx, gy, gz = seq.get_gradients()
        >>> round(float(gx(2e-3))), gy
        (312500, None)
        """
        return _get_gradients(
            self, trajectory_delay, gradient_offset, time_range, block_range
        )

    # -- what the sequence is ------------------------------------------

    def calc_rf_power(
        self, block_range=None, window_duration: float | None = None
    ) -> tuple[float, float, float, float]:
        """Return the RF's mean power, peak power, RMS amplitude and energy.

        As MATLAB Pulseq's ``calcRfPower``.

        Parameters
        ----------
        block_range : Sequence[int], default=None
            First and last block, 1-based and inclusive; all by default.
        window_duration : float, default=None
            Seconds. Energy, mean power and rms are then the largest over runs
            of whole blocks no longer than this, each divided by it.

        Returns
        -------
        mean_pwr : float
            Hz^2.
        peak_pwr : float
            Hz^2.
        rf_rms : float
            Hz.
        total_energy : float
            Hz^2 s.

        Notes
        -----
        Each pulse is read as :func:`pypulseqpp.calc_rf_power` reads it, a
        dynamic pTx pulse as the root-sum-square of its channels. The values
        are relative: divide ``rf_rms`` by gamma for tesla and the powers by
        gamma squared. :func:`pypulseqpp.safety.check_sar` gives SAR.

        Examples
        --------
        >>> import numpy as np
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_block_pulse(np.pi / 2, duration=1e-3))
        1
        >>> mean_pwr, peak_pwr, rf_rms, total_energy = seq.calc_rf_power()
        >>> round(peak_pwr), round(total_energy, 3)
        (62500, 62.5)
        """
        if self.num_blocks == 0:
            return 0.0, 0.0, 0.0, 0.0
        first, last = (
            (1, self.num_blocks)
            if block_range is None
            else _plot.blocks_for(self, block_range=block_range)
        )
        found = _cxx.rf_power(
            self._native,
            first,
            last,
            window=0.0 if window_duration is None else float(window_duration),
        )
        return (
            found["mean_power"],
            found["peak_power"],
            found["rms"],
            found["energy"],
        )

    def test_report(self) -> str:
        """Return sequence timing, encoding and gradient statistics, formatted as text.

        Returns
        -------
        str
            The statistics `test_report_dict` returns, one per line, with the
            timing check's findings.

        Notes
        -----
        Runs `check_timing`, which may record TotalDuration.

        Examples
        --------
        >>> import numpy as np
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_block_pulse(np.pi / 2, duration=1e-3))
        1
        >>> seq.add_block(pp.make_adc(num_samples=64, duration=3.2e-3))
        2
        >>> print(seq.test_report())
        Number of blocks: 2
        Number of events:
        RF:      1
        ...
        Event timing check passed successfully
        ...
        """
        return _report_text(_report_data(self))

    def test_report_dict(self) -> dict:
        """Return timing, encoding and gradient statistics.

        Returns
        -------
        dict[str, Any]
            ``num_blocks``, ``event_count`` and ``libraries``; ``duration``,
            ``TE`` and ``TR`` in seconds; ``flip_angles_deg``;
            ``unique_k_positions``; ``max_gradient`` and ``max_slew_rate``,
            each per axis and as a vector magnitude, in the file's units and
            in the scanner's; and ``timing_ok`` with ``timing_error_report``.
            A sequence whose encoding visits more than one position also
            carries ``dimensions``, ``spatial_resolution_mm``,
            ``repetitions`` and ``is_cartesian``.

        Notes
        -----
        Runs `check_timing`, which may record TotalDuration.

        Examples
        --------
        >>> import numpy as np
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_block_pulse(np.pi / 2, duration=1e-3))
        1
        >>> seq.add_block(pp.make_adc(num_samples=64, duration=3.2e-3))
        2
        >>> report = seq.test_report_dict()
        >>> report["num_blocks"], report["flip_angles_deg"], report["timing_ok"]
        (2, ..., True)
        """
        return _report_data(self)

    def rf_flip_angles(self) -> np.ndarray:
        """Return the flip angle of each RF event of the library, in degrees.

        Entry ``i`` is RF id ``i + 1``, the row order of :meth:`libraries`.
        The flip angle is the magnitude of the integral of the event's complex
        envelope times its amplitude, in turns, times 360. A dynamic pTx pulse
        is integrated channel by channel on its shared time base and the
        channels summed: the flip angle where every channel has unit, in-phase
        sensitivity. :meth:`test_report_dict` lists the distinct values.

        Examples
        --------
        >>> import numpy as np
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> for angle in (10, 20):
        ...     _ = seq.add_block(pp.make_sinc_pulse(np.deg2rad(angle), duration=2e-3))
        >>> seq.rf_flip_angles().round(1)
        array([10., 20.])
        """
        return np.asarray(_cxx.rf_flip_angles(self._native))

    def rf_channels(self) -> np.ndarray:
        """Return the transmit channels each RF event of the library holds.

        Entry ``i`` is RF id ``i + 1``. A dynamic pTx pulse holds its channels
        one after another over one time base, and the count is the number of
        samples at its first sample time when the times are that many
        identical copies, as the reference interpreter reads it; any other
        pulse holds one.

        Examples
        --------
        >>> import numpy as np
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> _ = seq.add_block(pp.make_ptx_pulse(100.0 * np.ones((2, 100))))
        >>> _ = seq.add_block(pp.make_block_pulse(np.pi / 2, duration=1e-3))
        >>> seq.rf_channels().tolist()
        [2, 1]
        """
        return np.asarray(_cxx.rf_channel_counts(self._native))

    # -- the repeating unit --------------------------------------------

    def _detect_tr(self) -> tuple[int, int]:
        """Detect the period of the block-definition stream.

        Returns
        -------
        size : int
            Blocks per repetition; the whole sequence when it does not
            repeat, and zero when it has no blocks.
        start : int
            1-based start of the first repetition, always 1.

        Notes
        -----
        The repetition is the shortest period of the block-definition stream
        from the first block that every block repeats, or failing that the
        shortest period by block structure, or the whole sequence: a slice
        acquired with its own preparation and dummy shots is one repetition,
        and a block played once makes the whole sequence one. ``start`` is 1.
        Records ``TRsize`` in sequence definitions; a recorded size shorter
        than the sequence that divides it and that the blocks repeat with is
        taken instead, so a longer hyper-TR can be declared. Structural edits
        invalidate the native detection cache.
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
            delay in the sequence that is not named here is left unchanged.

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

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_delay(1e-3), pp.make_soft_delay("TE", numID=0))
        1
        >>> seq.apply_soft_delay(TE=5e-3)
        >>> seq.block_durations[1]
        0.005
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
        """Return each soft delay's default value.

        A soft delay says how a block's duration follows from a value the
        console supplies: `duration = value / factor + offset`. Read the other
        way, the duration a block was built with says what value that is --
        and every block sharing a numeric id has to agree about it.

        Returns
        -------
        defaults : dict[str, float]
            The default value for each soft delay, by its hint.
        error_report : list[str]
            One line per disagreement found; empty when they all agree.
        limits : list[dict | None]
            Per numeric id, the default, the hint, the block it came from,
            and the range of values that keep the block duration positive.

        Raises
        ------
        ValueError
            If two numeric ids carry the same hint.

        Warns
        -----
        UserWarning
            If the numeric ids are not contiguous.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_delay(1e-3), pp.make_soft_delay("TE", numID=0))
        1
        >>> defaults, error_report, limits = seq.get_default_soft_delay_values()
        >>> defaults, error_report
        ({'TE': 0.001}, [])
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
        """Register an RF event and return its ids.

        Parameters
        ----------
        event : object
            An RF pulse.

        Returns
        -------
        rf_id : int
            The row it was stored as.
        shape_ids : list[int]
            Its magnitude, phase and time shapes; the time shape is 0 when
            the pulse lies on the RF raster.
        """
        stored = self._register(event, ("rf",), "register_rf_event", "an RF pulse")
        return stored["id"], stored["shapes"]

    def register_grad_event(self, event):
        """Register a gradient event and return its ids.

        Parameters
        ----------
        event : object
            A trapezoid or an arbitrary gradient.

        Returns
        -------
        int | tuple[int, list[int]]
            A trapezoid's row id on its own; an arbitrary waveform's row id
            with its waveform and time shape ids, the way the toolboxes
            report them.
        """
        stored = self._register(event, ("grad",), "register_grad_event", "a gradient")
        return (stored["id"], stored["shapes"]) if stored["shapes"] else stored["id"]

    def register_adc_event(self, event) -> tuple[int, int]:
        """Register an ADC event and return its ids.

        Parameters
        ----------
        event : object
            An ADC.

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
        """Register a label event and return its row id.

        Parameters
        ----------
        event : object
            A label set or increment.

        Returns
        -------
        int
            The row it was stored as.
        """
        return self._register(
            event, ("LABELSET", "LABELINC"), "register_label_event", "a label"
        )["id"]

    def register_control_event(self, event) -> int:
        """Register a trigger or digital-output event and return its row id.

        Parameters
        ----------
        event : object
            A trigger or a digital output.

        Returns
        -------
        int
            The row it was stored as.
        """
        return self._register(
            event, ("TRIGGERS",), "register_control_event", "a trigger"
        )["id"]

    def register_rotation_event(self, event) -> int:
        """Register a rotation event and return its row id.

        Parameters
        ----------
        event : object
            A rotation.

        Returns
        -------
        int
            The row it was stored as.
        """
        return self._register(
            event, ("ROTATIONS",), "register_rotation_event", "a rotation"
        )["id"]

    def register_rf_shim_event(self, event) -> int:
        """Register an RF shim event and return its row id.

        Parameters
        ----------
        event : object
            An RF shim.

        Returns
        -------
        int
            The row it was stored as.
        """
        return self._register(
            event, ("RF_SHIMS",), "register_rf_shim_event", "an RF shim"
        )["id"]

    def register_soft_delay_event(self, event) -> int:
        """Register a soft delay event and return its row id.

        Parameters
        ----------
        event : object
            A soft delay.

        Returns
        -------
        int
            The row it was stored as.
        """
        return self._register(
            event, ("DELAYS",), "register_soft_delay_event", "a soft delay"
        )["id"]

    # -- files ---------------------------------------------------------

    def libraries(self) -> SequenceLibraries:
        """Return the block table and every library, as a Pulseq file of the sequence holds them.

        The ids are this sequence's own. :meth:`write` with
        ``remove_duplicates=False`` writes these rows under these ids; by
        default it writes a collapsed copy, whose ids differ wherever two rows
        were equal. Taking the tables changes nothing, so a shape registered
        and not yet encoded, which the writer encodes on the way out, comes
        back as its samples.

        Returns
        -------
        pypulseqpp.io.SequenceLibraries
            A snapshot, in read-only arrays that no later edit reaches.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3))
        1
        >>> tables = seq.libraries()
        >>> tables.blocks.tolist(), tables.trapezoid_ids.tolist()
        ([[0, 1, 0, 0, 0, 0]], [1])
        """
        return _libraries(self._native)

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
        name : str | os.PathLike[str]
            Where to write it.
        create_signature : bool, default=True
            Sign the file, so a reader can tell it has not been edited.
        remove_duplicates : bool, default=True
            Collapse identical library rows first. Write a collapsed
            copy, leaving this sequence's event libraries unchanged.
        check_timing : bool, default=False
            Judge the timing first, and warn if anything is wrong.
        v141_compat : bool, default=False
            Write 1.4.1 instead, for an interpreter that predates 1.5.

        Returns
        -------
        str | None
            The signature written, or None if the file is unsigned. It is
            the signature of what was written, so with ``remove_duplicates``
            it belongs to the collapsed copy rather than to this sequence.

        Warns
        -----
        UserWarning
            If ``check_timing`` is set and the timing check finds anything.

        Examples
        --------
        >>> import pathlib, tempfile
        >>> import pypulseqpp as pp
        >>> folder = tempfile.mkdtemp()
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3))
        1
        >>> signature = seq.write(pathlib.Path(folder) / "gre.seq")
        >>> len(signature), seq.signature_type, seq.signature_file
        (32, 'md5', 'text')
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
        name : str | os.PathLike[str]
            Where to write it.
        create_signature : bool, default=True
            Append the signature section: an MD5 of everything above it, so a
            file says whether it is the file that was written.

        Returns
        -------
        str | None
            The signature written, or None when none was asked for.

        Examples
        --------
        >>> import pathlib, tempfile
        >>> import pypulseqpp as pp
        >>> folder = tempfile.mkdtemp()
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3))
        1
        >>> signature = seq.write_binary(pathlib.Path(folder) / "gre.bseq")
        >>> len(signature), seq.signature_file
        (32, 'bin')
        """
        written = _cxx.write_binary(self._native, create_signature)
        Path(name).write_bytes(written)
        return self._note_binary_signature(written)

    def write_v141(
        self, name, create_signature: bool = True, gamma=42576000.0, field=1.5
    ) -> str | None:
        """Write Pulseq 1.4.1 text, for an interpreter predating Pulseq 1.5.

        Parameters
        ----------
        name : str | os.PathLike[str]
            Where to write it.
        create_signature : bool, default=True
            Sign the file, so a reader can tell it has not been edited.
        gamma : float, default=42576000.0
            Gyromagnetic ratio in Hz/T.
        field : float, default=1.5
            Main field in T. With ``gamma``, converts ppm offsets, which 1.4.1
            has no column for, to absolute offsets.

        Returns
        -------
        str | None
            The MD5 signature written, or None if the file is unsigned.

        Raises
        ------
        RuntimeError
            If the sequence carries rotation or RF-shim extensions.

        Warns
        -----
        UserWarning
            When soft delays are omitted.

        Examples
        --------
        >>> import pathlib, tempfile
        >>> import pypulseqpp as pp
        >>> path = pathlib.Path(tempfile.mkdtemp()) / "gre.seq"
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3))
        1
        >>> _ = seq.write_v141(path)
        >>> path.read_text().splitlines()[3:6]
        ['[VERSION]', 'major 1', 'minor 4']
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
        file_path : str | os.PathLike[str] | typing.BinaryIO
            The file to read, or a binary file object whose ``read()``
            returns its contents, such as :class:`io.BytesIO` over bytes
            received without a file.
        detect_rf_use : bool, default=False
            Work out what each unlabelled pulse is for, from what it does.
            Before revision 1.5.0 the format had nowhere to record it, so a
            file older than that arrives with its pulses unlabelled. Pulses
            the file does label are left alone.
        remove_duplicates : bool, default=True
            Collapse identical library rows after reading.
        verify : bool, default=False
            Check the file against the signature it carries.

        Raises
        ------
        RuntimeError
            If ``verify`` is set and the signature the file carries is not the
            signature of its contents.
        TypeError
            If a file object returns text, as one opened in text mode does.

        Warns
        -----
        UserWarning
            If ``detect_rf_use`` is set and every pulse in the file already
            records what it is for.

        Examples
        --------
        >>> import pathlib, tempfile
        >>> import pypulseqpp as pp
        >>> path = pathlib.Path(tempfile.mkdtemp()) / "gre.seq"
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3))
        1
        >>> _ = seq.write(path)
        >>> loaded = pp.Sequence()
        >>> loaded.read(path, verify=True)
        >>> loaded.num_blocks, loaded.grad_raster_time
        (1, 2e-05)

        The same bytes, held in memory rather than in a file:

        >>> import io
        >>> received = pp.Sequence()
        >>> received.read(io.BytesIO(path.read_bytes()), verify=True)
        >>> received.num_blocks
        1
        """
        contents = _contents(file_path)
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
        """Major version of the Pulseq format this sequence is held as."""
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
        """Open the sequence in the SeqEyes viewer.

        SeqEyes is an optional dependency, installed with
        ``pip install 'pypulseqpp[plot]'``. It reads the sequence from a file,
        which holds only the selected range.

        Parameters
        ----------
        label : str, default=''
            Upstream's ADC label display. Ignored.
        show_blocks : bool, default=False
            Upstream's block-boundary grid. Ignored.
        save : bool, default=False
            Upstream's figure saving. Ignored.
        time_range : Sequence[float], default=(0, np.inf)
            The seconds to draw, measured from the start of the scan.
        time_disp : {'s', 'ms', 'us'}, default='s'
            Upstream's time unit. Ignored.
        grad_disp : {'kHz/m', 'mT/m'}, default='kHz/m'
            Upstream's gradient unit. Ignored.
        plot_now : bool, default=True
            Wait for the window to be closed before returning. When False,
            the window is left open and the returned viewer is live.
        clear : bool, default=True
            Upstream's figure clearing. Ignored.
        overlay : object, default=None
            Upstream's plot to overlay. Ignored.
        stacked : bool, default=False
            Upstream's single stacked figure. Ignored.
        show_guides : bool, default=False
            Upstream's cursor guides. Ignored.
        block_range : Sequence[int], default=None
            The first and last block to draw, 1-based and inclusive.
        tr_range : Sequence[int], default=None
            The first and last repetition to draw, 1-based and inclusive.
            A repetition is the period of the block definition stream from
            the first block; a sequence that does not repeat is one.

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
            If more than one range is given or a range is outside the
            sequence.

        Warns
        -----
        UserWarning
            When an ignored parameter is given a value other than its default.

        Notes
        -----
        The ignored parameters are upstream's, accepted so a script written
        for it runs: SeqEyes draws units, labels and block edges its own way,
        from its own settings.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3))
        1
        >>> viewer = seq.plot(block_range=(1, 1), plot_now=False)  # doctest: +SKIP
        >>> viewer.close()  # doctest: +SKIP
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
        axes_color=None,
        rf_color=None,
        gx_color=None,
        gy_color=None,
        gz_color=None,
        rf_plot: str = "abs",
        *,
        tr=None,
        max_underlays: int = 16,
        underlay_color=None,
        ax=None,
    ) -> SimpleNamespace:
        """Draw a publication-style diagram of one repetition, using mrsd.

        Rows are RF, the gradient axes z, y and x after each block's rotation,
        and ADC, each drawn from the waveform the sequence plays. Every row
        carries its own scale, set by the largest magnitude that channel reaches
        over the repetitions drawn, so heights are comparable within a row and
        not between rows. The other repetitions are drawn underneath in
        ``underlay_color``, which shows what changes from one to the next, and
        a TR interval is marked below.

        Parameters
        ----------
        time_range : Sequence[float], default=(0, np.inf)
            Upstream's window, in seconds. Given, the blocks it touches are
            drawn alone, without repetitions underneath.
        line_width : float, default=1.2
            Width of every drawn line, in points.
        axes_color : str | tuple[float, ...], default=None
            A Matplotlib colour for the baselines and the repetition marker.
            The package's faint ink by default.
        rf_color : str | tuple[float, ...], default=None
            A Matplotlib colour for the RF and ADC rows. The package's ink by
            default, which reads against a light and a dark background alike.
        gx_color : str | tuple[float, ...], default=None
            A Matplotlib colour for the x gradient row, as ``rf_color``.
        gy_color : str | tuple[float, ...], default=None
            A Matplotlib colour for the y gradient row, as ``rf_color``.
        gz_color : str | tuple[float, ...], default=None
            A Matplotlib colour for the z gradient row, as ``rf_color``.
        rf_plot : {'abs', 'real', 'imag'}, default='abs'
            Which part of the RF waveform to draw.
        tr : int, default=None
            1-based repetition to draw. By default, the one in which an axis
            reaches its largest magnitude.
        max_underlays : int, default=16
            At most this many repetitions, evenly spaced, are drawn underneath,
            together with those in which each axis reaches its most negative and
            most positive value; 0 draws none.
        underlay_color : str | tuple[float, ...], default=None
            A Matplotlib colour. The package's muted ink by default.
        ax : matplotlib.axes.Axes, default=None
            Axes to draw in; a new figure by default.

        Returns
        -------
        SimpleNamespace
            ``diagram``, the :class:`mrsd.Diagram`, whose ``annotate`` and
            ``interval`` add labels; ``tr``, the repetition drawn solid, and
            ``underlays``, those drawn underneath, 1-based (``None`` and
            empty for a ``time_range``).

        Notes
        -----
        Repetitions are detected from the block definitions. Choosing them is
        one compiled pass over the block table, and only the repetitions drawn
        are expanded, so the cost does not grow with the length of the scan. A
        sequence that does not repeat is one repetition, drawn whole.

        Examples
        --------
        >>> import numpy as np
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_block_pulse(np.pi / 2, duration=1e-3))
        1
        >>> seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3))
        2
        >>> drawn = seq.paper_plot()  # doctest: +SKIP
        >>> drawn.tr, drawn.underlays  # doctest: +SKIP
        (1, [])
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
        """Merge identical library rows and renumber the references to them.

        Parameters
        ----------
        in_place : bool, default=False
            Collapse this sequence. Otherwise a copy is collapsed and this
            one is left as it is.

        Returns
        -------
        Sequence
            The collapsed sequence: this one, or the copy.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> for _ in range(2):
        ...     _ = seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3))
        >>> print(seq.remove_duplicates())
        Sequence:
        blocks: 2
        rf_library: 0
        grad_library: 1
        ...
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
        """`write(filename, create_signature=False)`.

        Parameters
        ----------
        filename : str | os.PathLike[str]
            Where to write it.
        """
        self.write(filename, create_signature=False)

    def read_binary(self, filename) -> None:
        """`read(filename)`, which distinguishes text from binary itself.

        Parameters
        ----------
        filename : str | os.PathLike[str]
            The file to read.
        """
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
        """Always zero; no decoded block is cached."""
        return 0

    @property
    def event_cache_size(self) -> int:
        """Always zero; no registration result is cached."""
        return 0

    def clear_block_cache(self) -> None:
        """Compatibility no-op."""

    def clear_event_cache(self) -> None:
        """Compatibility no-op."""

    def clear_caches(self) -> None:
        """Compatibility no-op."""
