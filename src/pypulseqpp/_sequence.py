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
from ._check_timing import _limit, print_error_report
from ._check_timing import check_timing as _check_timing

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

    # -- the repeating unit --------------------------------------------

    def detect_tr(self) -> tuple[int, int]:
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
        The answer is recorded as the ``TRsize`` definition and read from
        there next time, so a sequence written and read back does not have to
        work it out again. ``TRsize`` is this package's own name, not one the
        Pulseq format defines; nothing writes it unless this is called.

        Adding or rewriting a block makes the answer stale, and so does
        collapsing duplicates, since that renumbers the definitions. Either
        way the next call works it out again.
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

    def write_binary(
        self,
        name,
        create_signature: bool = True,  # noqa: ARG002 -- the binary form is unsigned
    ) -> None:
        """Write the binary form, which a scanner parses faster.

        Parameters
        ----------
        name : str or Path
            Where to write it.
        create_signature : bool, default True
            Accepted for the text writer's signature; the binary format has
            no signature section, so a binary file is never signed.

        Returns
        -------
        None
            Always, since there is no signature to hand back.
        """
        Path(name).write_bytes(_cxx.write_binary(self._native))
        return None

    def write_v141(
        self, name, create_signature: bool = True, gamma=42576000.0, field=1.5
    ) -> str | None:
        """Write a Pulseq 1.4.1 file, for an interpreter that predates 1.5."""
        written = _cxx.write_text_v141(self._native, create_signature, gamma, field)
        Path(name).write_bytes(written)
        return self._note_signature(written, "text")

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
        self._native = _cxx.read(Path(file_path).read_bytes(), verify)
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
