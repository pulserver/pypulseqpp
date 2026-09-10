"""FOV scaling, rotation and translation of existing sequences."""

from __future__ import annotations

import numpy as np

from . import _ext as _cxx

__all__ = ["TransformFOV"]


def _quaternion_of(rotation):
    """Return a scalar-first quaternion from a matrix or SciPy rotation."""
    if hasattr(rotation, "as_quat"):
        return np.asarray(rotation.as_quat(canonical=True, scalar_first=True), float)
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(
            f"TransformFOV(): `rotation` must be (3, 3) or a rotation object, "
            f"got {matrix.shape}"
        )
    from scipy.spatial.transform import Rotation

    return np.asarray(
        Rotation.from_matrix(matrix).as_quat(canonical=True, scalar_first=True), float
    )


def _exempt_mask(seq, label, first, last):
    """Return exemption flags for an inclusive, 1-based block range.

    Labels start at zero at the selected range's first block.
    The label is sticky: setting it exempts that block and following blocks
    until cleared. Return None if the label is absent or no block is exempt.
    """
    found = _cxx.evaluate_labels(
        seq._native, evolution="blocks", first_block=first, last_block=last
    )
    flagged = found.get(label)
    if flagged is None:
        return None
    stop = len(seq) if last == 0 else last
    mask = np.zeros(stop - first + 1, dtype=np.uint8)
    seen = np.atleast_1d(flagged)[: mask.size]
    mask[: seen.size] = np.asarray(seen) != 0
    if not mask.any():
        return None
    return mask


def _runs_not_exempt(seq, label, first, last):
    """Return inclusive, 1-based ranges without the exemption label."""
    stop = len(seq) if last == 0 else last
    mask = _exempt_mask(seq, label, first, last)
    if mask is None:
        return [(first, stop)]

    runs = []
    at = None
    for offset, flag in enumerate(mask):
        block = first + offset
        if flag:
            if at is not None:
                runs.append((at, block - 1))
                at = None
        elif at is None:
            at = block
    if at is not None:
        runs.append((at, stop))
    return runs


class TransformFOV:
    """Apply a prescription to an existing sequence.

    Parameters
    ----------
    rotation : array_like or Rotation, optional
        Prescription orientation as a 3-by-3 matrix or SciPy rotation.
        Composed after the rotation already attached to each block.
    translation : sequence of float, optional
        Three offsets in logical coordinates, in metres.
    scale : sequence of float, optional
        Gradient amplitude multipliers along the three logical axes.
        A factor of zero disables encoding on that axis.
    transform : array_like, optional
        4-by-4 homogeneous matrix, mutually exclusive with ``rotation`` and
        ``translation``. Its translation is in the output frame and is
        converted to logical coordinates using the transpose of its rotation.
    use_rotation_extension : bool, default True
        Must be True; waveform-baked rotation is not implemented.
    system : Opts, optional
        Stored for compatibility; not used to validate transformed events.

    Attributes
    ----------
    block_k_origin : tuple of float
        Logical k-space position entering the next processed range, in 1/m.
        Reset at excitation and inverted at refocusing, at the RF centre.
    swept_k : tuple of float
        Cumulative logical gradient area, in 1/m, without RF resets.
        RF and ADC shift phases share this reference.

    Notes
    -----
    A nonzero translation updates both state vectors. Reuse them only for consecutive
    ranges; initialise them to the state entering the first selected block.
    Block ranges do not automatically integrate preceding blocks.
    """

    def __init__(
        self,
        *,
        rotation=None,
        translation=None,
        scale=None,
        transform=None,
        use_rotation_extension: bool = True,
        system=None,
    ) -> None:
        if transform is not None:
            if rotation is not None or translation is not None:
                raise ValueError(
                    "TransformFOV(): `transform` already carries a rotation and a "
                    "translation; give it or give those two, not both"
                )
            matrix = np.asarray(transform, dtype=float)
            if matrix.shape != (4, 4):
                raise ValueError(
                    f"TransformFOV(): `transform` must be (4, 4), got {matrix.shape}"
                )
            rotation = matrix[:3, :3]
            # A homogeneous matrix states its offset in the frame it rotates
            # into; everything here works in the logical one.
            translation = tuple(np.asarray(rotation, dtype=float).T @ matrix[:3, 3])
        elif rotation is None and translation is None and scale is None:
            raise ValueError(
                "TransformFOV(): give at least one of `rotation`, `translation`, "
                "`scale` or `transform`"
            )

        if not use_rotation_extension:
            raise NotImplementedError(
                "TransformFOV(use_rotation_extension=False) would bake the rotation "
                "into new gradient waveforms, one per orientation, which is not "
                "offered here: a rotation is an annotation, and the point of it is "
                "that a thousand orientations share one trajectory"
            )

        self.quaternion = None if rotation is None else _quaternion_of(rotation)
        self.translation = (
            None if translation is None else tuple(float(v) for v in translation)
        )
        self.scale = None if scale is None else tuple(float(v) for v in scale)
        self.use_rotation_extension = use_rotation_extension
        self.system = system
        self.block_k_origin = (0.0, 0.0, 0.0)
        self.swept_k = (0.0, 0.0, 0.0)

        for name, value in (("translation", self.translation), ("scale", self.scale)):
            if value is not None and len(value) != 3:
                raise ValueError(
                    f"TransformFOV(): `{name}` must be three numbers, got {len(value)}"
                )

    def apply_to_sequence(
        self, seq, *, time_range=None, block_range=None, in_place: bool = False
    ):
        """Apply scaling, rotation, then translation.

        ``NOSCL``, ``NOROT`` and ``NOPOS`` labels exempt blocks from the
        respective operations. Labels are evaluated from the selected range's
        start, without inheriting values from earlier blocks. A nonzero
        translation still integrates exempt blocks.

        Parameters
        ----------
        seq : Sequence
            Sequence to transform.
        time_range : sequence of float, optional
            Start and end times in seconds; selects all blocks they touch.
        block_range : sequence of int, optional
            Inclusive, 1-based block range; mutually exclusive with ``time_range``.
            Translation starts from this object's stored state, not from a scan
            of preceding blocks.
        in_place : bool, default False
            Modify ``seq`` rather than a copy. Nonzero translation updates this object's
            state in either case.

        Returns
        -------
        Sequence
            Transformed sequence; ``seq`` itself when ``in_place=True``.
        """
        target = seq if in_place else seq._copy()
        first, last = target._range_for(time_range, block_range)

        if self.scale is not None:
            for begins, ends in _runs_not_exempt(target, "NOSCL", first, last):
                _cxx.apply_fov_scale(
                    target._native, scale=self.scale, first=begins, last=ends
                )
        if self.quaternion is not None:
            for begins, ends in _runs_not_exempt(target, "NOROT", first, last):
                _cxx.apply_fov_rotation(
                    target._native,
                    quaternion=tuple(self.quaternion),
                    first=begins,
                    last=ends,
                )
        if self.translation is not None:
            # One walk, gated per block rather than one walk per stretch:
            # where k stands is a fact about everything played before it,
            # exempt or not, and a stretch skipped is a stretch of swept area
            # missing from every phase after it.
            moved = _cxx.apply_fov_shift(
                target._native,
                shift=self.translation,
                first=first,
                last=last,
                carry=self.swept_k,
                origin=self.block_k_origin,
                exempt=_exempt_mask(target, "NOPOS", first, last),
            )
            self.swept_k = moved["swept"]
            self.block_k_origin = moved["origin"]
        return target

    #: The reference toolbox's name for the same thing.
    apply_to_seq = apply_to_sequence

    def trajectories(self, seq, *, block_range=None):
        """Return ADC trajectories in unrotated logical coordinates, in 1/m.

        Uses ``block_k_origin`` as the position entering the selected range,
        without updating it. Block rotation extensions are not applied.

        Parameters
        ----------
        seq : Sequence
            Sequence to sample.
        block_range : sequence of int, optional
            Inclusive, 1-based range; defaults to the whole sequence.

        Returns
        -------
        list of (int, numpy.ndarray)
            Block index and a ``(3, n_samples)`` trajectory for each ADC block.
            Blocks without ADC samples are omitted.
        """
        first, last = seq._range_for(None, block_range)
        origins = _cxx.block_k_origins(
            seq._native, first=first, last=last, carry=self.block_k_origin
        )["origins"]
        stop = len(seq) if last == 0 else last

        found = []
        for block in range(first, stop + 1):
            sampled = _cxx.absolute_trajectory(
                seq._native, block, origin=tuple(origins[block - first])
            )
            if sampled.shape[1]:
                found.append((block, sampled))
        return found
