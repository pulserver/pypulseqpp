"""Diffusion tables: reading one from a file, and recording it in a sequence."""

from __future__ import annotations

__all__ = ["get_diffusion_table", "read_diffusion_table", "set_diffusion_table"]

import os
from pathlib import Path

import numpy as np

#: Definitions a sequence records its diffusion table under.
TABLE, B_VALUE, DIRECTIONS = "DiffusionTable", "DiffusionBValue", "DiffusionDirections"


def _rows(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split("#", 1)[0].split()
        if fields:
            rows.append([float(field) for field in fields])
    return rows


def _blocks(rows: list[list[float]], path: Path) -> dict[int, np.ndarray]:
    blocks: dict[int, np.ndarray] = {}
    start = 0
    while start < len(rows):
        if len(rows[start]) != 1:
            raise ValueError(f"{path}: expected a count line, got {rows[start]}")
        count = int(rows[start][0])
        block = rows[start + 1 : start + 1 + count]
        if len(block) != count or any(len(row) != 3 for row in block):
            raise ValueError(
                f"{path}: the block of {count} is not {count} rows of x y z"
            )
        blocks.setdefault(count, np.asarray(block, dtype=float).reshape(count, 3))
        start += 1 + count
    return blocks


def _fsl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stem = path.with_suffix("")
    bvals = next(
        (
            candidate
            for suffix in (".bval", ".bvals")
            if (candidate := stem.with_suffix(suffix)).is_file()
        ),
        None,
    )
    if bvals is None:
        raise ValueError(f"{path}: no .bval beside it")
    b = np.loadtxt(bvals, dtype=float, ndmin=1).ravel()
    vectors = np.loadtxt(path, dtype=float, ndmin=2)
    if vectors.shape[0] != 3 and vectors.shape[1] == 3:
        vectors = vectors.T
    if vectors.shape != (3, b.size):
        raise ValueError(f"{path}: {vectors.shape} vectors for {b.size} b-values")
    return b, vectors.T


def read_diffusion_table(
    path: str | os.PathLike[str],
    b_value: float | None = None,
    directions: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a diffusion table: one b-value and one direction per volume.

    Three layouts are read, told apart by the file:

    - ``x y z b`` per row (MRtrix's layout), b in s/mm².
    - An FSL pair, given as the ``.bvec`` path with the ``.bval`` beside it.
    - A block direction table: one or more blocks, each a line holding a count
      N followed by N rows ``x y z``. ``directions`` picks the block of that
      count, and a row's b-value is ``b_value * (x² + y² + z²)``, so that
      shorter vectors make lower shells and a zero row a b = 0 volume.

    ``#`` starts a comment, and values are separated by any whitespace.

    Parameters
    ----------
    path
        The table file.
    b_value : float, optional
        Largest b-value (s/mm²), which a block direction table's unit vectors
        reach. Required for that layout and ignored otherwise.
    directions : int, optional
        Count of the block to read from a block direction table; may be
        omitted when the file holds one block.

    Returns
    -------
    b : numpy.ndarray
        ``(volumes,)``, s/mm².
    g : numpy.ndarray
        ``(volumes, 3)``, unit directions along the sequence's gradient
        channels x, y and z, and zero where ``b`` is zero. Those channels turn
        with the prescription, so the directions are in its logical frame.

    Raises
    ------
    ValueError
        If the file matches none of the layouts, a block direction table is
        read without ``b_value``, or no block has ``directions`` rows.
    """
    path = Path(path)
    if path.suffix in (".bvec", ".bvecs"):
        b, vectors = _fsl(path)
    else:
        rows = _rows(path)
        if rows and all(len(row) == 4 for row in rows):
            table = np.asarray(rows, dtype=float)
            b, vectors = table[:, 3], table[:, :3]
        else:
            if b_value is None:
                raise ValueError(f"{path}: a block direction table needs b_value")
            blocks = _blocks(rows, path)
            if directions is None and len(blocks) == 1:
                directions = next(iter(blocks))
            if directions not in blocks:
                raise ValueError(
                    f"{path}: no block of {directions} directions; it has {sorted(blocks)}"
                )
            vectors = blocks[directions]
            b = b_value * np.sum(vectors**2, axis=1)

    norm = np.linalg.norm(vectors, axis=1)
    weighted = (b > 0) & (norm > 0)
    g = np.zeros_like(vectors)
    g[weighted] = vectors[weighted] / norm[weighted, None]
    return b, g


def set_diffusion_table(
    seq,
    path: str | os.PathLike[str],
    b_value: float | None = None,
    directions: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a diffusion table and record where it is in a sequence's definitions.

    The table is not copied: ``DiffusionTable`` records the path,
    ``DiffusionBValue`` and ``DiffusionDirections`` the arguments given, so
    that :func:`get_diffusion_table` reads the same table back from the file.

    Parameters
    ----------
    seq : Sequence
        The sequence that plays the table, one volume per row.
    path, b_value, directions
        As for :func:`read_diffusion_table`.

    Returns
    -------
    b, g : numpy.ndarray
        The table, as :func:`read_diffusion_table` returns it.
    """
    b, g = read_diffusion_table(path, b_value, directions)
    seq.set_definition(TABLE, os.fspath(Path(path).resolve()))
    if b_value is not None:
        seq.set_definition(B_VALUE, float(b_value))
    if directions is not None:
        seq.set_definition(DIRECTIONS, int(directions))
    return b, g


def get_diffusion_table(seq) -> tuple[np.ndarray, np.ndarray] | None:
    """Read the diffusion table a sequence's definitions name.

    Returns
    -------
    (b, g) or None
        As :func:`read_diffusion_table` returns them; None when the sequence
        names no table.
    """
    path = seq.get_definition(TABLE)
    if path in ("", None):
        return None
    if isinstance(path, list):
        path = " ".join(str(part) for part in path)

    def stated(name: str) -> float | None:
        value = seq.get_definition(name)
        if value in ("", None):
            return None
        return float(value[0] if isinstance(value, list) else value)

    b_value, directions = stated(B_VALUE), stated(DIRECTIONS)
    return read_diffusion_table(
        str(path), b_value, None if directions is None else int(directions)
    )
