"""Diffusion tables read from file and recorded in a sequence by path."""

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp.io import get_diffusion_table, read_diffusion_table, set_diffusion_table

BLOCKS = """# comment
3
1.0 0.0 0.0
0.0\t0.5\t0.0
0.0 0.0 0.0
2
0.0 0.0 1.0
0.6 0.8 0.0
"""


@pytest.fixture
def blocks(tmp_path):
    path = tmp_path / "table7.dat"
    path.write_text(BLOCKS)
    return path


def test_block_table_scales_b_by_the_squared_vector_length(blocks):
    b, g = read_diffusion_table(blocks, b_value=2000.0, directions=3)
    np.testing.assert_allclose(b, [2000.0, 500.0, 0.0])
    np.testing.assert_allclose(g, [[1, 0, 0], [0, 1, 0], [0, 0, 0]])


def test_block_table_reads_the_block_of_the_requested_count(blocks):
    b, g = read_diffusion_table(blocks, b_value=1000.0, directions=2)
    np.testing.assert_allclose(b, [1000.0, 1000.0])
    np.testing.assert_allclose(g, [[0, 0, 1], [0.6, 0.8, 0]])


def test_block_table_refuses_a_count_it_does_not_hold(blocks):
    with pytest.raises(ValueError, match=r"\[2, 3\]"):
        read_diffusion_table(blocks, b_value=1000.0, directions=5)


def test_block_table_needs_the_largest_b_value(blocks):
    with pytest.raises(ValueError, match="b_value"):
        read_diffusion_table(blocks, directions=3)


def test_four_column_table_keeps_its_b_values_and_normalises_directions(tmp_path):
    path = tmp_path / "scheme.b"
    path.write_text("0 0 0 0\n2 0 0 1000\n0 0.5 0.5 3000\n")
    b, g = read_diffusion_table(path)
    np.testing.assert_allclose(b, [0.0, 1000.0, 3000.0])
    np.testing.assert_allclose(g, [[0, 0, 0], [1, 0, 0], [0, 2**-0.5, 2**-0.5]])


def test_fsl_pair_is_read_from_the_bvec_with_the_bval_beside_it(tmp_path):
    (tmp_path / "dwi.bval").write_text("0 1000 1000\n")
    (tmp_path / "dwi.bvec").write_text("1 1 0\n0 0 1\n0 0 0\n")
    b, g = read_diffusion_table(tmp_path / "dwi.bvec")
    np.testing.assert_allclose(b, [0.0, 1000.0, 1000.0])
    np.testing.assert_allclose(g, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_sequence_records_the_path_not_the_table(blocks):
    seq = pp.Sequence()
    set_diffusion_table(seq, blocks, b_value=2000.0, directions=3)
    assert seq.get_definition("DiffusionTable") == str(blocks.resolve())
    assert "1.0 0.0 0.0" not in str(seq.definitions)


def test_table_read_back_from_a_written_sequence_matches_the_one_played(
    blocks, tmp_path
):
    seq = pp.Sequence()
    seq.add_block(pp.make_delay(1e-3))
    played = set_diffusion_table(seq, blocks, b_value=2000.0, directions=3)
    seq.write(tmp_path / "dwi.seq")
    read_back = get_diffusion_table(pp.io.read(tmp_path / "dwi.seq"))
    for expected, actual in zip(played, read_back, strict=True):
        np.testing.assert_allclose(actual, expected)


def test_sequence_without_a_table_has_none():
    assert get_diffusion_table(pp.Sequence()) is None
