"""The text reader's block table, whatever order the file lists it in."""

import numpy as np

import pypulseqpp as pp


def test_text_blocks_listed_out_of_order_are_played_in_number_order(tmp_path):
    seq = pp.Sequence(pp.Opts())
    for area in (100, 200, 300):
        seq.add_block(pp.make_trapezoid("x", area=area, duration=2e-3))
    path = tmp_path / "ordered.seq"
    seq.write(str(path))
    lines = path.read_text().splitlines()
    start = lines.index("[BLOCKS]") + 1
    while lines[start].startswith("#") or not lines[start].strip():
        start += 1
    lines[start], lines[start + 2] = lines[start + 2], lines[start]
    shuffled = tmp_path / "shuffled.seq"
    shuffled.write_text("\n".join(lines) + "\n")

    read = pp.io.read(str(shuffled))

    expected = pp.io.read(str(path)).block_events
    assert list(read.block_events) == list(expected)
    for block, events in expected.items():
        np.testing.assert_array_equal(read.block_events[block], events)
