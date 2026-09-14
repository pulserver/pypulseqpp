"""The API reference lists every public Sequence member or deliberately leaves it out."""

import re
from pathlib import Path

import pypulseqpp as pp

TEMPLATE = Path(__file__).parents[1] / "docs/_templates/autosummary/sequence.rst"

#: Members kept for scripts written against upstream PyPulseq that the reference
#: does not present: compatibility no-ops and flags, aliases, and the
#: registration and extension-numbering internals `add_block` and the codecs use.
HIDDEN = {
    "block_cache_size",
    "calculate_kspacePP",
    "clear_block_cache",
    "clear_caches",
    "clear_event_cache",
    "event_cache_size",
    "get_extension_type_ID",
    "get_extension_type_string",
    "get_raw_block_content_IDs",
    "install",
    "read_binary",
    "register_adc_event",
    "register_control_event",
    "register_grad_event",
    "register_label_event",
    "register_rf_event",
    "register_rf_shim_event",
    "register_rotation_event",
    "register_soft_delay_event",
    "set_extension_string_ID",
    "use_block_cache",
    "use_event_cache",
    "write_file",
}


def test_every_public_sequence_member_is_categorised_or_deliberately_hidden():
    listed = re.findall(r"~Sequence\.(\w+)", TEMPLATE.read_text())
    public = {name for name in dir(pp.Sequence) if not name.startswith("_")}

    assert len(listed) == len(set(listed)), "a member is listed twice"
    assert not set(listed) & HIDDEN
    assert set(listed) | HIDDEN == public
