"""The API reference lists every public Sequence member or deliberately leaves it out."""

import inspect
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


#: The `Attributes` section of the class docstring, up to the next section
#: heading. Matched against the cleaned docstring: Python 3.13 strips the common
#: indentation when it compiles one, and earlier versions keep it.
_ATTRIBUTES = re.compile(r"^Attributes\n-+\n(.*?)^\w+\n-+\n", re.M | re.S)


def _documented_attributes() -> list[str]:
    section = _ATTRIBUTES.search(inspect.cleandoc(pp.Sequence.__doc__))
    assert section, "Sequence's docstring has no Attributes section"
    return re.findall(r"^(\w+) : ", section.group(1), re.M)


def test_every_public_sequence_member_is_categorised_or_deliberately_hidden():
    """Methods are pages of their own; properties and attributes are the class's."""
    listed = re.findall(r"~Sequence\.(\w+)", TEMPLATE.read_text())
    attributes = _documented_attributes()
    public = {
        name
        for name in dir(pp.Sequence) + list(vars(pp.Sequence()))
        if not name.startswith("_")
    }

    assert len(listed) == len(set(listed)), "a member is listed twice"
    assert len(attributes) == len(set(attributes)), "an attribute is listed twice"
    assert not set(listed) & set(attributes)
    assert not (set(listed) | set(attributes)) & HIDDEN
    assert set(listed) | set(attributes) | HIDDEN == public


def test_no_property_has_a_page_of_its_own():
    listed = re.findall(r"~Sequence\.(\w+)", TEMPLATE.read_text())
    properties = {
        name for name, value in vars(pp.Sequence).items() if isinstance(value, property)
    }

    assert not set(listed) & properties
