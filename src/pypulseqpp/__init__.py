"""PyPulseq-compatible sequence design and analysis over a C++ core."""

from __future__ import annotations

import functools as _functools
import inspect as _inspect
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

import pypulseq as _pypulseq

from . import _events
from ._adiabatic import make_adiabatic_pulse as _make_adiabatic_pulse
from ._angles import (
    calc_golden_angles,
    calc_projection_shell,
    calc_raga_angles,
    calc_tiny_golden_angles,
    calc_uniform_angles,
)
from ._block_to_events import block_to_events as _block_to_events
from ._calc_rf_bandwidth import calc_rf_bandwidth as _calc_rf_bandwidth
from ._check_timing import check_timing as _check_timing
from ._check_timing import print_error_report as _print_error_report
from ._epi import calc_epi_order
from ._events import as_namespace, convert, interoperating
from ._gradients import concatenate_gradients as _concatenate_gradients
from ._gradients import make_crusher as _make_crusher
from ._gradients import make_phase_blip as _make_phase_blip
from ._gradients import make_phase_encoding as _make_phase_encoding
from ._gradients import make_wave_gradients as _make_wave_gradients
from ._make_hexagon_gradient_area import (
    make_hexagon_gradient_area as _make_hexagon_gradient_area,
)
from ._make_label import make_label as _make_label
from ._make_rf_shim import make_rf_shim as _make_rf_shim
from ._make_rotation import make_rotation as _make_rotation
from ._masks import (
    calc_calibration_lines,
    calc_sampled_lines,
    calc_sampled_pairs,
    make_caipirinha_mask,
    make_centric_order,
    make_linear_order,
    make_poisson_disc_mask,
    make_radial_adaptive_order,
    make_radial_order,
    make_random_mask,
    make_shuffling_order,
)
from ._opts import (
    MAX_GRAD_DERATE,
    MAX_SLEW_DERATE,
    apply_system_derates,
    cap_system,
    default_system,
)
from ._opts import Opts as _Opts
from ._ordering import calc_traversal_order
from ._ptx import make_ptx_pulse as _make_ptx_pulse
from ._ptx import split_ptx_pulse
from ._rf_pulses import make_2d_selective_pulse as _make_2d_selective_pulse
from ._rf_pulses import make_half_passages as _make_half_passages
from ._rf_pulses import make_slr_pulse as _make_slr_pulse
from ._rf_pulses import make_sms_pulse as _make_sms_pulse
from ._rf_pulses import make_spsp_pulse as _make_spsp_pulse
from ._rotate_3d import rotate_3d as _rotate_3d
from ._sampling import make_uniform_mask
from ._schedules import (
    make_phase_cycling_schedule,
    make_rf_spoiling_schedule,
    make_traps_schedule,
)
from ._sequence import Sequence as _Sequence
from ._sim_rf import bloch as _bloch
from ._sim_rf import sim_rf as _sim_rf
from ._timing import (
    calc_adc_timing,
    ceil_to_raster,
    quantize_readout_timing,
    round_to_raster,
)
from ._traj_to_grad import traj_to_grad as _traj_to_grad
from ._trajectories import (
    calc_radial_trajectory,
    calc_rosette_trajectory,
    calc_spiral_trajectory,
)
from ._transform_fov import TransformFOV as _TransformFOV

try:
    __version__ = _distribution_version(__name__)
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0.dev0"

#: Upstream names that are not re-exported: the shape codec and the unit
#: helper, which upstream exposes as modules rather than as vocabulary.
_EXCLUDED = {"compress_shape", "convert", "decompress_shape"}

#: Upstream callables that must not be wrapped. Wrapping a class would
#: replace its constructor with a plain function.
_UNWRAPPED = {"SigpyPulseOpts"}

#: Upstream names that resolve here but are not advertised. Reachable and
#: advertised are different promises: ``pp.x`` answers for any script written
#: against upstream, while ``__all__`` shows the vocabulary a sequence is
#: written in. ``eps`` is a raster comparison tolerance and ``round_half_up``
#: a rounding fix, neither of which describes a sequence;
#: ``make_sigpy_pulse`` is upstream's spelling of :func:`make_slr_pulse` and
#: ``SigpyPulseOpts`` its argument bundle, which has nothing here to
#: configure.
_UNADVERTISED = {"SigpyPulseOpts", "eps", "make_sigpy_pulse", "round_half_up"}

#: Upstream names that are modules rather than vocabulary: ``np``, ``math``,
#: ``importlib``, and the submodules upstream's own ``__init__`` happens to
#: touch. Filled in by the loop below, and kept out of ``__all__``.
_UPSTREAM_MODULES: set[str] = set()

for _name in dir(_pypulseq):
    if _name.startswith("_") or _name in _EXCLUDED:
        continue
    _value = getattr(_pypulseq, _name)
    if _inspect.ismodule(_value):
        _UPSTREAM_MODULES.add(_name)
        globals()[_name] = _value
        continue
    if callable(_value) and not isinstance(_value, type) and _name not in _UNWRAPPED:
        _value = interoperating(_value)
    globals()[_name] = _value
del _name, _value


def _fast_scale_grad(upstream):
    """`scale_grad` that stays in the slotted form for the common case.

    The hot one: a phase-encode loop scales the same prewinder once per
    shot, and the generic decorator would convert the event to a namespace
    and back on every call. A slotted gradient with no system to re-check
    against is scaled in place of that; anything else is upstream's.
    """
    scaled = _events.scaled_gradient

    @_functools.wraps(upstream)
    def scale_grad(grad, scale, system=None):
        if system is None:
            try:
                return scaled(grad, scale)
            except TypeError:
                pass
        return upstream(grad, scale, system=system)

    return scale_grad


scale_grad = _fast_scale_grad(globals()["scale_grad"])

# The factories, which hand back events with their fields in slots.
for _factory in _events.__all__:
    globals()[_factory] = getattr(_events, _factory)
del _factory

# Ours, until upstream has them.
make_label = _make_label
make_rf_shim = _make_rf_shim
TransformFOV = _TransformFOV
make_rotation = _make_rotation

# Upstream leaves this one in a module of the same name, and hands a list
# of events back inside a tuple rather than as the tuple its own docstring
# promises.
block_to_events = interoperating(_block_to_events)

# Ours, over upstream's: a hard pulse answers its own width rather than zero.
calc_rf_bandwidth = interoperating(_calc_rf_bandwidth)

# Ours, over upstream's: its two sweeps, with BIR-4 and GOIA-WURST beside them.
make_adiabatic_pulse = _make_adiabatic_pulse

# Ours, over upstream's: the rasters both vendors can play.
Opts = _Opts

# Pulseq's, where upstream has not ported them. Each goes through the
# interoperation decorator, because each is written against the namespaces
# upstream's own helpers build.
make_hexagon_gradient_area = interoperating(_make_hexagon_gradient_area)
rotate_3d = interoperating(_rotate_3d)
sim_rf = interoperating(_sim_rf)
bloch = _bloch

# Waveform and pulse design. Each goes through the interoperation decorator
# for the same reason: the bodies build with PyPulseq's own factories and
# helpers, which want namespaces.
concatenate_gradients = interoperating(_concatenate_gradients)
make_2d_selective_pulse = interoperating(_make_2d_selective_pulse)
make_crusher = interoperating(_make_crusher)
make_half_passages = interoperating(_make_half_passages)
make_phase_blip = interoperating(_make_phase_blip)
make_phase_encoding = interoperating(_make_phase_encoding)
make_slr_pulse = interoperating(_make_slr_pulse)
#: Upstream's spelling of the same design, which pulls in no sigpy: its
#: SigpyPulseOpts argument bundle has nothing here to configure.
make_sigpy_pulse = make_slr_pulse
make_sms_pulse = interoperating(_make_sms_pulse)
make_ptx_pulse = interoperating(_make_ptx_pulse)
make_spsp_pulse = interoperating(_make_spsp_pulse)
make_wave_gradients = interoperating(_make_wave_gradients)

# Ours, over upstream's: a path is re-parameterised within the limits rather
# than differentiated at whatever spacing it arrived with.
traj_to_grad = _traj_to_grad

# Ours outright: upstream's Sequence is a different implementation, and
# this is the one with the compiled core under it.
Sequence = _Sequence

# The timing check, which reads the compiled core's own tables. The name
# upstream carries here is its module rather than a callable.
check_timing = _check_timing
print_error_report = _print_error_report

#: The events every one of these returns are compiled, not namespaces.
SLOTTED = frozenset(_events.__all__) | {
    "make_extended_trapezoid_area",
    "make_label",
    "make_rf_shim",
}

#: The vocabulary a sequence is written in. Every upstream name still
#: resolves -- ``import pypulseqpp as pp`` answers ``pp.x`` for any script
#: written against ``pypulseq`` -- but the imports upstream's ``__init__``
#: happens to make, and the handful of helpers that describe arithmetic
#: rather than a sequence, are left out of what the namespace advertises.
__all__ = sorted(
    {
        name
        for name in globals()
        if not name.startswith("_")
        and name != "annotations"
        and name not in _UPSTREAM_MODULES
        and name not in _UNADVERTISED
    }
    | {"__version__"}
)


#: Why a name that looks as though it should be here is not. Reaching for one
#: gets this rather than a bare AttributeError, which would read as an
#: oversight.
_WITHHELD = {
    "add_ramps": (
        "not exposed: it joins a trajectory to zero with calc_ramp, and "
        "calc_ramp raises on any join needing more than zero intermediate "
        "points. traj_to_grad solves the same problem."
    ),
    "add_custom_label": (
        "there is nothing to register: make_label takes any name, writing "
        "and reading round-trip it, and a label an interpreter does not know "
        "it ignores."
    ),
    "compress_shape": "a shape codec rather than authoring vocabulary; import it from pypulseq.",
    "decompress_shape": "a shape codec rather than authoring vocabulary; import it from pypulseq.",
}


def __getattr__(name: str):
    reason = _WITHHELD.get(name)
    if reason is not None:
        raise AttributeError(f"pypulseqpp does not export {name!r}: {reason}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
