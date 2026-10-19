# C++ core internals

The rules the native sequence store keeps and the binding conventions that
preserve its cost. The concepts are in {doc}`../../explanations/cpp-core`.

## Language, layout and performance

- Native code is C++17 and Python is 3.10 or later. The Python-independent
  core is in `src/cpp/pulseq/`; the CPython and pybind11 bindings for
  `pypulseqpp._ext` are in `src/cpp/bindings/`.
- Per-block loops stay in C++. Measure before adding an allocation on the
  million-block path.
- Per-block binding calls use `METH_FASTCALL`, not pybind11 argument
  conversion. `add_block` registers all events in one native call and takes
  exactly seven arguments: RF, three gradients, ADC, extension and duration.
  A pybind11 binding would be a `builtin_function_or_method`; the fast one is a
  `method_descriptor`.
- Scalar event fields are `PyMemberDef` offsets. Waveform accessors expose
  samples at their stored amplitude.
- Substantial native work, including deduplication, compression and writing,
  releases the GIL.
- Run `benchmarks/throughput.py` before and after changing bindings and report
  the measurements, not an assertion of improvement.

## Block-table snapshots

Block-table arrays are copy-on-write snapshots. A view owns a share of the
buffer it points into; the sequence copies the buffer before writing to it
while a view is out. A view therefore never observes a later write, stays valid
when the table grows and survives destruction of the sequence. The
capacity-based fast-path ownership check is preserved. The `block_events` view
exposes the six columns the file carries as a strided view, leaving the
rotation column out.

## Derived analysis and the revision counter

Every native mutation increments a core edit counter (`Sequence` revision) that
never decreases and is unrelated to the file-format revision. Derived analysis
records the counter it was computed at; the Python cache guard is one
comparison of counters, not a per-block Python attribute update. Compatibility
flags such as `use_block_cache` do not control these caches or the native shape
registrations, and decoded blocks are not cached.

## Facade and event contracts

`pypulseqpp` re-exports upstream authoring vocabulary alongside its own
implementations. Upstream imports and arithmetic helpers need not appear in
`__all__`; they stay reachable. `block_to_events` is
a function here where upstream names a module.

The interoperability decorator `interoperating` converts compiled events to
namespaces for upstream functions and converts returned events back, recursively
through arguments and results. This applies to event consumers such as
`calc_duration`, `align`, `split_gradient` and `rotate`, not only to factories.
A wrapped PyPulseq function's docstring gets a note that its events go in as
namespaces and come back with their fields in slots. Our own functions that
build with upstream helpers pass through it too and keep their own docstrings.

Withheld names resolve to an `AttributeError` that states the reason:

| Name | Reason |
| --- | --- |
| `add_ramps` | Joins a trajectory to zero with `calc_ramp`, which raises on any join needing intermediate points; `traj_to_grad` solves the same problem. |
| `add_custom_label` | Nothing to register: `make_label` takes any name, which writing and reading round-trip. |
| `compress_shape`, `decompress_shape` | A shape codec, not authoring vocabulary; import from `pypulseq`. |

Not implemented: automatic labelling and sequence tiling. Scanner-host
integration, including an execution stream, belongs to Pulserver.

A dynamic pTx pulse is one arbitrary RF event holding every transmit channel's
waveform one after another over a shared time base, so its time shape restarts
once per channel (Roos et al., Magn Reson Med 2025,
doi:10.1002/mrm.30601). The channel count is the number of samples at the first
sample time, as the reference interpreter reads it. `make_ptx_pulse` writes the
layout and `split_ptx_pulse` reads it. The core's timing check judges one
channel's time base; flip-angle integrals take each channel on its own time
base and sum them, which is the flip where every channel has unit, in-phase
sensitivity. The rule is in `src/cpp/pulseq/channels.hpp`.

Waveforms are normalised beside scalar amplitudes. Amplitude changes preserve
shape registrations; waveform replacement invalidates them. Registration IDs
are sequence-local. Decoded blocks are independent event snapshots.

## Storage invariants

- Libraries and block indices are 1-based; a zero event ID means absence.
- Trapezoids and arbitrary gradients share file IDs but occupy separate tables.
- Shape roles are a bit mask recorded when events reference a shape;
  deduplication ORs the roles of merged shapes.
- Event definitions separate fixed timing and shape data from playout
  parameters. RF magnitude, phase and time shapes belong to definitions;
  arbitrary gradient waveforms belong to instances. ADC and extension choices do
  not distinguish block definitions. Pure delays share one definition
  independent of duration; triggers and digital outputs are not pure delays.
- Registration creates definitions. Deduplication rebuilds them after
  renumbering shapes and events, so definition IDs are not stable across it.
  Before deduplication, equal separately registered shapes can yield distinct
  definitions, so the repetition found before and after it can differ.
- A block carries at most one rotation and one RF shim; `add_block` refuses a
  second. `add_block(None)` raises, as upstream does.
