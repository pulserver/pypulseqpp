# Explanations

Conceptual background for the interfaces documented in {doc}`../api/index` and
applied in the {doc}`examples <../examples>`. These pages
establish the vocabulary, the models and the conventions that the rest of the
documentation assumes.

{doc}`pulseq/index`
: **Pulseq representation.** What a `.seq` file contains — blocks, events,
  libraries, shapes, extensions and definitions — how it is stored and
  deduplicated, and the rasters every event time is quantized to.

{doc}`design/index`
: **Sequence design in pypulseqpp.** The abstractions the package places above
  the file format: sequence modules, which solve one group of blocks at
  construction, and sequence applications, which add a prescription, a
  sampling order and a scan loop.

{doc}`safety/index`
: **Gradient, PNS and SAR constraints.** What each check of a finished sequence
  computes, the criterion it applies, the model or table it requires, and what
  it does not establish.

Each group opens with a landing page that states what its pages cover. A page
generally proceeds from the physical or computational concept to the model or
criterion that makes it precise, then to the consequences for the quantities a
user controls, and finally to the interface that represents it.
