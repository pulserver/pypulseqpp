========================
3D stack-of-stars MPRAGE
========================

:Family: MPRAGE
:Dimensionality: 3D, slab-selective
:Sampling: Radial in-plane, Cartesian partitions
:Readout: One inversion per shot, followed by the radial spokes of one partition as a spoiled gradient-echo train

3D MPRAGE on a stack of stars: one inversion per partition, then its spokes.

Each shot applies one inversion and then acquires every sampled in- plane view of a single partition, so the partition encode is constant within a shot and the number of shots equals the number of sampled partitions. The inversion time is measured from the centre of the inversion pulse to the centre of the first excitation of the train, and the repetition time is the interval between successive inversions.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: mprage_stack_of_stars3D_sequence

.. minigallery:: pypulseqpp.sequences.mprage_stack_of_stars3D_sequence
   :add-heading: Designed and drawn

See also
--------

Other mprage sequences: :doc:`mprage3D_sequence <mprage3D_sequence>`, :doc:`mprage_stack_of_spirals3D_sequence <mprage_stack_of_spirals3D_sequence>`.

The same sampling in another family: :doc:`gre_stack_of_stars3D_sequence <gre_stack_of_stars3D_sequence>`, :doc:`se_stack_of_stars3D_sequence <se_stack_of_stars3D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
