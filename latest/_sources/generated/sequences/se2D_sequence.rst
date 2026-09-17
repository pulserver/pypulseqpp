======================
2D Cartesian spin echo
======================

:Family: Spin echo
:Dimensionality: 2D, multi-slice
:Sampling: Cartesian
:Readout: One phase-encode line per excitation, read at a refocused echo

Multi-slice 2D Cartesian spin echo: one line per excitation.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: se2D_sequence

Representative configuration for documentation
----------------------------------------------

The configuration below is chosen to make the structure of the sequence
legible at the size of this page: the matrix is small, delays that would
otherwise dominate the diagram are short, and the figures are drawn from
the sequence this call designs. It is not a protocol recommendation.

.. code-block:: python

    from pypulseqpp import sequences

    seq = sequences.se2D_sequence(
        n_x=64,
        n_y=16,
        n_slices=1,
        te=0.014,
        tr=0.022,
        n_dummy=0,
        readout_oversampling=1.0,
    )

.. figure:: /generated/sequences/se2D_sequence-repetition.png
   :width: 100%

   One repetition. The remaining repetitions are drawn underneath in grey, so an event that changes between them appears as a band.

.. figure:: /generated/sequences/se2D_sequence-train.png
   :width: 100%

   Consecutive repetitions, showing how the encoding steps.

Related sequences
-----------------

Other spin echo sequences: :doc:`se3D_sequence <se3D_sequence>`, :doc:`se_radial2D_sequence <se_radial2D_sequence>`, :doc:`se_stack_of_stars3D_sequence <se_stack_of_stars3D_sequence>`, :doc:`se_spiral2D_sequence <se_spiral2D_sequence>`, :doc:`se_stack_of_spirals3D_sequence <se_stack_of_spirals3D_sequence>`, :doc:`se_propeller2D_sequence <se_propeller2D_sequence>`, :doc:`se_epi_propeller2D_sequence <se_epi_propeller2D_sequence>`, :doc:`se_stack_of_blades3D_sequence <se_stack_of_blades3D_sequence>`.

The same sampling in another family: :doc:`gre2D_sequence <gre2D_sequence>`, :doc:`gre3D_sequence <gre3D_sequence>`, :doc:`gre_multiecho2D_sequence <gre_multiecho2D_sequence>`, :doc:`gre_multiecho3D_sequence <gre_multiecho3D_sequence>`, :doc:`fse3D_sequence <fse3D_sequence>`, :doc:`mprage3D_sequence <mprage3D_sequence>`, :doc:`bssfp2D_sequence <bssfp2D_sequence>`, :doc:`bssfp3D_sequence <bssfp3D_sequence>`, :doc:`epi2D_sequence <epi2D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
