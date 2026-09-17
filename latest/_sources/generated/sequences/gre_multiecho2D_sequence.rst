=====================================
2D Cartesian multi-echo gradient echo
=====================================

:Family: Gradient echo
:Dimensionality: 2D, multi-slice
:Sampling: Cartesian
:Readout: One phase-encode line read at several echo times per repetition

RF-spoiled, multi-slice multi-echo 2D Cartesian gradient echo.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: gre_multiecho2D_sequence

Representative configuration for documentation
----------------------------------------------

The configuration below is chosen to make the structure of the sequence
legible at the size of this page: the matrix is small, delays that would
otherwise dominate the diagram are short, and the figures are drawn from
the sequence this call designs. It is not a protocol recommendation.

.. code-block:: python

    from pypulseqpp import sequences

    seq = sequences.gre_multiecho2D_sequence(
        n_x=64,
        n_y=16,
        n_slices=1,
        n_echoes=4,
        te=None,
        tr=0.012,
        n_dummy=0,
        readout_oversampling=1.0,
        readout_bandwidth_hz=200000.0,
    )

.. figure:: /generated/sequences/gre_multiecho2D_sequence-repetition.png
   :width: 100%

   One repetition. The remaining repetitions are drawn underneath in grey, so an event that changes between them appears as a band.

.. figure:: /generated/sequences/gre_multiecho2D_sequence-echoes.png
   :width: 100%

   The opening of one echo train, at the raster the events are played on.

Related sequences
-----------------

Other gradient echo sequences: :doc:`gre2D_sequence <gre2D_sequence>`, :doc:`gre3D_sequence <gre3D_sequence>`, :doc:`gre_multiecho3D_sequence <gre_multiecho3D_sequence>`, :doc:`gre_radial2D_sequence <gre_radial2D_sequence>`, :doc:`gre_stack_of_stars3D_sequence <gre_stack_of_stars3D_sequence>`, :doc:`gre_spiral2D_sequence <gre_spiral2D_sequence>`, :doc:`gre_stack_of_spirals3D_sequence <gre_stack_of_spirals3D_sequence>`, :doc:`gre_propeller2D_sequence <gre_propeller2D_sequence>`, :doc:`gre_stack_of_blades3D_sequence <gre_stack_of_blades3D_sequence>`.

The same sampling in another family: :doc:`se2D_sequence <se2D_sequence>`, :doc:`se3D_sequence <se3D_sequence>`, :doc:`fse3D_sequence <fse3D_sequence>`, :doc:`mprage3D_sequence <mprage3D_sequence>`, :doc:`bssfp2D_sequence <bssfp2D_sequence>`, :doc:`bssfp3D_sequence <bssfp3D_sequence>`, :doc:`epi2D_sequence <epi2D_sequence>`.

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
