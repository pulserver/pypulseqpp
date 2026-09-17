=================
3D zero echo time
=================

:Family: Zero echo time
:Dimensionality: 3D, non-selective
:Sampling: Radial half-spokes on a sphere
:Readout: A hard pulse transmitted with the readout gradient already at amplitude, one half-spoke per view

3D zero echo time: hard pulses on a readout gradient held on across each shell.

Prescription
------------

.. currentmodule:: pypulseqpp.sequences

.. autofunction:: zte3D_sequence

Representative configuration for documentation
----------------------------------------------

The configuration below is chosen to make the structure of the sequence
legible at the size of this page: the matrix is small, delays that would
otherwise dominate the diagram are short, and the figures are drawn from
the sequence this call designs. It is not a protocol recommendation.

.. code-block:: python

    from pypulseqpp import sequences

    seq = sequences.zte3D_sequence(
        n_x=32,
        n_dummy=0,
    )

.. figure:: /generated/sequences/zte3D_sequence-views.png
   :width: 100%

   Several consecutive views. The readout gradient stays on across the hard pulses, and its amplitude steps from one view to the next.

.. figure:: /generated/sequences/zte3D_sequence-kspace.png
   :width: 100%

   Sampling locations in k-space, coloured by shot.

Related sequences
-----------------

Every shipped sequence is listed in the :doc:`catalogue </sequences>`.
