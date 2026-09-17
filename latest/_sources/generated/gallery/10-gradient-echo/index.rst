

.. _sphx_glr_generated_gallery_10-gradient-echo:

===============
Gradient echoes
===============

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One excitation and one phase-encode line per repetition, with the transverse magnetisation spoiled by a gradient and by a quadratic RF phase increment before the next excitation. The workhorse of the family, and the sequence the other Cartesian variants are read against.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D Cartesian gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One excitation and one (line, partition) view per repetition over a slab. The second phase-encode axis replaces slice selection, so the slab is resolved by encoding rather than by the pulse.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D Cartesian gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One excitation per repetition, with the line read again at several echo times. The signal decays between echoes at a rate the tissue&#x27;s apparent transverse relaxation sets, so one repetition measures the decay rather than one point on it.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_multiecho2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_multiecho2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D Cartesian multi-echo gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The multi-echo readout over a slab: one excitation per (line, partition) view, with that view read at several echo times.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_multiecho3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_multiecho3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D Cartesian multi-echo gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One line of one rotating blade per repetition. A blade is a narrow band of parallel lines through the centre of k-space, and the blades are turned so that between them they cover the disc; each blade samples the centre, so a blade corrupted by motion can be detected and rejected.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_propeller2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_propeller2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D PROPELLER gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One full spoke through the centre of k-space per repetition. Every readout crosses the centre, so the acquisition is insensitive to motion between repetitions in a way a Cartesian one is not, and undersampling shows as streaks rather than as aliasing.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_radial2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_radial2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D radial gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One spiral interleaf per repetition, solved against the gradient amplitude and slew limits. An interleaf covers a disc rather than a line, so a plane is acquired in a few tens of repetitions.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_spiral2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_spiral2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D spiral gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="PROPELLER blades in the plane and Cartesian encoding along the slab axis.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_stack_of_blades3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_stack_of_blades3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-blades gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Spiral interleaves in the plane and Cartesian encoding along the slab axis, which is the most efficient of the stacks: a partition is covered by a few interleaves rather than by a few hundred lines.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_stack_of_spirals3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_stack_of_spirals3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-spirals gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Radial spokes in the plane and Cartesian encoding along the slab axis. The in-plane acquisition keeps the motion behaviour of a radial one; the partition axis keeps the efficiency of Cartesian encoding.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_stack_of_stars3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_stack_of_stars3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-stars gradient echo</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /generated/gallery/10-gradient-echo/gre2D_sequence
   /generated/gallery/10-gradient-echo/gre3D_sequence
   /generated/gallery/10-gradient-echo/gre_multiecho2D_sequence
   /generated/gallery/10-gradient-echo/gre_multiecho3D_sequence
   /generated/gallery/10-gradient-echo/gre_propeller2D_sequence
   /generated/gallery/10-gradient-echo/gre_radial2D_sequence
   /generated/gallery/10-gradient-echo/gre_spiral2D_sequence
   /generated/gallery/10-gradient-echo/gre_stack_of_blades3D_sequence
   /generated/gallery/10-gradient-echo/gre_stack_of_spirals3D_sequence
   /generated/gallery/10-gradient-echo/gre_stack_of_stars3D_sequence

