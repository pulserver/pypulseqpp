

.. _sphx_glr_generated_gallery_03-gre-to-epi:

=================================
From gradient echo to echo planar
=================================

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to acquire more than one echo per excitation, by following the readout gradient with further readouts of alternating polarity. Nothing else about the repetition changes, and the echoes land on the same k-space line at increasing echo times, which is what a T_2^* estimate is made from.">

.. only:: html

  .. image:: /generated/gallery/03-gre-to-epi/images/thumb/sphx_glr_01_multi_echo_thumb.png
    :alt:

  :doc:`/generated/gallery/03-gre-to-epi/01_multi_echo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multi-echo readouts</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to put a phase-encode blip between the echoes of the train of the previous page, so that one excitation acquires several k-space lines instead of the same line several times. The number of excitations the matrix is divided over is then a free parameter, and it decides both the scan time and how far off-resonance displaces the image.">

.. only:: html

  .. image:: /generated/gallery/03-gre-to-epi/images/thumb/sphx_glr_02_segmented_thumb.png
    :alt:

  :doc:`/generated/gallery/03-gre-to-epi/02_segmented`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Segmented echo planar</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to take the segmentation of the previous page to one shot, so that the whole matrix is acquired after a single excitation, and to measure the two things that limit such an acquisition: the decay of the signal over an echo train tens of milliseconds long, and the sensitivity of a train of alternating readouts to a delay between the gradient and the acquisition.">

.. only:: html

  .. image:: /generated/gallery/03-gre-to-epi/images/thumb/sphx_glr_03_epi_thumb.png
    :alt:

  :doc:`/generated/gallery/03-gre-to-epi/03_epi`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Single-shot echo planar</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /generated/gallery/03-gre-to-epi/01_multi_echo
   /generated/gallery/03-gre-to-epi/02_segmented
   /generated/gallery/03-gre-to-epi/03_epi

