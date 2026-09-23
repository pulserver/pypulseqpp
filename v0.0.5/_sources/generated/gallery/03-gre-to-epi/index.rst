

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

    <div class="sphx-glr-thumbcontainer" tooltip="The gradient echo of the first section acquires one echo per excitation. This lesson acquires several, by following the readout gradient with further readout gradients of alternating polarity. The rest of the repetition is unchanged, and the echoes sample the same k-space line at increasing echo times, from which a T_2^* estimate is computed.">

.. only:: html

  .. image:: /generated/gallery/03-gre-to-epi/images/thumb/sphx_glr_01_multi_echo_thumb.png
    :alt:

  :doc:`/generated/gallery/03-gre-to-epi/01_multi_echo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multi-echo readouts</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The multi-echo train of the previous lesson samples the same k-space line several times. This lesson adds a phase-encode blip between the echoes, so that one excitation acquires several k-space lines. The number of excitations (shots) over which the matrix is divided is then a free parameter, and it determines both the scan time and the off-resonance displacement in the image.">

.. only:: html

  .. image:: /generated/gallery/03-gre-to-epi/images/thumb/sphx_glr_02_segmented_thumb.png
    :alt:

  :doc:`/generated/gallery/03-gre-to-epi/02_segmented`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Segmented echo planar</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The previous lesson divided the matrix over several shots. This lesson takes the segmentation to one shot, so that the whole matrix is acquired after a single excitation, and measures the two effects that limit such an acquisition: the signal decay over an echo train tens of milliseconds long, and the sensitivity of a train of alternating readouts to a delay between the gradient waveform and the acquisition. The relationship between shot count, distortion and scan time is measured in /generated/gallery/03-gre-to-epi/02_segmented.">

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

