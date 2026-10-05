

.. _sphx_glr_generated_gallery_08-tours:

=====
Tours
=====

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A multi-echo gradient echo follows the readout gradient with further readout gradients of alternating polarity, each with its own ADC event. The rest of the repetition is unchanged, and the echoes sample the same k-space line at increasing echo times. The measured quantities are the echo spacing, which depends on the receiver bandwidth, and the number of echoes that fit in the repetition time at each bandwidth. Where the echoes of such a train land, and why the even echoes are acquired in reverse order, is measured in Course lesson 5.">

.. only:: html

  .. image:: /generated/gallery/08-tours/images/thumb/sphx_glr_01_multi_echo_thumb.png
    :alt:

  :doc:`/generated/gallery/08-tours/01_multi_echo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multi-echo readouts</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A segmented echo planar readout divides the phase-encode lines of the matrix over several excitations (shots), each acquiring an interleaved subset with a phase-encode blip between its echoes. The shot count determines both the scan time and the off-resonance displacement in the image. The measured relationship is the bandwidth per pixel along the phase-encode direction, which increases in proportion to the number of shots, against the number of excitations and hence the scan time.">

.. only:: html

  .. image:: /generated/gallery/08-tours/images/thumb/sphx_glr_02_segmented_epi_thumb.png
    :alt:

  :doc:`/generated/gallery/08-tours/02_segmented_epi`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Segmented echo planar</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A spiral readout acquires k-space along a spiral arm rather than along straight spokes. Its waveform cannot be written as a trapezoid and is solved numerically against the limits by SpiralReadout2D; here the module is used only for the arms it designs. This example establishes which of the system limits determines the duration of an arm.">

.. only:: html

  .. image:: /generated/gallery/08-tours/images/thumb/sphx_glr_03_spiral_thumb.png
    :alt:

  :doc:`/generated/gallery/08-tours/03_spiral`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Spiral readout</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example writes a Cartesian readout module that follows the SequenceModule contract and samples through the ramps of the readout lobe, and compares it with the shipped readout.">

.. only:: html

  .. image:: /generated/gallery/08-tours/images/thumb/sphx_glr_04_ramp_sampled_readout_thumb.png
    :alt:

  :doc:`/generated/gallery/08-tours/04_ramp_sampled_readout`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A ramp-sampled readout module</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example writes a non-Cartesian readout module: the trajectory is stated as a k-space path, solved into a gradient waveform under the gradient limits, and published as a module.">

.. only:: html

  .. image:: /generated/gallery/08-tours/images/thumb/sphx_glr_05_twisting_radial_readout_thumb.png
    :alt:

  :doc:`/generated/gallery/08-tours/05_twisting_radial_readout`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A twisting radial readout module</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /generated/gallery/08-tours/01_multi_echo
   /generated/gallery/08-tours/02_segmented_epi
   /generated/gallery/08-tours/03_spiral
   /generated/gallery/08-tours/04_ramp_sampled_readout
   /generated/gallery/08-tours/05_twisting_radial_readout

