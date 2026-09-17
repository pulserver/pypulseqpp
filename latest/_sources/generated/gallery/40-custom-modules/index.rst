

.. _sphx_glr_generated_gallery_40-custom-modules:

=======================
Custom sequence modules
=======================

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The shipped Cartesian readouts acquire on the flat top of the readout lobe, so the ramps carry area that is never sampled. Sampling through the ramps as well covers the same extent of k-space in a shorter lobe, at the price of samples that are no longer equally spaced in k and a reconstruction that has to regrid them.">

.. only:: html

  .. image:: /generated/gallery/40-custom-modules/images/thumb/sphx_glr_custom-cartesian-readout_thumb.png
    :alt:

  :doc:`/generated/gallery/40-custom-modules/custom-cartesian-readout`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A ramp-sampled readout module</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The shipped excitation modules design linear-phase SLR pulses, whose energy is symmetric about the middle of the pulse. A minimum-phase design concentrates the energy at the end instead, which shortens the interval between the pulse and the echo at the same duration and time-bandwidth product, at the cost of a higher peak B_1 and a slice-profile phase that is no longer linear.">

.. only:: html

  .. image:: /generated/gallery/40-custom-modules/images/thumb/sphx_glr_custom-excitation-module_thumb.png
    :alt:

  :doc:`/generated/gallery/40-custom-modules/custom-excitation-module`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A minimum-phase excitation module</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A radial spoke samples the centre of k-space far more densely than the periphery: at radius k, adjacent spokes of an N-interleaf set are 2\pi k / N apart, which exceeds the Nyquist spacing 1/\mathrm{FOV} beyond a transition radius">

.. only:: html

  .. image:: /generated/gallery/40-custom-modules/images/thumb/sphx_glr_custom-noncartesian-readout_thumb.png
    :alt:

  :doc:`/generated/gallery/40-custom-modules/custom-noncartesian-readout`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A twisting radial readout module</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /generated/gallery/40-custom-modules/custom-cartesian-readout
   /generated/gallery/40-custom-modules/custom-excitation-module
   /generated/gallery/40-custom-modules/custom-noncartesian-readout

