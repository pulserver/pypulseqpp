:orphan:

=====================
Gallery source pages
=====================

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

=================
Cartesian imaging
=================

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An echo-planar train samples the whole phase-encode axis after one excitation, so off-resonance accumulates along that axis instead of across repetitions. A spin at offset \Delta f acquires phase 2\pi \Delta f\, m\, \mathrm{esp} on echo m, which is linear in k_y and therefore a displacement of">

.. only:: html

  .. image:: /generated/gallery/01-cartesian/images/thumb/sphx_glr_01-epi-segmentation-and-acceleration_thumb.png
    :alt:

  :doc:`/generated/gallery/01-cartesian/01-epi-segmentation-and-acceleration`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Echo train length and geometric distortion in EPI</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A three-dimensional fast spin echo reads one (line, partition) view per refocused echo, so the amplitude the train has left at echo m becomes the weight of whichever view that echo reads. The ordering is the map from echo index to k-space position, and the weighting it produces is a filter applied to the image: its inverse Fourier transform is the point-spread function of the acquisition.">

.. only:: html

  .. image:: /generated/gallery/01-cartesian/images/thumb/sphx_glr_02-fse-train-length-and-point-spread_thumb.png
    :alt:

  :doc:`/generated/gallery/01-cartesian/02-fse-train-length-and-point-spread`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Echo train length, signal envelope and point spread</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

==========================
Non-Cartesian trajectories
==========================

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three limits bound the traversal of a spiral arm. Two are properties of the gradient system: the maximum amplitude and the maximum slew rate. The third follows from the receiver: with a dwell time \Delta t the trajectory may not advance further than 1/\mathrm{FOV} between samples, which caps the gradient amplitude at">

.. only:: html

  .. image:: /generated/gallery/02-non-cartesian/images/thumb/sphx_glr_01-spiral-design-under-hardware-limits_thumb.png
    :alt:

  :doc:`/generated/gallery/02-non-cartesian/01-spiral-design-under-hardware-limits`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gradient, slew and receiver limits on a spiral readout</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

================
RF pulse design
================

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slice-selective SLR pulse is specified by three numbers: the flip angle, the slice thickness and the time-bandwidth product. The duration is a fourth, and it is not independent of the rest, because the selection gradient has to place the pulse&#x27;s bandwidth across the slice:">

.. only:: html

  .. image:: /generated/gallery/03-rf-pulses/images/thumb/sphx_glr_01-slice-profile-and-time-bandwidth_thumb.png
    :alt:

  :doc:`/generated/gallery/03-rf-pulses/01-slice-profile-and-time-bandwidth`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Time-bandwidth product, slice profile and pulse duration</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /generated/gallery/01-cartesian/index.rst
   /generated/gallery/02-non-cartesian/index.rst
   /generated/gallery/03-rf-pulses/index.rst


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: gallery_python.zip </generated/gallery/gallery_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: gallery_jupyter.zip </generated/gallery/gallery_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
