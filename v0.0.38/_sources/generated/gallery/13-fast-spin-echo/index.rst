

.. _sphx_glr_generated_gallery_13-fast-spin-echo:

==============
Fast spin echo
==============

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Individually parameterized 3D FSE assigns different echo-train lengths and repetition times to the shots that acquire central and peripheral k-space [BUO25]_. From the central to the peripheral shots, TR and the minimum and maximum angles of the refocusing schedule [BUS08b]_ follow a cubic smooth-step transition between their two limits; the echo-train length follows the same transition, rounded to integer values. The views are assigned by an adaptive radial order. Longer trains and a different TR at the periphery can reduce scan time, while the contrast at the centre of k-space is set by the parameters of the central shots.">

.. only:: html

  .. image:: /generated/gallery/13-fast-spin-echo/images/thumb/sphx_glr_fse3D_adaptive_thumb.png
    :alt:

  :doc:`/generated/gallery/13-fast-spin-echo/fse3D_adaptive`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Individually optimized 3D fast spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slab-selective excitation is followed by a CPMG fast-spin-echo refocusing train, with one Cartesian (line, partition) view acquired at each echo. Refocusing angles below 180 degrees add stimulated-echo pathways to the echo signal [HEN88]_, and variable refocusing angles modulate the T2-dependent signal evolution along the train [BUS08a]_. Radial view ordering maps this evolution onto the (k_y, k_z) plane and therefore determines the modulation transfer function and image blurring [BUS08a]_. 3D FSE is used for T2- and proton-density-weighted structural imaging.">

.. only:: html

  .. image:: /generated/gallery/13-fast-spin-echo/images/thumb/sphx_glr_fse3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/13-fast-spin-echo/fse3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Conventional 3D fast spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Shuffled 3D FSE uses the same refocusing-train design as conventional FSE and differs in two separate choices. The support is a variable-density Poisson-disc draw, which determines which views are acquired. The ordering is that of T2 Shuffling [TAM17]_: each train is a group of contiguous views in raster order, and the echo position of each view within its train is random, which determines when each view is acquired. Together, the variable-density support and the random echo positions give each echo time a subset of views spread over the sampled extent without a regular pattern, the sampling condition of echo-resolved subspace reconstruction [TAM17]_; no reconstruction is performed here.">

.. only:: html

  .. image:: /generated/gallery/13-fast-spin-echo/images/thumb/sphx_glr_fse3D_shuffling_thumb.png
    :alt:

  :doc:`/generated/gallery/13-fast-spin-echo/fse3D_shuffling`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Shuffled echo-resolved 3D FSE</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /generated/gallery/13-fast-spin-echo/fse3D_adaptive
   /generated/gallery/13-fast-spin-echo/fse3D_sequence
   /generated/gallery/13-fast-spin-echo/fse3D_shuffling

