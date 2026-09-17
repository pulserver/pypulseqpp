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

===============
Getting started
===============

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The shortest complete workflow: system limits, events, blocks, a sequence, and the file it is written to. Two repetitions of a slice-selective gradient echo are enough to show every step.">

.. only:: html

  .. image:: /generated/gallery/01-getting-started/images/thumb/sphx_glr_basic-pulseq-sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/01-getting-started/basic-pulseq-sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A basic Pulseq sequence</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

===========================
Sequence module composition
===========================

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A sequence module solves the layout of one group of blocks at construction and exposes the resulting events for a scan loop to place. The architecture is described in /explanations/design/sequence-module; this page shows the interface running.">

.. only:: html

  .. image:: /generated/gallery/20-modules-overview/images/thumb/sphx_glr_sequence-module-composition_thumb.png
    :alt:

  :doc:`/generated/gallery/20-modules-overview/sequence-module-composition`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">What a sequence module publishes</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

=========================
RF and excitation modules
=========================

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The excitation family produces one RF pulse and the gradients that go with it. Every member reports the events it built and, through sim_rf, the Bloch response of the pulse it holds.">

.. only:: html

  .. image:: /generated/gallery/21-modules-rf/images/thumb/sphx_glr_excitation-modules_thumb.png
    :alt:

  :doc:`/generated/gallery/21-modules-rf/excitation-modules`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Excitation modules</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slice-selective SLR pulse is specified by three numbers: the flip angle, the slice thickness and the time-bandwidth product. The duration is a fourth, and it is not independent of the rest, because the selection gradient has to place the pulse&#x27;s bandwidth across the slice:">

.. only:: html

  .. image:: /generated/gallery/21-modules-rf/images/thumb/sphx_glr_slice-profile-and-time-bandwidth_thumb.png
    :alt:

  :doc:`/generated/gallery/21-modules-rf/slice-profile-and-time-bandwidth`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Time-bandwidth product, slice profile and pulse duration</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

===================
Preparation modules
===================

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Preparation modules put the magnetisation into a stated condition and acquire nothing. None of them defines a recovery interval: the interval between a preparation and the readout that follows belongs to the scan loop, so the same module serves a single-shot and a segmented acquisition.">

.. only:: html

  .. image:: /generated/gallery/22-modules-preparation/images/thumb/sphx_glr_preparation-modules_thumb.png
    :alt:

  :doc:`/generated/gallery/22-modules-preparation/preparation-modules`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Preparation modules</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

=========================
Cartesian readout modules
=========================

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A readout module solves the rest of the repetition around an excitation it is given: the prewinders, the phase-encode template at its largest step, the acquisition window and the spoiler. The Cartesian family differs in how many views one excitation reads.">

.. only:: html

  .. image:: /generated/gallery/23-modules-cartesian/images/thumb/sphx_glr_cartesian-readout-modules_thumb.png
    :alt:

  :doc:`/generated/gallery/23-modules-cartesian/cartesian-readout-modules`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cartesian readout modules</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

=============================
Non-Cartesian readout modules
=============================

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A non-Cartesian readout designs one base interleaf — its acquisition window and the gradients that prewind to and rewind from the centre of k-space — and the scan loop rotates it per shot with a ROTATIONS extension. One interleaf in the gradient library therefore serves the whole scan, however many angles it is played at.">

.. only:: html

  .. image:: /generated/gallery/24-modules-noncartesian/images/thumb/sphx_glr_noncartesian-readout-modules_thumb.png
    :alt:

  :doc:`/generated/gallery/24-modules-noncartesian/noncartesian-readout-modules`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Non-Cartesian readout modules</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three limits bound the traversal of a spiral arm. Two are properties of the gradient system: the maximum amplitude and the maximum slew rate. The third follows from the receiver: with a dwell time \Delta t the trajectory may not advance further than 1/\mathrm{FOV} between samples, which caps the gradient amplitude at">

.. only:: html

  .. image:: /generated/gallery/24-modules-noncartesian/images/thumb/sphx_glr_spiral-readout-limits_thumb.png
    :alt:

  :doc:`/generated/gallery/24-modules-noncartesian/spiral-readout-limits`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gradient, slew and receiver limits on a spiral readout</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

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

    <div class="sphx-glr-thumbcontainer" tooltip="One phase-encode line is read per repetition and the transverse magnetisation is spoiled between them, so the signal is a steady state of the flip angle, the repetition time and T1.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D Cartesian gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slab-selective excitation replaces the slice-selective one and the second phase encode samples the partition axis, so one repetition reads one (line, partition) view.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D Cartesian gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Each excitation is followed by several readout lobes, so one phase-encode line is sampled at several echo times and the decay across them measures T2*.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_multiecho2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_multiecho2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D Cartesian multi-echo gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The multi-echo readout of the two-dimensional sequence over a slab-selective excitation and a partition encode.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_multiecho3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_multiecho3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D Cartesian multi-echo gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Each shot reads a rectangular blade of Cartesian lines and the blades are rotated to cover k-space. Every blade samples the centre, so the shots can be registered against each other before reconstruction.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_propeller2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_propeller2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D PROPELLER gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Every repetition reads a full spoke through the centre of k-space, so the low spatial frequencies are sampled once per repetition rather than once per scan.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_radial2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_radial2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D radial gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One spiral interleaf is read per repetition, designed from the prescription rather than from a fixed shape, and rotated to each of n_shots angles.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_spiral2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_spiral2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D spiral gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="PROPELLER blades in the plane and a Cartesian partition encode along z.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_stack_of_blades3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_stack_of_blades3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-blades gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Spiral interleaves in the plane and a Cartesian partition encode along z.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_stack_of_spirals3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_stack_of_spirals3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-spirals gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Radial spokes in the plane and a Cartesian partition encode along z. The in-plane interleaf is one waveform for the whole scan, rotated per shot.">

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

===========
Spin echoes
===========

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A refocusing pulse between the excitation and the readout recovers the dephasing from static field inhomogeneity, so the acquired echo is weighted by T2 rather than by T2*.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D Cartesian spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One (line, partition) view is read per excitation, at a refocused echo.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D Cartesian spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A whole blade is read after one excitation as an echo-planar train, so a PROPELLER coverage is acquired in as many shots as there are blades.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_epi_propeller2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_epi_propeller2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D PROPELLER spin echo with echo-planar blades</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One line of one rotated blade is read at the refocused echo of each excitation.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_propeller2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_propeller2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D PROPELLER spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A full spoke through the centre of k-space is read at the refocused echo of each excitation.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_radial2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_radial2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D radial spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One spiral interleaf is read at the refocused echo of each excitation.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_spiral2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_spiral2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D spiral spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="PROPELLER blades in the plane, a Cartesian partition encode along z, and a refocused echo per view.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_stack_of_blades3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_stack_of_blades3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-blades spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Spiral interleaves in the plane, a Cartesian partition encode along z, and a refocused echo per view.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_stack_of_spirals3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_stack_of_spirals3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-spirals spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Radial spokes in the plane, a Cartesian partition encode along z, and a refocused echo per view.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_stack_of_stars3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_stack_of_stars3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-stars spin echo</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

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

    <div class="sphx-glr-thumbcontainer" tooltip="A three-dimensional fast spin echo reads one (line, partition) view per refocused echo, so the amplitude the train has left at echo m becomes the weight of whichever view that echo reads. The ordering is the map from echo index to k-space position, and the weighting it produces is a filter applied to the image: its inverse Fourier transform is the point-spread function of the acquisition.">

.. only:: html

  .. image:: /generated/gallery/13-fast-spin-echo/images/thumb/sphx_glr_fse-echo-ordering-and-point-spread_thumb.png
    :alt:

  :doc:`/generated/gallery/13-fast-spin-echo/fse-echo-ordering-and-point-spread`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Echo train length, signal envelope and point spread</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One excitation is followed by a CPMG train of refocusing pulses, and one (line, partition) view is read at each echo. The amplitude left at an echo weights whichever view that echo reads.">

.. only:: html

  .. image:: /generated/gallery/13-fast-spin-echo/images/thumb/sphx_glr_fse3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/13-fast-spin-echo/fse3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D fast spin echo</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

======
MPRAGE
======

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Each shot applies one inversion and then reads every sampled line of a single partition as a spoiled gradient-echo train, so the partition encode is constant within a shot.">

.. only:: html

  .. image:: /generated/gallery/12-mprage/images/thumb/sphx_glr_mprage3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/12-mprage/mprage3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D Cartesian MPRAGE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Each shot applies one inversion and then reads the interleaves of a single partition.">

.. only:: html

  .. image:: /generated/gallery/12-mprage/images/thumb/sphx_glr_mprage_stack_of_spirals3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/12-mprage/mprage_stack_of_spirals3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-spirals MPRAGE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Each shot applies one inversion and then reads the spokes of a single partition.">

.. only:: html

  .. image:: /generated/gallery/12-mprage/images/thumb/sphx_glr_mprage_stack_of_stars3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/12-mprage/mprage_stack_of_stars3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-stars MPRAGE</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

=============
Balanced SSFP
=============

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Every gradient axis returns to zero moment within each repetition and the RF phase alternates, so the steady state depends on the off-resonance accumulated over one repetition time.">

.. only:: html

  .. image:: /generated/gallery/14-bssfp/images/thumb/sphx_glr_bssfp2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/14-bssfp/bssfp2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D balanced SSFP</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The balanced gradient structure of the two-dimensional sequence over a partition-encoded slab, with each train opened by a half flip.">

.. only:: html

  .. image:: /generated/gallery/14-bssfp/images/thumb/sphx_glr_bssfp3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/14-bssfp/bssfp3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D balanced SSFP</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

===================
Echo-planar imaging
===================

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

  .. image:: /generated/gallery/15-epi/images/thumb/sphx_glr_epi-segmentation-and-acceleration_thumb.png
    :alt:

  :doc:`/generated/gallery/15-epi/epi-segmentation-and-acceleration`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Echo train length and geometric distortion in EPI</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One excitation is followed by a train of readout lobes of alternating polarity with phase-encode blips between them, so the whole phase-encode axis is covered in one or a few shots.">

.. only:: html

  .. image:: /generated/gallery/15-epi/images/thumb/sphx_glr_epi2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/15-epi/epi2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D echo-planar imaging</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slab-selective excitation and a partition encode over the echo-planar train, with the skipped-CAIPI lattice available when both encoded axes are undersampled.">

.. only:: html

  .. image:: /generated/gallery/15-epi/images/thumb/sphx_glr_epi3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/15-epi/epi3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D echo-planar imaging</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

==============
Zero echo time
==============

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The readout gradient is already at amplitude when the hard pulse is transmitted, so acquisition begins without a ramp and the trajectory starts at the centre of k-space.">

.. only:: html

  .. image:: /generated/gallery/16-zte/images/thumb/sphx_glr_zte3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/16-zte/zte3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D zero echo time</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

==================
Building sequences
==================

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A complete sequence separates three things: the prescription a user sets, the order the views are sampled in, and the blocks of one repetition. SequenceApp is the contract that keeps them apart, and every sequence the package ships is written against it. The architecture is described in /explanations/design/sequence-application.">

.. only:: html

  .. image:: /generated/gallery/30-building-sequences/images/thumb/sphx_glr_sequence-app-from-scratch_thumb.png
    :alt:

  :doc:`/generated/gallery/30-building-sequences/sequence-app-from-scratch`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A SequenceApp from scratch</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

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
   :includehidden:


   /generated/gallery/01-getting-started/index.rst
   /generated/gallery/20-modules-overview/index.rst
   /generated/gallery/21-modules-rf/index.rst
   /generated/gallery/22-modules-preparation/index.rst
   /generated/gallery/23-modules-cartesian/index.rst
   /generated/gallery/24-modules-noncartesian/index.rst
   /generated/gallery/10-gradient-echo/index.rst
   /generated/gallery/11-spin-echo/index.rst
   /generated/gallery/13-fast-spin-echo/index.rst
   /generated/gallery/12-mprage/index.rst
   /generated/gallery/14-bssfp/index.rst
   /generated/gallery/15-epi/index.rst
   /generated/gallery/16-zte/index.rst
   /generated/gallery/30-building-sequences/index.rst
   /generated/gallery/40-custom-modules/index.rst


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: gallery_python.zip </generated/gallery/gallery_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: gallery_jupyter.zip </generated/gallery/gallery_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
