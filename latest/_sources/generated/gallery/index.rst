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

    <div class="sphx-glr-thumbcontainer" tooltip="A two-repetition slice-selective gradient-echo sequence illustrates system limits, event construction, block timing, sequence assembly and Pulseq output.">

.. only:: html

  .. image:: /generated/gallery/01-getting-started/images/thumb/sphx_glr_basic-pulseq-sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/01-getting-started/basic-pulseq-sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A basic Pulseq sequence</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The package computes timing and six constraint checks over a finished sequence: event timing, gradient amplitude, slew rate, gradient continuity across block boundaries, peripheral nerve stimulation (PNS), mechanical resonance and specific absorption rate (SAR). Each check returns a verdict and the quantities used to determine it.">

.. only:: html

  .. image:: /generated/gallery/01-getting-started/images/thumb/sphx_glr_safety-checks_thumb.png
    :alt:

  :doc:`/generated/gallery/01-getting-started/safety-checks`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Sequence constraint checks</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="A sequence module contains a reusable block layout and named event templates. The composition below combines inversion preparation, a prescribed inversion delay and a Cartesian readout, then verifies the resulting pulse-centre interval. The object model is described in /explanations/design/sequence-module.">

.. only:: html

  .. image:: /generated/gallery/20-modules-overview/images/thumb/sphx_glr_sequence-module-composition_thumb.png
    :alt:

  :doc:`/generated/gallery/20-modules-overview/sequence-module-composition`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Sequence module composition</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="A non-Cartesian readout designs one base interleaf — its acquisition window and the gradients that prewind to and rewind from the centre of k-space — and the scan loop rotates it per shot with a ROTATIONS extension. The gradient library stores one interleaf; per-shot rotation extensions define its physical orientation.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="One excitation and one phase-encode line per repetition, with the transverse magnetisation spoiled by a gradient and by a quadratic RF phase increment before the next excitation. This is the reference implementation for the Cartesian gradient-echo variants.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="One excitation per repetition, with the line read again at several echo times. Signal amplitude across the echo train follows apparent transverse relaxation, providing multiple points on the decay curve per excitation.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="One full spoke through the centre of k-space per repetition. Every readout crosses the k-space origin. Angular undersampling produces streak artefacts rather than coherent Cartesian aliasing.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="Radial spokes in the plane and Cartesian encoding along the slab axis. The in-plane trajectory retains radial sampling properties, with Cartesian encoding along the partition axis.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="One excitation and one refocusing pulse per repetition, with the line read at the refocused echo. Refocusing undoes the dephasing that static field inhomogeneity causes, so the contrast follows the true transverse relaxation rather than the apparent one.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D Cartesian spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One excitation and one refocusing pulse per (line, partition) view over a slab.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D Cartesian spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One whole blade per excitation, read as an echo-planar train. The blade is acquired in one shot rather than a line at a time, so the scan is far shorter than a line-by-line PROPELLER and the blade carries the off-resonance behaviour of an echo-planar readout.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_epi_propeller2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_epi_propeller2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D PROPELLER spin echo with echo-planar blades</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One line of one rotating blade per excitation, read at the refocused echo.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_propeller2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_propeller2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D PROPELLER spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One full spoke per excitation, read at the refocused echo.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_radial2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_radial2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D radial spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One spiral interleaf per excitation, read at the refocused echo.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_spiral2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_spiral2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D spiral spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="PROPELLER blades in the plane and Cartesian encoding along the slab axis, read at the refocused echo.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_stack_of_blades3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_stack_of_blades3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-blades spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Spiral interleaves in the plane and Cartesian encoding along the slab axis, read at the refocused echo.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_stack_of_spirals3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_stack_of_spirals3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-spirals spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Radial spokes in the plane and Cartesian encoding along the slab axis, read at the refocused echo.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="One excitation followed by a CPMG train of refocusing pulses, with one (line, partition) view acquired per echo. Signal amplitude at echo m weights the corresponding k-space view. Echo ordering therefore determines the modulation transfer function and point-spread function.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="One inversion per shot, an inversion time, and then a spoiled gradient-echo train that reads the views of one partition. The acquisition time of central k-space relative to the inversion pulse determines the dominant inversion-recovery contrast. View ordering therefore defines the contrast weighting across k-space.">

.. only:: html

  .. image:: /generated/gallery/12-mprage/images/thumb/sphx_glr_mprage3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/12-mprage/mprage3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D Cartesian MPRAGE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One inversion per shot, followed by a spoiled gradient-echo train that reads the spiral interleaves of one partition. An interleaf covers far more of the plane than a line does, so a partition needs few readouts and the whole train sits close behind the inversion.">

.. only:: html

  .. image:: /generated/gallery/12-mprage/images/thumb/sphx_glr_mprage_stack_of_spirals3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/12-mprage/mprage_stack_of_spirals3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-spirals MPRAGE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One inversion per shot, followed by a spoiled gradient-echo train that reads the radial spokes of one partition. In-plane the acquisition is radial, so every spoke crosses the centre of k-space and the contrast the inversion time sets is carried by every readout rather than by a few central lines.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="Every gradient axis returns to zero moment within each repetition and the RF phase alternates, so the magnetisation reaches a steady state that carries both relaxation times. The train opens with a half flip, which places the magnetisation on the axis the steady state oscillates about.">

.. only:: html

  .. image:: /generated/gallery/14-bssfp/images/thumb/sphx_glr_bssfp2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/14-bssfp/bssfp2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D balanced SSFP</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The balanced gradient structure over a partition-encoded slab, with each train opened by a half flip.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="One excitation followed by a train of readout lobes of alternating polarity, with a phase-encoding blip between successive readouts. A single-shot train acquires the complete phase-encode axis after one excitation. Off-resonance phase accumulates across the train and produces displacement along that axis.">

.. only:: html

  .. image:: /generated/gallery/15-epi/images/thumb/sphx_glr_epi2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/15-epi/epi2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D echo-planar imaging</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One excitation per shot, followed by a train of readout lobes of alternating polarity that covers a shell of partitions. The sampled views form a CAIPIRINHA lattice. Phase-encode lines satisfy (y - n_y // 2) % ry == 0; the partition index advances by the CAIPI shift between adjacent lattice lines. Each shot acquires every n_shots-th lattice line, defining skipped-CAIPI sampling (Stirnberg and Stöcker, Magn Reson Med 2021, doi:10.1002/mrm.28486); one shot per shell is blipped-CAIPI.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="The readout gradient is already at amplitude when the hard pulse is transmitted, so acquisition begins as soon as the receiver is available and the echo time is a few tens of microseconds. Concurrent excitation and gradient encoding produce a spatially dependent RF bandwidth. Transmit/receive dead time leaves a central k-space gap.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="A complete sequence defines a prescription, sampling order and repetition kernel. SequenceApp separates these responsibilities and provides the base class for the shipped sequences. The architecture is described in /explanations/design/sequence-application.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="The shipped Cartesian readouts acquire on the flat top of the readout lobe, so the ramps carry area that is never sampled. Sampling through the ramps as well covers the same extent of k-space in a shorter lobe, with nonuniform ADC sampling locations that require regridding.">

.. only:: html

  .. image:: /generated/gallery/40-custom-modules/images/thumb/sphx_glr_custom-cartesian-readout_thumb.png
    :alt:

  :doc:`/generated/gallery/40-custom-modules/custom-cartesian-readout`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A ramp-sampled readout module</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The shipped excitation modules design linear-phase SLR pulses, whose energy is symmetric about the middle of the pulse. A minimum-phase design concentrates RF energy near the end of the waveform. For fixed duration and time-bandwidth product, this reduces the interval to the echo but increases peak B_1 and introduces nonlinear slice-profile phase.">

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
