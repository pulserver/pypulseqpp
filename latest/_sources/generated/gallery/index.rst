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

    <div class="sphx-glr-thumbcontainer" tooltip="A spoiled gradient-echo (SPGR) acquisition applies one low-flip-angle slice-selective excitation before an unbalanced Cartesian readout in each TR. Gradient spoiling and quadratic RF/receiver phase cycling suppress coherent residual transverse magnetisation. TR and flip angle primarily determine T1 weighting, with T2* decay during TE. SPGR is widely used for T1-weighted structural imaging.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D Cartesian gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 3D spoiled gradient-echo (SPGR) acquisition applies a low-flip-angle slab excitation before one Cartesian (line, partition) readout per TR. Gradient spoiling and quadratic RF/receiver phase cycling suppress coherent residual transverse magnetisation. TR, flip angle, and TE determine the T1 and T2* weighting. This sequence is used for high-resolution T1-weighted structural imaging.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D Cartesian gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A spoiled low-flip-angle excitation is followed by several Cartesian gradient echoes of the same phase-encode line. Gradient and RF spoiling suppress residual transverse coherence before the next TR. The echo train samples T2 decay while TR and flip angle determine the T1 weighting. Multi-echo GRE is used for T2/R2* mapping, susceptibility mapping, and structural imaging.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_multiecho2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_multiecho2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D Cartesian multi-echo gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slab-selective spoiled excitation is followed by several gradient echoes of one Cartesian (line, partition) view. Gradient and RF spoiling suppress residual transverse coherence between repetitions. The multiple echo times sample T2 decay; TR and flip angle determine the T1 weighting. Applications include high-resolution structural imaging, R2 mapping, and QSM.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_multiecho3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_multiecho3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D Cartesian multi-echo gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A spoiled low-flip-angle excitation is followed by one Cartesian line from a rotating PROPELLER blade. Gradient and RF spoiling suppress residual transverse coherence between repetitions. TR, flip angle, and TE determine contrast. The overlapping central k-space region supports motion estimation in structural imaging.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_propeller2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_propeller2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D PROPELLER gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A spoiled low-flip-angle excitation is followed by one radial spoke through k-space centre. Gradient and RF spoiling suppress residual transverse coherence before the next TR. TR and flip angle primarily determine T1 weighting, with T2* decay during TE. Radial SPGR is used for motion-robust dynamic and structural imaging.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_radial2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_radial2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D radial gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A spoiled low-flip-angle excitation is followed by one spiral interleaf. Gradient and RF spoiling suppress residual transverse coherence between repetitions. TR and flip angle primarily determine T1 weighting; off-resonance and T2* decay affect the spiral readout. Spiral SPGR supports rapid dynamic and structural imaging.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_spiral2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_spiral2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D spiral gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slab-selective spoiled excitation is followed by one line from a rotating in-plane PROPELLER blade with Cartesian partition encoding. Gradient and RF spoiling suppress residual transverse coherence. TR, flip angle, and TE determine contrast. The overlapping blade centres support motion-robust 3D structural imaging.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_stack_of_blades3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_stack_of_blades3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-blades gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slab-selective spoiled excitation is followed by one spiral interleaf with Cartesian partition encoding. Gradient and RF spoiling suppress residual transverse coherence between repetitions. TR, flip angle, and TE determine contrast. Stack-of-spirals SPGR supports rapid 3D structural and dynamic imaging.">

.. only:: html

  .. image:: /generated/gallery/10-gradient-echo/images/thumb/sphx_glr_gre_stack_of_spirals3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/10-gradient-echo/gre_stack_of_spirals3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-spirals gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slab-selective spoiled excitation is followed by one radial spoke with Cartesian partition encoding. Gradient and RF spoiling suppress residual transverse coherence between repetitions. TR and flip angle primarily determine T1 weighting. Stack-of-stars SPGR is used for motion-robust 3D structural and dynamic imaging.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="A slice-selective excitation and 180-degree refocusing pulse form one spin echo, followed by a Cartesian readout. Spoilers suppress unwanted coherence before the next TR. TE controls T2 weighting and TR controls longitudinal recovery. Spin echo is used for conventional T1-, T2-, and proton-density-weighted structural imaging.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D Cartesian spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slab-selective excitation and 180-degree refocusing pulse form one spin echo, followed by a Cartesian (line, partition) readout. Spoilers suppress unwanted coherence before the next TR. TE and TR determine T2 and longitudinal recovery weighting. 3D spin echo supports high-resolution structural imaging.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D Cartesian spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slice-selective excitation and 180-degree refocusing pulse form a spin echo, followed by an echo-planar readout of one rotating PROPELLER blade. Spoilers suppress unwanted coherence between shots. TE controls T2 weighting, while the EPI train introduces off-resonance sensitivity. This sequence supports rapid, motion-robust structural imaging.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_epi_propeller2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_epi_propeller2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D PROPELLER spin echo with echo-planar blades</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slice-selective excitation and 180-degree refocusing pulse form one spin echo, which reads one line of a rotating PROPELLER blade. Spoilers suppress unwanted coherence before the next TR. TE controls T2 weighting and TR controls longitudinal recovery. The overlapping blade centres support motion-robust structural imaging.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_propeller2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_propeller2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D PROPELLER spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slice-selective excitation and 180-degree refocusing pulse form one spin echo, followed by a radial spoke through k-space centre. Spoilers suppress unwanted coherence before the next TR. TE controls T2 weighting and TR controls longitudinal recovery. Radial spin echo supports motion-robust structural imaging.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_radial2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_radial2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D radial spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slice-selective excitation and 180-degree refocusing pulse form one spin echo, followed by a spiral interleaf. Spoilers suppress unwanted coherence before the next TR. TE controls T2 weighting; off-resonance affects the spiral readout. Spiral spin echo supports rapid T2-weighted structural imaging.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_spiral2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_spiral2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D spiral spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slab-selective excitation and 180-degree refocusing pulse form one spin echo, followed by a PROPELLER line with Cartesian partition encoding. Spoilers suppress unwanted coherence between repetitions. TE and TR determine T2 and longitudinal recovery weighting. The overlapping blade centres support motion-robust 3D structural imaging.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_stack_of_blades3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_stack_of_blades3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-blades spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slab-selective excitation and 180-degree refocusing pulse form one spin echo, followed by a spiral interleaf with Cartesian partition encoding. Spoilers suppress unwanted coherence between repetitions. TE controls T2 weighting; off-resonance affects the spiral readout. This sequence supports rapid 3D T2-weighted imaging.">

.. only:: html

  .. image:: /generated/gallery/11-spin-echo/images/thumb/sphx_glr_se_stack_of_spirals3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/11-spin-echo/se_stack_of_spirals3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-spirals spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slab-selective excitation and 180-degree refocusing pulse form one spin echo, followed by a radial spoke with Cartesian partition encoding. Spoilers suppress unwanted coherence between repetitions. TE and TR determine T2 and longitudinal recovery weighting. Stack-of-stars spin echo supports motion-robust 3D structural imaging.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="Individually parameterized 3D FSE assigns different echo-train lengths and repetition times to central and peripheral k-space. The refocusing schedules and radial view order vary smoothly between these limits. This coupling can reduce scan time while retaining a prescribed central-k-space contrast for high-resolution structural imaging.">

.. only:: html

  .. image:: /generated/gallery/13-fast-spin-echo/images/thumb/sphx_glr_fse3D_adaptive_thumb.png
    :alt:

  :doc:`/generated/gallery/13-fast-spin-echo/fse3D_adaptive`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Individually optimized 3D fast spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slab-selective excitation is followed by a CPMG fast-spin-echo refocusing train, with one Cartesian (line, partition) view acquired at each echo. Variable refocusing angles control stimulated-echo pathways and T2-dependent signal evolution. Radial view ordering assigns this evolution to k-space and therefore determines the modulation transfer function and image blurring. 3D FSE is used for T2- and proton-density-weighted structural imaging.">

.. only:: html

  .. image:: /generated/gallery/13-fast-spin-echo/images/thumb/sphx_glr_fse3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/13-fast-spin-echo/fse3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Conventional 3D fast spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Shuffled 3D FSE uses the same optimized refocusing train as conventional FSE, but distributes echo times over a variable-density Poisson-disc sampling pattern. The resulting incoherent contrast distribution can support echo-resolved or subspace reconstruction; no reconstruction is performed here.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="An inversion preparation is followed after the prescribed inversion delay by a train of low-flip-angle spoiled Cartesian gradient echoes. The inversion time is measured to the first excitation centre; the corresponding central ADC sample occurs one TE later. The ordering assigns recovery times within each inversion cycle to (line, partition) views. MPRAGE is used for high-resolution 3D T1-weighted structural imaging.">

.. only:: html

  .. image:: /generated/gallery/12-mprage/images/thumb/sphx_glr_mprage3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/12-mprage/mprage3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D Cartesian MPRAGE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An inversion preparation is followed by a train of low-flip-angle spoiled spiral gradient echoes with Cartesian partition encoding. Interleaf and partition order determine the recovery time of the acquired data within and between inversion cycles. Stack-of-spirals MPRAGE provides rapid T1-weighted 3D structural imaging.">

.. only:: html

  .. image:: /generated/gallery/12-mprage/images/thumb/sphx_glr_mprage_stack_of_spirals3D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/12-mprage/mprage_stack_of_spirals3D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D stack-of-spirals MPRAGE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An inversion preparation is followed by a train of low-flip-angle spoiled radial gradient echoes with Cartesian partition encoding. Each spoke crosses in-plane k-space centre; spoke and partition order determine the recovery time of the acquired data within and between inversion cycles. Stack-of-stars MPRAGE provides T1-weighted 3D structural imaging with radial sampling.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="A low-flip-angle excitation and balanced Cartesian gradient-echo readout repeat with alternating RF phase. Zero net gradient moment in every TR preserves transverse coherence and establishes a steady state governed by T2/T1 and off-resonance. A half-flip preparation reduces transient oscillation. 2D bSSFP is widely used for cardiac cine and dynamic cardiac imaging.">

.. only:: html

  .. image:: /generated/gallery/14-bssfp/images/thumb/sphx_glr_bssfp2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/14-bssfp/bssfp2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D balanced SSFP</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slab-selective low-flip-angle excitation and balanced Cartesian readout repeat with alternating RF phase and zero net gradient moment in every TR. The preserved transverse coherence establishes a high-SNR steady state governed by T2/T1 and off-resonance. A half-flip preparation reduces transient oscillation. 3D bSSFP is used for high-SNR structural imaging.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="A slice-selective excitation is followed by alternating readout gradients and phase-encode blips that acquire multiple Cartesian lines in one echo train. Spoilers suppress residual transverse coherence between repetitions. Off-resonance phase accumulates during the train and produces geometric distortion along the phase-encode axis. EPI supports rapid structural imaging and functional MRI.">

.. only:: html

  .. image:: /generated/gallery/15-epi/images/thumb/sphx_glr_epi2D_sequence_thumb.png
    :alt:

  :doc:`/generated/gallery/15-epi/epi2D_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D echo-planar imaging</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slab-selective excitation is followed by alternating readout gradients with phase-encode and partition blips. Segmented skipped-CAIPI traversal distributes a three-dimensional Cartesian lattice among shots. Spoilers suppress residual transverse coherence between repetitions; off-resonance accumulates during each echo train. 3D EPI supports rapid structural and functional imaging.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="A short non-selective excitation is applied while the radial readout gradient is already at amplitude. Acquisition begins after the transmit/receive dead time, without gradient-echo formation; spoiling suppresses residual transverse magnetisation between repetitions. Contrast depends on TR, flip angle, RF bandwidth, and very short-T2 decay. ZTE is used for anatomical imaging of short-T2 tissues and other minimal-TE applications.">

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
   /generated/gallery/21-modules-rf/index.rst
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
