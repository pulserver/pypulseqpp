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

=============
Pulseq basics
=============

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to build the smallest complete Pulseq sequence, a pulse-acquire experiment, and to establish the vocabulary the rest of the course adds to: the system limits a factory solves against, the events that carry a pulse and an acquisition window, the blocks that play them, the timing check, and the file that is written.">

.. only:: html

  .. image:: /generated/gallery/01-pulseq-basics/images/thumb/sphx_glr_01_fid_thumb.png
    :alt:

  :doc:`/generated/gallery/01-pulseq-basics/01_fid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Free induction decay</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to add a refocusing pulse to the pulse-acquire experiment of the previous page, so that the acquisition is centred on an echo at a prescribed echo time, and to place the crusher pair that keeps the signal of an imperfect refocusing pulse out of that acquisition.">

.. only:: html

  .. image:: /generated/gallery/01-pulseq-basics/images/thumb/sphx_glr_02_spin_echo_thumb.png
    :alt:

  :doc:`/generated/gallery/01-pulseq-basics/02_spin_echo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to turn the non-imaging experiments of the two previous pages into a two-dimensional acquisition: the excitation becomes slice-selective, the echo is formed by a gradient rather than by a refocusing pulse, and a phase encode moves the acquired line from one repetition to the next. This is the structure every Cartesian sequence in the course is a variation on.">

.. only:: html

  .. image:: /generated/gallery/01-pulseq-basics/images/thumb/sphx_glr_03_gradient_echo_thumb.png
    :alt:

  :doc:`/generated/gallery/01-pulseq-basics/03_gradient_echo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gradient echo</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

========
Spoiling
========

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to add a spoiler gradient to the gradient echo of the previous section, and to establish what it does and does not achieve: a spoiler winds the transverse magnetisation left at the end of a repetition through several cycles across a voxel, so that it integrates to nothing there, but it winds every repetition by the same amount and therefore leaves a coherent pathway that survives into the steady state.">

.. only:: html

  .. image:: /generated/gallery/02-spoiling/images/thumb/sphx_glr_01_gradient_spoiling_thumb.png
    :alt:

  :doc:`/generated/gallery/02-spoiling/01_gradient_spoiling`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gradient spoiling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to remove the coherent pathway the previous page was left with, by advancing the phase of the pulse and of the receiver by a quadratically increasing amount from one repetition to the next, and to measure which phase increments do so.">

.. only:: html

  .. image:: /generated/gallery/02-spoiling/images/thumb/sphx_glr_02_rf_spoiling_thumb.png
    :alt:

  :doc:`/generated/gallery/02-spoiling/02_rf_spoiling`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">RF spoiling</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

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

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to replace the phase encode of the Cartesian gradient echo with a rotation of the readout itself, so that every repetition acquires a spoke through the centre of k-space, and to establish how many spokes such an acquisition needs and what ordering them by the golden angle changes.">

.. only:: html

  .. image:: /generated/gallery/04-non-cartesian/images/thumb/sphx_glr_01_radial_thumb.png
    :alt:

  :doc:`/generated/gallery/04-non-cartesian/01_radial`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Radial sampling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to acquire k-space along a spiral arm rather than along straight lines, and to establish which of the system&#x27;s limits decides how long an arm takes. A spiral is the first trajectory of the course that cannot be written down as a trapezoid: its waveform is solved numerically against the limits, which is what SpiralReadout2D is for. The interface such a module presents is the subject of /generated/gallery/05-sequence-modules/02_readout; here it is used for the arms it designs.">

.. only:: html

  .. image:: /generated/gallery/04-non-cartesian/images/thumb/sphx_glr_02_spiral_thumb.png
    :alt:

  :doc:`/generated/gallery/04-non-cartesian/02_spiral`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Spiral readout</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

================
Sequence modules
================

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to design slice-selective excitations with the module that solves them, and to measure what the three numbers a selective pulse is specified by — flip angle, slice thickness and time-bandwidth product — do to the slice profile, to the selection gradient and to the peak B_1, and which combinations of them a gradient system admits.">

.. only:: html

  .. image:: /generated/gallery/05-sequence-modules/images/thumb/sphx_glr_01_excitation_thumb.png
    :alt:

  :doc:`/generated/gallery/05-sequence-modules/01_excitation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Excitation modules</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to replace the hand-built readout of the first sections with the module that designs one, and to use the two prescriptions the earlier pages solved by hand — a partial echo and a multi-echo train — as the check that the module reaches the same answers and states them.">

.. only:: html

  .. image:: /generated/gallery/05-sequence-modules/images/thumb/sphx_glr_02_readout_thumb.png
    :alt:

  :doc:`/generated/gallery/05-sequence-modules/02_readout`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Readout modules</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to assemble the modules of the two previous pages into a complete acquisition: a prescription, a sampling order, and a kernel that is played once per repetition. SequenceApp separates those three, and is the base class every shipped sequence is written against, so what is written here is what a sequence in /sequences is written as.">

.. only:: html

  .. image:: /generated/gallery/05-sequence-modules/images/thumb/sphx_glr_03_sequence_app_thumb.png
    :alt:

  :doc:`/generated/gallery/05-sequence-modules/03_sequence_app`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A sequence application</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

=================
Constraint checks
=================

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to run the checks the package computes over a finished sequence, and to read what each of them reports: the quantity it measured, the limit it compared it with, and where in the sequence the measurement came from.">

.. only:: html

  .. image:: /generated/gallery/06-checks/images/thumb/sphx_glr_01_constraint_checks_thumb.png
    :alt:

  :doc:`/generated/gallery/06-checks/01_constraint_checks`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Sequence constraint checks</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

==============
Custom modules
==============

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to write an excitation module of one&#x27;s own, by subclassing RfModule, and to measure what the design it implements gains and costs against the shipped one.">

.. only:: html

  .. image:: /generated/gallery/07-custom-modules/images/thumb/sphx_glr_01_excitation_module_thumb.png
    :alt:

  :doc:`/generated/gallery/07-custom-modules/01_excitation_module`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A minimum-phase excitation module</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to write a Cartesian readout module of one&#x27;s own, by following the SequenceModule contract, and to measure what it changes against the shipped readout.">

.. only:: html

  .. image:: /generated/gallery/07-custom-modules/images/thumb/sphx_glr_02_cartesian_readout_thumb.png
    :alt:

  :doc:`/generated/gallery/07-custom-modules/02_cartesian_readout`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A ramp-sampled readout module</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to write a non-Cartesian readout module of one&#x27;s own: to state a trajectory as a k-space path, have it solved into a waveform under the gradient limits, and publish the result as a module.">

.. only:: html

  .. image:: /generated/gallery/07-custom-modules/images/thumb/sphx_glr_03_noncartesian_readout_thumb.png
    :alt:

  :doc:`/generated/gallery/07-custom-modules/03_noncartesian_readout`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A twisting radial readout module</div>
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


.. toctree::
   :hidden:
   :includehidden:


   /generated/gallery/01-pulseq-basics/index.rst
   /generated/gallery/02-spoiling/index.rst
   /generated/gallery/03-gre-to-epi/index.rst
   /generated/gallery/04-non-cartesian/index.rst
   /generated/gallery/05-sequence-modules/index.rst
   /generated/gallery/06-checks/index.rst
   /generated/gallery/07-custom-modules/index.rst
   /generated/gallery/10-gradient-echo/index.rst
   /generated/gallery/11-spin-echo/index.rst
   /generated/gallery/13-fast-spin-echo/index.rst
   /generated/gallery/12-mprage/index.rst
   /generated/gallery/14-bssfp/index.rst
   /generated/gallery/15-epi/index.rst
   /generated/gallery/16-zte/index.rst


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: gallery_python.zip </generated/gallery/gallery_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: gallery_jupyter.zip </generated/gallery/gallery_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
