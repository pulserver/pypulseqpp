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

======
Course
======

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This course takes you from a 2D Cartesian gradient echo to a 3D radial MPRAGE, one step per lesson. This lesson starts with the gradient echo.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_01_first_gradient_echo_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/01_first_gradient_echo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">1. Your first gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 1 the echo landed wherever the blocks put it: right after the prewinder, in the middle of the readout. A protocol asks for an echo time and a repetition time, so in this lesson you put the echo exactly where the protocol asks for it, find the shortest echo time your readout allows, and shorten it further by acquiring only part of the echo.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_02_echo_and_repetition_time_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/02_echo_and_repetition_time`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2. Echo and repetition time</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="With a TR of 20 ms and a T2 of tens of milliseconds, your gradient echo still has transverse magnetisation when the next pulse arrives. That pulse refocuses part of it, and the leftover from one repetition adds to the signal of the next: after a few hundred repetitions the signal settles into a steady state that depends on T2 as well as on T1 and the flip angle. A spoiled gradient echo removes that leftover, so the steady state is the simple one that every T1-weighted protocol assumes.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_03_spoiling_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/03_spoiling`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3. Spoiling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scanner plays your sequence and hands the reconstruction a stream of readouts. To put each readout in its place, the reconstruction has to know which k-space line it is, which slice, which average, and what the field of view and the matrix are. None of that can be guessed from the samples: you write it into the sequence, as labels on the blocks and definitions in the file header.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_04_labels_and_metadata_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/04_labels_and_metadata`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">4. Labels and metadata</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="So far each repetition of your gradient echo acquired a horizontal line of k-space, moved up or down by the phase encoding. A radial acquisition acquires spokes instead: lines through the centre of k-space, each rotated by its own angle. Every spoke samples the centre, which makes radial imaging robust to motion, and the spokes can be ordered so that any subset of them covers k-space evenly.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_05_radial_sampling_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/05_radial_sampling`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">5. Radial sampling and gradient surgery</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="MPRAGE is the T1-weighted 3D scan of almost every brain protocol. An inversion pulse flips the longitudinal magnetisation; while it recovers, at a speed set by each tissue&#x27;s T1, a fast train of gradient echoes reads it out; then the magnetisation recovers before the next inversion. White and grey matter recover at different speeds, and the train catches them while they differ.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_06_radial_mprage_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/06_radial_mprage`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">6. Radial MPRAGE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A sequence that passes check_timing is one the scanner can play, block by block. Whether it should is a question about the whole scan: does any gradient exceed the amplitude or slew-rate limit once it is rotated, does the switching stimulate peripheral nerves, does the gradient waveform drive the coil at one of its mechanical resonances, how much RF power does the subject absorb. pypulseqpp computes each of these from the sequence, and this lesson runs them on your MPRAGE.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_07_hardware_and_safety_checks_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/07_hardware_and_safety_checks`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">7. Hardware and safety checks</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Your MPRAGE of lesson 6 is a script: the protocol sits in constants at the top, and changing the inversion time means editing the file. In this lesson you turn it into a sequence function, a function of the scanner and the protocol that returns the sequence. That is the form every sequence shipped with pypulseqpp takes, and the form a scanner console or a command line can drive: they read the protocol from the function&#x27;s signature and docstring, and call it with the values the user prescribes.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_08_sequence_functions_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/08_sequence_functions`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">8. Sequence functions</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Most of your MPRAGE function designs events: the inversion and its crusher, the slab-selective pulse, the merged readout, the partition encodings. The same parts appear in many sequences, and pypulseqpp ships them as sequence modules. A module designs one part of the repetition from a few physical parameters, publishes its events under fixed names, lays them out in blocks, and tells you its timing. The loop stays yours.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_09_sequence_modules_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/09_sequence_modules`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">9. Sequence modules</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The shipped modules cover the common parts of a sequence, but sooner or later you need one that is not there. A module is a small class, and writing one is the same work as designing its events by hand, done once and given a name.">

.. only:: html

  .. image:: /generated/gallery/01-course/images/thumb/sphx_glr_10_custom_sequence_module_thumb.png
    :alt:

  :doc:`/generated/gallery/01-course/10_custom_sequence_module`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">10. Custom sequence module</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

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

    <div class="sphx-glr-thumbcontainer" tooltip="The simplest signal you can acquire is the free induction decay (FID): a pulse tips the magnetisation into the transverse plane and the receiver records what the precessing magnetisation emits. This Tour builds that sequence, checks the flip angle of its pulses with a Bloch simulation, and then adds a refocusing pulse and a pair of crusher gradients so that the acquisition is centred on a spin echo at the echo time you prescribe.">

.. only:: html

  .. image:: /generated/gallery/02-tours/images/thumb/sphx_glr_01_fid_and_spin_echo_thumb.png
    :alt:

  :doc:`/generated/gallery/02-tours/01_fid_and_spin_echo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">FID and spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A slice-selective pulse is played together with a selection gradient, so that only the spins within a slab of the chosen thickness are on resonance. Three numbers specify it: the flip angle, the slice thickness and the time-bandwidth product. This Tour designs excitations with the excitation module, simulates their slice profiles, and measures how the time-bandwidth product and the duration affect the profile, the selection gradient and the peak B_1, and which combinations the gradient system admits.">

.. only:: html

  .. image:: /generated/gallery/02-tours/images/thumb/sphx_glr_02_slice_selective_excitation_thumb.png
    :alt:

  :doc:`/generated/gallery/02-tours/02_slice_selective_excitation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Slice-selective excitation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A gradient echo acquires one k-space line per excitation. Echo planar imaging (EPI) acquires the whole matrix after a single excitation. This Tour first follows the readout gradient with further readout gradients of alternating polarity, each forming an echo, and then adds a phase-encode blip between the echoes so that each echo acquires a different line.">

.. only:: html

  .. image:: /generated/gallery/02-tours/images/thumb/sphx_glr_03_epi_thumb.png
    :alt:

  :doc:`/generated/gallery/02-tours/03_epi`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Echo planar imaging</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A multi-echo gradient echo follows the first readout gradient with further readout gradients of alternating polarity, each with its own ADC event. Each gradient traverses the same k-space line, in the opposite direction to the one before it, so the train forms one echo per gradient and needs no rewinder between them. The echoes sample the line at increasing echo times, which is what lets you map T_2^* from a single repetition.">

.. only:: html

  .. image:: /generated/gallery/02-tours/images/thumb/sphx_glr_04_multi_echo_thumb.png
    :alt:

  :doc:`/generated/gallery/02-tours/04_multi_echo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multi-echo gradient echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A single-shot echo planar readout acquires the whole matrix after one excitation, and its long echo train distorts the image along the phase-encode direction. A segmented readout divides the phase-encode lines of the matrix over several excitations (shots). Each shot acquires an interleaved subset of the lines, with a phase-encode blip between its echoes, so a shot has a shorter echo train.">

.. only:: html

  .. image:: /generated/gallery/02-tours/images/thumb/sphx_glr_05_segmented_epi_thumb.png
    :alt:

  :doc:`/generated/gallery/02-tours/05_segmented_epi`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Segmented echo planar imaging</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A spiral readout acquires k-space along a spiral arm instead of along straight lines or spokes. The waveform of an arm is not a trapezoid: it is solved numerically against the system limits by SpiralReadout2D, and in this Tour you use the module only for the arms it designs.">

.. only:: html

  .. image:: /generated/gallery/02-tours/images/thumb/sphx_glr_06_spiral_thumb.png
    :alt:

  :doc:`/generated/gallery/02-tours/06_spiral`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Spiral readout</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Cartesian readouts you have used so far sample on the flat top of the readout lobe, so the area under the ramps is not sampled. Sampling through the ramps as well covers the same extent of k-space in a shorter lobe, at the cost of sample positions that are not evenly spaced. In this Tour you write a Cartesian readout module that samples through the ramps, compare its duration with the flat-top design at the same resolution, and measure the sample spacing along the line from the k-space analysis.">

.. only:: html

  .. image:: /generated/gallery/02-tours/images/thumb/sphx_glr_07_ramp_sampled_readout_thumb.png
    :alt:

  :doc:`/generated/gallery/02-tours/07_ramp_sampled_readout`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ramp-sampled readout</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A radial spoke samples the centre of k-space far more densely than the periphery: at radius k, adjacent spokes of an N-interleaf set are 2\pi k / N apart, which exceeds the Nyquist spacing 1/\mathrm{FOV} beyond a transition radius">

.. only:: html

  .. image:: /generated/gallery/02-tours/images/thumb/sphx_glr_08_twisting_radial_readout_thumb.png
    :alt:

  :doc:`/generated/gallery/02-tours/08_twisting_radial_readout`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Twisting radial readout</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The shipped excitation modules design linear-phase SLR pulses, whose energy is symmetric about the middle of the pulse. A minimum-phase design concentrates RF energy near the end of the waveform, so that at a fixed duration and time-bandwidth product the interval from the pulse to the echo is shorter. The peak B_1 is larger, and the phase of the slice profile is not linear.">

.. only:: html

  .. image:: /generated/gallery/02-tours/images/thumb/sphx_glr_09_minimum_phase_excitation_thumb.png
    :alt:

  :doc:`/generated/gallery/02-tours/09_minimum_phase_excitation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Minimum-phase excitation</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="A slice-selective excitation and 180-degree refocusing pulse form one spin echo, at which one line of a rotating PROPELLER blade is acquired. Spoilers suppress unwanted coherence before the next TR. TE controls T2 weighting and TR controls longitudinal recovery. The overlapping blade centres support motion-robust structural imaging.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="A slab-selective excitation is followed by alternating readout gradients with phase-encode and partition blips. The sampled (line, partition) views form a CAIPIRINHA lattice [BRE06]_, and segmented skipped-CAIPI traversal [STI21]_ distributes the lattice among shots. Spoilers suppress residual transverse coherence between repetitions; off-resonance accumulates during each echo train. 3D EPI supports rapid structural and functional imaging.">

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


   /generated/gallery/01-course/index.rst
   /generated/gallery/02-tours/index.rst
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
