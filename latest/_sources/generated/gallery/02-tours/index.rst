

.. _sphx_glr_generated_gallery_02-tours:

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

    <div class="sphx-glr-thumbcontainer" tooltip="The simplest signal you can acquire is the free induction decay (FID): a pulse tips the magnetization into the transverse plane and the receiver records what the precessing magnetization emits. This Tour builds that sequence, checks the flip angle of its pulses with a Bloch simulation, and then adds a refocusing pulse and a pair of crusher gradients so that the acquisition is centred on a spin echo at the echo time you prescribe.">

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


.. toctree::
   :hidden:

   /generated/gallery/02-tours/01_fid_and_spin_echo
   /generated/gallery/02-tours/02_slice_selective_excitation
   /generated/gallery/02-tours/03_epi
   /generated/gallery/02-tours/04_multi_echo
   /generated/gallery/02-tours/05_segmented_epi
   /generated/gallery/02-tours/06_spiral
   /generated/gallery/02-tours/07_ramp_sampled_readout
   /generated/gallery/02-tours/08_twisting_radial_readout
   /generated/gallery/02-tours/09_minimum_phase_excitation

