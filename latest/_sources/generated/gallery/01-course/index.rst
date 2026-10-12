

.. _sphx_glr_generated_gallery_01-course:

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

    <div class="sphx-glr-thumbcontainer" tooltip="With a TR of 20 ms and a T2 of tens of milliseconds, your gradient echo still has transverse magnetization when the next pulse arrives. That pulse refocuses part of it, and the leftover from one repetition adds to the signal of the next: after a few hundred repetitions the signal settles into a steady state that depends on T2 as well as on T1 and the flip angle. A spoiled gradient echo removes that leftover, so the steady state is the simple one that every T1-weighted protocol assumes.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="MPRAGE is the T1-weighted 3D scan of almost every brain protocol. An inversion pulse flips the longitudinal magnetization; while it recovers, at a speed set by each tissue&#x27;s T1, a fast train of gradient echoes reads it out; then the magnetization recovers before the next inversion. White and grey matter recover at different speeds, and the train catches them while they differ.">

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


.. toctree::
   :hidden:

   /generated/gallery/01-course/01_first_gradient_echo
   /generated/gallery/01-course/02_echo_and_repetition_time
   /generated/gallery/01-course/03_spoiling
   /generated/gallery/01-course/04_labels_and_metadata
   /generated/gallery/01-course/05_radial_sampling
   /generated/gallery/01-course/06_radial_mprage
   /generated/gallery/01-course/07_hardware_and_safety_checks
   /generated/gallery/01-course/08_sequence_functions
   /generated/gallery/01-course/09_sequence_modules
   /generated/gallery/01-course/10_custom_sequence_module

