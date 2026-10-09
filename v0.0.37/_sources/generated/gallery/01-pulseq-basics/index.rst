

.. _sphx_glr_generated_gallery_01-pulseq-basics:

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

    <div class="sphx-glr-thumbcontainer" tooltip="The smallest complete Pulseq sequence is a pulse-acquire experiment: one excitation pulse followed by one acquisition window. This first lesson of the course builds it and introduces the objects every later lesson extends: the system limits against which a factory designs an event, the RF and ADC events, the blocks in which events are played, the timing check, and the .seq file. A Bloch simulation of the stored pulse relates the transverse magnetisation to the flip angle.">

.. only:: html

  .. image:: /generated/gallery/01-pulseq-basics/images/thumb/sphx_glr_01_fid_thumb.png
    :alt:

  :doc:`/generated/gallery/01-pulseq-basics/01_fid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Free induction decay</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The previous lesson acquired a free induction decay directly after the excitation. This lesson adds a refocusing pulse, so that the acquisition is centred on a spin echo at a prescribed echo time, and a pair of crusher gradients about the refocusing pulse. The crushers leave the spin-echo pathway rephased and dephase the coherence pathways the pulse does not refocus, such as the free induction decay an imperfect refocusing pulse produces.">

.. only:: html

  .. image:: /generated/gallery/01-pulseq-basics/images/thumb/sphx_glr_02_spin_echo_thumb.png
    :alt:

  :doc:`/generated/gallery/01-pulseq-basics/02_spin_echo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Spin echo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The two previous lessons acquired signal without spatial encoding. This lesson turns the experiment into a two-dimensional acquisition: the excitation becomes slice-selective, the echo is formed by reversing a gradient rather than by a refocusing pulse, and a phase encode changes the acquired k-space line from one repetition to the next. Every Cartesian sequence in the later lessons extends this structure.">

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


.. toctree::
   :hidden:

   /generated/gallery/01-pulseq-basics/01_fid
   /generated/gallery/01-pulseq-basics/02_spin_echo
   /generated/gallery/01-pulseq-basics/03_gradient_echo

