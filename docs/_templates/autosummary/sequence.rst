{{ name | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

Blocks
------

.. autosummary::
   :toctree:
   :nosignatures:

   ~Sequence.add_block
   ~Sequence.set_block
   ~Sequence.get_block
   ~Sequence.find_block_by_time
   ~Sequence.remove_duplicates
   ~Sequence.repetition
   ~Sequence.block_rotations
   ~Sequence.block_shims

Definitions and labels
----------------------

.. autosummary::
   :toctree:
   :nosignatures:

   ~Sequence.set_definition
   ~Sequence.get_definition
   ~Sequence.copy_definitions
   ~Sequence.evaluate_labels
   ~Sequence.label_blocks
   ~Sequence.add_trid
   ~Sequence.get_or_create_trid_id

Soft delays
-----------

.. autosummary::
   :toctree:
   :nosignatures:

   ~Sequence.apply_soft_delay
   ~Sequence.get_default_soft_delay_values

Reading and writing
-------------------

.. autosummary::
   :toctree:
   :nosignatures:

   ~Sequence.read
   ~Sequence.write
   ~Sequence.write_binary
   ~Sequence.write_v141
   ~Sequence.libraries

Waveforms and k-space
---------------------

.. autosummary::
   :toctree:
   :nosignatures:

   ~Sequence.waveforms
   ~Sequence.waveforms_and_times
   ~Sequence.get_gradients
   ~Sequence.sound
   ~Sequence.calculate_kspace
   ~Sequence.adc_kspace
   ~Sequence.adc_echoes
   ~Sequence.adc_times
   ~Sequence.rf_times
   ~Sequence.rf_gradients
   ~Sequence.duration

Checks and reports
------------------

.. autosummary::
   :toctree:
   :nosignatures:

   ~Sequence.check_timing
   ~Sequence.test_report
   ~Sequence.test_report_dict
   ~Sequence.calc_rf_power
   ~Sequence.rf_flip_angles
   ~Sequence.rf_channels
   ~Sequence.gradient_statistics

Gradient edits
--------------

.. autosummary::
   :toctree:
   :nosignatures:

   ~Sequence.flip_grad_axis
   ~Sequence.mod_grad_axis

Plotting
--------

.. autosummary::
   :toctree:
   :nosignatures:

   ~Sequence.plot
   ~Sequence.paper_plot

{% if module + "." + objname in gallery_backreferences %}
.. minigallery:: {{ module }}.{{ objname }}
   :add-heading: Examples using ``{{ objname }}``
{% endif %}
