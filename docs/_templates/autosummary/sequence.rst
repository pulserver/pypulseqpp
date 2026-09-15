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

Definitions and labels
----------------------

.. autosummary::
   :toctree:
   :nosignatures:

   ~Sequence.set_definition
   ~Sequence.get_definition
   ~Sequence.copy_definitions
   ~Sequence.evaluate_labels
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

Waveforms and k-space
---------------------

.. autosummary::
   :toctree:
   :nosignatures:

   ~Sequence.waveforms
   ~Sequence.waveforms_and_times
   ~Sequence.get_gradients
   ~Sequence.calculate_kspace
   ~Sequence.adc_times
   ~Sequence.rf_times
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
