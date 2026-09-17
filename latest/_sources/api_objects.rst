:orphan:

API object index
================

The stub page of every documented object. The API pages carry the same
object lists as tables, each entry linking the stub this page writes;
writing them here keeps them out of the toctree the sidebar is built
from.

.. currentmodule:: pypulseqpp.sequences

.. autosummary::
   :toctree: generated
   :nosignatures:

   SequenceApp

.. currentmodule:: pypulseqpp.cli

.. autosummary::
   :toctree: generated
   :nosignatures:

   run
   write_sequence

.. currentmodule:: pypulseqpp

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_adc
   make_delay
   make_digital_output_pulse
   make_label
   make_rf_shim
   make_rotation
   make_soft_delay
   make_trigger

.. autosummary::
   :toctree: generated
   :nosignatures:

   align
   block_to_events
   calc_duration
   rotate

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_supported_labels
   enable_trace
   disable_trace

.. autosummary::
   :toctree: generated
   :nosignatures:

   interoperating
   convert
   as_namespace

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_arbitrary_grad
   make_crusher
   make_extended_trapezoid
   make_extended_trapezoid_area
   make_hexagon_gradient_area
   make_phase_blip
   make_phase_encoding
   make_trapezoid
   make_wave_gradients

.. autosummary::
   :toctree: generated
   :nosignatures:

   add_gradients
   calc_ramp
   concatenate_gradients
   points_to_waveform
   rotate_3d
   scale_grad
   split_gradient
   split_gradient_at

.. currentmodule:: pypulseqpp.sequences

.. autosummary::
   :toctree: generated
   :nosignatures:

   SequenceModule
   RfModule

.. autosummary::
   :toctree: generated
   :nosignatures:

   NonSelectiveExcitation
   NonSelectiveRefocusing
   SpatialSelectiveExcitation
   SpatialSelectiveRefocusing
   SpatialSelective2DExcitation
   FrequencySelectiveExcitation
   SpspExcitation
   SmsExcitation
   MultibandExcitation

.. autosummary::
   :toctree: generated
   :nosignatures:

   InversionPreparation
   BlochSiegertPreparation
   DiffusionPreparation
   FatSaturation
   IhMtPreparation
   MtPreparation
   OffResonanceSaturation
   T1T2Preparation
   T2Preparation

.. autosummary::
   :toctree: generated
   :nosignatures:

   LineReadout2D
   LineReadout3D
   EpiReadout2D
   EpiReadout3D
   FseReadout2D
   FseReadout3D
   BssfpReadout2D
   BssfpReadout3D
   PropellerReadout2D
   PropellerStackReadout

.. autosummary::
   :toctree: generated
   :nosignatures:

   NonCartesianReadout
   RadialReadout2D
   RadialStackReadout
   RadialProjectionReadout
   SpiralReadout2D
   SpiralStackReadout
   SpiralProjectionReadout
   SpiralNavigator
   RosetteReadout2D
   RosetteStackReadout
   RosetteProjectionReadout
   ZteReadout

.. autosummary::
   :toctree: generated
   :nosignatures:

   NonCartesianGradient
   Arbitrary
   Spiral
   Rosette

.. currentmodule:: pypulseqpp.plot

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot
   paper_plot

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_kspace
   plot_rf

.. currentmodule:: pypulseqpp

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_adiabatic_pulse
   make_arbitrary_rf
   make_block_pulse
   make_gauss_pulse
   make_half_passages
   make_sinc_pulse

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_slr_pulse
   make_recursive_slr_pulses
   make_sms_pulse
   make_spsp_pulse
   make_2d_selective_pulse

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_gslider_pulse
   make_hadamard_pulse
   make_pins_pulse

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_b1_selective_pulse
   make_b1_gslider_pulse
   make_b1_hadamard_pulse
   make_bloch_siegert_pulse

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_ptx_pulse
   split_ptx_pulse
   calc_rf_shim
   make_spokes_pulse

.. autosummary::
   :toctree: generated
   :nosignatures:

   calc_rf_bandwidth
   calc_rf_center
   calc_rf_power
   sim_bloch
   sim_rf

.. currentmodule:: pypulseqpp.safety

.. autosummary::
   :toctree: generated
   :nosignatures:

   check_max_grad
   check_max_slew
   check_grad_continuity

.. autosummary::
   :toctree: generated
   :nosignatures:

   check_mech_resonance
   read_forbidden_bands
   ForbiddenBand

.. autosummary::
   :toctree: generated
   :nosignatures:

   check_pns
   read_safe_model
   ChronaxieModel

.. autosummary::
   :toctree: generated
   :nosignatures:

   check_sar
   read_vops
   example_vops
   VopModel

.. currentmodule:: pypulseqpp

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/sequence.rst

   Sequence

.. autosummary::
   :toctree: generated
   :nosignatures:

   TransformFOV

.. autosummary::
   :toctree: generated
   :nosignatures:

   Opts
   default_system
   apply_system_derates
   cap_system
   MAX_GRAD_DERATE
   MAX_SLEW_DERATE

.. autosummary::
   :toctree: generated
   :nosignatures:

   calc_adc_segments
   calc_adc_timing
   quantize_readout_timing
   round_to_raster
   ceil_to_raster

.. autosummary::
   :toctree: generated
   :nosignatures:

   calc_radial_trajectory
   calc_spiral_trajectory
   calc_rosette_trajectory
   traj_to_grad

