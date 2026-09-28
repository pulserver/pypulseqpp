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
   ProtocolParameter

.. currentmodule:: pypulseqpp.cli

.. autosummary::
   :toctree: generated
   :nosignatures:

   run

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
   align
   block_to_events
   calc_absolute_offsets
   calc_duration
   rotate
   get_supported_labels
   enable_trace
   disable_trace
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
   add_gradients
   calc_ramp
   concatenate_gradients
   points_to_waveform
   rotate_3d
   scale_grad
   split_gradient
   split_gradient_at
   gradient_sound

.. currentmodule:: pypulseqpp.io

.. autosummary::
   :toctree: generated
   :nosignatures:

   read
   read_chain
   write
   SequenceLibraries
   Shape

.. currentmodule:: pypulseqpp.sequences

.. autosummary::
   :toctree: generated
   :nosignatures:

   SequenceModule
   RfModule
   NonSelectiveExcitation
   NonSelectiveRefocusing
   SpatialSelectiveExcitation
   SpatialSelectiveRefocusing
   SpatialSelective2DExcitation
   FrequencySelectiveExcitation
   SpspExcitation
   SmsExcitation
   MultibandExcitation
   InversionPreparation
   BlochSiegertPreparation
   DiffusionPreparation
   FatSaturation
   IhMtPreparation
   MtPreparation
   OffResonanceSaturation
   T1T2Preparation
   T2Preparation
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
   SAMPLING
   MAGNITUDE
   SIGNED
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
   make_slr_pulse
   make_recursive_slr_pulses
   make_sms_pulse
   make_spsp_pulse
   make_2d_selective_pulse
   make_gslider_pulse
   make_hadamard_pulse
   make_pins_pulse
   make_b1_selective_pulse
   make_b1_gslider_pulse
   make_b1_hadamard_pulse
   make_bloch_siegert_pulse
   make_ptx_pulse
   split_ptx_pulse
   calc_rf_shim
   make_spokes_pulse
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
   check_mech_resonance
   mech_resonance_spectrum
   read_forbidden_bands
   ForbiddenBand
   check_pns
   read_safe_model
   ChronaxieModel
   check_sar
   read_vops
   example_vops
   VopModel

.. currentmodule:: pypulseqpp

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_cartesian_axis_sampling
   make_cartesian_plane_sampling
   make_caipirinha_mask
   make_poisson_disc_mask
   make_random_mask
   make_traversal_order
   make_linear_order
   make_centric_order
   make_radial_order
   make_radial_adaptive_order
   make_shuffling_order
   make_epi_shot_offsets
   calc_uniform_angles
   calc_golden_angles
   calc_tiny_golden_angles
   calc_raga_angles
   calc_projection_shell
   make_rf_spoiling_schedule
   make_phase_cycling_schedule
   make_traps_schedule

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/sequence.rst

   Sequence

.. autosummary::
   :toctree: generated
   :nosignatures:

   TransformFOV
   Opts
   default_system
   apply_system_derates
   cap_system
   MAX_GRAD_DERATE
   MAX_SLEW_DERATE

.. autosummary::
   :toctree: generated
   :nosignatures:

   Isochromats

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

