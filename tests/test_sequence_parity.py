"""A sequence written as a function writes the files its SequenceApp form writes.

``tests/legacy/`` holds the SequenceApp form of each such sequence, unedited.
For the default protocol, for each protocol in ``NON_DEFAULT`` and for system
limits above and below the sequence's own, the files of the application's chain
of prescans and main sequence are the files of the function's result, byte for
byte. The function takes the same system and protocol as the application and
documents them identically; one that returns a chain describes the list in its
Returns and its Examples.
"""

import dataclasses
import importlib.util
import inspect
import re
from pathlib import Path

import pytest
from zoo import FUNCTIONS, LEGACY, SMALL, legacy_application

import pypulseqpp as pp
from pypulseqpp import sequences
from pypulseqpp.sequences._app import _split_sections

#: The sequences whose SequenceApp form is kept in ``tests/legacy``.
LEGACY_NAMES = sorted(path.stem for path in LEGACY.glob("*_sequence.py"))

#: Whether torchsim, which designs the trains of ``flip_modulation="optimized"``, is installed.
TORCHSIM = importlib.util.find_spec("torchsim") is not None

#: Protocols that exercise what the default one leaves out, per sequence. The
#: first of each changes every parameter; the others reach the packets, the
#: gating and the echo train a prescription of one slice or one echo does not.
NON_DEFAULT = {
    "gre2D_sequence": [
        {
            "fov_x": 0.24,
            "fov_y": 0.2,
            "n_x": 32,
            "n_y": 24,
            "n_slices": 3,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "flip_angle_deg": 20.0,
            "te": None,
            "tr": None,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "partial_fourier_x": 0.75,
            "partial_fourier_y": 0.75,
            "n_dummy": 2,
            "readout_oversampling": 1.0,
            "n_acs_y": 6,
        },
        # Seven slices dealt into packets of four and three.
        {
            "n_x": 32,
            "n_y": 16,
            "n_slices": 7,
            "slice_spacing": 1e-3,
            "te": 4e-3,
            "tr": 30e-3,
            "n_dummy": 3,
            "n_acs_y": 0,
        },
    ],
    "se2D_sequence": [
        {
            "fov_x": 0.24,
            "fov_y": 0.2,
            "n_x": 32,
            "n_y": 24,
            "n_slices": 3,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "te": None,
            "tr": None,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "partial_fourier_x": 0.75,
            "partial_fourier_y": 0.75,
            "n_dummy": 2,
            "readout_oversampling": 1.0,
            "n_acs_y": 6,
        },
        # Five slices dealt into packets of two, two and one.
        {"n_x": 32, "n_y": 16, "n_slices": 5, "te": 22e-3, "tr": 70e-3, "n_acs_y": 0},
    ],
    "gre_multiecho2D_sequence": [
        # A bipolar train, which reads even echoes backwards and takes a full echo.
        {
            "fov_x": 0.24,
            "fov_y": 0.2,
            "n_x": 32,
            "n_y": 24,
            "n_slices": 2,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "flip_angle_deg": 20.0,
            "te": 6e-3,
            "tr": None,
            "n_echoes": 3,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "partial_fourier_y": 0.75,
            "n_dummy": 2,
            "readout_oversampling": 1.0,
            "n_acs_y": 6,
            "flyback": False,
        },
        # A monopolar train with a partial echo and a wait after every echo but
        # the last, in packets of three and two.
        {
            "n_x": 32,
            "n_y": 16,
            "n_slices": 5,
            "n_echoes": 3,
            "flyback": True,
            "echo_spacing": 6e-3,
            "te": 4e-3,
            "tr": 60e-3,
            "partial_fourier_x": 0.75,
            "n_dummy": 2,
            "n_acs_y": 0,
        },
    ],
    "bssfp2D_sequence": [
        # Prospective gating at a TR above the shortest.
        {
            "fov_x": 0.28,
            "fov_y": 0.25,
            "n_x": 64,
            "n_y": 16,
            "n_slices": 2,
            "slice_thickness": 5e-3,
            "slice_spacing": 1e-3,
            "flip_angle_deg": 60.0,
            "tr": 5e-3,
            "readout_bandwidth_hz": 50e3,
            "ry": 2,
            "partial_fourier_y": 0.75,
            "n_phases": 3,
            "n_dummy": 2,
            "readout_oversampling": 2.0,
            "n_acs_y": 4,
            "gating": "prospective",
            "heart_rate_bpm": 300.0,
            "views_per_segment": 4,
            "trigger_delay": 5e-3,
        },
        {
            "n_x": 64,
            "n_y": 16,
            "n_slices": 2,
            "readout_bandwidth_hz": 50e3,
            "n_dummy": 2,
            "n_acs_y": 0,
            "gating": "retrospective",
            "heart_rate_bpm": 120.0,
            "views_per_segment": 4,
        },
        # Ungated, three slices, undersampled with a calibration block.
        {
            "n_x": 64,
            "n_y": 24,
            "n_slices": 3,
            "slice_spacing": 2e-3,
            "tr": 4e-3,
            "readout_bandwidth_hz": 50e3,
            "ry": 2,
            "n_acs_y": 6,
            "partial_fourier_y": 0.75,
            "n_dummy": 0,
        },
    ],
    "gre_radial2D_sequence": [
        # A TE above the shortest, undersampled, in one packet.
        {
            "fov": 0.24,
            "n": 32,
            "n_slices": 3,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "flip_angle_deg": 20.0,
            "te": 5e-3,
            "tr": None,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "n_dummy": 2,
            "readout_oversampling": 1.0,
        },
        # Seven slices dealt into packets of three, two and two.
        {"n": 32, "n_slices": 7, "te": None, "tr": 17e-3, "ry": 3, "n_dummy": 3},
        # A TR that holds one slice: three packets, no dummies.
        {"n": 16, "n_slices": 3, "te": None, "tr": 6e-3, "n_dummy": 0},
    ],
    "gre_spiral2D_sequence": [
        # A dual-density interleaf, every second one of four.
        {
            "fov": 0.24,
            "n": 32,
            "n_slices": 3,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "flip_angle_deg": 20.0,
            "te": 4e-3,
            "tr": None,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "n_dummy": 2,
            "n_shots": 4,
            "density": "dual",
            "periphery_undersampling": 3.0,
            "transition_speed": 8.0,
        },
        # A variable-density interleaf, every third one of six, in packets of
        # three and two.
        {
            "n": 32,
            "n_slices": 5,
            "n_shots": 6,
            "ry": 3,
            "density": "variable",
            "periphery_undersampling": 2.5,
            "te": None,
            "tr": 20e-3,
            "n_dummy": 3,
        },
        # An odd number of constant-density interleaves, two slices, no dummies.
        {"n": 24, "n_slices": 2, "n_shots": 5, "te": None, "tr": None, "n_dummy": 0},
    ],
    "gre_propeller2D_sequence": [
        # Blades of eight lines, every second of the seven that cover the disc.
        {
            "fov": 0.24,
            "n": 32,
            "n_slices": 3,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "flip_angle_deg": 20.0,
            "te": None,
            "tr": None,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "n_dummy": 2,
            "blade_width": 8,
        },
        # Blades of seven lines, an odd width, in packets of three and two.
        {
            "n": 32,
            "n_slices": 5,
            "blade_width": 7,
            "te": 4e-3,
            "tr": 25e-3,
            "ry": 2,
            "n_dummy": 3,
        },
        # Blades as wide as the matrix, which two orientations cover.
        {
            "n": 24,
            "n_slices": 2,
            "blade_width": 24,
            "te": None,
            "tr": None,
            "n_dummy": 0,
        },
    ],
    "se_radial2D_sequence": [
        # A TE above the shortest, which delays the refocusing pulse.
        {
            "fov": 0.24,
            "n": 32,
            "n_slices": 3,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "te": 20e-3,
            "tr": None,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "n_dummy": 2,
            "readout_oversampling": 1.0,
        },
        # A TE off the block raster, which is rounded up, in packets of three
        # and two.
        {
            "n": 32,
            "n_slices": 5,
            "te": 16.253e-3,
            "tr": 60e-3,
            "ry": 3,
            "n_dummy": 1,
        },
        # A TR that holds one slice: three packets, no dummies.
        {"n": 24, "n_slices": 3, "te": None, "tr": 15e-3, "n_dummy": 0},
    ],
    "se_spiral2D_sequence": [
        # A dual-density interleaf, every second one of four, with a delayed
        # refocusing pulse.
        {
            "fov": 0.24,
            "n": 32,
            "n_slices": 3,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "te": 20e-3,
            "tr": None,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "n_dummy": 2,
            "n_shots": 4,
            "density": "dual",
            "periphery_undersampling": 3.0,
            "transition_speed": 8.0,
        },
        # A variable-density interleaf at a TE off the block raster, in packets
        # of three and two.
        {
            "n": 32,
            "n_slices": 5,
            "n_shots": 6,
            "ry": 3,
            "density": "variable",
            "periphery_undersampling": 2.5,
            "te": 16.253e-3,
            "tr": 70e-3,
            "n_dummy": 1,
        },
        # An odd number of constant-density interleaves, two slices, no dummies.
        {"n": 24, "n_slices": 2, "n_shots": 5, "te": None, "tr": None, "n_dummy": 0},
    ],
    "se_propeller2D_sequence": [
        # Blades of eight lines, every second of the seven that cover the disc,
        # at the shortest TE the slower readout admits.
        {
            "fov": 0.24,
            "n": 32,
            "n_slices": 3,
            "slice_thickness": 4e-3,
            "slice_spacing": 1.5e-3,
            "te": None,
            "tr": None,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "n_dummy": 2,
            "blade_width": 8,
        },
        # Blades of seven lines at a TE off the block raster, in packets of
        # three and two.
        {
            "n": 32,
            "n_slices": 5,
            "blade_width": 7,
            "te": 16.253e-3,
            "tr": 70e-3,
            "ry": 2,
            "n_dummy": 1,
        },
        # Blades as wide as the matrix, which two orientations cover.
        {
            "n": 24,
            "n_slices": 2,
            "blade_width": 24,
            "te": 20e-3,
            "tr": 100e-3,
            "n_dummy": 0,
        },
    ],
    "gre_stack_of_stars3D_sequence": [
        # A spectral-spatial excitation, golden-angle partition shifts and a
        # calibration block at every tilt.
        {
            "fov": 0.24,
            "n": 32,
            "fov_z": 0.1,
            "n_z": 8,
            "flip_angle_deg": 20.0,
            "te": 9e-3,
            "tr": 20e-3,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "rz": 2,
            "partial_fourier_z": 0.75,
            "n_dummy": 2,
            "excitation": "spsp",
            "partition_angle_shift": "golden",
            "n_acs_z": 2,
            "readout_oversampling": 1.0,
        },
        # A hard pulse without a slab gradient, tiny golden-angle shifts, one
        # spoke in three and a TR above the shortest.
        {
            "n": 32,
            "n_z": 8,
            "te": 3e-3,
            "tr": 10e-3,
            "ry": 3,
            "rz": 2,
            "n_acs_z": 4,
            "n_dummy": 0,
            "excitation": "nonselective",
            "partition_angle_shift": "tiny_golden",
        },
        # Partial Fourier along the partitions, dummies and golden-angle shifts.
        {
            "n": 32,
            "n_z": 8,
            "partial_fourier_z": 0.75,
            "n_dummy": 3,
            "partition_angle_shift": "golden",
        },
    ],
    "se_stack_of_stars3D_sequence": [
        # An echo time off the block raster, which is rounded up.
        {
            "fov": 0.24,
            "n": 32,
            "fov_z": 0.1,
            "n_z": 8,
            "te": 24.013e-3,
            "tr": 40e-3,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "rz": 2,
            "partial_fourier_z": 0.75,
            "n_dummy": 2,
            "excitation": "spsp",
            "partition_angle_shift": "golden",
            "n_acs_z": 2,
            "readout_oversampling": 1.0,
        },
        # The shortest TE and TR, with a hard pulse and a calibration block.
        {
            "n": 32,
            "n_z": 8,
            "te": None,
            "tr": None,
            "ry": 3,
            "rz": 2,
            "n_acs_z": 4,
            "excitation": "nonselective",
            "partition_angle_shift": "tiny_golden",
        },
        # The default TR above the shortest, with dummies.
        {"n": 32, "n_z": 4, "n_dummy": 2},
    ],
    "gre_stack_of_spirals3D_sequence": [
        # Variable density, a spectral-spatial excitation and a calibration block
        # at every tilt.
        {
            "fov": 0.24,
            "n": 32,
            "fov_z": 0.1,
            "n_z": 8,
            "flip_angle_deg": 20.0,
            "te": 9e-3,
            "tr": 20e-3,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "rz": 2,
            "partial_fourier_z": 0.75,
            "n_dummy": 2,
            "excitation": "spsp",
            "partition_angle_shift": "golden",
            "n_acs_z": 2,
            "n_shots": 4,
            "density": "variable",
            "periphery_undersampling": 1.5,
            "transition_speed": 8.0,
        },
        # Dual density with a hard pulse and tiny golden-angle shifts.
        {
            "n": 32,
            "n_z": 8,
            "n_shots": 8,
            "density": "dual",
            "excitation": "nonselective",
            "partition_angle_shift": "tiny_golden",
            "rz": 2,
            "n_acs_z": 4,
            "n_dummy": 0,
        },
        # Constant density with a TR above the shortest and partial Fourier.
        {
            "n": 32,
            "n_z": 8,
            "n_shots": 4,
            "tr": 12e-3,
            "partial_fourier_z": 0.75,
            "n_dummy": 1,
        },
    ],
    "se_stack_of_spirals3D_sequence": [
        # An echo time off the block raster, which is rounded up.
        {
            "fov": 0.24,
            "n": 32,
            "fov_z": 0.1,
            "n_z": 8,
            "te": 24.013e-3,
            "tr": 40e-3,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "rz": 2,
            "partial_fourier_z": 0.75,
            "n_dummy": 2,
            "excitation": "spsp",
            "partition_angle_shift": "golden",
            "n_acs_z": 2,
            "n_shots": 4,
            "density": "variable",
            "periphery_undersampling": 1.5,
            "transition_speed": 8.0,
        },
        # The shortest TE and TR, with dual density and a hard pulse.
        {
            "n": 32,
            "n_z": 8,
            "te": None,
            "tr": None,
            "n_shots": 8,
            "density": "dual",
            "excitation": "nonselective",
            "partition_angle_shift": "tiny_golden",
            "rz": 2,
            "n_acs_z": 4,
        },
        # The default TR above the shortest, with dummies.
        {"n": 32, "n_z": 4, "n_shots": 4, "n_dummy": 2},
    ],
    "gre_stack_of_blades3D_sequence": [
        # A spectral-spatial excitation, golden-angle partition shifts and a
        # calibration block at every tilt.
        {
            "fov": 0.24,
            "n": 32,
            "fov_z": 0.1,
            "n_z": 8,
            "flip_angle_deg": 20.0,
            "te": 9e-3,
            "tr": 20e-3,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "rz": 2,
            "partial_fourier_z": 0.75,
            "n_dummy": 2,
            "excitation": "spsp",
            "partition_angle_shift": "golden",
            "n_acs_z": 2,
            "blade_width": 8,
        },
        # A hard pulse, four-line blades, tiny golden-angle shifts and one blade
        # in three.
        {
            "n": 32,
            "n_z": 8,
            "te": 3e-3,
            "blade_width": 4,
            "ry": 3,
            "rz": 2,
            "n_acs_z": 4,
            "n_dummy": 0,
            "excitation": "nonselective",
            "partition_angle_shift": "tiny_golden",
        },
        # A TR above the shortest, with dummies.
        {"n": 32, "n_z": 4, "blade_width": 8, "tr": 10e-3, "n_dummy": 3},
    ],
    "se_stack_of_blades3D_sequence": [
        # An echo time off the block raster, which is rounded up.
        {
            "fov": 0.24,
            "n": 32,
            "fov_z": 0.1,
            "n_z": 8,
            "te": 24.013e-3,
            "tr": 40e-3,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "rz": 2,
            "partial_fourier_z": 0.75,
            "n_dummy": 2,
            "excitation": "spsp",
            "partition_angle_shift": "golden",
            "n_acs_z": 2,
            "blade_width": 8,
        },
        # The shortest TE and TR, with a hard pulse and four-line blades.
        {
            "n": 32,
            "n_z": 8,
            "te": None,
            "tr": None,
            "blade_width": 4,
            "ry": 3,
            "rz": 2,
            "n_acs_z": 4,
            "excitation": "nonselective",
            "partition_angle_shift": "tiny_golden",
        },
        # The default TR above the shortest, with dummies.
        {"n": 32, "n_z": 4, "blade_width": 8, "n_dummy": 2},
    ],
    "zte3D_sequence": [
        {
            "fov": 0.24,
            "n": 24,
            "flip_angle_deg": 5.0,
            "tr": 4e-4,
            "readout_bandwidth_hz": 200e3,
            "r": 2,
            "n_shots": 4,
            "scheme": "meridian",
            "n_dummy": 1,
            "readout_oversampling": 1.0,
            "n_gain_calibration_readouts": 2,
        },
        # The shells left to the design, which balances the spacing within one
        # against the spacing between them.
        {"n": 16, "n_shots": None, "n_dummy": 0},
        # Meridian shells, one in two, with two dummy shells.
        {"n": 24, "n_shots": 4, "scheme": "meridian", "r": 2, "n_dummy": 2},
    ],
    "gre3D_sequence": [
        # A nonselective pulse, undersampling along both phase encodes with a
        # CAIPIRINHA shift, the whole grid with an elliptical calibration region,
        # and a wave on the partition encode.
        {
            "fov_x": 0.24,
            "fov_y": 0.2,
            "fov_z": 0.1,
            "n_x": 32,
            "n_y": 24,
            "n_z": 6,
            "flip_angle_deg": 20.0,
            "te": 6e-3,
            "tr": 16e-3,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "rz": 2,
            "caipi_shift": 1,
            "partial_fourier_x": 0.75,
            "partial_fourier_y": 0.75,
            "partial_fourier_z": 0.75,
            "n_dummy": 2,
            "excitation": "nonselective",
            "readout_oversampling": 1.0,
            "n_acs_y": 6,
            "n_acs_z": 4,
            "elliptical_sampling": False,
            "elliptical_acs": True,
            "wave": "partition",
            "wave_cycles": 3,
            "wave_amplitude": 8e-3,
        },
        # A spectral-spatial pulse, waves on both phase encodes with the
        # wave-free reference, and a TR above the shortest.
        {
            "n_x": 32,
            "n_y": 16,
            "n_z": 8,
            "excitation": "spsp",
            "n_dummy": 3,
            "ry": 2,
            "rz": 2,
            "n_acs_y": 4,
            "n_acs_z": 4,
            "wave": "both",
            "wave_cycles": 2,
            "wave_amplitude": 8e-3,
            "tr": 20e-3,
        },
        # The shortest TE and TR with a partial echo and a wave on the phase
        # encode, undersampled along the phase encode alone.
        {
            "n_x": 32,
            "n_y": 18,
            "n_z": 6,
            "ry": 3,
            "rz": 1,
            "n_acs_y": 6,
            "n_acs_z": 6,
            "partial_fourier_x": 0.75,
            "wave": "phase",
            "wave_cycles": 2,
            "wave_amplitude": 6e-3,
            "n_dummy": 0,
        },
    ],
    "se3D_sequence": [
        # A nonselective pulse, undersampling along both phase encodes with a
        # CAIPIRINHA shift, the whole grid with an elliptical calibration region,
        # and a wave on the partition encode, at a TE and a TR above the shortest.
        {
            "fov_x": 0.24,
            "fov_y": 0.2,
            "fov_z": 0.1,
            "n_x": 32,
            "n_y": 24,
            "n_z": 6,
            "te": 12e-3,
            "tr": 40e-3,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "rz": 2,
            "caipi_shift": 1,
            "partial_fourier_x": 0.75,
            "partial_fourier_y": 0.75,
            "partial_fourier_z": 0.75,
            "n_dummy": 2,
            "excitation": "nonselective",
            "readout_oversampling": 1.0,
            "n_acs_y": 6,
            "n_acs_z": 4,
            "elliptical_sampling": False,
            "elliptical_acs": True,
            "wave": "partition",
            "wave_cycles": 3,
            "wave_amplitude": 8e-3,
        },
        # A spectral-spatial pulse at the shortest TE and TR, with waves on both
        # phase encodes and the wave-free reference.
        {
            "n_x": 32,
            "n_y": 16,
            "n_z": 8,
            "excitation": "spsp",
            "te": None,
            "tr": None,
            "n_dummy": 1,
            "ry": 2,
            "rz": 2,
            "n_acs_y": 4,
            "n_acs_z": 4,
            "wave": "both",
            "wave_cycles": 2,
            "wave_amplitude": 8e-3,
        },
        # A partial echo and a wave on the phase encode, undersampled along the
        # phase encode alone.
        {
            "n_x": 32,
            "n_y": 18,
            "n_z": 6,
            "te": None,
            "tr": None,
            "ry": 3,
            "rz": 1,
            "n_acs_y": 6,
            "n_acs_z": 6,
            "partial_fourier_x": 0.75,
            "wave": "phase",
            "wave_cycles": 2,
            "wave_amplitude": 6e-3,
        },
    ],
    "bssfp3D_sequence": [
        # A slab excitation and two phase cycles, a catalyst and a train each,
        # undersampled along both phase encodes with a CAIPIRINHA shift.
        {
            "fov_x": 0.24,
            "fov_y": 0.2,
            "fov_z": 0.1,
            "n_x": 64,
            "n_y": 24,
            "n_z": 6,
            "flip_angle_deg": 60.0,
            "tr": 6e-3,
            "readout_bandwidth_hz": 50e3,
            "ry": 2,
            "rz": 2,
            "caipi_shift": 1,
            "partial_fourier_y": 0.75,
            "partial_fourier_z": 0.75,
            "excitation": "slab",
            "readout_oversampling": 1.0,
            "n_acs_y": 6,
            "n_acs_z": 4,
            "elliptical_acs": True,
            "n_phase_cycles": 2,
        },
        # Three phase cycles at a TR above the shortest.
        {"n_x": 64, "n_y": 16, "n_z": 4, "n_phase_cycles": 3, "tr": 4e-3},
        # One phase cycle at the shortest TR, undersampled along the phase
        # encode alone, with a partial partition extent.
        {
            "n_x": 64,
            "n_y": 20,
            "n_z": 6,
            "ry": 2,
            "n_acs_y": 6,
            "n_acs_z": 4,
            "partial_fourier_z": 0.75,
        },
    ],
    "gre_multiecho3D_sequence": [
        # A bipolar train, which reads even echoes backwards and takes a full echo,
        # at an echo spacing above the shortest, with a nonselective pulse,
        # undersampling along both phase encodes and a wave on the partition encode.
        {
            "fov_x": 0.24,
            "fov_y": 0.2,
            "fov_z": 0.1,
            "n_x": 32,
            "n_y": 24,
            "n_z": 6,
            "flip_angle_deg": 20.0,
            "te": 3e-3,
            "tr": 14e-3,
            "n_echoes": 3,
            "readout_bandwidth_hz": 200e3,
            "ry": 2,
            "rz": 2,
            "caipi_shift": 1,
            "partial_fourier_y": 0.75,
            "partial_fourier_z": 0.75,
            "n_dummy": 2,
            "excitation": "nonselective",
            "readout_oversampling": 1.0,
            "n_acs_y": 6,
            "n_acs_z": 4,
            "elliptical_sampling": False,
            "elliptical_acs": True,
            "wave": "partition",
            "wave_cycles": 3,
            "wave_amplitude": 8e-3,
            "echo_spacing": 2e-3,
            "flyback": False,
        },
        # A monopolar train with a partial echo and a wait after every echo but
        # the last, a spectral-spatial pulse and waves on both phase encodes.
        {
            "n_x": 32,
            "n_y": 16,
            "n_z": 8,
            "n_echoes": 3,
            "flyback": True,
            "echo_spacing": 5e-3,
            "te": 8e-3,
            "tr": 40e-3,
            "partial_fourier_x": 0.75,
            "excitation": "spsp",
            "n_dummy": 2,
            "ry": 2,
            "rz": 2,
            "n_acs_y": 4,
            "n_acs_z": 4,
            "wave": "both",
            "wave_cycles": 2,
            "wave_amplitude": 8e-3,
        },
        # A bipolar train of two echoes at the shortest spacing, with a wave on
        # the phase encode, undersampled along the phase encode alone.
        {
            "n_x": 32,
            "n_y": 18,
            "n_z": 6,
            "n_echoes": 2,
            "flyback": False,
            "ry": 3,
            "rz": 1,
            "n_acs_y": 6,
            "n_acs_z": 6,
            "wave": "phase",
            "wave_cycles": 2,
            "wave_amplitude": 6e-3,
            "n_dummy": 0,
        },
    ],
    "se_epi_propeller2D_sequence": [
        # Golden-angle blades, the slices in reverse order, a TE above the
        # shortest, and dummy blades.
        {
            "fov": 0.24,
            "n_x": 32,
            "blade_width": 8,
            "n_blades": 3,
            "angle_scheme": "golden",
            "n_slices": 3,
            "slice_thickness": 4e-3,
            "slice_gap": 1e-3,
            "slice_order": "reverse",
            "te": 50e-3,
            "tr": 400e-3,
            "readout_bandwidth_hz": 200e3,
            "crusher_cycles": 3.0,
            "n_dummy": 1,
            "n_gain_calibration_readouts": 5,
        },
        # Five slices dealt into passes of three and two, centre out.
        {
            "n_x": 32,
            "blade_width": 8,
            "n_slices": 5,
            "slice_order": "center_out",
            "te": 40e-3,
            "tr": 150e-3,
            "n_dummy": 2,
        },
        # The shortest TE, and a TR that holds four slices in one pass.
        {
            "n_x": 32,
            "blade_width": 8,
            "n_blades": 3,
            "n_slices": 4,
            "te": None,
            "tr": 100e-3,
        },
    ],
    "epi2D_sequence": [
        # Multiband and segmented, accelerated with calibration lines and
        # partial Fourier, with fat saturation, a time series and the volume
        # output.
        {
            "fov_x": 0.24,
            "fov_y": 0.2,
            "n_x": 32,
            "n_y": 24,
            "n_slices": 4,
            "slice_thickness": 4e-3,
            "slice_spacing": 1e-3,
            "flip_angle_deg": 60.0,
            "te": 9e-3,
            "tr": 90e-3,
            "n_frames": 2,
            "readout_bandwidth_hz": 250e3,
            "ry": 2,
            "partial_fourier_y": 0.75,
            "n_shots": 2,
            "multiband": 2,
            "fat_saturation": True,
            "n_dummy": 1,
            "readout_oversampling": 1.5,
            "n_acs_y": 8,
            "volume_output": True,
        },
        # Seven slices dealt into packets of three, two and two by a TR too
        # short for one, with calibration lines.
        {
            "n_x": 32,
            "n_y": 24,
            "n_slices": 7,
            "slice_spacing": 1e-3,
            "tr": 30e-3,
            "ry": 2,
            "n_acs_y": 8,
            "partial_fourier_y": 0.75,
        },
        # Three shots of three slices over two frames, with a TE above the
        # shortest and no dummies.
        {
            "n_x": 32,
            "n_y": 24,
            "n_slices": 3,
            "n_frames": 2,
            "n_shots": 3,
            "te": 12e-3,
            "tr": 170e-3,
            "n_dummy": 0,
        },
    ],
    "epi3D_sequence": [
        # A hard pulse, skipped-CAIPI shells under partial Fourier on both
        # axes with a calibration rectangle, a time series and the volume
        # output.
        {
            "fov_x": 0.24,
            "fov_y": 0.2,
            "fov_z": 0.08,
            "n_x": 32,
            "n_y": 24,
            "n_z": 8,
            "flip_angle_deg": 15.0,
            "te": 6e-3,
            "tr": 60e-3,
            "n_frames": 2,
            "readout_bandwidth_hz": 250e3,
            "ry": 2,
            "rz": 2,
            "partial_fourier_y": 0.75,
            "partial_fourier_z": 0.75,
            "n_shots": 2,
            "n_dummy": 1,
            "excitation": "nonselective",
            "readout_oversampling": 1.5,
            "n_acs_y": 8,
            "n_acs_z": 4,
            "volume_output": True,
        },
        # A spectral-spatial pulse and three-partition shells under partial
        # Fourier.
        {
            "n_x": 32,
            "n_y": 16,
            "n_z": 12,
            "rz": 3,
            "partial_fourier_z": 0.75,
            "excitation": "spsp",
            "n_dummy": 1,
            "n_acs_y": 8,
            "n_acs_z": 4,
        },
        # A 2x4 lattice with a CAIPI shift of two, in two shots over two
        # frames.
        {
            "n_x": 32,
            "n_y": 16,
            "n_z": 16,
            "ry": 2,
            "rz": 4,
            "n_shots": 2,
            "n_frames": 2,
            "n_acs_y": 8,
            "n_acs_z": 4,
        },
    ],
    "fse3D_sequence": [
        # Every parameter changed except the order, which individually
        # parameterized trains require to be radial: a hard-pulse train of
        # designed refocusing angles, shorter and faster at the periphery,
        # undersampled with a CAIPIRINHA shift, wave-encoded on one channel
        # and navigated.
        {
            "fov_x": 0.2,
            "fov_y": 0.18,
            "fov_z": 0.12,
            "n_x": 32,
            "n_y": 16,
            "n_z": 8,
            "te": 20e-3,
            "tr": 0.6,
            "etl": 8,
            "refocusing_angle_deg": 120.0,
            "readout_bandwidth_hz": 100e3,
            "ry": 2,
            "rz": 2,
            "caipi_shift": 1,
            "partial_fourier_x": 0.75,
            "partial_fourier_y": 0.75,
            "partial_fourier_z": 0.75,
            "n_dummy": 1,
            "excitation": "nonselective",
            "readout_oversampling": 1.0,
            "esp": 8e-3,
            "n_acs_y": 4,
            "n_acs_z": 2,
            "elliptical_acs": True,
            "flip_modulation": "optimized",
            "tr_periphery": 0.4,
            "etl_periphery": 5,
            "wave": "phase",
            "wave_cycles": 2,
            "wave_amplitude": 8e-3,
            "navigator": True,
        },
        # A designed train whose TE echo is past the fifth.
        {
            "n_x": 32,
            "n_y": 16,
            "n_z": 8,
            "etl": 10,
            "te": 70e-3,
            "esp": 10e-3,
            "tr": 0.4,
            "refocusing_angle_deg": 120.0,
            "flip_modulation": "optimized",
        },
        # Shuffled Poisson-disc views, with the calibration region acquired
        # again wave-free.
        {
            "n_x": 32,
            "n_y": 16,
            "n_z": 8,
            "etl": 8,
            "te": 20e-3,
            "tr": 0.3,
            "ry": 2,
            "rz": 2,
            "n_acs_y": 4,
            "n_acs_z": 2,
            "ordering": "shuffling",
            "wave": "both",
            "wave_cycles": 2,
            "wave_amplitude": 8e-3,
        },
        # The shortest TR and the first echo as TE, with dummy trains.
        {
            "n_x": 32,
            "n_y": 16,
            "n_z": 8,
            "etl": 6,
            "te": None,
            "tr": None,
            "ry": 2,
            "rz": 2,
            "caipi_shift": 1,
            "n_acs_y": 4,
            "n_acs_z": 4,
            "wave": "both",
            "wave_cycles": 2,
            "wave_amplitude": 8e-3,
            "n_dummy": 2,
        },
        # A periphery with longer trains and a longer TR than the centre.
        {
            "n_x": 32,
            "n_y": 16,
            "n_z": 8,
            "etl": 8,
            "te": 20e-3,
            "tr": 0.3,
            "etl_periphery": 12,
            "tr_periphery": 0.4,
        },
    ],
    "mprage3D_sequence": [
        # Every parameter changed: shuffled, undersampled, wave-encoded on one
        # channel, a hard pulse, and navigators in the recovery.
        {
            "fov_x": 0.2,
            "fov_y": 0.18,
            "fov_z": 0.12,
            "n_x": 32,
            "n_y": 16,
            "n_z": 8,
            "flip_angle_deg": 12.0,
            "te": 5e-3,
            "esp": 12e-3,
            "ti": 0.15,
            "tr": 1.0,
            "readout_bandwidth_hz": 100e3,
            "ry": 2,
            "rz": 2,
            "caipi_shift": 1,
            "partial_fourier_x": 0.75,
            "partial_fourier_y": 0.75,
            "partial_fourier_z": 0.75,
            "n_dummy": 2,
            "excitation": "nonselective",
            "readout_oversampling": 1.0,
            "n_acs_y": 4,
            "n_acs_z": 2,
            "elliptical_acs": True,
            "ordering": "shuffling",
            "wave": "partition",
            "wave_cycles": 2,
            "wave_amplitude": 8e-3,
            "navigator": True,
        },
        # The shortest TI and TR with a spectral-spatial pulse, a CAIPIRINHA
        # shift and an elliptical calibration region.
        {
            "n_x": 32,
            "n_y": 16,
            "n_z": 8,
            "excitation": "spsp",
            "ti": None,
            "tr": None,
            "ry": 2,
            "rz": 2,
            "caipi_shift": 1,
            "n_acs_y": 4,
            "n_acs_z": 4,
            "elliptical_acs": True,
            "partial_fourier_z": 0.75,
        },
        # Wave-CAIPI on both channels, with the calibration region acquired
        # again wave-free ahead of the image shots and no dummy shot.
        {
            "n_x": 32,
            "n_y": 16,
            "n_z": 8,
            "ti": 0.1,
            "tr": 0.3,
            "ry": 2,
            "n_acs_y": 4,
            "n_acs_z": 2,
            "wave": "both",
            "wave_cycles": 2,
            "wave_amplitude": 8e-3,
            "n_dummy": 0,
        },
        # Poisson-disc support, played in shuffled order.
        {
            "n_x": 32,
            "n_y": 16,
            "n_z": 8,
            "ti": 0.1,
            "tr": 0.3,
            "ordering": "shuffling",
            "ry": 2,
            "rz": 2,
            "n_acs_y": 4,
            "n_acs_z": 2,
        },
    ],
    "mprage_stack_of_spirals3D_sequence": [
        # Every parameter changed: a dual-density spiral under a hard pulse,
        # undersampled in angle and partition, with navigators.
        {
            "fov": 0.22,
            "n": 32,
            "fov_z": 0.12,
            "n_z": 8,
            "flip_angle_deg": 12.0,
            "te": 2e-3,
            "esp": 30e-3,
            "ti": 0.15,
            "tr": 1.5,
            "readout_bandwidth_hz": 125e3,
            "ry": 2,
            "rz": 2,
            "partial_fourier_z": 0.75,
            "n_dummy": 2,
            "excitation": "nonselective",
            "partition_angle_shift": "tiny_golden",
            "n_acs_z": 2,
            "n_shots": 8,
            "density": "dual",
            "periphery_undersampling": 3.0,
            "transition_speed": 8.0,
            "navigator": True,
        },
        # The shortest TI and TR, variable density, a spectral-spatial pulse
        # and no turn between partitions.
        {
            "n": 32,
            "n_z": 8,
            "n_shots": 4,
            "ti": None,
            "tr": None,
            "excitation": "spsp",
            "partition_angle_shift": "none",
            "density": "variable",
        },
        # Interleaves and partitions undersampled, with the central
        # partitions acquired as calibration.
        {
            "n": 32,
            "n_z": 8,
            "n_shots": 8,
            "ti": 0.1,
            "tr": 0.5,
            "ry": 2,
            "rz": 3,
            "n_acs_z": 2,
            "density": "dual",
            "n_dummy": 0,
        },
    ],
    "mprage_stack_of_stars3D_sequence": [
        # Every parameter changed: a hard pulse, angular and partition
        # undersampling, and navigators.
        {
            "fov": 0.22,
            "n": 32,
            "fov_z": 0.12,
            "n_z": 8,
            "flip_angle_deg": 12.0,
            "te": 2e-3,
            "esp": 8e-3,
            "ti": 0.15,
            "tr": 1.5,
            "readout_bandwidth_hz": 125e3,
            "ry": 3,
            "rz": 2,
            "partial_fourier_z": 0.75,
            "n_dummy": 2,
            "excitation": "nonselective",
            "partition_angle_shift": "tiny_golden",
            "n_acs_z": 2,
            "readout_oversampling": 1.0,
            "navigator": True,
        },
        # The shortest TI and TR with a spectral-spatial pulse and no turn
        # between partitions.
        {
            "n": 32,
            "n_z": 8,
            "ti": None,
            "tr": None,
            "ry": 2,
            "excitation": "spsp",
            "partition_angle_shift": "none",
        },
        # Spokes and partitions undersampled, with the central partitions
        # acquired as calibration.
        {
            "n": 32,
            "n_z": 8,
            "ti": 0.1,
            "tr": 0.5,
            "ry": 4,
            "rz": 3,
            "n_acs_z": 2,
            "n_dummy": 0,
        },
    ],
}

#: The sections a function that returns a chain documents for itself.
CHAIN_SECTIONS = ("Returns", "Examples")

#: ``(max_grad in mT/m, max_slew in T/m/s)`` of a system above the limits a
#: sequence is designed under, and of one below them.
SYSTEMS = {"above": (150, 400), "below": (30, 100)}


def hardware(max_grad, max_slew):
    return pp.Opts(
        max_grad=max_grad, grad_unit="mT/m", max_slew=max_slew, slew_unit="T/m/s"
    )


def cases():
    """``(name, limits, protocol)``: the default, each non-default, each system."""
    found = []
    for name in LEGACY_NAMES:
        found.append(pytest.param(name, None, {}, id=f"{name}-default"))
        for index, protocol in enumerate(NON_DEFAULT.get(name, ())):
            designed = protocol.get("flip_modulation") == "optimized"
            found.append(
                pytest.param(
                    name,
                    None,
                    protocol,
                    id=f"{name}-non-default-{index}",
                    marks=pytest.mark.skipif(
                        designed and not TORCHSIM, reason="needs torchsim"
                    ),
                )
            )
        for label, limits in SYSTEMS.items():
            found.append(
                pytest.param(name, limits, SMALL[name], id=f"{name}-{label}-limits")
            )
    return found


def application_chain(name, limits, protocol):
    """The prescans and the main sequence of the SequenceApp form, in play order."""
    system = None if limits is None else hardware(*limits)
    app = legacy_application(name)(system, **protocol)
    return [app.design(prescan) for prescan in app.prescans()] + [app.design()]


def function_chain(name, limits, protocol):
    """What the function returns, as a list of sequences."""
    system = None if limits is None else hardware(*limits)
    result = getattr(sequences, name).main(system, **protocol)
    return [result] if isinstance(result, pp.Sequence) else list(result)


def written(directory, chain):
    """``(file name, bytes)`` of each file ``sequences.write`` writes for ``chain``."""
    directory.mkdir()
    paths = sequences.write(directory / "scan.seq", chain)
    return [(Path(path).name, Path(path).read_bytes()) for path in paths]


def plays_prescans(name):
    """Whether the application of ``name`` has prescans, so its function returns a chain."""
    return legacy_application(name).prescans is not sequences.SequenceApp.prescans


def normalised(text):
    """``text`` with a role on a class attribute as the literal of a module constant.

    A reference to the application's ``prescans`` method is dropped: a function
    has no such method.
    """
    text = re.sub(r":attr:`([^`]+)`", r"``\1``", text)
    return re.sub(r" \(:meth:`prescans`\)", "", text)


def test_every_sequence_with_a_legacy_application_is_written_as_a_function():
    assert LEGACY_NAMES
    assert set(LEGACY_NAMES) <= set(FUNCTIONS)


def test_every_sequence_with_a_legacy_application_has_a_non_default_protocol():
    assert set(NON_DEFAULT) == set(LEGACY_NAMES)
    assert all(NON_DEFAULT[name] for name in LEGACY_NAMES)


@pytest.mark.parametrize(("name", "limits", "protocol"), cases())
def test_a_function_writes_the_files_its_application_writes(
    tmp_path, name, limits, protocol
):
    from_application = written(
        tmp_path / "application", application_chain(name, limits, protocol)
    )
    from_function = written(
        tmp_path / "function", function_chain(name, limits, protocol)
    )

    assert [file for file, _ in from_function] == [file for file, _ in from_application]
    for (file, made), (_, wanted) in zip(from_function, from_application, strict=True):
        assert made == wanted, file


@pytest.mark.parametrize(("name", "limits", "protocol"), cases())
def test_every_protocol_compared_is_a_design_that_passes_its_timing_check(
    name, limits, protocol
):
    for seq in function_chain(name, limits, protocol):
        is_ok, errors = seq.check_timing()
        assert is_ok, errors


@pytest.mark.parametrize("name", LEGACY_NAMES)
def test_a_function_takes_the_system_and_the_protocol_its_application_takes(name):
    shipped = inspect.signature(getattr(sequences, name).main, eval_str=True)
    legacy = inspect.signature(legacy_application(name).function())

    def parameters(signature):
        return [
            (p.name, p.kind, p.default, p.annotation)
            for p in signature.parameters.values()
        ]

    def protocol(function):
        return {
            key: dataclasses.replace(entry, description=normalised(entry.description))
            for key, entry in sequences.parameters(function).items()
        }

    assert parameters(shipped) == parameters(legacy)
    assert protocol(getattr(sequences, name).main) == protocol(
        legacy_application(name).function()
    )


@pytest.mark.parametrize("name", LEGACY_NAMES)
def test_a_function_documents_what_its_application_documents(name):
    """The protocol's own entries are compared through ``sequences.parameters``.

    A function that returns a chain documents the chain in its Returns and its
    Examples, where the application's ``main`` documents the main sequence.
    """
    shipped = _split_sections(inspect.getdoc(getattr(sequences, name).main))
    legacy = _split_sections(inspect.getdoc(legacy_application(name).main))
    chain = plays_prescans(name)

    assert shipped[0] == normalised(legacy[0])
    assert shipped[1] == normalised(legacy[1])
    assert [heading for heading, _ in shipped[2]] == [
        heading for heading, _ in legacy[2]
    ]
    for (heading, body), (_, wanted) in zip(shipped[2], legacy[2], strict=True):
        if heading != "Parameters" and not (chain and heading in CHAIN_SECTIONS):
            assert body == normalised(wanted), heading
    if chain:
        assert dict(shipped[2])["Returns"].startswith("list of pypulseqpp.Sequence\n")
