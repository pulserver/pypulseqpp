# Pulseq basics

Authoring and inspecting a sequence at the level of the file format.

The first page builds a two-dimensional gradient-echo sequence from individual
events and blocks, and writes it as a `.seq` file. The second reads a sequence
back and runs the analyses and hardware checks that decide whether it can be
played: the k-space trajectory it samples, its timing against the rasters, and
its gradient waveforms against amplitude, slew-rate, nerve-stimulation and
mechanical-resonance limits.
