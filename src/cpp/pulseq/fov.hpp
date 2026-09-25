/**
 * @file fov.hpp
 * @brief Channel-axis k-space integration and field-of-view transformations.
 *
 * Translations are in metres, along the channel axes or, through each block's
 * rotation, the logical axes; k-space coordinates are in 1/m.
 * Their dot product is a phase in cycles; event phase offsets and ADC
 * modulation are in radians, while RF phase shapes store cycles.
 */

#ifndef PULSEQ_FOV_HPP
#define PULSEQ_FOV_HPP

#include <array>
#include <cstdint>
#include <vector>

#include "pulseq/sequence.hpp"

namespace pulseq
{

    /**
     * Update the trajectory origin at an RF centre.
     *
     * Excitation (including undefined use) resets k to zero; refocusing reverses
     * k. Other uses leave the origin unchanged. @p at and @p origin are in 1/m.
     * @return Whether the origin changed.
     */
    bool advance_origin(char use, const double at[3], double origin[3]);

    /**
     * Return k-space at each block's start on the channel axes (1/m).
     *
     * Excitation resets and refocusing reverses k at the RF centre; undefined
     * RF use is treated as excitation. Block rotation extensions are not applied.
     *
     * @param first  First block, 1-based inclusive.
     * @param last   Last block inclusive, or 0 for the sequence end.
     * @param carry  Incoming k at @p first, updated in place to outgoing k at
     *               @p last. Earlier blocks are not integrated; initialise to
     *               zero at the sequence start and reuse across consecutive chunks.
     */
    std::vector<std::array<double, 3>> block_k_origins(
        const Sequence& seq, int first, int last, double carry[3]);

    /**
     * Return ADC-sampled k-space on the channel axes (1/m).
     *
     * @param block   1-based block index.
     * @param origin  Incoming k at the block start, as from block_k_origins().
     * @return Three sample vectors, or three empty vectors for a block without ADC.
     *         Block rotation extensions are not applied.
     */
    std::array<std::vector<double>, 3> absolute_trajectory(
        const Sequence& seq, int block, const double origin[3]);

    /**
     * Multiply gradient amplitudes by @p scale on each channel axis.
     *
     * FOV size varies inversely with the multiplier; zero suppresses encoding
     * on that axis. New event rows preserve blocks outside the selected range.
     * Shapes are unchanged. Block indices are 1-based inclusive; last=0 means end.
     */
    void apply_fov_scale(
        Sequence& seq, const double scale[3], int first, int last);

    /**
     * Compose the scalar-first @p quaternion after each block's existing rotation.
     *
     * The result is stored in a ROTATIONS extension; waveforms are not resampled.
     * Block indices are 1-based inclusive; last=0 means the sequence end.
     *
     * @param reflected_axis  Channel axis 0, 1 or 2 of an improper prescription
     *                M = R D, where R is @p quaternion and D negates that axis;
     *                -1 for a proper one. A block playing R_b g then plays
     *                M R_b g as the rotation R (D R_b D) of the gradient D g:
     *                the gradient on that channel axis is negated and the
     *                block's own rotation is conjugated by D before R is
     *                composed after it.
     */
    void apply_fov_rotation(
        Sequence& seq, const double quaternion[4], int first, int last,
        int reflected_axis = -1);

    /** Which events an FOV shift may modify. */
    enum class FovShiftScope
    {
        /**
         * Modify RF only; the consumer must apply ADC translation from the trajectory.
         */
        RfOnly,
        /**
         * Modify RF and ADC frequency, phase and modulation.
         */
        RfAndAdc,
    };

    /**
     * Apply a translation in metres to RF and, optionally, ADC events.
     *
     * Constant gradients require only frequency and phase offsets. Residual
     * phase under varying gradients is stored in RF phase shapes (cycles) or
     * ADC modulation (radians). RF phase is referenced to the pulse centre.
     * ADC phase is referenced to the nearest k-space approach, shared across
     * playouts of the same block/ADC definition within the selected range.
     *
     * A moved event is registered as a new row and the block repointed to it;
     * the row it named is left unchanged for the blocks that share it outside
     * the range or exempt. Blocks whose moved event is equal share one row.
     *
     * Phase uses the unbroken gradient integral in @p carry, not the
     * excitation-reset trajectory in @p origin. Resetting the phase integral
     * between excitation and readout would give them inconsistent references.
     * Fractional cycles are accumulated per segment to limit roundoff.
     *
     * @param first   First block, 1-based inclusive.
     * @param last    Last block inclusive, or 0 for the sequence end.
     * @param carry   Incoming unbroken gradient integral (1/m), updated in place.
     * @param origin  Incoming excitation/refocusing-aware k (1/m), updated in place.
     * @param exempt  One byte per selected block, nonzero to suppress edits while
     *                still advancing both integrals. Null exempts nothing.
     * @param through_rotation  Translate along the logical axes: move a block
     *                that carries a rotation R by the gradients it plays, R g,
     *                and advance both integrals by them, for a design whose
     *                rotations are its own. False moves it along its channel
     *                axes, by g, for a sequence whose rotations are a
     *                prescription composed onto it.
     *
     * Earlier blocks are not integrated. A zero translation returns without
     * advancing either integral.
     */
    void apply_fov_shift(
        Sequence& seq,
        const double shift_m[3],
        FovShiftScope scope,
        int first,
        int last,
        double carry[3],
        double origin[3],
        const unsigned char* exempt = nullptr,
        bool through_rotation = false);

    /**
     * The gradient each RF pulse plays under, one entry per block with RF, in
     * play order, along the channel axes.
     *
     * An axis is steady when its gradient holds one value from the pulse's
     * first sample to its last, which is when apply_fov_shift() moves the
     * pulse by a frequency and a phase offset alone; an axis without a
     * gradient is steady at zero.
     */
    struct RfGradients
    {
        /** 1-based block of each pulse. */
        std::vector<int32_t> block;
        /** Pulses x 3: 1 where the gradient along x, y or z is steady. */
        std::vector<uint8_t> steady;
        /** Pulses x 3: the gradient along x, y and z at the pulse's centre, in Hz/m. */
        std::vector<double> gradient;
    };

    /** Find the gradient each RF pulse plays under. */
    RfGradients rf_gradients(const Sequence& seq);

} // namespace pulseq

#endif /* PULSEQ_FOV_HPP */
