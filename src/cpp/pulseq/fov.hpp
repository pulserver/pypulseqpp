/**
 * @file fov.hpp
 * @brief Logical-frame k-space integration and field-of-view transformations.
 *
 * Translations are in logical metres and k-space coordinates in 1/m.
 * Their dot product is a phase in cycles; event phase offsets and ADC
 * modulation are in radians, while RF phase shapes store cycles.
 */

#ifndef PULSEQ_FOV_HPP
#define PULSEQ_FOV_HPP

#include <array>
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
     * Return k-space at each block's start in the unrotated logical frame (1/m).
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
     * Return ADC-sampled k-space in the unrotated logical frame (1/m).
     *
     * @param block   1-based block index.
     * @param origin  Incoming k at the block start, as from block_k_origins().
     * @return Three sample vectors, or three empty vectors for a block without ADC.
     *         Block rotation extensions are not applied.
     */
    std::array<std::vector<double>, 3> absolute_trajectory(
        const Sequence& seq, int block, const double origin[3]);

    /**
     * Multiply gradient amplitudes by @p scale on each logical axis.
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
     */
    void apply_fov_rotation(
        Sequence& seq, const double quaternion[4], int first, int last);

    /** What a shift is allowed to write on. */
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
     * Apply translation in logical metres to RF and, optionally, ADC events.
     *
     * Constant gradients require only frequency and phase offsets. Residual
     * phase under varying gradients is stored in RF phase shapes (cycles) or
     * ADC modulation (radians). RF phase is referenced to the pulse centre.
     * ADC phase is referenced to the nearest k-space approach, shared across
     * playouts of the same block/ADC definition within the selected range.
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
        const unsigned char* exempt = nullptr);

} // namespace pulseq

#endif /* PULSEQ_FOV_HPP */
