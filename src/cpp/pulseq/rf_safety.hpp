/**
 * @file rf_safety.hpp
 * @brief RF power of a block range, and time-averaged VOP SAR per repetition.
 */

#ifndef PULSEQ_RF_SAFETY_HPP
#define PULSEQ_RF_SAFETY_HPP

#include <complex>
#include <vector>

#include "pulseq/sequence.hpp"

namespace pulseq
{

    /** MATLAB Pulseq's Sequence.calcRfPower, in Pulseq's Hz units. */
    struct RfPower
    {
        double mean_power = 0.0; /**< Hz^2 */
        double peak_power = 0.0; /**< Hz^2 */
        double rms = 0.0;        /**< Hz   */
        double energy = 0.0;     /**< Hz^2 s */
    };

    /**
     * RF power over blocks @p first to @p last, 1-based and inclusive.
     *
     * Each pulse is resampled at the midpoints of a @p dt grid over its shape
     * duration and zero outside its samples, as mr.calcRfPower does; a
     * dynamic pTx pulse sums |b_c|^2 over its channels at each time. Mean
     * power and rms divide by the blocks' total duration. With @p window > 0,
     * energy and rms are the largest over runs of whole blocks, a run being
     * shortened from its front until it is no longer than @p window, and both
     * are divided by @p window, as MATLAB does.
     */
    RfPower rf_power(const Sequence& seq, int first, int last, double window, double dt);

    /** VOPs, and how a pulse drives the channels they are written for. */
    struct SarModel
    {
        int channels = 0;
        /** VOP-major, row-major: vops[(k * channels + i) * channels + j]. */
        std::vector<std::complex<double>> vops;
        /** Global SAR matrix, row-major; empty when there is none. */
        std::vector<std::complex<double>> global;
        /** Channel drive per Hz of RF amplitude, per channel. */
        std::vector<double> drive;
        /** Channel weights of a single-channel pulse played without a shim. */
        std::vector<std::complex<double>> default_shim;
        /** Resampling step, as for rf_power. */
        double dt = 1e-6;
        /** Per-VOP SAR each window is compared with; empty for none. */
        std::vector<double> reference;
    };

    /** One averaging window: a prologue, one repetition, a tail, or everything. */
    struct SarWindow
    {
        int first = 0;
        int last = 0;
        double duration = 0.0;
        /** Largest VOP SAR over the window, in the VOPs' units, and which VOP. */
        double local = 0.0;
        int vop = -1;
        double global = 0.0;
        /**
         * Largest over VOPs of this window's SAR over the reference's for that
         * VOP, and which VOP; infinite where a VOP the reference leaves cold
         * is heated.
         */
        double ratio = 0.0;
        int ratio_vop = -1;
    };

    struct SarReport
    {
        std::vector<SarWindow> windows;
        /** Per-VOP SAR of the window with the largest local SAR. */
        std::vector<double> worst;
    };

    /**
     * Time-averaged SAR of each window of the sequence.
     *
     * The windows are the blocks before the first full repetition, each
     * repetition of @p size blocks from @p start, and the blocks after the
     * last; or the whole sequence when @p size is 0. A pulse drives channel c
     * with drive_c * s_c * b_c(t): b is its per-channel waveform in Hz (one
     * channel repeated on every channel for a single-channel pulse), and s its
     * block's RF shim, the default shim for a single-channel pulse without
     * one, or ones. Each VOP Q_k contributes v^H Q_k v integrated over the
     * window and divided by its duration.
     *
     * @throws std::invalid_argument  When a pulse or shim has a channel count
     *                                other than the model's, or than one.
     */
    SarReport vop_sar(const Sequence& seq, const SarModel& model, int size, int start);

} // namespace pulseq

#endif /* PULSEQ_RF_SAFETY_HPP */
