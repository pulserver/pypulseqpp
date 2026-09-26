/**
 * @file bloch.hpp
 * @brief Isochromats whose magnetisation the Bloch equation carries from one
 *        block to the next.
 *
 * The magnetisation turns about the field b = (Re b1, Im b1, bz), in Hz, in
 * the frame rotating at the reference frequency, as the magnetic moment of a
 * nucleus of positive gyromagnetic ratio precesses: dM/dt = 2 pi M x b, a
 * clockwise turn seen from the tip of b. Free precession, with a
 * piecewise-linear gradient, is integrated exactly. An RF pulse is a sequence
 * of steps holding b1 constant, each a rotation about the step's mean field
 * between two half steps of relaxation.
 */

#ifndef PULSEQ_BLOCH_HPP
#define PULSEQ_BLOCH_HPP

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace pulseq
{

    class GradientAreas;

    /** What each isochromat is; entry i of every vector describes isochromat i. */
    struct IsochromatProperties
    {
        /** Position, in m. */
        std::vector<double> x, y, z;
        /** Equilibrium longitudinal magnetisation. */
        std::vector<double> proton_density;
        /** Relaxation times, in s; infinity for none. */
        std::vector<double> t1, t2;
        /** Precession frequency at rest, in Hz from the reference frequency. */
        std::vector<double> off_resonance;
        /** Transmit sensitivities, isochromat-major, `transmit_channels` per
         *  isochromat. None: one channel of unit sensitivity. */
        size_t transmit_channels = 0;
        std::vector<std::complex<double>> transmit;
        /** Receive sensitivities, isochromat-major, `coils` per isochromat.
         *  None: one coil of unit sensitivity. */
        size_t coils = 0;
        std::vector<std::complex<double>> receive;
    };

    /** The events one block plays, timed in s from the block's start. */
    struct BlockEvents
    {
        double duration = 0.0;

        /** Per axis, the corners of a piecewise-linear gradient in Hz/m,
         *  zero outside them. */
        std::array<const double*, 3> gradient_times{};
        std::array<const double*, 3> gradient_values{};
        std::array<size_t, 3> gradient_corners{};

        /**
         * The transverse field of an RF pulse, in Hz: `rf_channels` rows of
         * `rf_steps` values, channel-major, each held for `rf_step` from
         * `rf_start` on. Without transmit sensitivities the channels are
         * summed; with them, the channels are theirs.
         */
        double rf_start = 0.0;
        double rf_step = 0.0;
        size_t rf_steps = 0;
        size_t rf_channels = 0;
        const std::complex<double>* rf = nullptr;

        /** Increasing ADC sample times, none inside the RF pulse. */
        const double* adc_times = nullptr;
        size_t adc_samples = 0;
    };

    /**
     * A set of isochromats and their magnetisation.
     *
     * Free precession is applied when the magnetisation is next needed -- by
     * an RF pulse, an ADC sample or a read -- so blocks without either cost
     * nothing per isochromat.
     */
    class Isochromats
    {
    public:
        /**
         * @param properties  What each isochromat is; every vector as long as
         *                    the positions, or empty for its default.
         * @param threads     Worker threads; 0 for every core.
         * @throws std::invalid_argument on vectors of the wrong length.
         */
        Isochromats(IsochromatProperties properties, size_t threads = 0);

        size_t size() const
        {
            return count_;
        }
        size_t coils() const
        {
            return coils_;
        }
        size_t transmit_channels() const
        {
            return properties_.transmit_channels;
        }

        /** The time played since construction or the last reset, in s. */
        double elapsed() const
        {
            const std::lock_guard<std::mutex> held(mutex_);
            return elapsed_;
        }

        /** Put every isochromat at equilibrium, along +z, and the clock at zero. */
        void reset();

        /** Write the magnetisation, (size(), 3) row-major, to @p into. */
        void magnetization(double* into);

        /** Replace the magnetisation with (size(), 3) row-major @p from. */
        void set_magnetization(const double* from);

        /**
         * Play @p block, writing what each coil receives at each ADC sample to
         * @p signal, coil-major: signal[c * adc_samples + k], the sum over the
         * isochromats of the receive sensitivity times Mx + i My.
         *
         * @throws std::invalid_argument on events the engine cannot play: an
         *         ADC sample inside the RF pulse, times out of order, or RF
         *         channels that differ in number from the transmit
         *         sensitivities.
         */
        void play(const BlockEvents& block, std::complex<double>* signal);

    private:
        /** Isochromats that see one field during an RF pulse. */
        struct Grouping
        {
            int mode = 0;
            double direction[3] = {0.0, 0.0, 0.0};
            std::vector<uint32_t> group_of;
            std::vector<uint32_t> representative;
        };

        void check();
        void lay_out_receive();
        void classify();
        void flush();
        void advance(const GradientAreas& areas, double& now, double to);
        void excite(const BlockEvents& block, const GradientAreas& areas);
        void acquire(
            const BlockEvents& block,
            const GradientAreas& areas,
            size_t first,
            size_t last,
            std::complex<double>* signal);
        const Grouping& grouping(int mode, const double direction[3]);

        IsochromatProperties properties_;
        size_t count_ = 0;
        size_t coils_ = 1;
        size_t threads_ = 1;

        std::vector<double> mx_, my_, mz_;
        /** Receive sensitivities, coil-major: entry c * size() + i. */
        std::vector<double> receive_re_, receive_im_;
        /** Classes of isochromats equal in everything but position. */
        std::vector<uint32_t> class_of_;

        /** Free precession not yet applied: gradient area in 1/m, and time. */
        double pending_area_[3] = {0.0, 0.0, 0.0};
        double pending_time_ = 0.0;
        double elapsed_ = 0.0;

        std::vector<Grouping> groupings_;

        /** Held by every call that reads or changes the magnetisation. */
        mutable std::mutex mutex_;
    };

} // namespace pulseq

#endif /* PULSEQ_BLOCH_HPP */
