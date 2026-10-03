/**
 * @file pns.cpp
 * @brief SAFE and chronaxie nerve responses.  See pns.hpp.
 *
 * The SAFE response follows the Python translation of
 * https://github.com/filip-szczepankiewicz/safe_pns_prediction that upstream
 * PyPulseq ships (BSD 3-Clause, Copyright (c) 2018 Filip Szczepankiewicz and
 * Thomas Witzel; LICENSES/safe_pns_prediction-BSD-3-Clause.txt). Its
 * truncated convolution with (1 - alpha)^k is the recursion carried here.
 */

#include "pulseq/pns.hpp"

#include "pulseq/raster.hpp"

#include <algorithm>
#include <cmath>

#if defined(__SSE__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace pulseq
{

    namespace
    {

        /** Chronaxies of c / (c + t)^2 kept before the tail is cut. */
        constexpr double kChronaxieKernelSpan = 20.0;

        /** Samples read per pass over the raster. */
        constexpr int64_t kChunk = 4096;

        /**
         * Subnormal results flushed to zero for the guard's lifetime, on x86.
         *
         * Filter memory decaying through a silent stretch spends thousands of
         * samples as subnormal numbers, each step of which costs many times a
         * normal one. Values below 2.2e-308 carry nothing a response needs.
         */
        class FlushSubnormals
        {
        public:
#if defined(__SSE__) || defined(_M_X64)
            FlushSubnormals() : saved_(_mm_getcsr()) { _mm_setcsr(saved_ | kFtzDaz); }
            ~FlushSubnormals() { _mm_setcsr(saved_); }

        private:
            static constexpr unsigned int kFtzDaz = 0x8040u;
            unsigned int saved_;
#endif
        };

        /** Three RC low-passes per axis, from rest. */
        class Safe
        {
        public:
            /** Its memory never ends, so a silent stretch is read sample by sample. */
            static constexpr bool kEnds = false;

            Safe(const std::array<SafeAxis, 3>& axes, double dt) : axes_(axes)
            {
                const double dt_ms = dt * 1e3;
                for (int axis = 0; axis < 3; ++axis)
                    for (int stage = 0; stage < 3; ++stage)
                        alpha_[axis][stage] = dt_ms / (axes[axis].tau_ms[stage] + dt_ms);
            }

            double respond(int axis, double slew)
            {
                const double* alpha = alpha_[axis];
                double* state = state_[axis];
                state[0] = alpha[0] * slew + (1.0 - alpha[0]) * state[0];
                state[1] = alpha[1] * std::fabs(slew) + (1.0 - alpha[1]) * state[1];
                state[2] = alpha[2] * slew + (1.0 - alpha[2]) * state[2];
                const SafeAxis& c = axes_[axis];
                return (c.a[0] * std::fabs(state[0]) + c.a[1] * state[1] +
                        c.a[2] * std::fabs(state[2])) /
                    c.stim_limit * c.g_scale;
            }

        private:
            std::array<SafeAxis, 3> axes_;
            double alpha_[3][3];
            double state_[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        };

        /**
         * Quadrature of c / (c + t)^2 = c \int s e^{-s (c + t)} ds over u = ln(s c).
         *
         * Each node is one exponential, so the kernel integrated over a raster
         * interval is a sum of first-order recursions. In u the nodes do not
         * depend on c or the raster; over 20 chronaxies the sum is within
         * 1e-5 of the kernel's unit area, in total absolute error per tap.
         */
        constexpr double kNodeFirst = -7.0;
        constexpr double kNodeStep = 0.6;
        constexpr int kNodes = 17;

        /** The chronaxie kernel over each axis's slew history, cut after its span. */
        class Chronaxie
        {
        public:
            /** The response is exactly zero once a span of samples has no slew. */
            static constexpr bool kEnds = true;

            Chronaxie(double chronaxie, const double (&rheobase)[3],
                      const double (&alpha)[3], double dt)
            {
                length_ = static_cast<size_t>(kChronaxieKernelSpan * chronaxie / dt) + 1;
                for (int k = 0; k < kNodes; ++k)
                {
                    const double s =
                        std::exp(kNodeFirst + kNodeStep * static_cast<double>(k)) / chronaxie;
                    decay_[k] = std::exp(-s * dt);
                    weight_[k] = chronaxie * kNodeStep * s * std::exp(-s * chronaxie) *
                        -std::expm1(-s * dt);
                    /* What a slew contributes once it is `length_` samples old,
                     * subtracted then so the kernel ends where the cut is. */
                    expired_[k] = weight_[k] * std::pow(decay_[k], static_cast<double>(length_));
                }
                /* Powers of each decay up to the span, for silent stretches. */
                powers_.resize(static_cast<size_t>(kNodes) * (length_ + 1));
                for (int k = 0; k < kNodes; ++k)
                {
                    double power = 1.0;
                    for (size_t d = 0; d <= length_; ++d, power *= decay_[k])
                        powers_[static_cast<size_t>(k) * (length_ + 1) + d] = power;
                    expired_sum_ += expired_[k];
                }
                for (int axis = 0; axis < 3; ++axis)
                {
                    scale_[axis] = alpha[axis] / rheobase[axis];
                    history_[axis].assign(length_, 0.0);
                }
            }

            double respond(int axis, double slew)
            {
                std::vector<double>& history = history_[axis];
                size_t& at = at_[axis];
                const double oldest = history[at];
                history[at] = slew;
                at = at + 1 == length_ ? 0 : at + 1;
                double* state = state_[axis];
                double sum = 0.0;
                for (int k = 0; k < kNodes; ++k)
                {
                    state[k] = decay_[k] * state[k] + weight_[k] * slew - expired_[k] * oldest;
                    sum += state[k];
                }
                held_[axis] += std::fabs(slew) - std::fabs(oldest);
                return std::fabs(sum) * scale_[axis];
            }

            /**
             * Upper bound on the response at every later sample while the slew is zero.
             *
             * Without the cut, the response is the positive states' sum less
             * the negative states' magnitude, each decaying, so neither
             * exceeds its present value. What the cut takes away is at most
             * the expiry weights times the slew magnitude still held.
             */
            double quiet_bound(int axis) const
            {
                double positive = 0.0;
                double negative = 0.0;
                for (int k = 0; k < kNodes; ++k)
                {
                    if (state_[axis][k] > 0.0)
                        positive += state_[axis][k];
                    else
                        negative -= state_[axis][k];
                }
                const double cut = expired_sum_ * std::max(held_[axis], 0.0);
                return (std::max(positive, negative) + cut) * scale_[axis] * (1.0 + 1e-12);
            }

            /** Samples of zero slew after which the response is zero. */
            int64_t span() const
            {
                return static_cast<int64_t>(length_);
            }

            /** Rest, as after a span of zero slew. */
            void clear()
            {
                for (int axis = 0; axis < 3; ++axis)
                {
                    std::fill(history_[axis].begin(), history_[axis].end(), 0.0);
                    std::fill(state_[axis], state_[axis] + kNodes, 0.0);
                    held_[axis] = 0.0;
                }
            }

            /**
             * The filter after @p count samples of zero slew, fewer than the span,
             * without their responses: only the slews leaving the history cost work.
             */
            void rest(int64_t count)
            {
                const size_t m = static_cast<size_t>(count);
                const size_t stride = length_ + 1;
                for (int axis = 0; axis < 3; ++axis)
                {
                    std::vector<double>& history = history_[axis];
                    size_t& at = at_[axis];
                    double leaving[kNodes] = {};
                    for (size_t j = 0; j < m; ++j)
                    {
                        const double oldest = history[at];
                        if (oldest != 0.0)
                        {
                            history[at] = 0.0;
                            held_[axis] -= std::fabs(oldest);
                            for (int k = 0; k < kNodes; ++k)
                                leaving[k] += oldest * powers_[static_cast<size_t>(k) * stride + m - 1 - j];
                        }
                        at = at + 1 == length_ ? 0 : at + 1;
                    }
                    for (int k = 0; k < kNodes; ++k)
                        state_[axis][k] = powers_[static_cast<size_t>(k) * stride + m] * state_[axis][k] -
                            expired_[k] * leaving[k];
                }
            }

        private:
            size_t length_ = 0;
            double decay_[kNodes];
            double weight_[kNodes];
            double expired_[kNodes];
            double state_[3][kNodes] = {};
            std::vector<double> powers_;
            double expired_sum_ = 0.0;
            /** Sum of |slew| over each axis's history. */
            double held_[3] = {0.0, 0.0, 0.0};
            double scale_[3] = {0.0, 0.0, 0.0};
            std::vector<double> history_[3];
            size_t at_[3] = {0, 0, 0};
        };

        void note(PnsPeak& peak, double value, double time, int block)
        {
            if (value > peak.value)
            {
                peak.value = value;
                peak.time = time;
                peak.block = block;
            }
        }

        template <typename Model>
        class Run
        {
        public:
            Run(PhysicalRaster& raster, Model& model, const PnsOptions& options, PnsReport& out)
                : raster_(raster), model_(model), options_(options), out_(out),
                  dt_(raster.raster()), to_slew_(1.0 / (dt_ * options.gamma))
            {
                for (int axis = 0; axis < 3; ++axis)
                    samples_[axis].resize(static_cast<size_t>(kChunk));
            }

            void operator()()
            {
                if constexpr (Model::kEnds)
                {
                    if (!options_.keep_trace)
                    {
                        skipping();
                        return;
                    }
                }
                while (read(kChunk) > 0)
                {
                }
                out_.samples = n_;
            }

        private:
            void skipping()
            {
                bool steady = false;
                int64_t ahead;
                while ((ahead = raster_.run(&steady)) > 0)
                {
                    if (!steady || ahead == 1)
                        read(std::min(ahead, kChunk));
                    else
                    {
                        /* The first sample may step; the rest repeat it. */
                        read(1);
                        quiet(ahead - 1);
                    }
                }
                out_.samples = n_;
            }

            /**
             * Zero slew for @p count samples. Skipped where the response cannot
             * pass a peak already found, and otherwise read up to the kernel's
             * end; past that end the filter is at rest.
             */
            void quiet(int64_t count)
            {
                const int64_t span = model_.span();
                double squared = 0.0;
                bool below = true;
                for (int axis = 0; axis < 3; ++axis)
                {
                    const double bound = model_.quiet_bound(axis);
                    squared += bound * bound;
                    below = below && bound <= out_.axes[static_cast<size_t>(axis)].value;
                }
                below = below && std::sqrt(squared) <= out_.norm.value;
                if (count < span)
                {
                    if (!below)
                    {
                        read_all(count);
                        return;
                    }
                    model_.rest(count);
                }
                else
                {
                    const int64_t read_through = below ? 0 : span;
                    read_all(read_through);
                    model_.clear();
                    count -= read_through;
                }
                raster_.skip(count);
                n_ += count;
            }

            void read_all(int64_t count)
            {
                while (count > 0)
                    count -= read(std::min(count, kChunk));
            }

            int64_t read(int64_t count)
            {
                const int64_t got =
                    raster_.read(count, samples_[0].data(), samples_[1].data(), samples_[2].data());
                const int block = raster_.block();
                for (int64_t i = 0; i < got; ++i, ++n_)
                {
                    const double time = (static_cast<double>(n_) + 0.5) * dt_;
                    double squared = 0.0;
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        const double g = samples_[axis][static_cast<size_t>(i)];
                        const double response =
                            model_.respond(axis, (g - previous_[axis]) * to_slew_);
                        previous_[axis] = g;
                        squared += response * response;
                        note(out_.axes[static_cast<size_t>(axis)], response, time, block);
                        if (options_.keep_trace)
                            out_.trace_axes[static_cast<size_t>(axis)].push_back(response);
                    }
                    const double norm = std::sqrt(squared);
                    note(out_.norm, norm, time, block);
                    if (options_.keep_trace)
                        out_.trace_norm.push_back(norm);
                }
                return got;
            }

            PhysicalRaster& raster_;
            Model& model_;
            const PnsOptions& options_;
            PnsReport& out_;
            const double dt_;
            const double to_slew_;
            std::vector<double> samples_[3];
            double previous_[3] = {0.0, 0.0, 0.0};
            int64_t n_ = 0;
        };

        template <typename Model>
        void run(PhysicalRaster& raster, Model& model, const PnsOptions& options, PnsReport& out)
        {
            Run<Model>(raster, model, options, out)();
        }

    } // namespace

    PnsReport pns(const Sequence& seq, const PnsModel& model, const PnsOptions& options)
    {
        PnsReport out;
        const FlushSubnormals flush;
        PhysicalRaster raster(seq, options.rotation);
        out.raster = raster.raster();
        if (model.kind == PnsModel::Kind::Safe)
        {
            Safe safe(model.safe, out.raster);
            run(raster, safe, options, out);
        }
        else
        {
            Chronaxie chronaxie(model.chronaxie, model.rheobase, model.alpha, out.raster);
            run(raster, chronaxie, options, out);
        }
        return out;
    }

} // namespace pulseq
