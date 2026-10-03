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
#include <memory>
#include <unordered_map>

#if defined(__SSE__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace pulseq
{

    namespace
    {

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
            static constexpr bool kLinear = false;
            static constexpr int kStates = 3;

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
         * depend on c or the raster; the sum is within 1e-5 of the kernel's
         * unit area, in total absolute error per tap, out to 2000 chronaxies.
         */
        constexpr double kNodeFirst = -11.0;
        constexpr double kNodeStep = 0.6;
        constexpr int kNodes = 24;

        /** The chronaxie kernel over each axis's slew history. */
        class Chronaxie
        {
        public:
            /** The response is a linear function of the filter state and the slews. */
            static constexpr bool kLinear = true;
            static constexpr int kStates = kNodes;

            Chronaxie(double chronaxie, const double (&rheobase)[3],
                      const double (&alpha)[3], double dt)
            {
                for (int k = 0; k < kNodes; ++k)
                {
                    const double s =
                        std::exp(kNodeFirst + kNodeStep * static_cast<double>(k)) / chronaxie;
                    log_decay_[k] = -s * dt;
                    decay_[k] = std::exp(-s * dt);
                    weight_[k] = chronaxie * kNodeStep * s * std::exp(-s * chronaxie) *
                        -std::expm1(-s * dt);
                }
                for (int axis = 0; axis < 3; ++axis)
                    scale_[axis] = alpha[axis] / rheobase[axis];
            }

            double respond(int axis, double slew)
            {
                double* state = state_[axis];
                double sum = 0.0;
                for (int k = 0; k < kNodes; ++k)
                {
                    state[k] = decay_[k] * state[k] + weight_[k] * slew;
                    sum += state[k];
                }
                return std::fabs(sum) * scale_[axis];
            }

            /** The state after one more sample of slew @p slew, without the response. */
            void advanced(int axis, double slew, double* out) const
            {
                for (int k = 0; k < kNodes; ++k)
                    out[k] = decay_[k] * state_[axis][k] + weight_[k] * slew;
            }

            double* state(int axis)
            {
                return state_[axis];
            }

            /** Each decay raised to @p count. */
            void decayed(int64_t count, double* out) const
            {
                for (int k = 0; k < kNodes; ++k)
                    out[k] = std::exp(log_decay_[k] * static_cast<double>(count));
            }

            /**
             * Upper bound, over every later sample with no slew, on the response
             * to the state @p state: the sum of its positive entries less the
             * magnitude of its negative ones, each decaying, so the response
             * stays within the larger of the two at present.
             */
            double bound(int axis, const double* state) const
            {
                double positive = 0.0;
                double negative = 0.0;
                for (int k = 0; k < kNodes; ++k)
                {
                    if (state[k] > 0.0)
                        positive += state[k];
                    else
                        negative -= state[k];
                }
                return std::max(positive, negative) * scale_[axis];
            }

        private:
            double log_decay_[kNodes];
            double decay_[kNodes];
            double weight_[kNodes];
            double state_[3][kNodes] = {};
            double scale_[3] = {0.0, 0.0, 0.0};
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

        /** What makes two blocks play one physical gradient over their samples. */
        struct BlockKey
        {
            int32_t gradient[3];
            int32_t rotation;
            int64_t samples;
            /** Time from the block's start to its first sample centre, in ps. */
            int64_t offset;

            bool operator==(const BlockKey& other) const
            {
                return gradient[0] == other.gradient[0] && gradient[1] == other.gradient[1] &&
                    gradient[2] == other.gradient[2] && rotation == other.rotation &&
                    samples == other.samples && offset == other.offset;
            }
        };

        struct BlockKeyHash
        {
            size_t operator()(const BlockKey& key) const
            {
                uint64_t h = 1469598103934665603ull;
                const auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
                for (int32_t g : key.gradient)
                    mix(static_cast<uint32_t>(g));
                mix(static_cast<uint32_t>(key.rotation));
                mix(static_cast<uint64_t>(key.samples));
                mix(static_cast<uint64_t>(key.offset));
                return static_cast<size_t>(h);
            }
        };

        /**
         * One block's response by linearity, from its last full evaluation.
         *
         * With the state c after the block's first sample split into what the
         * block brings from rest and what it carries in, the response at its
         * n-th sample is sum_k decay_k^n c_k plus the response of the block
         * from rest, and the state it leaves is decay^(N-1) c plus that
         * response's final state. A block whose carried state differs from the
         * remembered one by d responds within the bound on d of what it did.
         */
        template <int States>
        struct Remembered
        {
            double carried[3][States];
            double from_rest[3][States];
            double peak[3];
            double peak_norm;
            double last[3];
        };

        /** Samples read between bound checks while a held gradient's response decays. */
        constexpr int64_t kQuietStep = 64;

        /** Relative excess over the peak found that a skipped block may hide. */
        constexpr double kPeakTolerance = 1e-9;

        template <typename Model>
        class Run
        {
        public:
            Run(const Sequence& seq, PhysicalRaster& raster, Model& model,
                const PnsOptions& options, PnsReport& out)
                : seq_(seq), raster_(raster), model_(model), options_(options), out_(out),
                  dt_(raster.raster()), to_slew_(1.0 / (dt_ * options.gamma))
            {
                for (int axis = 0; axis < 3; ++axis)
                    samples_[axis].resize(static_cast<size_t>(kChunk));
            }

            void operator()()
            {
                if constexpr (Model::kLinear)
                {
                    if (!options_.keep_trace)
                    {
                        by_block();
                        return;
                    }
                }
                while (read(kChunk) > 0)
                {
                }
                out_.samples = n_;
            }

        private:
            static constexpr int kStates = Model::kStates;

            void by_block()
            {
                int64_t count;
                while ((count = raster_.enter_block()) > 0)
                {
                    const int32_t* row = seq_.block_events() +
                        static_cast<size_t>(raster_.block() - 1) * BLOCK_WIDTH;
                    const double offset =
                        (static_cast<double>(raster_.position()) + 0.5) * dt_ - raster_.block_start();
                    const BlockKey key = {{row[1], row[2], row[3]},
                                          row[BLOCK_ROTATION_COLUMN],
                                          count,
                                          static_cast<int64_t>(std::llround(offset * 1e12))};
                    double carried[3][kStates];
                    for (int axis = 0; axis < 3; ++axis)
                        model_.advanced(axis, -previous_[axis] * to_slew_, carried[axis]);

                    auto found = memory_.find(key);
                    if (found == memory_.end())
                    {
                        /* Remembered from its second appearance on: most blocks
                         * that appear once never appear again. */
                        memory_.emplace(key, nullptr);
                        evaluate(count);
                        continue;
                    }
                    if (found->second && within(*found->second, carried))
                    {
                        const Remembered<kStates>& block = *found->second;
                        double decay[kStates];
                        model_.decayed(count - 1, decay);
                        for (int axis = 0; axis < 3; ++axis)
                        {
                            double* state = model_.state(axis);
                            for (int k = 0; k < kStates; ++k)
                                state[k] = decay[k] * carried[axis][k] + block.from_rest[axis][k];
                            previous_[axis] = block.last[axis];
                        }
                        raster_.skip(count);
                        n_ += count;
                        continue;
                    }
                    if (!found->second)
                        found->second = std::make_unique<Remembered<kStates>>();
                    Remembered<kStates>& block = *found->second;
                    for (int axis = 0; axis < 3; ++axis)
                        block_peak_[axis] = 0.0;
                    block_peak_norm_ = 0.0;
                    evaluate(count);
                    double decay[kStates];
                    model_.decayed(count - 1, decay);
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        const double* state = model_.state(axis);
                        for (int k = 0; k < kStates; ++k)
                        {
                            block.carried[axis][k] = carried[axis][k];
                            block.from_rest[axis][k] = state[k] - decay[k] * carried[axis][k];
                        }
                        block.peak[axis] = block_peak_[axis];
                        block.last[axis] = previous_[axis];
                    }
                    block.peak_norm = block_peak_norm_;
                }
                out_.samples = n_;
            }

            /** Whether the block, carrying this state in, cannot pass a peak found. */
            bool within(const Remembered<kStates>& block, const double (&carried)[3][kStates]) const
            {
                double squared = 0.0;
                for (int axis = 0; axis < 3; ++axis)
                {
                    double difference[kStates];
                    for (int k = 0; k < kStates; ++k)
                        difference[k] = carried[axis][k] - block.carried[axis][k];
                    const double bound = model_.bound(axis, difference);
                    squared += bound * bound;
                    if (block.peak[axis] + bound >
                        out_.axes[static_cast<size_t>(axis)].value * (1.0 + kPeakTolerance))
                        return false;
                }
                return block.peak_norm + std::sqrt(squared) <=
                    out_.norm.value * (1.0 + kPeakTolerance);
            }

            /**
             * The block's @p count samples, read where the slew changes. Where
             * the gradient holds, the state only decays: once its bound is
             * within the peaks found, the rest of the stretch is skipped and
             * the bound stands in for the block's own peak.
             */
            void evaluate(int64_t count)
            {
                while (count > 0)
                {
                    bool steady = false;
                    const int64_t ahead = std::min(count, raster_.run(&steady));
                    if (!steady || ahead == 1)
                    {
                        read_all(ahead);
                        count -= ahead;
                        continue;
                    }
                    /* The first sample may step; the rest repeat it. */
                    read(1);
                    int64_t quiet = ahead - 1;
                    count -= ahead;
                    while (quiet > 0)
                    {
                        double bounds[3];
                        double squared = 0.0;
                        bool below = true;
                        for (int axis = 0; axis < 3; ++axis)
                        {
                            bounds[axis] = model_.bound(axis, model_.state(axis));
                            squared += bounds[axis] * bounds[axis];
                            below = below &&
                                bounds[axis] <= out_.axes[static_cast<size_t>(axis)].value;
                        }
                        const double norm = std::sqrt(squared);
                        if (below && norm <= out_.norm.value)
                        {
                            double decay[kStates];
                            model_.decayed(quiet, decay);
                            for (int axis = 0; axis < 3; ++axis)
                            {
                                double* state = model_.state(axis);
                                for (int k = 0; k < kStates; ++k)
                                    state[k] *= decay[k];
                                block_peak_[axis] = std::max(block_peak_[axis], bounds[axis]);
                            }
                            block_peak_norm_ = std::max(block_peak_norm_, norm);
                            raster_.skip(quiet);
                            n_ += quiet;
                            break;
                        }
                        quiet -= read(std::min(quiet, kQuietStep));
                    }
                }
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
                        block_peak_[axis] = std::max(block_peak_[axis], response);
                        note(out_.axes[static_cast<size_t>(axis)], response, time, block);
                        if (options_.keep_trace)
                            out_.trace_axes[static_cast<size_t>(axis)].push_back(response);
                    }
                    const double norm = std::sqrt(squared);
                    block_peak_norm_ = std::max(block_peak_norm_, norm);
                    note(out_.norm, norm, time, block);
                    if (options_.keep_trace)
                        out_.trace_norm.push_back(norm);
                }
                return got;
            }

            const Sequence& seq_;
            PhysicalRaster& raster_;
            Model& model_;
            const PnsOptions& options_;
            PnsReport& out_;
            const double dt_;
            const double to_slew_;
            std::vector<double> samples_[3];
            double previous_[3] = {0.0, 0.0, 0.0};
            double block_peak_[3] = {0.0, 0.0, 0.0};
            double block_peak_norm_ = 0.0;
            int64_t n_ = 0;
            std::unordered_map<BlockKey, std::unique_ptr<Remembered<kStates>>, BlockKeyHash>
                memory_;
        };

        template <typename Model>
        void run(const Sequence& seq, PhysicalRaster& raster, Model& model,
                 const PnsOptions& options, PnsReport& out)
        {
            Run<Model>(seq, raster, model, options, out)();
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
            run(seq, raster, safe, options, out);
        }
        else
        {
            Chronaxie chronaxie(model.chronaxie, model.rheobase, model.alpha, out.raster);
            run(seq, raster, chronaxie, options, out);
        }
        return out;
    }

} // namespace pulseq
