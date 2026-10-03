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

            /** Responses to @p count samples of each axis's slew, written over them. */
            void respond_chunk(int64_t count, double* const (&slew)[3])
            {
                for (int axis = 0; axis < 3; ++axis)
                    for (int64_t i = 0; i < count; ++i)
                        slew[axis][i] = respond(axis, slew[axis][i]);
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

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11 && defined(__x86_64__) && \
    defined(__linux__)
/* Resolved once per process to the widest vector unit the processor has. */
#define PULSEQ_PNS_CLONES \
    __attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
#else
#define PULSEQ_PNS_CLONES
#endif

        static_assert(kNodes % 8 == 0, "sums are taken in eight lanes");

        /** Sum of the nodes, in eight independent lanes so that it vectorises. */
        inline double lane_sum(const double (&values)[kNodes])
        {
            double lanes[8];
            for (int j = 0; j < 8; ++j)
                lanes[j] = values[j];
            for (int k = 8; k < kNodes; k += 8)
                for (int j = 0; j < 8; ++j)
                    lanes[j] += values[k + j];
            return ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5])) +
                ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]));
        }

        /**
         * Advance every axis's filter over @p count samples of slew, writing the
         * signed sum of each axis's state after each sample.
         */
        PULSEQ_PNS_CLONES
        void filter_chunk(int64_t count, const double* const (&slew)[3], double (&state)[3][kNodes],
                          const double (&decay)[kNodes], const double (&weight)[kNodes],
                          double* const (&out)[3])
        {
            /* Local copies, so that writing the output cannot alias the state. */
            double held[3][kNodes];
            double a[kNodes];
            double b[kNodes];
            for (int k = 0; k < kNodes; ++k)
            {
                a[k] = decay[k];
                b[k] = weight[k];
            }
            for (int axis = 0; axis < 3; ++axis)
                for (int k = 0; k < kNodes; ++k)
                    held[axis][k] = state[axis][k];
            for (int64_t i = 0; i < count; ++i)
            {
                double sums[3];
                for (int axis = 0; axis < 3; ++axis)
                {
                    const double x = slew[axis][i];
                    for (int k = 0; k < kNodes; ++k)
                        held[axis][k] = a[k] * held[axis][k] + b[k] * x;
                    sums[axis] = lane_sum(held[axis]);
                }
                for (int axis = 0; axis < 3; ++axis)
                    out[axis][i] = sums[axis];
            }
            for (int axis = 0; axis < 3; ++axis)
                for (int k = 0; k < kNodes; ++k)
                    state[axis][k] = held[axis][k];
        }

        /**
         * A repeated block's answer by linearity, when it stands.
         *
         * Writes into @p carried the state after the block's first sample
         * without the block's own slew; when the response it implies stays
         * within @p limit (each axis, then the norm), writes the state the
         * block leaves and returns true.
         */
        PULSEQ_PNS_CLONES
        bool repeat_block(double (&state)[3][kNodes], const double (&entering)[3],
                          const double (&decay)[kNodes], const double (&weight)[kNodes],
                          const double (&scale)[3], const double (&was)[3][kNodes],
                          const double (&from_rest)[3][kNodes], const double (&power)[kNodes],
                          const double (&peak)[3], double peak_norm, const double (&limit)[4],
                          double (&carried)[3][kNodes])
        {
            double squared = 0.0;
            bool stands = true;
            for (int axis = 0; axis < 3; ++axis)
            {
                double positive[kNodes];
                double negative[kNodes];
                for (int k = 0; k < kNodes; ++k)
                {
                    carried[axis][k] = decay[k] * state[axis][k] + weight[k] * entering[axis];
                    const double difference = carried[axis][k] - was[axis][k];
                    positive[k] = std::max(difference, 0.0);
                    negative[k] = std::max(-difference, 0.0);
                }
                const double bound =
                    std::max(lane_sum(positive), lane_sum(negative)) * scale[axis];
                squared += bound * bound;
                stands = stands && peak[axis] + bound <= limit[axis];
            }
            if (!stands || peak_norm + std::sqrt(squared) > limit[3])
                return false;
            for (int axis = 0; axis < 3; ++axis)
                for (int k = 0; k < kNodes; ++k)
                    state[axis][k] = power[k] * carried[axis][k] + from_rest[axis][k];
            return true;
        }

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

            /** Responses to @p count samples of each axis's slew, written over them. */
            void respond_chunk(int64_t count, double* const (&slew)[3])
            {
                filter_chunk(count, {slew[0], slew[1], slew[2]}, state_, decay_, weight_, slew);
                for (int axis = 0; axis < 3; ++axis)
                    for (int64_t i = 0; i < count; ++i)
                        slew[axis][i] = std::fabs(slew[axis][i]) * scale_[axis];
            }

            /** See repeat_block; @p was, @p from_rest and @p power are a remembered block's. */
            bool repeat(const double (&entering)[3], const double (&was)[3][kNodes],
                        const double (&from_rest)[3][kNodes], const double (&power)[kNodes],
                        const double (&peak)[3], double peak_norm, const double (&limit)[4],
                        double (&carried)[3][kNodes])
            {
                return repeat_block(state_, entering, decay_, weight_, scale_, was, from_rest,
                                    power, peak, peak_norm, limit, carried);
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
                double positive[kNodes];
                double negative[kNodes];
                for (int k = 0; k < kNodes; ++k)
                {
                    positive[k] = std::max(state[k], 0.0);
                    negative[k] = std::max(-state[k], 0.0);
                }
                return std::max(lane_sum(positive), lane_sum(negative)) * scale_[axis];
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

        /**
         * Resolution, in rasters, at which two blocks' sample offsets are told
         * apart. Blocks closer than this are answered as one: their samples
         * sit within it of each other on the same waveform.
         */
        constexpr double kOffsetStep = 1e-4;

        /** What makes two blocks play one physical gradient over their samples. */
        struct BlockKey
        {
            int32_t gradient[3];
            int32_t rotation;
            int64_t samples;
            /**
             * Time from the block's start to its first sample centre, in units
             * of kOffsetStep rasters: coarse enough that the rounding of block
             * start times summed over a long sequence does not split a key.
             */
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
            /** Each decay raised to the block's sample count less one. */
            double decay[States];
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
                                          static_cast<int64_t>(std::llround(offset / (dt_ * kOffsetStep)))};
                    auto found = memory_.find(key);
                    if (found == memory_.end())
                    {
                        /* Remembered from its second appearance on: most blocks
                         * that appear once never appear again. */
                        memory_.emplace(key, nullptr);
                        evaluate(count);
                        continue;
                    }
                    double carried[3][kStates];
                    const double entering[3] = {-previous_[0] * to_slew_, -previous_[1] * to_slew_,
                                                -previous_[2] * to_slew_};
                    if (found->second)
                    {
                        const Remembered<kStates>& block = *found->second;
                        const double t = 1.0 + kPeakTolerance;
                        const double limit[4] = {out_.axes[0].value * t, out_.axes[1].value * t,
                                                 out_.axes[2].value * t, out_.norm.value * t};
                        if (model_.repeat(entering, block.carried, block.from_rest, block.decay,
                                          block.peak, block.peak_norm, limit, carried))
                        {
                            for (int axis = 0; axis < 3; ++axis)
                                previous_[axis] = block.last[axis];
                            raster_.skip(count);
                            n_ += count;
                            continue;
                        }
                    }
                    else
                    {
                        for (int axis = 0; axis < 3; ++axis)
                            model_.advanced(axis, entering[axis], carried[axis]);
                    }
                    if (!found->second)
                        found->second = std::make_unique<Remembered<kStates>>();
                    Remembered<kStates>& block = *found->second;
                    for (int axis = 0; axis < 3; ++axis)
                        block_peak_[axis] = 0.0;
                    block_peak_norm_ = 0.0;
                    evaluate(count);
                    model_.decayed(count - 1, block.decay);
                    const double* decay = block.decay;
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
                            const double* decay = powers(quiet);
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

            /** Each decay raised to @p count, computed once per distinct count. */
            const double* powers(int64_t count)
            {
                auto found = powers_.find(count);
                if (found == powers_.end())
                {
                    found = powers_.emplace(count, std::array<double, kStates>()).first;
                    model_.decayed(count, found->second.data());
                }
                return found->second.data();
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
                double* const slews[3] = {samples_[0].data(), samples_[1].data(),
                                          samples_[2].data()};
                for (int axis = 0; axis < 3; ++axis)
                {
                    double* slew = slews[axis];
                    double previous = previous_[axis];
                    for (int64_t i = 0; i < got; ++i)
                    {
                        const double g = slew[i];
                        slew[i] = (g - previous) * to_slew_;
                        previous = g;
                    }
                    previous_[axis] = previous;
                }
                model_.respond_chunk(got, slews);
                const int block = raster_.block();
                for (int64_t i = 0; i < got; ++i, ++n_)
                {
                    const double time = (static_cast<double>(n_) + 0.5) * dt_;
                    double squared = 0.0;
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        const double response = slews[axis][i];
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
            std::unordered_map<int64_t, std::array<double, kStates>> powers_;
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
