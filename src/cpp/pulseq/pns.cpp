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

namespace pulseq
{

    namespace
    {

        /** Chronaxies of c / (c + t)^2 kept before the tail is cut. */
        constexpr double kChronaxieKernelSpan = 20.0;

        /** Samples read per pass over the raster. */
        constexpr int64_t kChunk = 4096;

        /** Three RC low-passes per axis, from rest. */
        class Safe
        {
        public:
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
                return std::fabs(sum) * scale_[axis];
            }

        private:
            size_t length_ = 0;
            double decay_[kNodes];
            double weight_[kNodes];
            double expired_[kNodes];
            double state_[3][kNodes] = {};
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
        void run(PhysicalRaster& raster, Model& model, const PnsOptions& options, PnsReport& out)
        {
            const double dt = raster.raster();
            const double to_slew = 1.0 / (dt * options.gamma);
            std::vector<double> samples[3];
            for (int axis = 0; axis < 3; ++axis)
                samples[axis].resize(static_cast<size_t>(kChunk));
            double previous[3] = {0.0, 0.0, 0.0};

            int64_t n = 0;
            int64_t got;
            while ((got = raster.read(
                        kChunk, samples[0].data(), samples[1].data(), samples[2].data())) > 0)
            {
                const int block = raster.block();
                for (int64_t i = 0; i < got; ++i, ++n)
                {
                    const double time = (static_cast<double>(n) + 0.5) * dt;
                    double squared = 0.0;
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        const double g = samples[axis][static_cast<size_t>(i)];
                        const double response = model.respond(axis, (g - previous[axis]) * to_slew);
                        previous[axis] = g;
                        squared += response * response;
                        note(out.axes[static_cast<size_t>(axis)], response, time, block);
                        if (options.keep_trace)
                            out.trace_axes[static_cast<size_t>(axis)].push_back(response);
                    }
                    const double norm = std::sqrt(squared);
                    note(out.norm, norm, time, block);
                    if (options.keep_trace)
                        out.trace_norm.push_back(norm);
                }
            }
            out.samples = n;
        }

    } // namespace

    PnsReport pns(const Sequence& seq, const PnsModel& model, const PnsOptions& options)
    {
        PnsReport out;
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
