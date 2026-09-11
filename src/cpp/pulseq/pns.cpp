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

        /** The chronaxie kernel over each axis's slew history. */
        class Chronaxie
        {
        public:
            Chronaxie(double chronaxie, double rheobase, double alpha, double dt)
            {
                const size_t length =
                    static_cast<size_t>(kChronaxieKernelSpan * chronaxie / dt) + 1;
                const double s_min = rheobase / alpha;
                kernel_.resize(length);
                /* Integrated over each interval, as a product rather than a
                 * difference of reciprocals, which cancels once t >> c. */
                for (size_t i = 0; i < length; ++i)
                {
                    const double t = static_cast<double>(i) * dt;
                    kernel_[i] = chronaxie * dt / (s_min * (chronaxie + t) * (chronaxie + t + dt));
                }
                for (int axis = 0; axis < 3; ++axis)
                    history_[axis].assign(2 * length, 0.0);
            }

            double respond(int axis, double slew)
            {
                /* Each slew is written twice, a kernel length apart, so the
                 * last `length` of them are always contiguous, newest last. */
                const size_t length = kernel_.size();
                std::vector<double>& history = history_[axis];
                size_t& at = at_[axis];
                at = at + 1 == length ? 0 : at + 1;
                history[at] = slew;
                history[at + length] = slew;
                const double* newest = history.data() + at + length;
                double sum = 0.0;
                for (size_t i = 0; i < length; ++i)
                    sum += kernel_[i] * newest[-static_cast<std::ptrdiff_t>(i)];
                return std::fabs(sum);
            }

        private:
            std::vector<double> kernel_;
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
