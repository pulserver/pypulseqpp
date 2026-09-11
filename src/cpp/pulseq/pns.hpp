/**
 * @file pns.hpp
 * @brief Peripheral nerve stimulation of the physical-axis gradient, SAFE or chronaxie.
 */

#ifndef PULSEQ_PNS_HPP
#define PULSEQ_PNS_HPP

#include <array>
#include <cstdint>
#include <vector>

#include "pulseq/sequence.hpp"

namespace pulseq
{

    /** SAFE coefficients of one axis (Hebrank and Gebhardt, ISMRM 2000). */
    struct SafeAxis
    {
        double a[3] = {0.0, 0.0, 0.0};
        /** Low-pass time constants, in ms. */
        double tau_ms[3] = {0.0, 0.0, 0.0};
        /** In T/m/s. */
        double stim_limit = 1.0;
        double g_scale = 1.0;
    };

    struct PnsModel
    {
        enum class Kind
        {
            Safe,
            Chronaxie
        };
        Kind kind = Kind::Safe;
        std::array<SafeAxis, 3> safe;
        /** Chronaxie in s, rheobase in T/m/s; one set for all three axes. */
        double chronaxie = 0.0;
        double rheobase = 0.0;
        double alpha = 1.0;
    };

    struct PnsOptions
    {
        /** Prescription rotation, logical to physical after each block's own. */
        double rotation[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
        /** Gyromagnetic ratio in Hz/T, to turn Hz/m into T/m. */
        double gamma = 42.576e6;
        /** Also return the response at every sample. */
        bool keep_trace = false;
    };

    /** The strongest response, as a fraction of threshold, and where. */
    struct PnsPeak
    {
        double value = 0.0;
        double time = 0.0;
        int block = 0;
    };

    struct PnsReport
    {
        /** Root-sum-square over the axes, and each axis on its own. */
        PnsPeak norm;
        std::array<PnsPeak, 3> axes;
        int64_t samples = 0;
        double raster = 0.0;
        /** Per sample when asked for, at (n + 1/2) raster. */
        std::vector<double> trace_norm;
        std::array<std::vector<double>, 3> trace_axes;
    };

    /**
     * Nerve response to the slew of every physical axis, over the whole sequence.
     *
     * The gradient is read at raster centres (PhysicalRaster), from rest; the
     * slew of sample n is its difference from sample n - 1 over one raster,
     * zero before the first. SAFE filters each axis's slew with its own three
     * RC low-passes, as the Szczepankiewicz-Witzel implementation does, and
     * normalises by stim_limit / g_scale. The chronaxie model convolves the
     * slew with c / (c + t)^2 integrated over each raster interval, normalised
     * by rheobase / alpha, so a rectangular slew S held for tau responds with
     * S alpha tau / (rheobase (c + tau)); the kernel is cut after
     * 20 chronaxies. Both are evaluated in one pass with the filter's own
     * memory carried along, which is the whole-timeline answer.
     */
    PnsReport pns(const Sequence& seq, const PnsModel& model, const PnsOptions& options);

} // namespace pulseq

#endif /* PULSEQ_PNS_HPP */
