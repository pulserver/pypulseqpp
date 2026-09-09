/**
 * @file fov.cpp
 * @brief Where the trajectory stands, block by block.  See fov.hpp.
 */

#include "pulseq/fov.hpp"

#include "pulseq/corners.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace pulseq
{

    namespace
    {

        constexpr double kEps = 1e-12;

        /**
         * What one gradient sweeps between the start of its block and @p when.
         *
         * The corners are a polyline, so this is the area under it clipped at
         * @p when -- and clipped rather than interpolated to the raster,
         * because where a pulse acts is where it acts.
         */
        double swept_by(const Corners& drawn, std::vector<double>& corner_times, double when)
        {
            if (drawn.values.empty())
                return 0.0;
            drawn.at(0.0, corner_times);

            double area = 0.0;
            for (size_t i = 0; i + 1 < corner_times.size(); ++i)
            {
                const double from = corner_times[i];
                const double to = corner_times[i + 1];
                if (when <= from)
                    break;
                const double span = to - from;
                if (span <= kEps)
                    continue;
                const double first = drawn.values[i];
                const double last = drawn.values[i + 1];
                if (when >= to)
                {
                    area += 0.5 * (first + last) * span;
                    continue;
                }
                /* Part of a segment: the polyline is straight across it, so
                 * what it sweeps is a trapezium of its own. */
                const double along = (when - from) / span;
                const double reached = first + along * (last - first);
                area += 0.5 * (first + reached) * (when - from);
                break;
            }
            return area;
        }

    } // namespace

    bool advance_origin(char use, const double at[3], double origin[3])
    {
        /* A pulse that records no use is read as an excitation, which is what
         * `calculate_kspace` reads it as -- one sequence cannot have two
         * stories about where its trajectory restarts. Before revision 1.5.0
         * the format had nowhere to write a use, so a file older than that
         * arrives with every pulse undefined; `detect_rf_use` is what fills
         * them in from what each pulse does. */
        if (use == 'e' || use == 'u')
        {
            /* A new trajectory: k is zero from here. */
            for (int axis = 0; axis < 3; ++axis)
                origin[axis] = 0.0;
            return true;
        }
        if (use == 'r')
        {
            /* Turned around: what was k is now -k, which is what walks a spin
             * echo back towards the origin. */
            for (int axis = 0; axis < 3; ++axis)
                origin[axis] = -at[axis];
            return true;
        }
        return false;
    }

    std::vector<std::array<double, 3>> block_k_origins(
        const Sequence& seq, int first, int last, double carry[3])
    {
        const int blocks = seq.num_blocks();
        const int from = first > 1 ? first : 1;
        const int to = (last > 0 && last < blocks) ? last : blocks;

        std::vector<std::array<double, 3>> out;
        if (from > to)
            return out;
        out.reserve(static_cast<size_t>(to - from + 1));

        CornerCache corners(seq);
        const int32_t* events = seq.block_events();
        const std::vector<char>& uses = seq.rf_uses();

        std::vector<double> corner_times;

        for (int index = from; index <= to; ++index)
        {
            const int32_t* row = events + static_cast<size_t>(index - 1) * BLOCK_WIDTH;

            out.push_back({carry[0], carry[1], carry[2]});

            /* Where the block's pulse acts, if it has one, and what it is for.
             * The centre rather than the block boundary: a refocusing between
             * two crushers has each crusher counted on its own side. */
            double acts_at = -1.0;
            char use = 'u';
            const int32_t rf_id = row[0];
            if (rf_id > 0)
            {
                const double* rf = seq.rf_library().row(rf_id);
                acts_at = rf[5] + rf[4];
                use = rf_id <= static_cast<int32_t>(uses.size())
                    ? uses[static_cast<size_t>(rf_id) - 1]
                    : 'u';
            }

            double whole[3];
            double before[3];
            double at_pulse[3];
            for (int axis = 0; axis < 3; ++axis)
            {
                const Corners& drawn = corners[row[1 + axis]];
                whole[axis] = swept_by(drawn, corner_times, 1e30);
                before[axis] =
                    acts_at >= 0.0 ? swept_by(drawn, corner_times, acts_at) : 0.0;
                at_pulse[axis] = carry[axis] + before[axis];
            }

            if (acts_at >= 0.0 && advance_origin(use, at_pulse, carry))
            {
                /* The origin moved where the pulse acts, so only what the
                 * block sweeps after it counts towards the next block. */
                for (int axis = 0; axis < 3; ++axis)
                    carry[axis] += whole[axis] - before[axis];
            }
            else
            {
                for (int axis = 0; axis < 3; ++axis)
                    carry[axis] += whole[axis];
            }
        }
        return out;
    }

} // namespace pulseq
