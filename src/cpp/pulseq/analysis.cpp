/**
 * @file analysis.cpp
 * @brief What a sequence is, read off the libraries and the trajectory.
 *        See analysis.hpp.
 */

#include "pulseq/analysis.hpp"

#include "pulseq/shape.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <unordered_map>

namespace pulseq
{

    namespace
    {

        constexpr double kTwoPi = 6.28318530717958647692;

        std::vector<double> decompressed(const ShapeLibrary& shapes, int id)
        {
            if (id < 1 || id > shapes.size())
                return {};
            return decompress_shape(
                shapes.samples(id), shapes.num_compressed(id),
                shapes.num_uncompressed(id));
        }

        /** Where each sample of one pulse sits, in seconds from its start. */
        std::vector<double> sample_times(
            const std::vector<double>& time_shape,
            size_t samples,
            double rf_raster)
        {
            if (!time_shape.empty())
            {
                std::vector<double> times = time_shape;
                for (size_t i = 0; i < times.size(); ++i)
                    times[i] *= rf_raster;
                return times;
            }

            std::vector<double> times(samples);
            for (size_t i = 0; i < samples; ++i)
                times[i] = (static_cast<double>(i) + 0.5) * rf_raster;
            return times;
        }

        /** One column of the binned trajectory, addressed by its sample. */
        struct Column
        {
            const std::vector<int32_t>* keys;
            int axes;
            int count;

            size_t operator()(int sample) const
            {
                size_t hash = 1469598103934665603ull;
                for (int axis = 0; axis < axes; ++axis)
                {
                    const int32_t key = (*keys)[static_cast<size_t>(axis) * count + sample];
                    hash ^= static_cast<size_t>(static_cast<uint32_t>(key));
                    hash *= 1099511628211ull;
                }
                return hash;
            }

            bool operator()(int left, int right) const
            {
                for (int axis = 0; axis < axes; ++axis)
                {
                    const size_t row = static_cast<size_t>(axis) * count;
                    if ((*keys)[row + left] != (*keys)[row + right])
                        return false;
                }
                return true;
            }
        };

        double median_of(std::vector<double> values)
        {
            if (values.empty())
                return 0.0;
            std::sort(values.begin(), values.end());
            const size_t middle = values.size() / 2;
            return values.size() % 2 ? values[middle]
                                     : 0.5 * (values[middle - 1] + values[middle]);
        }

    } // namespace

    std::vector<double> flip_angles(const Sequence& sequence)
    {
        const Table& library = sequence.rf_library();
        const ShapeLibrary& shapes = sequence.shape_library();
        const double rf_raster = sequence.rf_raster_time();

        std::vector<double> angles;
        angles.reserve(static_cast<size_t>(library.size()));

        for (int id = 1; id <= library.size(); ++id)
        {
            const double* row = library.row(id);
            const std::vector<double> magnitude = decompressed(shapes, static_cast<int>(row[1]));
            const std::vector<double> phase = decompressed(shapes, static_cast<int>(row[2]));
            const std::vector<double> times = sample_times(
                decompressed(shapes, static_cast<int>(row[3])), magnitude.size(), rf_raster);

            std::complex<double> turns(0.0, 0.0);
            const size_t samples = std::min(magnitude.size(), times.size());
            for (size_t i = 0; i + 1 < samples; ++i)
            {
                const double angle = kTwoPi * (i < phase.size() ? phase[i] : 0.0);
                const double weight = row[0] * magnitude[i] * (times[i + 1] - times[i]);
                turns += std::complex<double>(
                    weight * std::cos(angle), weight * std::sin(angle));
            }
            angles.push_back(std::abs(turns) * 360.0);
        }

        return angles;
    }

    KspaceCoverage kspace_coverage(
        const double* samples,
        int axes,
        int count,
        double threshold)
    {
        KspaceCoverage found;
        found.unique_positions.assign(static_cast<size_t>(axes), 0.0);
        if (axes <= 0 || count <= 0 || !(threshold > 0.0))
            return found;

        std::vector<int32_t> keys(static_cast<size_t>(axes) * count);
        for (size_t i = 0; i < keys.size(); ++i)
            keys[i] = static_cast<int32_t>(std::nearbyint(samples[i] / threshold));

        // Which samples land on the same position, and how many do.
        const Column column{&keys, axes, count};
        std::unordered_map<int, int, Column, Column> positions(
            static_cast<size_t>(count), column, column);
        std::vector<double> visits;
        std::vector<int> first_visit;
        for (int sample = 0; sample < count; ++sample)
        {
            const std::unordered_map<int, int, Column, Column>::const_iterator known =
                positions.find(sample);
            if (known == positions.end())
            {
                positions.emplace(sample, static_cast<int>(visits.size()));
                visits.push_back(1.0);
                first_visit.push_back(sample);
            }
            else
            {
                visits[static_cast<size_t>(known->second)] += 1.0;
            }
        }

        found.repeats_min = *std::min_element(visits.begin(), visits.end());
        found.repeats_max = *std::max_element(visits.begin(), visits.end());
        found.repeats_median = median_of(visits);

        // How many distinct coordinates each axis takes over the positions
        // visited for the first time -- the encoding once, without its
        // repeats. A coordinate one cell away from a coordinate already seen
        // is the same one: a position reached along two different ramps can
        // land either side of a cell boundary.
        double grid = 1.0;
        for (int axis = 0; axis < axes; ++axis)
        {
            const size_t row = static_cast<size_t>(axis) * count;
            std::unordered_map<int32_t, int> coordinates;
            int distinct = 0;
            for (size_t i = 0; i < first_visit.size(); ++i)
            {
                const int32_t key = keys[row + first_visit[i]];
                if (coordinates.count(key) || coordinates.count(key + 1) ||
                    coordinates.count(key - 1))
                    continue;
                coordinates.emplace(key, distinct++);
            }
            found.unique_positions[static_cast<size_t>(axis)] = distinct;
            grid *= distinct;
        }

        found.is_cartesian = grid == static_cast<double>(first_visit.size());
        return found;
    }

} // namespace pulseq
