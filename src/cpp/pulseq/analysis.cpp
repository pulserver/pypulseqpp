/**
 * @file analysis.cpp
 * @brief What a sequence is, read off the libraries and the trajectory.
 *        See analysis.hpp.
 */

#include "pulseq/analysis.hpp"

#include "pulseq/channels.hpp"
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

        /**
         * How far one RF row's envelope tips the magnetisation, in turns per
         * unit amplitude.
         *
         * The integral of the complex envelope over the pulse: an amplitude
         * in hertz over a time in seconds is a number of turns. The amplitude
         * is left out, being real and so a scale on the answer. A dynamic pTx
         * pulse is integrated channel by channel on its shared time base and
         * the channels summed: the flip where every channel has unit,
         * in-phase sensitivity.
         */
        double integrate_envelope(const Sequence& sequence, int id)
        {
            const double* row = sequence.rf_library().row(id);
            const ShapeLibrary& shapes = sequence.shape_library();
            const std::vector<double> magnitude =
                decompressed(shapes, static_cast<int>(row[1]));
            const std::vector<double> phase = decompressed(shapes, static_cast<int>(row[2]));
            const std::vector<double> times = sample_times(
                decompressed(shapes, static_cast<int>(row[3])), magnitude.size(),
                sequence.rf_raster_time());

            std::complex<double> turns(0.0, 0.0);
            const size_t samples = std::min(magnitude.size(), times.size());
            const size_t per_channel = std::max<size_t>(1, samples / rf_channels(times));
            for (size_t i = 0; i + 1 < samples; ++i)
            {
                if ((i + 1) % per_channel == 0)
                    continue;
                const double angle = kTwoPi * (i < phase.size() ? phase[i] : 0.0);
                const double weight = magnitude[i] * (times[i + 1] - times[i]);
                turns += std::complex<double>(
                    weight * std::cos(angle), weight * std::sin(angle));
            }
            return std::abs(turns);
        }

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

    std::vector<double> rf_flip_angles(const Sequence& sequence)
    {
        const Table& library = sequence.rf_library();
        const std::vector<int32_t>& definition_of = sequence.rf_definitions();
        const int definitions = sequence.num_rf_definitions();

        // How far one turn of the envelope tips, before any amplitude: the
        // envelope belongs to the definition, so a pulse swept over a
        // thousand amplitudes is integrated once and multiplied a thousand
        // times.
        std::vector<double> envelope(static_cast<size_t>(definitions) + 1, -1.0);

        std::vector<double> angles;
        angles.reserve(static_cast<size_t>(library.size()));
        for (int id = 1; id <= library.size(); ++id)
        {
            const int32_t definition =
                id <= static_cast<int>(definition_of.size())
                ? definition_of[static_cast<size_t>(id) - 1]
                : 0;
            const bool known = definition >= 1 && definition <= definitions;

            double& turns = known ? envelope[static_cast<size_t>(definition)] : envelope[0];
            if (!known || turns < 0.0)
                turns = integrate_envelope(sequence, id);

            angles.push_back(std::fabs(library.row(id)[0]) * turns * 360.0);
        }
        return angles;
    }

    std::vector<double> flip_angles(const Sequence& sequence)
    {
        std::vector<double> angles = rf_flip_angles(sequence);
        std::sort(angles.begin(), angles.end());
        angles.erase(std::unique(angles.begin(), angles.end()), angles.end());
        return angles;
    }

    std::vector<int> rf_channel_counts(const Sequence& sequence)
    {
        const Table& library = sequence.rf_library();
        const ShapeLibrary& shapes = sequence.shape_library();
        std::vector<int> counts;
        counts.reserve(static_cast<size_t>(library.size()));
        for (int id = 1; id <= library.size(); ++id)
        {
            const double* row = library.row(id);
            counts.push_back(static_cast<int>(
                rf_channels(decompressed(shapes, static_cast<int>(row[3])))));
        }
        return counts;
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


    LabelEvolution evaluate_labels(
        const Sequence& seq,
        LabelEvolutionAt at,
        int first_block,
        int last_block,
        const std::vector<std::pair<std::string, int32_t>>& start)
    {
        LabelEvolution out;

        /* By label id rather than by name: what a block carries is a number,
         * and turning it into a name once per label beats once per block. */
        std::unordered_map<int, size_t> column;
        std::vector<int32_t> now;
        const auto column_for = [&](int label_id) {
            auto found = column.find(label_id);
            if (found != column.end())
                return found->second;
            const size_t made = out.names.size();
            out.names.push_back(seq.label_name(label_id));
            out.values.emplace_back();
            now.push_back(0);
            column.emplace(label_id, made);
            return made;
        };

        for (const auto& given : start)
        {
            const int id = seq.find_label_id(given.first);
            /* A name the table does not know cannot be carried by a block
             * either, so it is reported as it was given and never changes. */
            if (id == 0)
            {
                out.names.push_back(given.first);
                out.values.emplace_back();
                now.push_back(given.second);
                continue;
            }
            now[column_for(id)] = given.second;
        }

        const int blocks = seq.num_blocks();
        const int first = first_block > 1 ? first_block : 1;
        const int last = (last_block > 0 && last_block < blocks) ? last_block : blocks;

        const int32_t* events = seq.block_events();
        const IntTable& links = seq.extensions_library();
        const IntTable& sets = seq.label_set_library();
        const IntTable& increments = seq.label_inc_library();
        const int set_type = seq.find_extension_type_id("LABELSET");
        const int inc_type = seq.find_extension_type_id("LABELINC");

        int recorded = 0;
        const auto record = [&]() {
            ++recorded;
            for (size_t i = 0; i < out.values.size(); ++i)
                out.values[i].push_back(now[i]);
        };

        for (int index = first; index <= last; ++index)
        {
            const int32_t* row = events + static_cast<size_t>(index - 1) * BLOCK_WIDTH;

            bool labelled = false;
            for (int32_t node = row[5]; node > 0 && node <= links.size();)
            {
                const int32_t* link = links.row(node);
                const int32_t kind = link[0];
                const int32_t ref = link[1];
                node = link[2];
                const bool setting = set_type != 0 && kind == set_type;
                const bool adding = inc_type != 0 && kind == inc_type;
                if (!setting && !adding)
                    continue;
                const IntTable& carrying = setting ? sets : increments;
                if (ref < 1 || ref > carrying.size())
                    continue;
                const int32_t* what = carrying.row(ref);
                const size_t which = column_for(static_cast<int>(what[1]));
                /* A column made here starts blank, so the points already
                 * recorded read as the zero a label is before it is set. */
                out.values[which].resize(static_cast<size_t>(recorded), 0);
                now[which] = setting ? what[0] : now[which] + what[0];
                labelled = true;
            }

            const bool wanted = at == LabelEvolutionAt::Blocks ||
                (at == LabelEvolutionAt::Adc && row[4] > 0) ||
                (at == LabelEvolutionAt::Label && labelled);
            if (wanted)
                record();
        }

        /* One point is not an evolution, and the reference toolbox says so:
         * asked for one it hands back what the labels finish at instead, and
         * a sequence that never reaches the thing being recorded -- no ADC,
         * say -- is asked the same question. */
        if (at == LabelEvolutionAt::End || recorded < 2)
        {
            for (auto& column_values : out.values)
                column_values.clear();
            recorded = 0;
            record();
        }
        else
        {
            for (auto& column_values : out.values)
                column_values.resize(static_cast<size_t>(recorded), 0);
        }
        return out;
    }

    std::vector<int32_t> label_blocks(const Sequence& seq, const std::string& label, bool setting)
    {
        std::vector<int32_t> out;
        const int id = seq.find_label_id(label);
        const int type = seq.find_extension_type_id(setting ? "LABELSET" : "LABELINC");
        if (id == 0 || type == 0)
            return out;

        const IntTable& links = seq.extensions_library();
        const IntTable& statements = setting ? seq.label_set_library() : seq.label_inc_library();
        const int32_t* events = seq.block_events();
        const int blocks = seq.num_blocks();
        for (int index = 1; index <= blocks; ++index)
        {
            const int32_t* row = events + static_cast<size_t>(index - 1) * BLOCK_WIDTH;
            for (int32_t node = row[5]; node > 0 && node <= links.size();)
            {
                const int32_t* link = links.row(node);
                node = link[2];
                if (link[0] != type || link[1] < 1 || link[1] > statements.size())
                    continue;
                if (statements.row(link[1])[1] == id)
                {
                    out.push_back(index);
                    break;
                }
            }
        }
        return out;
    }

} // namespace pulseq
