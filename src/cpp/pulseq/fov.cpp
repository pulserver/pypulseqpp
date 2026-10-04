/**
 * @file fov.cpp
 * @brief Where the trajectory stands, block by block.  See fov.hpp.
 */

#include "pulseq/fov.hpp"

#include "pulseq/corners.hpp"
#include "pulseq/shape.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace pulseq
{

    namespace
    {

        constexpr double kEps = 1e-12;
        constexpr double kPi = 3.14159265358979323846;

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


        /** The fractional part, which is all a phase in turns means. */
        double turns(double value)
        {
            return value - std::floor(value);
        }

        /**
         * One gradient as the block plays it: the corners, timed from the
         * block's start, and what it is worth in phase.
         *
         * The times are materialised once per block per axis; the values are
         * the cache's own, so a readout of ten thousand corners is not copied
         * to be integrated.
         */
        struct Played
        {
            std::vector<double> times;
            const std::vector<double>* values = nullptr;

            size_t count() const
            {
                return values == nullptr ? 0 : values->size();
            }

            /** What the gradient is at @p when; zero outside what it plays. */
            double at(double when) const
            {
                const size_t n = count();
                if (n == 0 || when < times.front() || when > times.back())
                    return 0.0;
                const auto found =
                    std::upper_bound(times.begin(), times.begin() + static_cast<long>(n), when);
                const size_t after = static_cast<size_t>(found - times.begin());
                if (after == 0)
                    return (*values)[0];
                if (after >= n)
                    return (*values)[n - 1];
                const double span = times[after] - times[after - 1];
                if (span <= kEps)
                    return (*values)[after];
                const double along = (when - times[after - 1]) / span;
                return (*values)[after - 1] +
                    along * ((*values)[after] - (*values)[after - 1]);
            }

            /** Whether it holds one value across `[from, to]`. */
            bool constant_over(double from, double to) const
            {
                const size_t n = count();
                if (n == 0)
                    return true;
                const double first = at(from);
                for (size_t i = 0; i < n; ++i)
                {
                    if (times[i] < from || times[i] > to)
                        continue;
                    if (std::fabs((*values)[i] - first) > 1e-10)
                        return false;
                }
                return std::fabs(at(to) - first) <= 1e-10;
            }

            /**
             * The phase, in turns, that @p shift has accumulated by @p when.
             *
             * Segment by segment, taking the fraction of each: across a scan
             * the whole is thousands of turns and only the fraction is a
             * phase, so summing first would spend the precision on the part
             * that is thrown away.
             */
            /** The area under it up to @p when, in Hz/m times seconds. */
            double swept(double when) const
            {
                const size_t n = count();
                double total = 0.0;
                for (size_t i = 0; i + 1 < n; ++i)
                {
                    const double from = times[i];
                    if (when <= from)
                        break;
                    const double to = times[i + 1];
                    const double span = to - from;
                    if (span <= kEps)
                        continue;
                    const double first = (*values)[i];
                    const double last = (*values)[i + 1];
                    if (when >= to)
                    {
                        total += 0.5 * (first + last) * span;
                        continue;
                    }
                    const double along = (when - from) / span;
                    total += 0.5 * (first + first + along * (last - first)) * (when - from);
                    break;
                }
                return total;
            }

            double swept_turns(double when, double shift) const
            {
                const size_t n = count();
                double total = 0.0;
                for (size_t i = 0; i + 1 < n; ++i)
                {
                    const double from = times[i];
                    if (when <= from)
                        break;
                    const double to = times[i + 1];
                    const double span = to - from;
                    if (span <= kEps)
                        continue;
                    const double first = (*values)[i];
                    const double last = (*values)[i + 1];
                    if (when >= to)
                    {
                        total = turns(total + turns(0.5 * (first + last) * span * shift));
                        continue;
                    }
                    const double along = (when - from) / span;
                    const double reached = first + along * (last - first);
                    total = turns(
                        total + turns(0.5 * (first + reached) * (when - from) * shift));
                    break;
                }
                return total;
            }
        };



        /**
         * When each sample of a pulse is played, measured from its delay.
         *
         * A pulse without times of its own is sampled at the centre of each
         * RF raster interval, which is where the format puts it; one with a
         * time shape says where its samples sit and those are in raster
         * units.
         */
        void when_at(const Sequence& seq, const double* rf, std::vector<double>& into)
        {
            const ShapeLibrary& shapes = seq.shape_library();
            const int magnitude = static_cast<int>(rf[1]);
            const int time_shape = static_cast<int>(rf[3]);
            into.clear();
            if (magnitude < 1 || magnitude > shapes.size())
                return;
            const size_t count =
                static_cast<size_t>(shapes.num_uncompressed(magnitude));
            const double raster = seq.rf_raster_time();
            if (time_shape >= 1 && time_shape <= shapes.size())
            {
                const std::vector<double> ticks = decompress_shape(
                    shapes.samples(time_shape),
                    shapes.num_compressed(time_shape),
                    shapes.num_uncompressed(time_shape));
                into.reserve(ticks.size());
                for (size_t i = 0; i < ticks.size(); ++i)
                    into.push_back(ticks[i] * raster);
                return;
            }
            into.reserve(count);
            for (size_t i = 0; i < count; ++i)
                into.push_back((static_cast<double>(i) + 0.5) * raster);
        }

        /**
         * The first and last sample times of the pulse @p rf, from the start
         * of its block, with every sample time from its delay in @p moment.
         * A gradient is steady under the pulse when it holds one value across
         * this span.
         */
        std::pair<double, double> pulse_span(
            const Sequence& seq, const double* rf, std::vector<double>& moment)
        {
            const double delay = rf[5];
            when_at(seq, rf, moment);
            if (moment.empty())
                return {delay, delay};
            return {delay + moment.front(), delay + moment.back()};
        }

        /**
         * A phase shape with @p added turns folded into it, as a new row.
         *
         * Registered rather than written over: a shape is shared by every
         * event that names it, and only the ones this block plays are being
         * moved. Deduplication collapses the copies afterwards, and the added
         * turns depend on the pulse and the gradient under it and not on
         * where the block sits, so every block playing that pair lands on one
         * row.
         */
        int phase_shape_with(
            Sequence& seq, int existing, const std::vector<double>& added, double whole)
        {
            const ShapeLibrary& shapes = seq.shape_library();
            std::vector<double> samples(added.size(), 0.0);
            if (existing >= 1 && existing <= shapes.size())
            {
                const std::vector<double> held = decompress_shape(
                    shapes.samples(existing),
                    shapes.num_compressed(existing),
                    shapes.num_uncompressed(existing));
                for (size_t i = 0; i < samples.size() && i < held.size(); ++i)
                    samples[i] = held[i];
            }
            /* Wrapped to one turn, in whatever a turn is here: an RF phase
             * shape is stored in turns and an ADC's in radians. */
            for (size_t i = 0; i < samples.size(); ++i)
            {
                const double sum = samples[i] + added[i];
                samples[i] = sum - whole * std::floor(sum / whole);
            }
            return seq.shape_library().append_raw(
                samples.data(), static_cast<int>(samples.size()));
        }

        /**
         * The line a moving readout's offsets are taken along: through k at
         * the window's centre, at the rate between the samples either side
         * of it, both read off the samples rather than the gradient. A
         * receiver that knows only the samples' k draws the same line, so
         * it can compute the modulation from the shift alone.
         *
         * @param slope  Set to the rate, in Hz/m.
         * @param swept  Set to the turns @p shift has swept at the centre.
         */
        void centre_chord(
            const Played& played, double shift, int samples, double dwell, double delay,
            double& slope, double& swept)
        {
            const double centre = 0.5 * static_cast<double>(samples - 1);
            const int below = static_cast<int>(std::floor(centre));
            const auto at = [&](double index)
            { return played.swept(delay + dwell * (index + 0.5)); };
            double k = 0.0;
            if (samples % 2 != 0)
            {
                k = at(below);
                slope = (at(below + 1) - at(below - 1)) / (2.0 * dwell);
            }
            else
            {
                const double low = at(below);
                const double high = at(below + 1);
                k = 0.5 * (low + high);
                slope = (high - low) / dwell;
            }
            swept = turns(shift * k);
        }

        /**
         * Phase shapes already registered, keyed by everything the added
         * turns are computed from: the shape they are folded into, the
         * event's timing, and the shift and corners of each axis whose
         * gradient moves under it, all timed from the block's start. Equal
         * keys give equal samples, so the readouts and pulses a sequence
         * repeats share one row rather than registering one each.
         */
        class ShapeMemo
        {
        public:
            void start(double tag)
            {
                key_.clear();
                add(tag);
            }

            void add(double value)
            {
                key_.append(reinterpret_cast<const char*>(&value), sizeof value);
            }

            void add(int axis, double shift, double at, const Played& played)
            {
                add(static_cast<double>(axis));
                add(shift);
                add(at);
                const size_t n = played.count();
                add(static_cast<double>(n));
                key_.append(
                    reinterpret_cast<const char*>(played.times.data()), n * sizeof(double));
                key_.append(
                    reinterpret_cast<const char*>(played.values->data()), n * sizeof(double));
            }

            /** The row for the current key, made by @p make on first use. */
            template <typename Make> int row(Make make)
            {
                const auto found = rows_.find(key_);
                if (found != rows_.end())
                    return found->second;
                const int made = make();
                rows_.emplace(key_, made);
                return made;
            }

        private:
            std::string key_;
            std::unordered_map<std::string, int> rows_;
        };

        /** An event row with a shift folded in: what it was, frequency, phase, shape. */
        using MovedRow = std::tuple<int32_t, double, double, double>;

        /**
         * Moved rows already registered, in one open-addressed array: almost
         * every row of a shifted readout is new, and a node per row costs an
         * allocation and a cache miss per lookup.
         */
        class MovedRows
        {
        public:
            /**
             * The row a moved event is played from, registered on first use.
             *
             * A row is shared by every block that names it, and what the shift
             * adds depends on where each block sits, so a moved event is a new
             * row rather than an edit of the shared one. Blocks whose event
             * comes out the same share the new row.
             */
            template <typename Register>
            int32_t row(const MovedRow& key, Register add)
            {
                if (2 * (entries_.size() + 1) > slots_.size())
                    grow();
                const uint64_t h = hash(key);
                const size_t mask = slots_.size() - 1;
                for (size_t at = static_cast<size_t>(h) & mask;; at = (at + 1) & mask)
                {
                    const uint64_t slot = slots_[at];
                    if (slot == 0)
                    {
                        const int32_t made = static_cast<int32_t>(add());
                        entries_.push_back({key, made});
                        slots_[at] = pack(h, entries_.size());
                        return made;
                    }
                    if ((slot >> 32) == (h >> 32) &&
                        entries_[static_cast<uint32_t>(slot) - 1].first == key)
                        return entries_[static_cast<uint32_t>(slot) - 1].second;
                }
            }

            /** Room for `rows` rows without growing. */
            void reserve(size_t rows)
            {
                entries_.reserve(rows);
                size_t size = 1024;
                while (size < 2 * rows)
                    size *= 2;
                if (size > slots_.size())
                {
                    slots_.resize(size);
                    rehash();
                }
            }

        private:
            static uint64_t mix(uint64_t x)
            {
                x += 0x9E3779B97F4A7C15ull;
                x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
                x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
                return x ^ (x >> 31);
            }

            /* -0.0 is folded onto 0.0, which the tuple calls equal. */
            static uint64_t hash(const MovedRow& key)
            {
                uint64_t h = mix(static_cast<uint64_t>(std::get<0>(key)));
                for (const double value : {std::get<1>(key), std::get<2>(key), std::get<3>(key)})
                {
                    uint64_t bits;
                    const double folded = value + 0.0;
                    std::memcpy(&bits, &folded, sizeof bits);
                    h = mix(h ^ bits);
                }
                return h;
            }

            /* The hash's high half beside the entry's position from 1; 0 is
             * an empty slot. */
            static uint64_t pack(uint64_t h, size_t entry)
            {
                return (h & 0xFFFFFFFF00000000ull) | static_cast<uint32_t>(entry);
            }

            void grow()
            {
                slots_.resize(slots_.empty() ? 1024 : 2 * slots_.size());
                rehash();
            }

            void rehash()
            {
                std::fill(slots_.begin(), slots_.end(), 0);
                const size_t mask = slots_.size() - 1;
                for (size_t entry = 0; entry < entries_.size(); ++entry)
                {
                    const uint64_t h = hash(entries_[entry].first);
                    size_t at = static_cast<size_t>(h) & mask;
                    while (slots_[at] != 0)
                        at = (at + 1) & mask;
                    slots_[at] = pack(h, entry + 1);
                }
            }

            /* Slots hold eight bytes and the keys sit apart, read only when
             * the hashes agree: the table a long scan probes stays small. */
            std::vector<uint64_t> slots_;
            std::vector<std::pair<MovedRow, int32_t>> entries_;
        };


        /**
         * Integrate gradient corners at nondecreasing sample times.
         *
         * Carry both the area and fractional phase in turns; accumulate fractional
         * phase per segment to avoid loss of precision over long acquisitions.
         */
        struct Sweep
        {
            const Played* played = nullptr;
            size_t corner = 0;
            double behind = 0.0;
            double behind_turns = 0.0;
            double shift = 0.0;

            void restart(const Played& over, double by)
            {
                played = &over;
                corner = 0;
                behind = 0.0;
                behind_turns = 0.0;
                shift = by;
            }

            /**
             * What is swept by @p when, which must not go backwards.
             *
             * @param in_turns  If given, the fractional part in turns.
             */
            double upto(double when, double* in_turns = nullptr)
            {
                const size_t n = played->count();
                double partial = 0.0;
                while (corner + 1 < n)
                {
                    const double from = played->times[corner];
                    if (when <= from)
                        break;
                    const double to = played->times[corner + 1];
                    const double span = to - from;
                    if (span <= kEps)
                    {
                        ++corner;
                        continue;
                    }
                    const double first = (*played->values)[corner];
                    const double last = (*played->values)[corner + 1];
                    if (when >= to)
                    {
                        const double whole = 0.5 * (first + last) * span;
                        behind += whole;
                        behind_turns = turns(behind_turns + turns(whole * shift));
                        ++corner;
                        continue;
                    }
                    const double along = (when - from) / span;
                    const double reached = first + along * (last - first);
                    partial = 0.5 * (first + reached) * (when - from);
                    break;
                }
                if (in_turns != nullptr)
                    *in_turns = turns(behind_turns + turns(partial * shift));
                return behind + partial;
            }
        };

        /**
         * Move @p origin past the block @p row plays, and say what it swept.
         *
         * The trajectory restarts where an excitation acts and turns over
         * where a refocusing does, both at the pulse's own centre, so what
         * the block sweeps on either side of the pulse is counted on its own
         * side -- a refocusing between two crushers has each crusher on the
         * side it belongs to.
         */
        void advance_walk(
            const Sequence& seq,
            const int32_t* row,
            const Played played[3],
            double origin[3],
            double* swept_out = nullptr)
        {
            double acts_at = -1.0;
            char use = 'u';
            const int32_t rf_id = row[0];
            if (rf_id > 0)
            {
                const std::vector<char>& uses = seq.rf_uses();
                const double* rf = seq.rf_library().row(rf_id);
                acts_at = rf[5] + rf[4];
                use = rf_id <= static_cast<int32_t>(uses.size())
                    ? uses[static_cast<size_t>(rf_id) - 1]
                    : 'u';
            }

            double whole[3] = {0.0, 0.0, 0.0};
            double before[3] = {0.0, 0.0, 0.0};
            double at_pulse[3];
            for (int axis = 0; axis < 3; ++axis)
            {
                if (played[axis].values != nullptr)
                {
                    whole[axis] = played[axis].swept(1e30);
                    if (acts_at >= 0.0)
                        before[axis] = played[axis].swept(acts_at);
                }
                at_pulse[axis] = origin[axis] + before[axis];
                if (swept_out != nullptr)
                    swept_out[axis] = whole[axis];
            }

            if (acts_at >= 0.0 && advance_origin(use, at_pulse, origin))
            {
                for (int axis = 0; axis < 3; ++axis)
                    origin[axis] += whole[axis] - before[axis];
            }
            else
            {
                for (int axis = 0; axis < 3; ++axis)
                    origin[axis] += whole[axis];
            }
        }

        /**
         * The chain @p ext names, rebuilt without any node of @p type_id.
         *
         * A block turns one way or no way, so the rotation it carried is
         * replaced rather than added to. Everything else in the chain keeps
         * its order.
         */
        int32_t rechained_without(Sequence& seq, int32_t ext, int type_id)
        {
            std::vector<std::pair<int32_t, int32_t>> kept;
            for (int32_t node = ext; node > 0 && node <= seq.extensions_library().size();)
            {
                const int32_t* link = seq.extensions_library().row(node);
                if (link[0] != type_id)
                    kept.push_back({link[0], link[1]});
                node = link[2];
            }
            int32_t rebuilt = 0;
            for (size_t i = kept.size(); i-- > 0;)
            {
                rebuilt = static_cast<int32_t>(
                    seq.chain_extension(kept[i].first, kept[i].second, rebuilt));
            }
            return rebuilt;
        }

        /**
         * The rotation a block's row carries, as a matrix; false for none.
         *
         * A rotated block plays R g, so a translation d in the logical frame
         * is, along that block's channel axes, R^T d, and what it sweeps along
         * them is R times that in the logical frame.
         */
        bool block_rotation(const Sequence& seq, const int32_t* row, double matrix[3][3])
        {
            const int32_t turn = row[BLOCK_ROTATION_COLUMN];
            if (turn < 1 || turn > seq.rotation_library().size())
                return false;
            rotation_matrix(seq.rotation_library().row(turn), matrix);
            return true;
        }

        /** Turn @p vector by the transpose of @p matrix, in place. */
        void unrotate(const double matrix[3][3], double vector[3])
        {
            const double x = vector[0];
            const double y = vector[1];
            const double z = vector[2];
            for (int axis = 0; axis < 3; ++axis)
                vector[axis] =
                    matrix[0][axis] * x + matrix[1][axis] * y + matrix[2][axis] * z;
        }

        /**
         * advance_walk() for a block played through @p matrix: the walk is
         * taken along the block's channel axes and handed back in the logical
         * frame, where a reset and a turn-over mean the same.
         */
        void advance_turned_walk(
            const Sequence& seq,
            const int32_t* row,
            const Played played[3],
            const double matrix[3][3],
            double origin[3],
            double swept[3])
        {
            unrotate(matrix, origin);
            advance_walk(seq, row, played, origin, swept);
            rotate(matrix, origin);
            rotate(matrix, swept);
        }

    } // namespace

    bool advance_origin(char use, const double at[3], double origin[3])
    {
        /**
         * Treat undefined RF use as excitation, matching calculate_kspace().
         */
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


    std::array<std::vector<double>, 3> absolute_trajectory(
        const Sequence& seq, int block, const double origin[3])
    {
        std::array<std::vector<double>, 3> out;
        if (block < 1 || block > seq.num_blocks())
            return out;

        const int32_t* row =
            seq.block_events() + static_cast<size_t>(block - 1) * BLOCK_WIDTH;
        const int32_t adc_id = row[4];
        if (adc_id <= 0)
            return out;

        const double* adc = seq.adc_library().row(adc_id);
        const int samples = static_cast<int>(adc[0]);
        const double dwell = adc[1];
        const double delay = adc[2];
        if (samples <= 0)
            return out;

        CornerCache corners(seq);
        Played played;
        for (int axis = 0; axis < 3; ++axis)
        {
            const Corners& drawn = corners[row[1 + axis]];
            std::vector<double>& into = out[static_cast<size_t>(axis)];
            into.assign(static_cast<size_t>(samples), origin[axis]);
            if (drawn.values.empty())
                continue;
            /* An axis that is flat across the window still sweeps k, so it
             * gets its samples like any other; only an axis the block does
             * not drive at all is the origin repeated. */
            drawn.at(0.0, played.times);
            played.values = &drawn.values;
            Sweep along_axis;
            along_axis.restart(played, 0.0);
            for (int i = 0; i < samples; ++i)
            {
                const double when = delay + dwell * (static_cast<double>(i) + 0.5);
                into[static_cast<size_t>(i)] = origin[axis] + along_axis.upto(when);
            }
        }
        return out;
    }

    void apply_fov_scale(
        Sequence& seq, const double scale[3], int first, int last)
    {
        const int blocks = seq.num_blocks();
        const int from = first > 1 ? first : 1;
        const int to = (last > 0 && last < blocks) ? last : blocks;
        if (from > to)
            return;
        if (scale[0] == 1.0 && scale[1] == 1.0 && scale[2] == 1.0)
            return;

        /* One rewritten row per row-and-axis actually met: a phase-encode
         * table plays one readout ten thousand times, and it is one row
         * before and one row after. */
        std::map<std::pair<int32_t, int>, int32_t> rewritten;

        for (int index = from; index <= to; ++index)
        {
            Block block = seq.get_block(index);
            int32_t named[3] = {block.gx, block.gy, block.gz};
            for (int axis = 0; axis < 3; ++axis)
            {
                const int32_t id = named[axis];
                if (id <= 0 || scale[axis] == 1.0)
                    continue;
                const auto key = std::make_pair(id, axis);
                auto found = rewritten.find(key);
                if (found == rewritten.end())
                {
                    const int at = seq.grad_row(id);
                    if (seq.grad_kind(id) == GradKind::Trap)
                    {
                        double made[TRAP_WIDTH];
                        const double* was = seq.trap_library().row(at);
                        for (int c = 0; c < TRAP_WIDTH; ++c)
                            made[c] = was[c];
                        made[0] *= scale[axis];
                        found = rewritten.emplace(key, seq.register_trap(made)).first;
                    }
                    else
                    {
                        double made[ARB_WIDTH];
                        const double* was = seq.arb_library().row(at);
                        for (int c = 0; c < ARB_WIDTH; ++c)
                            made[c] = was[c];
                        made[0] *= scale[axis];
                        made[1] *= scale[axis];
                        made[2] *= scale[axis];
                        found =
                            rewritten.emplace(key, seq.register_arbitrary(made)).first;
                    }
                }
                named[axis] = found->second;
            }
            block.gx = named[0];
            block.gy = named[1];
            block.gz = named[2];
            seq.set_block(index, block);
        }
    }

    namespace
    {
        /**
         * The reflection half of an improper prescription M = R D: the
         * gradient on channel axis @p axis negated over blocks @p from to
         * @p to. Nothing for an axis of -1.
         */
        void negate_channel(Sequence& seq, int axis, int from, int to)
        {
            if (axis < -1 || axis > 2)
                throw std::invalid_argument("reflected_axis must be -1, 0, 1 or 2");
            if (axis < 0)
                return;
            double negate[3] = {1.0, 1.0, 1.0};
            negate[axis] = -1.0;
            apply_fov_scale(seq, negate, from, to);
        }

        /**
         * D R D, in place, for the reflection D of channel axis @p axis: the
         * same turn about the reflected axis, so the vector components off it
         * change sign.
         */
        void conjugate_by_reflection(double quaternion[ROTATION_WIDTH], int axis)
        {
            for (int other = 0; other < 3; ++other)
                if (other != axis)
                    quaternion[other + 1] = -quaternion[other + 1];
        }
    } // namespace

    void apply_fov_rotation(
        Sequence& seq, const double quaternion[4], int first, int last,
        int reflected_axis)
    {
        const int blocks = seq.num_blocks();
        const int from = first > 1 ? first : 1;
        const int to = (last > 0 && last < blocks) ? last : blocks;
        if (from > to)
            return;
        negate_channel(seq, reflected_axis, from, to);

        const int type_id = seq.extension_type_id("ROTATIONS");
        std::map<int32_t, int32_t> composed;

        for (int index = from; index <= to; ++index)
        {
            Block block = seq.get_block(index);
            const int32_t already = block.rot;

            int32_t turn = 0;
            auto found = composed.find(already);
            if (found != composed.end())
            {
                turn = found->second;
            }
            else
            {
                double made[ROTATION_WIDTH];
                if (already >= 1 && already <= seq.rotation_library().size())
                {
                    /* Applied after what the block already carries: a module
                     * that placed itself keeps its orientation inside the
                     * prescription's. */
                    double was[ROTATION_WIDTH];
                    std::copy_n(seq.rotation_library().row(already), ROTATION_WIDTH, was);
                    if (reflected_axis >= 0)
                        conjugate_by_reflection(was, reflected_axis);
                    made[0] = quaternion[0] * was[0] - quaternion[1] * was[1] -
                        quaternion[2] * was[2] - quaternion[3] * was[3];
                    made[1] = quaternion[0] * was[1] + quaternion[1] * was[0] +
                        quaternion[2] * was[3] - quaternion[3] * was[2];
                    made[2] = quaternion[0] * was[2] - quaternion[1] * was[3] +
                        quaternion[2] * was[0] + quaternion[3] * was[1];
                    made[3] = quaternion[0] * was[3] + quaternion[1] * was[2] -
                        quaternion[2] * was[1] + quaternion[3] * was[0];
                }
                else
                {
                    for (int i = 0; i < ROTATION_WIDTH; ++i)
                        made[i] = quaternion[i];
                }
                turn = static_cast<int32_t>(seq.register_rotation(made));
                composed.emplace(already, turn);
            }

            /* The chain is rebuilt without whatever rotation it carried, and
             * the new one put on the front: a block turns one way or no way. */
            block.ext = rechained_without(seq, block.ext, type_id);
            block.ext = static_cast<int32_t>(
                seq.chain_extension(type_id, turn, block.ext));
            seq.set_block(index, block);
        }
    }

    void apply_fov_shift(
        Sequence& seq,
        const double shift_m[3],
        FovShiftScope scope,
        int first,
        int last,
        double carry[3],
        double origin[3],
        const unsigned char* exempt,
        bool through_rotation)
    {
        const int blocks = seq.num_blocks();
        const int from = first > 1 ? first : 1;
        const int to = (last > 0 && last < blocks) ? last : blocks;
        if (from > to)
            return;

        bool moves = false;
        for (int axis = 0; axis < 3; ++axis)
            moves = moves || std::fabs(shift_m[axis]) > 0.0;

        if (!moves)
            return;

        CornerCache corners(seq);
        const int32_t* events = std::as_const(seq).block_events();
        Played played[3];

        /** An axis whose gradient moves under an event, and what it was worth. */
        struct Turning
        {
            int axis;
            double slope;
            double swept;
            double at;
        };
        std::vector<Turning> turning;
        std::vector<double> moment;
        std::vector<double> added;
        Sweep sweeping;
        MovedRows moved_rf;
        MovedRows moved_adc;
        if (scope == FovShiftScope::RfAndAdc)
            moved_adc.reserve(static_cast<size_t>(to - from + 1));
        ShapeMemo memo;

        for (int index = from; index <= to; ++index)
        {
            /* Copied: repointing the block below can move the table. */
            int32_t row[BLOCK_WIDTH];
            std::copy_n(
                std::as_const(seq).block_events() +
                    static_cast<size_t>(index - 1) * BLOCK_WIDTH,
                BLOCK_WIDTH, row);
            int32_t rf_to = row[0];
            int32_t adc_to = row[4];

            /* An exempt block is walked and not written: a module that placed
             * itself keeps the phase it was designed with, and what it swept
             * still counts towards where everything after it stands. */
            const bool writes =
                exempt == nullptr || exempt[static_cast<size_t>(index - from)] == 0;

            for (int axis = 0; axis < 3; ++axis)
            {
                const Corners& drawn = corners[row[1 + axis]];
                played[axis].values = drawn.values.empty() ? nullptr : &drawn.values;
                if (played[axis].values != nullptr)
                    drawn.at(0.0, played[axis].times);
            }

            /* What the block is entered with: the shift against everything
             * swept before it. Every event in the block carries it, and that
             * is the point -- what a readout is measured by is its phase
             * against the phase its own excitation was given, so the two have
             * to be counted from the same place. */
            double entering = 0.0;
            for (int axis = 0; axis < 3; ++axis)
                entering = turns(entering + turns(shift_m[axis] * carry[axis]));

            /* The translation along the block's channel axes: a rotated
             * block plays them turned. */
            double turned[3][3];
            const bool rotated = through_rotation && block_rotation(seq, row, turned);
            double shift[3] = {shift_m[0], shift_m[1], shift_m[2]};
            if (rotated)
                unrotate(turned, shift);

            const int32_t rf_id = row[0];
            if (rf_id > 0 && writes)
            {
                turning.clear();
                double rf[RF_WIDTH];
                std::copy_n(std::as_const(seq).rf_library().row(rf_id), RF_WIDTH, rf);
                const double delay = rf[5];
                /* The pulse acts at the centre its designer recorded, which
                 * is what the format carries the field for. */
                const double centre = delay + rf[4];
                const auto [opens, closes] = pulse_span(seq, rf, moment);
                double frequency = 0.0;
                double phase = entering;
                for (int axis = 0; axis < 3; ++axis)
                {
                    if (std::fabs(shift[axis]) == 0.0 || played[axis].values == nullptr)
                        continue;
                    /* A pulse under a gradient that does not change is a
                     * frequency and a phase; one under a gradient that does
                     * needs its shape, and is referenced to its own centre so
                     * that what the pulse does is untouched. Asked of the
                     * whole pulse: a gradient flat under the first half and
                     * ramping under the second is not a steady one. */
                    const bool steady = played[axis].constant_over(opens, closes);
                    const double at = steady ? delay : centre;
                    const double slope = played[axis].at(at);
                    const double swept = played[axis].swept_turns(at, shift[axis]);
                    frequency += shift[axis] * slope;
                    phase = turns(
                        phase +
                        turns(swept - shift[axis] * slope * (at - delay)));
                    if (!steady)
                        turning.push_back({axis, slope, swept, at});
                }
                rf[8] += frequency;
                rf[9] += 2.0 * kPi * phase;

                if (!turning.empty())
                {
                    /* The gradient moves under the pulse, so two numbers
                     * cannot say what the shift does to it: what is left over
                     * goes into the phase the pulse is played with, sample by
                     * sample and referenced to the pulse's own centre so that
                     * what the pulse *does* is untouched. */
                    memo.start(1.0);
                    for (const int field : {1, 2, 3, 4, 5})
                        memo.add(rf[field]);
                    for (const Turning& axis : turning)
                        memo.add(axis.axis, shift[axis.axis], axis.at, played[axis.axis]);
                    rf[2] = static_cast<double>(memo.row([&] {
                    added.assign(moment.size(), 0.0);
                    for (const Turning& axis : turning)
                    {
                        /* The pulse's samples run forwards, so the corners
                         * under them are walked once rather than once per
                         * sample. */
                        sweeping.restart(played[axis.axis], shift[axis.axis]);
                        for (size_t i = 0; i < moment.size(); ++i)
                        {
                            double swept_here = 0.0;
                            sweeping.upto(moment[i] + delay, &swept_here);
                            added[i] = turns(
                                added[i] +
                                turns(
                                    swept_here -
                                    axis.slope * (moment[i] - rf[4]) * shift[axis.axis] -
                                    axis.swept));
                        }
                    }
                    return phase_shape_with(seq, static_cast<int>(rf[2]), added, 1.0);
                    }));
                }
                const char use =
                    std::as_const(seq).rf_uses()[static_cast<size_t>(rf_id) - 1];
                rf_to = moved_rf.row({rf_id, rf[8], rf[9], rf[2]},
                                  [&] { return seq.register_rf(rf, use); });
            }

            const int32_t adc_id = row[4];
            if (adc_id > 0 && writes && scope == FovShiftScope::RfAndAdc)
            {
                turning.clear();
                double adc[ADC_WIDTH];
                std::copy_n(std::as_const(seq).adc_library().row(adc_id), ADC_WIDTH, adc);
                const int samples = static_cast<int>(adc[0]);
                const double dwell = adc[1];
                const double delay = adc[2];
                const double opens = delay + 0.5 * dwell;
                const double closes =
                    delay + (static_cast<double>(samples) - 0.5) * dwell;
                /* Referenced to the window's centre, as the reference
                 * toolbox does: the offsets carry the phase and frequency
                 * there, the modulation what curves about it. */
                const double echo = delay + 0.5 * dwell * static_cast<double>(samples);
                double frequency = 0.0;
                double phase = entering;
                for (int axis = 0; axis < 3; ++axis)
                {
                    if (std::fabs(shift[axis]) == 0.0 || played[axis].values == nullptr)
                        continue;
                    const bool steady = played[axis].constant_over(opens, closes);
                    const double at = steady ? delay : echo;
                    double slope = played[axis].at(at);
                    double swept = played[axis].swept_turns(at, shift[axis]);
                    if (!steady && samples >= 3)
                        centre_chord(played[axis], shift[axis], samples, dwell, delay, slope, swept);
                    frequency += shift[axis] * slope;
                    phase = turns(
                        phase +
                        turns(swept - shift[axis] * slope * (at - delay)));
                    if (!steady)
                        turning.push_back({axis, slope, swept, at});
                }
                adc[5] += frequency;
                adc[6] += 2.0 * kPi * phase;

                if (!turning.empty())
                {
                    /**
                     * Store residual phase curvature; constant gradients need no modulation shape.
                     */
                    memo.start(2.0);
                    for (const int field : {0, 1, 2, 7})
                        memo.add(adc[field]);
                    for (const Turning& axis : turning)
                        memo.add(axis.axis, shift[axis.axis], axis.at, played[axis.axis]);
                    adc[7] = static_cast<double>(memo.row([&] {
                    added.assign(static_cast<size_t>(samples), 0.0);
                    for (const Turning& axis : turning)
                    {
                        sweeping.restart(played[axis.axis], shift[axis.axis]);
                        for (int i = 0; i < samples; ++i)
                        {
                            const double when =
                                delay + dwell * (static_cast<double>(i) + 0.5);
                            double swept_here = 0.0;
                            sweeping.upto(when, &swept_here);
                            const double left = swept_here - axis.swept -
                                shift[axis.axis] * axis.slope * (when - axis.at);
                            added[static_cast<size_t>(i)] = turns(
                                added[static_cast<size_t>(i)] + turns(left));
                        }
                    }
                    for (size_t i = 0; i < added.size(); ++i)
                        added[i] *= 2.0 * kPi;
                    return phase_shape_with(seq, static_cast<int>(adc[7]), added, 2.0 * kPi);
                    }));
                }
                adc_to = moved_adc.row({adc_id, adc[5], adc[6], adc[7]},
                                   [&] { return seq.register_adc(adc); });
            }

            if (rf_to != row[0] || adc_to != row[4])
            {
                Block block = seq.get_block(index);
                block.rf = rf_to;
                block.adc = adc_to;
                seq.set_block(index, block);
            }

            /**
             * Advance the unbroken phase integral separately from the RF-reset origin.
             * RF and ADC must retain a common phase reference across excitation.
             */
            double swept[3];
            if (rotated)
                advance_turned_walk(seq, row, played, turned, origin, swept);
            else
                advance_walk(seq, row, played, origin, swept);
            for (int axis = 0; axis < 3; ++axis)
                carry[axis] += swept[axis];
        }
    }

    RfGradients rf_gradients(const Sequence& seq)
    {
        RfGradients out;
        CornerCache corners(seq);
        const int32_t* events = seq.block_events();
        Played played[3];
        std::vector<double> moment;
        for (int index = 1; index <= seq.num_blocks(); ++index)
        {
            const int32_t* row = events + static_cast<size_t>(index - 1) * BLOCK_WIDTH;
            if (row[0] <= 0)
                continue;
            const double* rf = seq.rf_library().row(row[0]);
            const double centre = rf[5] + rf[4];
            const auto [opens, closes] = pulse_span(seq, rf, moment);
            out.block.push_back(index);
            for (int axis = 0; axis < 3; ++axis)
            {
                const Corners& drawn = corners[row[1 + axis]];
                played[axis].values = drawn.values.empty() ? nullptr : &drawn.values;
                if (played[axis].values != nullptr)
                    drawn.at(0.0, played[axis].times);
                out.steady.push_back(played[axis].constant_over(opens, closes) ? 1 : 0);
                out.gradient.push_back(played[axis].at(centre));
            }
        }
        return out;
    }

    namespace
    {
        /* What a readout block sweeps from its start at each sample, and the
         * echoes found for each k it has started from. */
        struct Readout
        {
            std::array<std::vector<double>, 3> swept;
            std::map<std::array<double, 3>, std::array<int32_t, 2>> echoes;
            uint8_t moving[3] = {0, 0, 0};
        };

        /* Excitation, refocusing and inversion at the readout ask for the
         * reset-aware k-space the walk does not keep. */
        bool acquires_beside_rf_of_unknown_role(const Sequence& seq)
        {
            const int blocks = seq.num_blocks();
            const int32_t* events = seq.block_events();
            const std::vector<char>& uses = seq.rf_uses();
            for (int index = 1; index <= blocks; ++index)
            {
                const int32_t* row = events + static_cast<size_t>(index - 1) * BLOCK_WIDTH;
                if (row[4] <= 0 || row[0] <= 0)
                    continue;
                const size_t at = static_cast<size_t>(row[0]) - 1;
                const char use = at < uses.size() ? uses[at] : 'u';
                if (use == 'e' || use == 'u' || use == 'r')
                    return true;
            }
            return false;
        }

        void sweep_readout(const Played played[3], const double* adc, int n, bool rotated,
                           const double matrix[3][3], Readout& readout,
                           std::vector<double>& distance)
        {
            const double dwell = adc[1];
            const double delay = adc[2];
            for (int axis = 0; axis < 3; ++axis)
            {
                std::vector<double>& into = readout.swept[static_cast<size_t>(axis)];
                into.assign(static_cast<size_t>(n), 0.0);
                if (played[axis].values == nullptr)
                    continue;
                Sweep along;
                along.restart(played[axis], 0.0);
                for (int i = 0; i < n; ++i)
                    into[static_cast<size_t>(i)] =
                        along.upto(delay + dwell * (static_cast<double>(i) + 0.5));
            }
            if (rotated)
            {
                for (size_t i = 0; i < static_cast<size_t>(n); ++i)
                {
                    double v[3] = {readout.swept[0][i], readout.swept[1][i], readout.swept[2][i]};
                    rotate(matrix, v);
                    for (size_t axis = 0; axis < 3; ++axis)
                        readout.swept[axis][i] = v[axis];
                }
            }
            /* Which axes move does not depend on where the readout starts: a
             * span is a difference. */
            int32_t unused[2];
            find_echo(readout.swept, 0, n, readout.moving, unused, distance);
        }

        const std::array<int32_t, 2>& readout_echo(Readout& readout, const double origin[3], int n,
                                                   std::array<std::vector<double>, 3>& k,
                                                   std::vector<double>& distance)
        {
            std::array<double, 3> start{};
            for (size_t axis = 0; axis < 3; ++axis)
                start[axis] = readout.moving[axis] ? origin[axis] : 0.0;
            auto [echo, unseen] = readout.echoes.try_emplace(start);
            if (unseen)
            {
                for (size_t axis = 0; axis < 3; ++axis)
                {
                    k[axis].resize(static_cast<size_t>(n));
                    for (size_t i = 0; i < static_cast<size_t>(n); ++i)
                        k[axis][i] = origin[axis] + readout.swept[axis][i];
                }
                uint8_t moving[3];
                find_echo(k, 0, n, moving, echo->second.data(), distance);
            }
            return echo->second;
        }
    } // namespace

    bool walked_adc_echoes(const Sequence& seq, AdcEchoes& out)
    {
        if (acquires_beside_rf_of_unknown_role(seq))
            return false;

        const int blocks = seq.num_blocks();
        const int32_t* events = seq.block_events();
        const Table& adcs = seq.adc_library();
        std::map<std::array<int32_t, 5>, Readout> readouts;
        CornerCache corners(seq);
        Played played[3];
        double origin[3] = {0.0, 0.0, 0.0};
        double matrix[3][3];
        std::array<std::vector<double>, 3> k;
        std::vector<double> distance;
        int64_t samples = 0;
        for (int index = 1; index <= blocks; ++index)
        {
            const int32_t* row = events + static_cast<size_t>(index - 1) * BLOCK_WIDTH;
            const bool rotated = block_rotation(seq, row, matrix);
            for (int axis = 0; axis < 3; ++axis)
            {
                const Corners& drawn = corners[row[1 + axis]];
                played[axis].values = drawn.values.empty() ? nullptr : &drawn.values;
                if (played[axis].values != nullptr)
                    drawn.at(0.0, played[axis].times);
            }

            if (row[4] > 0)
            {
                const double* adc = adcs.row(row[4]);
                const int n = static_cast<int>(std::lround(adc[0]));
                out.block.push_back(index);
                out.num_samples.push_back(n);
                out.first_sample.push_back(samples);
                samples += n;

                const std::array<int32_t, 5> key{
                    row[1], row[2], row[3], row[4], rotated ? row[BLOCK_ROTATION_COLUMN] : 0};
                auto [found, fresh] = readouts.try_emplace(key);
                Readout& readout = found->second;
                if (fresh)
                    sweep_readout(played, adc, n, rotated, matrix, readout, distance);
                const std::array<int32_t, 2>& echo = readout_echo(readout, origin, n, k, distance);
                out.moving.insert(out.moving.end(), readout.moving, readout.moving + 3);
                out.echo.insert(out.echo.end(), echo.begin(), echo.end());
            }

            double swept[3];
            if (rotated)
                advance_turned_walk(seq, row, played, matrix, origin, swept);
            else
                advance_walk(seq, row, played, origin, swept);
        }
        return true;
    }

} // namespace pulseq
