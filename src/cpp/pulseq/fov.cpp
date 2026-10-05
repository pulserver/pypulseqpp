/**
 * @file fov.cpp
 * @brief Where the trajectory stands, block by block.  See fov.hpp.
 */

#include "pulseq/fov.hpp"

#include "pulseq/corners.hpp"
#include "pulseq/parallel.hpp"
#include "pulseq/shape.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

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
        /* The fraction of a turn, value - floor(value); the integer
         * conversion is what a target without SSE4.1 rounds with inline,
         * where std::floor is a library call. */
        double turns(double value)
        {
            if (!(std::fabs(value) < 4.5e15))
                return value - std::floor(value);
            double whole = static_cast<double>(static_cast<int64_t>(value));
            if (whole > value)
                whole -= 1.0;
            return value - whole;
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

        /** An axis whose gradient moves under an event, and what it was worth. */
        struct Turning
        {
            int axis;
            double slope;
            double swept;
            double at;
        };

        /**
         * What a shift along the block's channel axes adds to a readout's
         * frequency, in Hz, and phase, in turns from @p entering; the axes
         * whose gradient moves across the window go into @p turning.
         */
        void readout_offsets(
            const double* adc, const Played played[3], const double shift[3], double entering,
            std::vector<Turning>& turning, double& frequency, double& phase)
        {
            turning.clear();
            const int samples = static_cast<int>(adc[0]);
            const double dwell = adc[1];
            const double delay = adc[2];
            const double opens = delay + 0.5 * dwell;
            const double closes = delay + (static_cast<double>(samples) - 0.5) * dwell;
            /* Referenced to the window's centre, as the reference toolbox
             * does: the offsets carry the phase and frequency there, the
             * modulation what curves about it. */
            const double echo = delay + 0.5 * dwell * static_cast<double>(samples);
            frequency = 0.0;
            phase = entering;
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
                phase = turns(phase + turns(swept - shift[axis] * slope * (at - delay)));
                if (!steady)
                    turning.push_back({axis, slope, swept, at});
            }
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
            /** A key, built field by field. */
            class Key
            {
            public:
                void start(double tag)
                {
                    bytes_.clear();
                    add(tag);
                }

                void add(double value)
                {
                    bytes_.append(reinterpret_cast<const char*>(&value), sizeof value);
                }

                void add(int axis, double shift, double at, const Played& played)
                {
                    add(static_cast<double>(axis));
                    add(shift);
                    add(at);
                    const size_t n = played.count();
                    add(static_cast<double>(n));
                    bytes_.append(
                        reinterpret_cast<const char*>(played.times.data()), n * sizeof(double));
                    bytes_.append(
                        reinterpret_cast<const char*>(played.values->data()), n * sizeof(double));
                }

            private:
                friend class ShapeMemo;
                std::string bytes_;
            };

            void start(double tag) { key_.start(tag); }
            void add(double value) { key_.add(value); }
            void add(int axis, double shift, double at, const Played& played)
            {
                key_.add(axis, shift, at, played);
            }

            /** The row for the current key, made by @p make on first use. */
            template <typename Make> int row(Make make)
            {
                const auto found = rows_.find(key_.bytes_);
                if (found != rows_.end())
                    return found->second;
                const int made = make();
                rows_.emplace(key_.bytes_, made);
                return made;
            }

            /** The row already made for @p key, or -1. Safe beside other finds. */
            int find(const Key& key) const
            {
                const auto found = rows_.find(key.bytes_);
                return found == rows_.end() ? -1 : found->second;
            }

        private:
            Key key_;
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
                return row(key, hash(key), add);
            }

            /** row() for a key whose hash() is @p h. */
            template <typename Register>
            int32_t row(const MovedRow& key, uint64_t h, Register add)
            {
                if (2 * (entries_.size() + 1) > slots_.size())
                    grow();
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

            /** Start loading the slot a key of hash @p h is probed from. */
            void prefetch(uint64_t h) const
            {
#if defined(__GNUC__) || defined(__clang__)
                if (!slots_.empty())
                    __builtin_prefetch(&slots_[static_cast<size_t>(h) & (slots_.size() - 1)]);
#else
                (void)h;
#endif
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

        private:
            static uint64_t mix(uint64_t x)
            {
                x += 0x9E3779B97F4A7C15ull;
                x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
                x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
                return x ^ (x >> 31);
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
         * What a block sweeps along its channel axes, whole and up to the
         * centre of its pulse, and how that pulse moves the trajectory.
         */
        struct Walk
        {
            double whole[3] = {0.0, 0.0, 0.0};
            double before[3] = {0.0, 0.0, 0.0};
            bool pulsed = false;
            char use = 'u';

            /** Move @p origin past the block. */
            void move(double origin[3]) const
            {
                double at_pulse[3];
                for (int axis = 0; axis < 3; ++axis)
                    at_pulse[axis] = origin[axis] + before[axis];
                if (pulsed && advance_origin(use, at_pulse, origin))
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
        };

        void walk_of(const Sequence& seq, const int32_t* row, const Played played[3], Walk& walk)
        {
            double acts_at = -1.0;
            const int32_t rf_id = row[0];
            if (rf_id > 0)
            {
                const std::vector<char>& uses = seq.rf_uses();
                const double* rf = seq.rf_library().row(rf_id);
                acts_at = rf[5] + rf[4];
                walk.use = rf_id <= static_cast<int32_t>(uses.size())
                    ? uses[static_cast<size_t>(rf_id) - 1]
                    : 'u';
            }
            walk.pulsed = acts_at >= 0.0;
            for (int axis = 0; axis < 3; ++axis)
            {
                if (played[axis].values == nullptr)
                    continue;
                walk.whole[axis] = played[axis].swept(1e30);
                if (walk.pulsed)
                    walk.before[axis] = played[axis].swept(acts_at);
            }
        }

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
            Walk walk;
            walk_of(seq, row, played, walk);
            walk.move(origin);
            if (swept_out != nullptr)
                for (int axis = 0; axis < 3; ++axis)
                    swept_out[axis] = walk.whole[axis];
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
        /* A block's rotation is read off its chain, so every block naming
         * one chain is given one rebuilt chain. */
        std::unordered_map<int32_t, int32_t> rechained;

        for (int index = from; index <= to; ++index)
        {
            const int32_t was_chain =
                std::as_const(seq).block_events()[static_cast<size_t>(index - 1) * BLOCK_WIDTH + 5];
            const auto known = rechained.find(was_chain);
            if (known != rechained.end())
            {
                seq.set_block_ext(index, known->second);
                continue;
            }
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
            rechained.emplace(was_chain, block.ext);
            seq.set_block_ext(index, block.ext);
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

        const Sequence& view = seq;
        CornerCache corners(view);
        /* Every gradient's corners are worked out here, so the threads below
         * only read the cache. */
        for (int32_t id = 1; id <= view.num_gradients(); ++id)
            corners[id];

        std::vector<Turning> turning;
        std::vector<double> moment;
        std::vector<double> added;
        Sweep sweeping;
        MovedRows moved_rf;
        MovedRows moved_adc;
        if (scope == FovShiftScope::RfAndAdc)
            moved_adc.reserve(static_cast<size_t>(to - from + 1));
        ShapeMemo memo;

        /** A block of the window being shifted. */
        struct Step
        {
            Walk walk;
            double turned[3][3];
            bool rotated;
            /* The shift against everything swept before the block. */
            double entering;
            /* A readout under steady gradients, and its moved offsets. */
            bool readout;
            double frequency;
            double phase;
            double shape;
            uint64_t key;
        };

        const auto draw = [&corners](const int32_t* row, Played played[3])
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                const Corners& drawn = corners[row[1 + axis]];
                played[axis].values = drawn.values.empty() ? nullptr : &drawn.values;
                if (played[axis].values != nullptr)
                    drawn.at(0.0, played[axis].times);
            }
        };
        /* The translation along the block's channel axes: a rotated block
         * plays them turned. */
        const auto shift_of = [shift_m](const Step& step, double shift[3])
        {
            for (int axis = 0; axis < 3; ++axis)
                shift[axis] = shift_m[axis];
            if (step.rotated)
                unrotate(step.turned, shift);
        };
        /* An exempt block is walked and not written: a module that placed
         * itself keeps the phase it was designed with, and what it swept
         * still counts towards where everything after it stands. */
        const auto writes_at = [exempt, from](int index)
        { return exempt == nullptr || exempt[static_cast<size_t>(index - from)] == 0; };

        /* Windows of blocks: what each block sweeps and the readouts under
         * steady gradients are worked out in parallel, the running integral
         * and every registration in block order, so the result does not
         * depend on the number of threads. */
        constexpr int kWindow = 1 << 16;
        const unsigned workers = worker_count(static_cast<size_t>(to - from + 1));
        std::vector<Step> steps(static_cast<size_t>(std::min(kWindow, to - from + 1)));
        for (int start = from; start <= to; start += kWindow)
        {
            const int count = std::min(kWindow, to - start + 1);
            const int32_t* events =
                view.block_events() + static_cast<size_t>(start - 1) * BLOCK_WIDTH;

            parallel_ranges(static_cast<size_t>(count), workers, [&](size_t lo, size_t hi)
            {
                Played played[3];
                for (size_t i = lo; i < hi; ++i)
                {
                    const int32_t* row = events + i * BLOCK_WIDTH;
                    Step& step = steps[i];
                    step.walk = Walk();
                    draw(row, played);
                    walk_of(view, row, played, step.walk);
                    step.rotated = through_rotation && block_rotation(view, row, step.turned);
                    step.readout = false;
                }
            });

            /**
             * Advance the unbroken phase integral separately from the RF-reset origin.
             * RF and ADC must retain a common phase reference across excitation.
             */
            for (int i = 0; i < count; ++i)
            {
                Step& step = steps[static_cast<size_t>(i)];
                /* What the block is entered with: the shift against everything
                 * swept before it. Every event in the block carries it, and
                 * that is the point -- what a readout is measured by is its
                 * phase against the phase its own excitation was given, so
                 * the two have to be counted from the same place. */
                double entering = 0.0;
                for (int axis = 0; axis < 3; ++axis)
                    entering = turns(entering + turns(shift_m[axis] * carry[axis]));
                step.entering = entering;

                double swept[3] = {step.walk.whole[0], step.walk.whole[1], step.walk.whole[2]};
                if (step.rotated)
                {
                    unrotate(step.turned, origin);
                    step.walk.move(origin);
                    rotate(step.turned, origin);
                    rotate(step.turned, swept);
                }
                else
                {
                    step.walk.move(origin);
                }
                for (int axis = 0; axis < 3; ++axis)
                    carry[axis] += swept[axis];
            }

            if (scope == FovShiftScope::RfAndAdc)
            {
                parallel_ranges(static_cast<size_t>(count), workers, [&](size_t lo, size_t hi)
                {
                    Played played[3];
                    std::vector<Turning> bent;
                    ShapeMemo::Key shape_key;
                    for (size_t i = lo; i < hi; ++i)
                    {
                        const int32_t* row = events + i * BLOCK_WIDTH;
                        if (row[4] <= 0 || !writes_at(start + static_cast<int>(i)))
                            continue;
                        Step& step = steps[i];
                        draw(row, played);
                        double shift[3];
                        shift_of(step, shift);
                        const double* adc = view.adc_library().row(row[4]);
                        double frequency = 0.0;
                        double phase = 0.0;
                        readout_offsets(adc, played, shift, step.entering, bent, frequency, phase);
                        double shape = adc[7];
                        if (!bent.empty())
                        {
                            /* A phase shape already made is looked up; a new
                             * one is made below, in order. */
                            shape_key.start(2.0);
                            for (const int field : {0, 1, 2, 7})
                                shape_key.add(adc[field]);
                            for (const Turning& axis : bent)
                                shape_key.add(axis.axis, shift[axis.axis], axis.at, played[axis.axis]);
                            const int made = memo.find(shape_key);
                            if (made < 0)
                                continue;
                            shape = static_cast<double>(made);
                        }
                        step.readout = true;
                        step.frequency = adc[5] + frequency;
                        step.phase = adc[6] + 2.0 * kPi * phase;
                        step.shape = shape;
                        step.key = MovedRows::hash({row[4], step.frequency, step.phase, shape});
                    }
                });
            }

            /* How far ahead a readout's slot is loaded: the table is far
             * larger than the cache and nearly every readout lands on a new
             * slot. */
            constexpr int kAhead = 16;
            Played played[3];
            for (int i = 0; i < count; ++i)
            {
                const int index = start + i;
                const Step& step = steps[static_cast<size_t>(i)];
                if (i + kAhead < count && steps[static_cast<size_t>(i + kAhead)].readout)
                    moved_adc.prefetch(steps[static_cast<size_t>(i + kAhead)].key);
                /* Copied: repointing the block below can move the table. */
                int32_t row[BLOCK_WIDTH];
                std::copy_n(
                    view.block_events() + static_cast<size_t>(index - 1) * BLOCK_WIDTH,
                    BLOCK_WIDTH, row);
                int32_t rf_to = row[0];
                int32_t adc_to = row[4];
                const bool writes = writes_at(index);
                const bool serial_readout =
                    row[4] > 0 && writes && scope == FovShiftScope::RfAndAdc && !step.readout;
                const bool serial_pulse = row[0] > 0 && writes;
                if (!serial_pulse && !serial_readout && !step.readout)
                    continue;
                if (serial_pulse || serial_readout)
                    draw(row, played);
                double shift[3];
                shift_of(step, shift);
                const double entering = step.entering;

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
                if (step.readout)
                {
                    double adc[ADC_WIDTH];
                    std::copy_n(std::as_const(seq).adc_library().row(adc_id), ADC_WIDTH, adc);
                    adc[5] = step.frequency;
                    adc[6] = step.phase;
                    adc[7] = step.shape;
                    adc_to = moved_adc.row({adc_id, adc[5], adc[6], adc[7]}, step.key,
                                       [&] { return seq.register_adc(adc); });
                }
                else if (adc_id > 0 && writes && scope == FovShiftScope::RfAndAdc)
                {
                    turning.clear();
                    double adc[ADC_WIDTH];
                    std::copy_n(std::as_const(seq).adc_library().row(adc_id), ADC_WIDTH, adc);
                    const int samples = static_cast<int>(adc[0]);
                    const double dwell = adc[1];
                    const double delay = adc[2];
                    double frequency = 0.0;
                    double phase = 0.0;
                    readout_offsets(adc, played, shift, entering, turning, frequency, phase);
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
                    seq.set_block_rf_adc(index, rf_to, adc_to);
            }
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
        /* What a readout block sweeps from its first sample at each sample,
         * kept once among the sweeps that are equal, what it sweeps from its
         * start to the first sample, and the echoes found for each k it has
         * started from. */
        struct Readout
        {
            int32_t id = 0;
            std::array<double, 3> lead{};
            std::map<std::array<double, 3>, std::array<int32_t, 2>> echoes;
            uint8_t moving[3] = {0, 0, 0};
            /** Swept in a batch not yet finished. */
            bool pending = false;
        };

        /* A readout path from its first sample, axis after axis, and its hash. */
        struct Path
        {
            std::vector<double> samples;
            size_t hash = 0;

            Path(const std::array<std::vector<double>, 3>& swept,
                 const std::array<double, 3>& lead, size_t m)
                : samples(3 * m), hash(3 * m)
            {
                for (size_t axis = 0; axis < 3; ++axis)
                    for (size_t i = 0; i < m; ++i)
                        samples[axis * m + i] = swept[axis][i] - lead[axis];
                for (const double v : samples)
                {
                    const float single = static_cast<float>(v);
                    uint32_t bits;
                    std::memcpy(&bits, &single, sizeof bits);
                    hash ^= std::hash<uint32_t>{}(bits) + 0x9e3779b97f4a7c15ULL + (hash << 6) +
                        (hash >> 2);
                }
            }
            /* Equal in single precision, which an MRD trajectory carries: a
             * join to the block before rounds differently under each rotation. */
            bool operator==(const Path& other) const
            {
                if (samples.size() != other.samples.size())
                    return false;
                for (size_t i = 0; i < samples.size(); ++i)
                    if (static_cast<float>(samples[i]) != static_cast<float>(other.samples[i]))
                        return false;
                return true;
            }
        };
        struct PathHash
        {
            size_t operator()(const Path& path) const { return path.hash; }
        };

        /* Distinct readout paths from the first sample, numbered in order of
         * appearance, until dropped. */
        class DistinctPaths
        {
          public:
            explicit DistinctPaths(size_t limit) : limit_(limit) {}

            /* The number of a path already kept, or -1. Safe across threads
             * while none adds. */
            int32_t find(const Path& path) const
            {
                const auto known = index_.find(path);
                return known == index_.end() ? -1 : known->second;
            }

            /* The path's number, or -1 once more than the limit are distinct. */
            int32_t add(Path&& path)
            {
                if (!keep_)
                    return -1;
                auto [known, unseen] =
                    index_.try_emplace(std::move(path), static_cast<int32_t>(by_id_.size()));
                if (unseen)
                    by_id_.push_back(&known->first.samples);
                if (by_id_.size() <= limit_)
                    return known->second;
                keep_ = false;
                decltype(index_)().swap(index_);
                decltype(by_id_)().swap(by_id_);
                return -1;
            }
            bool kept() const { return keep_; }
            void export_to(std::vector<std::array<std::vector<double>, 3>>& sweeps) const
            {
                for (const std::vector<double>* path : by_id_)
                {
                    const size_t m = path->size() / 3;
                    std::array<std::vector<double>, 3> kept;
                    for (size_t axis = 0; axis < 3; ++axis)
                        kept[axis].assign(path->begin() + static_cast<std::ptrdiff_t>(axis * m),
                                          path->begin() + static_cast<std::ptrdiff_t>((axis + 1) * m));
                    sweeps.push_back(std::move(kept));
                }
            }

          private:
            std::unordered_map<Path, int32_t, PathHash> index_;
            std::vector<const std::vector<double>*> by_id_;
            size_t limit_;
            bool keep_ = true;
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

        /* One axis swept from the block's start at each sample, memoized by
         * the gradient and the ADC, which are all it depends on. */
        using AxisSweeps = std::map<std::array<int32_t, 2>, std::vector<double>>;
        constexpr size_t MEMO_LIMIT = 4096;
        /** How near two gradient corners are to be one, as `adc_kspace` joins them. */
        constexpr double kJoined = 1e-9;

        /* One axis swept from the block's start at each sample. */
        void fill_axis_sweep(const Played& played, const double* adc, int n,
                             std::vector<double>& into)
        {
            into.assign(static_cast<size_t>(n), 0.0);
            if (played.values == nullptr)
                return;
            const double dwell = adc[1];
            const double delay = adc[2];
            Sweep along;
            along.restart(played, 0.0);
            for (int i = 0; i < n; ++i)
                into[static_cast<size_t>(i)] =
                    along.upto(delay + dwell * (static_cast<double>(i) + 0.5));
        }

        const std::vector<double>& axis_sweep(const Played& played, int32_t gradient,
                                              int32_t adc_id, const double* adc, int n,
                                              AxisSweeps& memo)
        {
            auto [found, fresh] = memo.try_emplace(std::array<int32_t, 2>{gradient, adc_id});
            if (fresh)
                fill_axis_sweep(played, adc, n, found->second);
            return found->second;
        }

        void rotate_sweep(int n, const double matrix[3][3],
                          std::array<std::vector<double>, 3>& swept)
        {
            for (size_t i = 0; i < static_cast<size_t>(n); ++i)
            {
                double v[3] = {swept[0][i], swept[1][i], swept[2][i]};
                rotate(matrix, v);
                for (size_t axis = 0; axis < 3; ++axis)
                    swept[axis][i] = v[axis];
            }
        }

        /* The readout's sweep from its block's start, rotated, into @p swept. */
        void sweep_readout(const Played played[3], const int32_t* row, const double* adc, int n,
                           bool rotated, const double matrix[3][3], AxisSweeps& memo,
                           std::array<std::vector<double>, 3>& swept)
        {
            for (int axis = 0; axis < 3; ++axis)
                swept[static_cast<size_t>(axis)] =
                    axis_sweep(played[axis], row[1 + axis], row[4], adc, n, memo);
            if (rotated)
                rotate_sweep(n, matrix, swept);
        }

        /* The readout's echo when it starts at @p origin; @p swept is the
         * readout's sweep from its block's start, filled by @p sweep on
         * demand. */
        template <class Fill>
        const std::array<int32_t, 2>& readout_echo(Readout& readout, const double origin[3],
                                                   Fill&& sweep, int n,
                                                   std::array<std::vector<double>, 3>& swept,
                                                   std::array<std::vector<double>, 3>& k,
                                                   std::vector<double>& distance)
        {
            std::array<double, 3> start{};
            for (size_t axis = 0; axis < 3; ++axis)
                start[axis] = readout.moving[axis] ? origin[axis] + readout.lead[axis] : 0.0;
            auto [echo, unseen] = readout.echoes.try_emplace(start);
            if (unseen)
            {
                sweep();
                const size_t m = static_cast<size_t>(n);
                for (size_t axis = 0; axis < 3; ++axis)
                {
                    k[axis].resize(m);
                    for (size_t i = 0; i < m; ++i)
                        k[axis][i] = start[axis] + swept[axis][i] - readout.lead[axis];
                }
                uint8_t moving[3];
                find_echo(k, 0, n, moving, echo->second.data(), distance);
            }
            return echo->second;
        }
        /** Zero the weights `waveforms_and_times` does not play an axis with. */
        void drop_faint_weights(double matrix[3][3])
        {
            for (int into = 0; into < 3; ++into)
                for (int from = 0; from < 3; ++from)
                    if (std::fabs(matrix[into][from]) < 1e-6)
                        matrix[into][from] = 0.0;
        }

        /**
         * What the gradients of the block before left, for the block after
         * to start from as `adc_kspace` joins them: on every physical axis
         * when all three play on both sides, else axis by axis between two
         * unrotated blocks. A file can store the two sides of a join with
         * different rounding.
         */
        class Carry
        {
          public:
            /** Give @p played the values carried where they start; true when one changed. */
            bool join(Played played[3], bool rotated, const double matrix[3][3],
                      const bool starts[3])
            {
                double step[3] = {0.0, 0.0, 0.0};
                if (all(starts) && all(carries_))
                    physical_step(played, rotated, matrix, step);
                else if (!rotated && !rotated_)
                    axis_step(played, starts, step);
                bool changed = false;
                for (int axis = 0; axis < 3; ++axis)
                {
                    if (step[axis] == 0.0)
                        continue;
                    joined_[axis] = *played[axis].values;
                    joined_[axis][0] += step[axis];
                    played[axis].values = &joined_[axis];
                    changed = true;
                }
                return changed;
            }

            /** Remember where the block's gradients end. */
            void record(const Played played[3], bool rotated, const double matrix[3][3],
                        const bool ends[3])
            {
                for (int axis = 0; axis < 3; ++axis)
                {
                    carries_[axis] = ends[axis];
                    carried_[axis] = ends[axis] ? played[axis].values->back() : 0.0;
                }
                rotated_ = rotated;
                if (rotated && all(ends))
                    rotate(matrix, carried_);
            }

          private:
            static bool all(const bool flags[3]) { return flags[0] && flags[1] && flags[2]; }

            void physical_step(const Played played[3], bool rotated, const double matrix[3][3],
                               double step[3]) const
            {
                double first[3];
                for (int axis = 0; axis < 3; ++axis)
                    first[axis] = played[axis].values->front();
                if (rotated)
                    rotate(matrix, first);
                for (int axis = 0; axis < 3; ++axis)
                    step[axis] = carried_[axis] - first[axis];
                if (rotated)
                    unrotate(matrix, step);
            }

            void axis_step(const Played played[3], const bool starts[3], double step[3]) const
            {
                for (int axis = 0; axis < 3; ++axis)
                    if (starts[axis] && carries_[axis])
                        step[axis] = carried_[axis] - played[axis].values->front();
            }

            double carried_[3] = {0.0, 0.0, 0.0};
            bool carries_[3] = {false, false, false};
            bool rotated_ = false;
            std::array<std::vector<double>, 3> joined_;
        };

        /** Set @p played to a block's gradients, and whether each starts and ends with it. */
        void load_block(CornerCache& corners, const int32_t* row, double duration,
                        Played played[3], bool starts[3], bool ends[3])
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                const Corners& drawn = corners[row[1 + axis]];
                played[axis].values = drawn.values.empty() ? nullptr : &drawn.values;
                starts[axis] = ends[axis] = false;
                if (played[axis].values == nullptr)
                    continue;
                drawn.at(0.0, played[axis].times);
                starts[axis] = played[axis].times.front() <= kJoined;
                ends[axis] = played[axis].times.back() >= duration - kJoined;
            }
        }

        /** Each readout's moving axes, echo, origin and sweep, appended to an AdcEchoes. */
        class EchoWalk
        {
          public:
            EchoWalk(size_t distinct_limit, AdcEchoes& out) : paths_(distinct_limit), out_(out) {}

            /**
             * One readout, played as @p played from @p origin; @p joined if it was.
             *
             * A readout whose sweep is not known yet is swept in a batch,
             * across threads; its entries are filled when the batch is.
             */
            void readout(const int32_t* row, const double* adc, const Played played[3],
                         bool rotated, const double matrix[3][3], bool joined,
                         const double origin[3])
            {
                out_.rotation.push_back(rotated ? rotation_id(matrix) : -1);
                /* A joined readout is not the readout its ids name. */
                Readout* readout = nullptr;
                if (!joined)
                {
                    const std::array<int32_t, 5> key{
                        row[1], row[2], row[3], row[4], rotated ? row[BLOCK_ROTATION_COLUMN] : 0};
                    auto [found, fresh] = readouts_.try_emplace(key);
                    readout = &found->second;
                    if (!fresh)
                    {
                        if (readout->pending)
                            finish_batch();
                        known(*readout, row, adc, played, rotated, matrix, origin);
                        return;
                    }
                    readout->pending = true;
                }
                defer(readout, row, adc, played, rotated, matrix, origin);
                if (used_ >= kBatch)
                    finish_batch();
            }

            /** The distinct sweeps, or no origins and sweeps once too many were. */
            void finish()
            {
                finish_batch();
                if (paths_.kept())
                {
                    paths_.export_to(out_.sweeps);
                    return;
                }
                out_.origin.clear();
                out_.sweep.clear();
                out_.rotation.clear();
                out_.rotations.clear();
            }

          private:
            /** Readouts swept together. */
            static constexpr size_t kBatch = 4096;

            /* A readout to sweep, with what it was played as, and its results. */
            struct Pending
            {
                Readout* readout;
                const int32_t* row;
                const double* adc;
                int n;
                bool rotated;
                double matrix[3][3];
                double origin[3];
                std::array<std::vector<double>, 3> values;
                std::array<std::vector<double>, 3> times;
                std::array<std::vector<double>, 3> swept;
                std::array<double, 3> lead{};
                std::array<int32_t, 2> echo{};
                uint8_t moving[3] = {0, 0, 0};
                size_t slot = 0;
                std::unique_ptr<Path> path;
                /** The number of a path kept before the batch, or -1. */
                int32_t known_path = -1;
            };

            /* The readout's rotation, numbered among the distinct ones. */
            int32_t rotation_id(const double matrix[3][3])
            {
                std::array<double, 9> key;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        key[static_cast<size_t>(3 * a + b)] = matrix[a][b];
                auto [found, fresh] =
                    rotation_ids_.try_emplace(key, static_cast<int32_t>(rotation_ids_.size()));
                if (fresh)
                    out_.rotations.insert(out_.rotations.end(), key.begin(), key.end());
                return found->second;
            }

            /* A readout whose sweep is known, from a new origin. */
            void known(Readout& readout, const int32_t* row, const double* adc,
                       const Played played[3], bool rotated, const double matrix[3][3],
                       const double origin[3])
            {
                const int n = static_cast<int>(std::lround(adc[0]));
                const auto sweep = [&] {
                    if (memo_.size() > MEMO_LIMIT)
                        memo_.clear();
                    sweep_readout(played, row, adc, n, rotated, matrix, memo_, swept_);
                };
                const std::array<int32_t, 2>& echo =
                    readout_echo(readout, origin, sweep, n, swept_, k_, distance_);
                out_.moving.insert(out_.moving.end(), readout.moving, readout.moving + 3);
                out_.echo.insert(out_.echo.end(), echo.begin(), echo.end());
                for (size_t axis = 0; axis < 3; ++axis)
                    out_.origin.push_back(origin[axis] + readout.lead[axis]);
                out_.sweep.push_back(readout.id);
            }

            void defer(Readout* readout, const int32_t* row, const double* adc,
                       const Played played[3], bool rotated, const double matrix[3][3],
                       const double origin[3])
            {
                if (used_ == batch_.size())
                    batch_.emplace_back();
                Pending& job = batch_[used_++];
                job.readout = readout;
                job.row = row;
                job.adc = adc;
                job.n = static_cast<int>(std::lround(adc[0]));
                job.rotated = rotated;
                for (int a = 0; a < 3; ++a)
                {
                    job.origin[a] = origin[a];
                    for (int b = 0; b < 3; ++b)
                        job.matrix[a][b] = rotated ? matrix[a][b] : 0.0;
                    std::vector<double>& values = job.values[static_cast<size_t>(a)];
                    std::vector<double>& times = job.times[static_cast<size_t>(a)];
                    if (played[a].values != nullptr)
                    {
                        values.assign(played[a].values->begin(), played[a].values->end());
                        times.assign(played[a].times.begin(), played[a].times.end());
                    }
                    else
                    {
                        values.clear();
                        times.clear();
                    }
                }
                job.slot = out_.sweep.size();
                out_.moving.insert(out_.moving.end(), 3, 0);
                out_.echo.insert(out_.echo.end(), 2, -1);
                out_.origin.insert(out_.origin.end(), 3, 0.0);
                out_.sweep.push_back(-1);
            }

            /* Sweep a readout, find which axes move and its echo from its
             * origin; with @p keep, its path before the rotation. */
            static void sweep_one(Pending& job, bool keep, std::array<std::vector<double>, 3>& k,
                                  std::vector<double>& distance)
            {
                Played played[3];
                for (size_t a = 0; a < 3; ++a)
                    if (!job.values[a].empty())
                    {
                        played[a].values = &job.values[a];
                        played[a].times = std::move(job.times[a]);
                    }
                for (size_t a = 0; a < 3; ++a)
                    fill_axis_sweep(played[a], job.adc, job.n, job.swept[a]);
                const size_t m = static_cast<size_t>(job.n);
                if (keep)
                {
                    std::array<double, 3> lead{};
                    for (size_t axis = 0; axis < 3; ++axis)
                        lead[axis] = m > 0 ? job.swept[axis][0] : 0.0;
                    job.path = std::make_unique<Path>(job.swept, lead, m);
                }
                if (job.rotated)
                    rotate_sweep(job.n, job.matrix, job.swept);
                moving_axes(job.swept, 0, job.n, job.moving);
                for (size_t axis = 0; axis < 3; ++axis)
                {
                    job.lead[axis] = m > 0 ? job.swept[axis][0] : 0.0;
                    const double start =
                        job.moving[axis] ? job.origin[axis] + job.lead[axis] : 0.0;
                    k[axis].resize(m);
                    for (size_t i = 0; i < m; ++i)
                        k[axis][i] = start + job.swept[axis][i] - job.lead[axis];
                }
                uint8_t unused[3];
                find_echo(k, 0, job.n, unused, job.echo.data(), distance);
                for (size_t a = 0; a < 3; ++a)
                    job.times[a] = std::move(played[a].times);
            }

            void finish_batch()
            {
                if (used_ == 0)
                    return;
                const bool keep = paths_.kept();
                parallel_ranges(used_, worker_count(used_, 64),
                                [&](size_t first, size_t last) {
                                    std::array<std::vector<double>, 3> k;
                                    std::vector<double> distance;
                                    for (size_t i = first; i < last; ++i)
                                    {
                                        Pending& job = batch_[i];
                                        sweep_one(job, keep, k, distance);
                                        job.known_path = keep ? paths_.find(*job.path) : -1;
                                    }
                                });
                for (size_t i = 0; i < used_; ++i)
                {
                    Pending& job = batch_[i];
                    int32_t id = -1;
                    if (keep && paths_.kept())
                        id = job.known_path >= 0 ? job.known_path
                                                 : paths_.add(std::move(*job.path));
                    job.path.reset();
                    if (job.readout != nullptr)
                    {
                        Readout& readout = *job.readout;
                        readout.id = id;
                        readout.lead = job.lead;
                        std::copy(job.moving, job.moving + 3, readout.moving);
                        std::array<double, 3> start{};
                        for (size_t axis = 0; axis < 3; ++axis)
                            start[axis] =
                                readout.moving[axis] ? job.origin[axis] + readout.lead[axis] : 0.0;
                        readout.echoes.emplace(start, job.echo);
                        readout.pending = false;
                    }
                    std::copy(job.moving, job.moving + 3, out_.moving.begin() + 3 * job.slot);
                    std::copy(job.echo.begin(), job.echo.end(), out_.echo.begin() + 2 * job.slot);
                    for (size_t axis = 0; axis < 3; ++axis)
                        out_.origin[3 * job.slot + axis] = job.origin[axis] + job.lead[axis];
                    out_.sweep[job.slot] = id;
                }
                used_ = 0;
            }

            std::map<std::array<int32_t, 5>, Readout> readouts_;
            std::map<std::array<double, 9>, int32_t> rotation_ids_;
            AxisSweeps memo_;
            DistinctPaths paths_;
            AdcEchoes& out_;
            std::vector<Pending> batch_;
            size_t used_ = 0;
            std::array<std::vector<double>, 3> k_;
            std::array<std::vector<double>, 3> swept_;
            std::vector<double> distance_;
        };

    } // namespace

    bool walked_adc_echoes(const Sequence& seq, AdcEchoes& out)
    {
        if (acquires_beside_rf_of_unknown_role(seq))
            return false;

        const int blocks = seq.num_blocks();
        const int32_t* events = seq.block_events();
        const Table& adcs = seq.adc_library();
        const double* durations = seq.block_durations();
        /* Sweeps are kept while one readout in 16 at most has a path of its
         * own; past that they would hold most of the samples. */
        size_t acquiring = 0;
        for (int index = 1; index <= blocks; ++index)
            acquiring += events[static_cast<size_t>(index - 1) * BLOCK_WIDTH + 4] > 0;
        out.block.reserve(acquiring);
        out.num_samples.reserve(acquiring);
        out.first_sample.reserve(acquiring);
        out.moving.reserve(3 * acquiring);
        out.echo.reserve(2 * acquiring);
        out.origin.reserve(3 * acquiring);
        out.sweep.reserve(acquiring);
        out.rotation.reserve(acquiring);
        EchoWalk echoes(std::max<size_t>(4096, acquiring / 16), out);
        Carry carry;
        CornerCache corners(seq);
        Played played[3];
        double origin[3] = {0.0, 0.0, 0.0};
        double matrix[3][3];
        int64_t samples = 0;
        for (int index = 1; index <= blocks; ++index)
        {
            const int32_t* row = events + static_cast<size_t>(index - 1) * BLOCK_WIDTH;
            const bool rotated = block_rotation(seq, row, matrix);
            if (rotated)
                drop_faint_weights(matrix);
            bool starts[3];
            bool ends[3];
            load_block(corners, row, durations[index - 1], played, starts, ends);
            const bool joined = carry.join(played, rotated, matrix, starts);
            carry.record(played, rotated, matrix, ends);

            if (row[4] > 0)
            {
                const double* adc = adcs.row(row[4]);
                const int n = static_cast<int>(std::lround(adc[0]));
                out.block.push_back(index);
                out.num_samples.push_back(n);
                out.first_sample.push_back(samples);
                samples += n;
                echoes.readout(row, adc, played, rotated, matrix, joined, origin);
            }

            double step[3];
            if (rotated)
                advance_turned_walk(seq, row, played, matrix, origin, step);
            else
                advance_walk(seq, row, played, origin, step);
        }
        echoes.finish();
        return true;
    }

} // namespace pulseq
