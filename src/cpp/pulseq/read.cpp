/**
 * @file read.cpp
 * @brief Pulseq text parsing and conversion of supported legacy versions.
 *
 * Parsing precedes registration because blocks may refer to libraries later
 * in the file. Legacy conversion derives RF centres, gradient endpoints and
 * block durations before building the sequence.
 */

#include "pulseq/read.hpp"

#include "pulseq/md5.hpp"
#include "pulseq/shape.hpp"
#include "pulseq/binary.hpp"
#include "pulseq/parsed.hpp"
#include "pulseq/sequence.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulseq
{

    namespace
    {
        /* ============================================================== */
        /*  Scanning                                                      */
        /* ============================================================== */

        [[noreturn]] void fail(int line, const std::string& what)
        {
            throw std::runtime_error("read(): line " + std::to_string(line) + ": " + what);
        }

        inline bool is_space(char c)
        {
            return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f';
        }

        /**
         * One line's fields, taken left to right.
         *
         * Numbers are read straight out of the file's own buffer: the tokens
         * are whitespace separated and the buffer is a std::string, so strtod
         * stops on the separator and never runs off the end.
         */
        class Row
        {
        public:
            Row(const char* begin, const char* end, int line)
                : cursor_(begin), end_(end), line_(line)
            {
            }

            bool more()
            {
                skip();
                return cursor_ < end_;
            }

            double number(const char* what)
            {
                skip();
                if (cursor_ >= end_)
                    fail(line_, std::string("expected ") + what);
                char* stop = nullptr;
                const double value = std::strtod(cursor_, &stop);
                if (stop == cursor_ || stop > end_)
                    fail(line_, std::string("expected ") + what + ", found \"" + word() + "\"");
                cursor_ = stop;
                return value;
            }

            /** A number if there is one, @p fallback if the row ended. */
            double optional(double fallback)
            {
                return more() ? number("a number") : fallback;
            }

            int32_t integer(const char* what)
            {
                return static_cast<int32_t>(std::lround(number(what)));
            }

            /** The next whitespace-delimited word. */
            std::string word()
            {
                skip();
                const char* begin = cursor_;
                while (cursor_ < end_ && !is_space(*cursor_))
                    ++cursor_;
                return std::string(begin, cursor_);
            }

            /** Everything left, with the surrounding space taken off. */
            std::string rest()
            {
                skip();
                const char* stop = end_;
                while (stop > cursor_ && is_space(stop[-1]))
                    --stop;
                std::string value(cursor_, stop);
                cursor_ = end_;
                return value;
            }

            int line() const
            {
                return line_;
            }

        private:
            void skip()
            {
                while (cursor_ < end_ && is_space(*cursor_))
                    ++cursor_;
            }

            const char* cursor_;
            const char* end_;
            int line_;
        };

        /** A line worth looking at: not blank, not a comment. */
        struct Significant
        {
            const char* begin;
            const char* end;
            int line;
        };

        std::vector<Significant> significant_lines(const std::string& text)
        {
            std::vector<Significant> lines;
            const char* cursor = text.data();
            const char* const end = cursor + text.size();
            int number = 0;
            while (cursor < end)
            {
                const char* stop = static_cast<const char*>(
                    std::memchr(cursor, '\n', static_cast<size_t>(end - cursor)));
                const char* line_end = stop ? stop : end;
                ++number;

                const char* begin = cursor;
                while (begin < line_end && is_space(*begin))
                    ++begin;
                const char* trimmed = line_end;
                while (trimmed > begin && is_space(trimmed[-1]))
                    --trimmed;
                if (begin < trimmed && *begin != '#')
                    lines.push_back({begin, trimmed, number});

                cursor = stop ? stop + 1 : end;
            }
            return lines;
        }

        /* ============================================================== */
        /*  What a file says                                              */
        /* ============================================================== */
        //
        // Every library is keyed by the id the file gave it rather than
        // collected in the order the rows appear, so a file whose rows are out
        // of order still registers them in the numbering it declared.

        /** A definition's value: numbers when every word is one, else text. */
        Definition definition_from(const std::string& value)
        {
            std::vector<double> numbers;
            const char* cursor = value.c_str();
            const char* const end = cursor + value.size();
            while (true)
            {
                while (cursor < end && is_space(*cursor))
                    ++cursor;
                if (cursor >= end)
                    break;
                char* stop = nullptr;
                const double number = std::strtod(cursor, &stop);
                if (stop == cursor)
                    return Definition(value);
                numbers.push_back(number);
                cursor = stop;
            }
            if (numbers.empty())
                return Definition(value);
            return Definition(std::move(numbers));
        }

        /* ============================================================== */
        /*  Parsing                                                       */
        /* ============================================================== */

        void parse_shapes(const std::vector<Significant>& lines, size_t& at, Parsed& out)
        {
            // A shape is `shape_id N`, `num_samples M`, then M or fewer
            // samples -- fewer when the encoding paid off. The samples run
            // until the next shape or the next section.
            while (at < lines.size() && lines[at].begin[0] != '[')
            {
                Row header(lines[at].begin, lines[at].end, lines[at].line);
                if (header.word() != "shape_id")
                    fail(lines[at].line, "expected shape_id");
                const int id = header.integer("a shape id");
                ++at;

                if (at >= lines.size())
                    fail(lines[at - 1].line, "shape ends before num_samples");
                Row count_line(lines[at].begin, lines[at].end, lines[at].line);
                if (count_line.word() != "num_samples")
                    fail(lines[at].line, "expected num_samples");
                const int num_uncompressed = count_line.integer("a sample count");
                ++at;

                std::vector<double> samples;
                while (at < lines.size() && lines[at].begin[0] != '[' &&
                       std::strncmp(lines[at].begin, "shape_id", 8) != 0)
                {
                    Row row(lines[at].begin, lines[at].end, lines[at].line);
                    while (row.more())
                        samples.push_back(row.number("a sample"));
                    ++at;
                }
                out.shapes.emplace(id, std::make_pair(num_uncompressed, std::move(samples)));
            }
        }

        void parse_extensions(const std::vector<Significant>& lines, size_t& at, Parsed& out)
        {
            // The chain list first, then one `extension NAME id` header per
            // specification and that specification's rows under it.
            std::string current;
            while (at < lines.size() && lines[at].begin[0] != '[')
            {
                Row row(lines[at].begin, lines[at].end, lines[at].line);
                if (std::strncmp(lines[at].begin, "extension ", 10) == 0)
                {
                    row.word();
                    current = row.word();
                    out.extension_types[current] = row.integer("an extension type id");
                    ++at;
                    continue;
                }

                const int id = row.integer("an id");
                if (current.empty())
                {
                    std::array<int32_t, EXTENSION_WIDTH> chain{};
                    chain[0] = row.integer("an extension type");
                    chain[1] = row.integer("an extension reference");
                    chain[2] = row.integer("the next link");
                    out.chains.emplace(id, chain);
                }
                else if (current == "TRIGGERS")
                {
                    std::array<double, TRIGGER_WIDTH> trigger{};
                    trigger[0] = row.number("a trigger type");
                    trigger[1] = row.number("a channel");
                    trigger[2] = row.number("a delay") * 1e-6;
                    trigger[3] = row.number("a duration") * 1e-6;
                    out.triggers.emplace(id, trigger);
                }
                else if (current == "LABELSET" || current == "LABELINC")
                {
                    ParsedLabel label;
                    label.value = row.integer("a label value");
                    label.name = row.word();
                    (current == "LABELSET" ? out.label_set : out.label_inc).emplace(id, label);
                }
                else if (current == "ROTATIONS")
                {
                    std::array<double, ROTATION_WIDTH> quaternion{};
                    for (int i = 0; i < ROTATION_WIDTH; ++i)
                        quaternion[i] = row.number("a quaternion component");
                    out.rotations.emplace(id, quaternion);
                }
                else if (current == "RF_SHIMS")
                {
                    // The channel count is written for a reader that wants it;
                    // the values that follow say the same thing.
                    const int32_t channels = row.integer("a channel count");
                    std::vector<double> values;
                    values.reserve(static_cast<size_t>(channels) * 2);
                    while (row.more())
                        values.push_back(row.number("a shim value"));
                    out.shims.emplace(id, std::move(values));
                }
                else if (current == "DELAYS")
                {
                    SoftDelay delay;
                    delay.num = row.integer("a delay number");
                    delay.offset = row.number("an offset") * 1e-6;
                    delay.factor = row.number("a factor");
                    delay.hint = row.rest();
                    out.soft_delays.emplace(id, std::move(delay));
                }
                else
                {
                    fail(lines[at].line, "unknown extension specification \"" + current + "\"");
                }
                ++at;
            }
        }

        Parsed parse(const std::string& text)
        {
            const std::vector<Significant> lines = significant_lines(text);
            Parsed out;

            std::string section;
            size_t at = 0;
            while (at < lines.size())
            {
                const Significant& line = lines[at];
                if (line.begin[0] == '[')
                {
                    section.assign(line.begin, line.end);
                    ++at;
                    if (section == "[BLOCKS]" && out.combined() < 1002000)
                        fail(
                            line.line,
                            "this is a Pulseq " + std::to_string(out.major) + "." +
                                std::to_string(out.minor) + "." + std::to_string(out.revision) +
                                " file, and 1.2.0 is the oldest the format is defined from");
                    if (section == "[SHAPES]")
                        parse_shapes(lines, at, out);
                    else if (section == "[EXTENSIONS]")
                        parse_extensions(lines, at, out);
                    else if (section == "[SIGNATURE]")
                    {
                        // Taken from the header line the scan is standing on:
                        // the section's own comments quote its name, so
                        // searching the text for it finds the wrong one.
                        out.signature_offset = static_cast<size_t>(line.begin - text.data());
                    }
                    continue;
                }

                Row row(line.begin, line.end, line.line);
                if (section == "[VERSION]")
                {
                    const std::string field = row.word();
                    const int value = row.integer("a version number");
                    if (field == "major")
                        out.major = value;
                    else if (field == "minor")
                        out.minor = value;
                    else if (field == "revision")
                        out.revision = value;
                }
                else if (section == "[DEFINITIONS]")
                {
                    const std::string key = row.word();
                    out.definitions.emplace(key, definition_from(row.rest()));
                }
                else if (section == "[BLOCKS]")
                {
                    const int id = row.integer("a block number");
                    ParsedBlock block;
                    const int32_t duration = row.integer("a block duration");
                    // Before 1.4 that column is an index into `[DELAYS]`
                    // rather than a count of rasters, and the duration is
                    // whatever the block's own events take; see upgrade().
                    if (out.combined() < 1004000)
                        out.block_delay.emplace(id, duration);
                    else
                        block.ticks = static_cast<long>(duration);
                    // 1.2 has no extension column; everything else does.
                    const int columns =
                        out.combined() <= 1002001 ? BLOCK_FILE_COLUMNS - 1 : BLOCK_FILE_COLUMNS;
                    for (int column = 0; column < columns; ++column)
                        block.events[static_cast<size_t>(column)] = row.integer("an event id");
                    out.blocks.emplace(id, block);
                }
                else if (section == "[RF]")
                {
                    const int id = row.integer("an RF id");
                    std::array<double, RF_WIDTH> event{};
                    event[0] = row.number("an amplitude");
                    event[1] = row.number("a magnitude shape");
                    event[2] = row.number("a phase shape");
                    // Three layouts. 1.5 has the centre and the ppm terms;
                    // 1.4 has neither; 1.3 has no time shape either, so the
                    // delay is where the time shape would be.
                    if (out.combined() >= 1004000)
                        event[3] = row.number("a time shape");
                    if (out.combined() >= 1005000)
                    {
                        event[4] = row.number("a center") * 1e-6;
                        event[5] = row.number("a delay") * 1e-6;
                        event[6] = row.number("a frequency in ppm");
                        event[7] = row.number("a phase in ppm");
                        event[8] = row.number("a frequency offset");
                        event[9] = row.number("a phase offset");
                    }
                    else
                    {
                        // The centre is derived once every shape is known;
                        // see upgrade().
                        event[5] = row.number("a delay") * 1e-6;
                        event[8] = row.number("a frequency offset");
                        event[9] = row.number("a phase offset");
                    }
                    out.rf.emplace(id, event);
                    // The initial of what the pulse is for, and only 1.5 files
                    // carry it; an older one leaves it undefined.
                    out.rf_use.emplace(id, row.more() ? row.word()[0] : 'u');
                }
                else if (section == "[GRADIENTS]")
                {
                    const int id = row.integer("a gradient id");
                    std::array<double, ARB_WIDTH> event{};
                    event[0] = row.number("an amplitude");
                    if (out.combined() >= 1005000)
                    {
                        event[1] = row.number("the first sample");
                        event[2] = row.number("the last sample");
                    }
                    else
                    {
                        // 1.4 carries neither, and neither can be defaulted:
                        // they are recovered from the block table in
                        // upgrade(). Not-a-number marks them as unset.
                        event[1] = event[2] = std::numeric_limits<double>::quiet_NaN();
                    }
                    event[3] = row.number("an amplitude shape");
                    if (out.combined() >= 1004000)
                        event[4] = row.number("a time shape");
                    event[5] = row.number("a delay") * 1e-6;
                    out.arbitrary.emplace(id, event);
                }
                else if (section == "[TRAP]")
                {
                    const int id = row.integer("a gradient id");
                    std::array<double, TRAP_WIDTH> event{};
                    event[0] = row.number("an amplitude");
                    event[1] = row.number("a rise time") * 1e-6;
                    event[2] = row.number("a flat time") * 1e-6;
                    event[3] = row.number("a fall time") * 1e-6;
                    event[4] = row.number("a delay") * 1e-6;
                    out.trapezoid.emplace(id, event);
                }
                else if (section == "[ADC]")
                {
                    const int id = row.integer("an ADC id");
                    std::array<double, ADC_WIDTH> event{};
                    event[0] = row.number("a sample count");
                    event[1] = row.number("a dwell time") * 1e-9;
                    event[2] = row.number("a delay") * 1e-6;
                    if (out.combined() >= 1005000)
                    {
                        event[3] = row.number("a frequency in ppm");
                        event[4] = row.number("a phase in ppm");
                        event[5] = row.number("a frequency offset");
                        event[6] = row.number("a phase offset");
                        event[7] = row.number("a phase shape");
                    }
                    else
                    {
                        // 1.4 has the two offsets and nothing else; the ppm
                        // terms and the phase shape are zero.
                        event[5] = row.number("a frequency offset");
                        event[6] = row.number("a phase offset");
                    }
                    out.adc.emplace(id, event);
                }
                else if (section == "[DELAYS]")
                {
                    const int id = row.integer("a delay id");
                    out.delays.emplace(id, row.number("a delay") * 1e-6);
                }
                else if (section == "[SIGNATURE]")
                {
                    const std::string field = row.word();
                    if (field == "Hash")
                    {
                        out.signature = row.rest();
                        out.has_signature = true;
                    }
                }
                ++at;
            }
            return out;
        }

        /** A raster from `[DEFINITIONS]`, or @p fallback where it says nothing. */
        double raster(const Parsed& parsed, const char* key, double fallback)
        {
            auto found = parsed.definitions.find(key);
            if (found == parsed.definitions.end() || found->second.numbers().empty())
                return fallback;
            return found->second.numbers().front();
        }

        /* ============================================================== */
        /*  Reading a file older than 1.5.0                               */
        /* ============================================================== */
        //
        // Three things a 1.4 file does not carry, none of which has a
        // sensible default: an RF pulse's center, and an arbitrary
        // gradient's first and last sample. All three are recovered the way
        // the reference toolbox recovers them -- the center from the pulse's
        // own envelope, the edges from walking the block table -- so a 1.4
        // file and the 1.5 file written from the same sequence read alike.

        /** The samples of shape @p id, decompressed; empty if there is none. */
        std::vector<double> samples_of(const Parsed& parsed, int id)
        {
            auto found = parsed.shapes.find(id);
            if (id <= 0 || found == parsed.shapes.end())
                return {};
            return decompress_shape(
                found->second.second.data(),
                static_cast<int>(found->second.second.size()),
                found->second.first);
        }

        /** The sample times of a shape on @p raster, or the default raster. */
        std::vector<double> times_of(
            const Parsed& parsed, int time_shape, size_t count, double raster, double offset)
        {
            std::vector<double> times = samples_of(parsed, time_shape);
            if (!times.empty())
            {
                for (double& t : times)
                    t *= raster;
                return times;
            }
            times.resize(count);
            for (size_t i = 0; i < count; ++i)
                times[i] = (static_cast<double>(i) + offset) * raster;
            return times;
        }

        /**
         * The center of an RF pulse: the middle of its peak.
         *
         * Where the envelope holds its maximum over several samples the
         * center is the middle of that run, which is what makes it the centre
         * of a flat-topped pulse rather than the first sample of the plateau.
         */
        void restore_rf_centers(Parsed& parsed, double rf_raster)
        {
            for (auto& entry : parsed.rf)
            {
                std::array<double, RF_WIDTH>& row = entry.second;
                const std::vector<double> magnitude =
                    samples_of(parsed, static_cast<int>(row[1]));
                if (magnitude.empty())
                    continue;

                double peak = 0.0;
                for (const double sample : magnitude)
                    peak = std::max(peak, std::fabs(sample));
                const double threshold = peak * 0.99999;

                size_t first = 0, last = 0;
                bool seen = false;
                for (size_t i = 0; i < magnitude.size(); ++i)
                    if (std::fabs(magnitude[i]) >= threshold)
                    {
                        if (!seen)
                            first = i;
                        last = i;
                        seen = true;
                    }
                if (!seen)
                    continue;

                const std::vector<double> times = times_of(
                    parsed, static_cast<int>(row[3]), magnitude.size(), rf_raster, 0.5);
                row[4] = (times[first] + times[last]) / 2.0;
            }
        }

        /**
         * An arbitrary gradient's first and last sample, from the block table.
         *
         * The last is the waveform's own end -- read off it for an extended
         * trapezoid, extrapolated for a gradient on the plain raster, exactly
         * as the factory would have. The first is where the axis was left by
         * the block before, which is zero unless the previous gradient ran to
         * that block's end, so this has to be a walk in playing order.
         */
        void restore_gradient_edges(Parsed& parsed, double grad_raster, double block_raster)
        {
            double previous_last[3] = {0.0, 0.0, 0.0};
            for (const auto& entry : parsed.blocks)
            {
                const ParsedBlock& block = entry.second;
                const double block_duration =
                    static_cast<double>(block.ticks) * block_raster;
                int32_t handled[3] = {0, 0, 0};

                for (int axis = 0; axis < 3; ++axis)
                {
                    const int32_t id = block.events[static_cast<size_t>(axis) + 1];
                    auto gradient = parsed.arbitrary.find(id);
                    if (id == 0)
                    {
                        previous_last[axis] = 0.0;
                        continue;
                    }
                    if (gradient == parsed.arbitrary.end())
                        continue; // a trapezoid, which carries no edges

                    std::array<double, ARB_WIDTH>& row = gradient->second;
                    const double delay = row[5];
                    if (delay > 0.0)
                        previous_last[axis] = 0.0;
                    if (!std::isnan(row[1]))
                        continue; // already known

                    const std::vector<double> waveform =
                        samples_of(parsed, static_cast<int>(row[3]));
                    if (waveform.size() < 2)
                        continue;

                    // `first` and `last` are in the file's units, where a
                    // stored shape is normalised, so the amplitude goes back
                    // on before either is derived from the samples.
                    const double amplitude = row[0];
                    const int time_shape = static_cast<int>(row[4]);
                    const double first = previous_last[axis];
                    double last;
                    double duration;
                    if (time_shape != 0)
                    {
                        last = amplitude * waveform.back();
                        const std::vector<double> times = times_of(
                            parsed, time_shape, waveform.size(), grad_raster, 1.0);
                        duration = delay + times.back();
                    }
                    else
                    {
                        // The same linear extrapolation the factory makes.
                        last = amplitude *
                               (3.0 * waveform.back() - waveform[waveform.size() - 2]) * 0.5;
                        duration =
                            delay + static_cast<double>(waveform.size()) * grad_raster;
                    }

                    previous_last[axis] =
                        duration + std::numeric_limits<double>::epsilon() < block_duration
                            ? 0.0
                            : last;

                    // One gradient can be played on two axes in one block;
                    // its row is written once.
                    bool already = false;
                    for (int earlier = 0; earlier < axis; ++earlier)
                        already = already || handled[earlier] == id;
                    handled[axis] = id;
                    if (already)
                        continue;

                    row[1] = first;
                    row[2] = last;
                }
            }

            // A gradient no block plays keeps nothing to derive from.
            for (auto& entry : parsed.arbitrary)
            {
                if (std::isnan(entry.second[1]))
                    entry.second[1] = 0.0;
                if (std::isnan(entry.second[2]))
                    entry.second[2] = 0.0;
            }
        }

        /**
         * Re-encode every shape, decoding it even where the counts agree.
         *
         * Before 1.4 a shape whose encoded length happened to equal its
         * sample count was indistinguishable from one that was never encoded,
         * so the two cannot be told apart by length alone. Decoding with that
         * rule suspended and encoding again settles it: afterwards equal
         * counts really do mean the samples are the samples.
         */
        void normalise_shapes(Parsed& parsed)
        {
            for (auto& entry : parsed.shapes)
            {
                const int uncompressed = entry.second.first;
                std::vector<double>& stored = entry.second.second;
                const std::vector<double> samples = decompress_shape(
                    stored.data(), static_cast<int>(stored.size()), uncompressed, true);
                if (samples.empty())
                    continue;
                stored = compress_shape(samples.data(), static_cast<int>(samples.size()));
            }
        }

        /**
         * The rise and fall a pre-1.4 file leaves off a zero trapezoid.
         *
         * A gradient of no amplitude was written with no ramp, which later
         * revisions do not allow: one raster of the flat time becomes the
         * ramp, on each side that is missing one.
         */
        void restore_trapezoid_ramps(Parsed& parsed, double grad_raster)
        {
            for (auto& entry : parsed.trapezoid)
            {
                std::array<double, TRAP_WIDTH>& row = entry.second;
                if (row[0] != 0.0)
                    continue;
                if (row[1] == 0.0 && row[2] > 0.0)
                {
                    row[1] = grad_raster;
                    row[2] -= grad_raster;
                }
                if (row[3] == 0.0 && row[2] > 0.0)
                {
                    row[2] -= grad_raster;
                    row[3] = grad_raster;
                }
            }
        }

        /** How long a shape lasts, on its own time shape or on the raster. */
        double shape_duration(const Parsed& parsed, int time_shape, size_t count, double raster)
        {
            if (time_shape > 0)
            {
                const std::vector<double> times = samples_of(parsed, time_shape);
                if (!times.empty())
                    return std::ceil(
                               (times.back() * raster - std::numeric_limits<double>::epsilon()) /
                               raster) *
                           raster;
            }
            return static_cast<double>(count) * raster;
        }

        /**
         * How long a block lasts, from what it plays.
         *
         * Before 1.4 the block table records a delay rather than a duration,
         * and the duration is the longest thing in the block -- which is what
         * later revisions store outright.
         */
        void restore_block_durations(
            Parsed& parsed, double rf_raster, double grad_raster, double block_raster)
        {
            const int triggers = [&parsed] {
                auto found = parsed.extension_types.find("TRIGGERS");
                return found == parsed.extension_types.end() ? 0 : found->second;
            }();

            for (auto& entry : parsed.blocks)
            {
                ParsedBlock& block = entry.second;
                auto delay = parsed.block_delay.find(entry.first);
                double duration = 0.0;
                if (delay != parsed.block_delay.end() && delay->second > 0)
                {
                    auto value = parsed.delays.find(delay->second);
                    if (value != parsed.delays.end())
                        duration = value->second;
                }

                auto rf = parsed.rf.find(block.events[0]);
                if (rf != parsed.rf.end())
                {
                    const std::vector<double> magnitude =
                        samples_of(parsed, static_cast<int>(rf->second[1]));
                    duration = std::max(
                        duration,
                        rf->second[5] + shape_duration(
                                            parsed,
                                            static_cast<int>(rf->second[3]),
                                            magnitude.size(),
                                            rf_raster));
                }

                for (int axis = 1; axis <= 3; ++axis)
                {
                    const int32_t id = block.events[static_cast<size_t>(axis)];
                    auto trapezoid = parsed.trapezoid.find(id);
                    if (trapezoid != parsed.trapezoid.end())
                    {
                        const std::array<double, TRAP_WIDTH>& row = trapezoid->second;
                        duration = std::max(duration, row[4] + row[1] + row[2] + row[3]);
                        continue;
                    }
                    auto gradient = parsed.arbitrary.find(id);
                    if (gradient == parsed.arbitrary.end())
                        continue;
                    const std::vector<double> waveform =
                        samples_of(parsed, static_cast<int>(gradient->second[3]));
                    duration = std::max(
                        duration,
                        gradient->second[5] + shape_duration(
                                                  parsed,
                                                  static_cast<int>(gradient->second[4]),
                                                  waveform.size(),
                                                  grad_raster));
                }

                auto adc = parsed.adc.find(block.events[4]);
                if (adc != parsed.adc.end())
                    duration = std::max(
                        duration, adc->second[2] + adc->second[0] * adc->second[1]);

                // A trigger is played too, and it is reached through the chain.
                for (int32_t node = block.events[5]; node > 0;)
                {
                    auto link = parsed.chains.find(node);
                    if (link == parsed.chains.end())
                        break;
                    if (triggers != 0 && link->second[0] == triggers)
                    {
                        auto trigger = parsed.triggers.find(link->second[1]);
                        if (trigger != parsed.triggers.end())
                            duration = std::max(
                                duration, trigger->second[2] + trigger->second[3]);
                    }
                    node = link->second[2];
                }

                block.ticks = static_cast<long>(std::lround(duration / block_raster));
            }
        }

        /** Bring what a pre-1.5 file said up to what a 1.5 file says. */
        void upgrade(Parsed& parsed)
        {
            if (parsed.combined() >= 1005000)
                return;

            const double rf_raster = raster(parsed, "RadiofrequencyRasterTime", 1e-6);
            const double grad_raster = raster(parsed, "GradientRasterTime", 10e-6);
            const double block_raster = raster(parsed, "BlockDurationRaster", 10e-6);

            if (parsed.combined() < 1004000)
            {
                normalise_shapes(parsed);
                restore_trapezoid_ramps(parsed, grad_raster);
                restore_block_durations(parsed, rf_raster, grad_raster, block_raster);
                // A file this old declares none of the rasters, so what was
                // assumed above is written down rather than left implicit.
                const std::pair<const char*, double> assumed[4] = {
                    {"GradientRasterTime", grad_raster},
                    {"RadiofrequencyRasterTime", rf_raster},
                    {"AdcRasterTime", raster(parsed, "AdcRasterTime", 100e-9)},
                    {"BlockDurationRaster", block_raster},
                };
                for (const auto& entry : assumed)
                    if (parsed.definitions.find(entry.first) == parsed.definitions.end())
                        parsed.definitions.emplace(entry.first, Definition(entry.second));
            }

            restore_rf_centers(parsed, rf_raster);
            restore_gradient_edges(parsed, grad_raster, block_raster);

            // The sequence is now what a 1.5 file holds, so it says so. What
            // it cannot say is what each pulse is *for*: 1.4 carries no `use`
            // column, and every pulse stays undefined rather than being
            // guessed at from its flip angle.
            parsed.minor = 5;
            parsed.revision = 0;
        }

        /* ============================================================== */
        /*  Building                                                      */
        /* ============================================================== */

        /** Check that registering in file order reproduced the file's ids. */
        void expect(int registered, int declared, const char* what)
        {
            if (registered != declared)
                throw std::runtime_error(
                    std::string("read(): ") + what + " " + std::to_string(declared) +
                    " was registered as " + std::to_string(registered) +
                    "; the file numbers them with a gap");
        }

    } // namespace

        Sequence build_sequence(const Parsed& parsed)
    {
        Sequence seq;
        seq.set_version(parsed.major, parsed.minor, parsed.revision);
        seq.set_rasters(
            raster(parsed, "RadiofrequencyRasterTime", 1e-6),
            raster(parsed, "GradientRasterTime", 10e-6),
            raster(parsed, "AdcRasterTime", 100e-9),
            raster(parsed, "BlockDurationRaster", 10e-6));
        for (const auto& entry : parsed.definitions)
            seq.set_definition(entry.first, entry.second);

        // Before any label row, so a number past the builtin table resolves
        // to the name the file gave it rather than minting a fresh one.
        auto custom = parsed.definitions.find("CustomLabels");
        if (custom != parsed.definitions.end())
        {
            std::istringstream names(
                custom->second.kind() == Definition::Kind::Text ? custom->second.text()
                                                                : std::string());
            std::string name;
            while (names >> name)
                seq.label_id(name);
        }

        // Before the chains, so a chain names the type the file said.
        for (const auto& entry : parsed.extension_types)
            seq.set_extension_type_id(entry.first, entry.second);

        for (const auto& entry : parsed.shapes)
            expect(
                seq.register_shape(
                    entry.second.first,
                    entry.second.second.data(),
                    static_cast<int>(entry.second.second.size())),
                entry.first,
                "shape");

        for (const auto& entry : parsed.rf)
        {
            auto use = parsed.rf_use.find(entry.first);
            expect(
                seq.register_rf(
                    entry.second.data(), use == parsed.rf_use.end() ? 'u' : use->second),
                entry.first,
                "RF event");
        }

        // Trapezoids and arbitrary gradients share one numbering, so they
        // are registered by walking that numbering rather than a table.
        {
            auto trap = parsed.trapezoid.begin();
            auto arb = parsed.arbitrary.begin();
            while (trap != parsed.trapezoid.end() || arb != parsed.arbitrary.end())
            {
                const bool take_trap =
                    arb == parsed.arbitrary.end() ||
                    (trap != parsed.trapezoid.end() && trap->first < arb->first);
                if (take_trap)
                {
                    expect(seq.register_trap(trap->second.data()), trap->first, "gradient");
                    ++trap;
                }
                else
                {
                    expect(seq.register_arbitrary(arb->second.data()), arb->first, "gradient");
                    ++arb;
                }
            }
        }

        for (const auto& entry : parsed.adc)
            expect(seq.register_adc(entry.second.data()), entry.first, "ADC event");
        for (const auto& entry : parsed.triggers)
            expect(seq.register_trigger(entry.second.data()), entry.first, "trigger");
        for (const auto& entry : parsed.rotations)
            expect(seq.register_rotation(entry.second.data()), entry.first, "rotation");
        // A file that names its labels is believed; one that only numbers
        // them is taken at its number, which is Pulseq's own for every label
        // Pulseq defines and is all the binary form carries for those.
        const auto label = [&seq](const ParsedLabel& row) {
            return row.name.empty() ? row.label_id : seq.label_id(row.name);
        };
        for (const auto& entry : parsed.label_set)
            expect(
                seq.register_label_set(entry.second.value, label(entry.second)),
                entry.first,
                "LABELSET");
        for (const auto& entry : parsed.label_inc)
            expect(
                seq.register_label_inc(entry.second.value, label(entry.second)),
                entry.first,
                "LABELINC");
        for (const auto& entry : parsed.shims)
            expect(
                seq.register_rf_shim(
                    entry.second.data(), static_cast<int>(entry.second.size())),
                entry.first,
                "RF shim");
        for (const auto& entry : parsed.soft_delays)
            expect(seq.register_soft_delay(entry.second), entry.first, "soft delay");

        // Appended rather than chained: the file's own table is
        // reproduced, duplicate links and all, because collapsing them
        // here would renumber what the blocks point at.
        for (const auto& entry : parsed.chains)
            expect(
                seq.append_extension(
                    entry.second[0], entry.second[1], entry.second[2]),
                entry.first,
                "extension chain");

        // Last: a block is split into a definition and an instance as it
        // is added, which needs every event it names to be registered.
        const double tick = seq.block_duration_raster();
        for (const auto& entry : parsed.blocks)
        {
            Block block;
            block.rf = entry.second.events[0];
            block.gx = entry.second.events[1];
            block.gy = entry.second.events[2];
            block.gz = entry.second.events[3];
            block.adc = entry.second.events[4];
            block.ext = entry.second.events[5];
            block.duration = static_cast<double>(entry.second.ticks) * tick;
            expect(seq.add_block(block), entry.first, "block");
        }
    return seq;
    }

    /* ================================================================== */
    /*  Entry points                                                      */
    /* ================================================================== */

    Sequence read(const std::string& contents, bool verify)
    {
        if (is_binary(contents))
            return build_sequence(parse_binary(contents));

        Parsed parsed = parse(contents);
        if (verify && parsed.has_signature)
        {
            // The newline before the header belongs to the digest, and the
            // blank line before that does not: the writer hashes everything it
            // had emitted, then appends "\n[SIGNATURE]\n".
            const size_t hashed = parsed.signature_offset ? parsed.signature_offset - 1 : 0;
            const std::string expected = md5_hex(contents.data(), hashed);
            if (expected != parsed.signature)
                throw std::runtime_error(
                    "read(): the file's signature does not match its contents (recorded " +
                    parsed.signature + ", computed " + expected + ")");
        }
        upgrade(parsed);
        return build_sequence(parsed);
    }

    Sequence read_file(const std::string& path, bool verify)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
            throw std::runtime_error("read(): cannot open " + path);
        std::ostringstream contents;
        contents << file.rdbuf();
        return read(contents.str(), verify);
    }

} // namespace pulseq
