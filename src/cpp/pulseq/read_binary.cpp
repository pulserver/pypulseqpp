/**
 * @file read_binary.cpp
 * @brief The Pulseq binary sequence reader.
 *
 * The inverse of write_binary.cpp, and the same shape as the text reader: the
 * file is taken apart into a Parsed and handed to the one builder, so both
 * forms of a sequence file become a Sequence by the same rules.
 *
 * Little-endian, as every writer of this format produces. A file written the
 * other way round is recognised -- the version triple is what gives it away,
 * since a major version of 1 read backwards is an enormous number -- and
 * refused by name rather than read as nonsense.
 */

#include "pulseq/binary.hpp"
#include "pulseq/parsed.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulseq
{

    namespace
    {
        /** A reader over the file's bytes that will not run off the end. */
        class Cursor
        {
        public:
            Cursor(const char* begin, const char* end) : begin_(begin), at_(begin), end_(end)
            {
            }

            bool done() const
            {
                return at_ >= end_;
            }

            size_t offset() const
            {
                return static_cast<size_t>(at_ - begin_);
            }

            [[noreturn]] void fail(const std::string& what) const
            {
                throw std::runtime_error(
                    "read(): byte " + std::to_string(offset()) + ": " + what);
            }

            void need(size_t bytes, const char* what) const
            {
                if (static_cast<size_t>(end_ - at_) < bytes)
                    fail(std::string("the file ends in the middle of ") + what);
            }

            uint64_t u64(const char* what)
            {
                need(8, what);
                uint64_t value = 0;
                for (int i = 0; i < 8; ++i)
                    value |= static_cast<uint64_t>(static_cast<unsigned char>(at_[i])) << (8 * i);
                at_ += 8;
                return value;
            }

            int64_t i64(const char* what)
            {
                return static_cast<int64_t>(u64(what));
            }

            int32_t i32(const char* what)
            {
                need(4, what);
                uint32_t value = 0;
                for (int i = 0; i < 4; ++i)
                    value |= static_cast<uint32_t>(static_cast<unsigned char>(at_[i])) << (8 * i);
                at_ += 4;
                return static_cast<int32_t>(value);
            }

            double f64(const char* what)
            {
                const uint64_t bits = u64(what);
                double value;
                std::memcpy(&value, &bits, sizeof(value));
                return value;
            }

            double f32(const char* what)
            {
                need(4, what);
                uint32_t bits = 0;
                for (int i = 0; i < 4; ++i)
                    bits |= static_cast<uint32_t>(static_cast<unsigned char>(at_[i])) << (8 * i);
                at_ += 4;
                float value;
                std::memcpy(&value, &bits, sizeof(value));
                return static_cast<double>(value);
            }

            /** A NUL-terminated string. */
            std::string cstring(const char* what)
            {
                const void* stop = std::memchr(at_, '\0', static_cast<size_t>(end_ - at_));
                if (!stop)
                    fail(std::string("the file ends in the middle of ") + what);
                const char* terminator = static_cast<const char*>(stop);
                std::string value(at_, terminator);
                at_ = terminator + 1;
                return value;
            }

            std::string bytes(size_t count, const char* what)
            {
                need(count, what);
                std::string value(at_, count);
                at_ += count;
                return value;
            }

            void skip(size_t count, const char* what)
            {
                need(count, what);
                at_ += count;
            }

        private:
            const char* begin_;
            const char* at_;
            const char* end_;
        };

        /** Seconds from the whole picoseconds the format carries. */
        double seconds(int64_t picoseconds)
        {
            return static_cast<double>(picoseconds) * 1e-12;
        }

        /** A count, checked: a corrupt one would otherwise size an allocation. */
        int64_t count_of(Cursor& in, const char* what)
        {
            const int64_t count = in.i64(what);
            if (count < 0)
                in.fail(std::string("a negative number of ") + what);
            return count;
        }

        void read_definitions(Cursor& in, Parsed& out)
        {
            const int64_t count = count_of(in, "definitions");
            for (int64_t i = 0; i < count; ++i)
            {
                const int32_t length = in.i32("a definition name length");
                if (length < 0)
                    in.fail("a negative definition name length");
                const std::string key = in.bytes(static_cast<size_t>(length), "a definition name");
                const int32_t values = in.i32("a definition value count");
                if (values < 0)
                    in.fail("a negative definition value count");
                const char tag = in.bytes(1, "a definition type")[0];

                if (tag == 'c')
                {
                    out.definitions.emplace(
                        key, Definition(in.bytes(static_cast<size_t>(values), "a definition")));
                    continue;
                }
                if (tag != 'f' && tag != 'i')
                    in.fail(std::string("a definition tagged '") + tag + "', which means nothing");

                std::vector<double> numbers;
                numbers.reserve(static_cast<size_t>(values));
                for (int32_t v = 0; v < values; ++v)
                    numbers.push_back(
                        tag == 'i' ? static_cast<double>(in.i32("a definition value"))
                                   : in.f64("a definition value"));
                out.definitions.emplace(
                    key,
                    tag == 'i' ? Definition::integers(std::move(numbers))
                               : Definition(std::move(numbers)));
            }
        }

        void read_blocks(Cursor& in, Parsed& out)
        {
            const int64_t count = count_of(in, "blocks");
            for (int64_t i = 0; i < count; ++i)
            {
                ParsedBlock block;
                block.ticks = static_cast<long>(in.i64("a block duration"));
                for (int column = 0; column < BLOCK_WIDTH; ++column)
                    block.events[static_cast<size_t>(column)] = in.i32("a block event");
                out.blocks.emplace(static_cast<int>(i) + 1, block);
            }
        }

        void read_rf(Cursor& in, Parsed& out)
        {
            const int64_t count = count_of(in, "RF events");
            for (int64_t i = 0; i < count; ++i)
            {
                const int id = in.i32("an RF id");
                std::array<double, RF_WIDTH> event{};
                event[0] = in.f64("an amplitude");
                event[1] = in.i32("a magnitude shape");
                event[2] = in.i32("a phase shape");
                event[3] = in.i32("a time shape");
                event[4] = seconds(in.i64("a center"));
                event[5] = seconds(in.i64("a delay"));
                event[6] = in.f64("a frequency in ppm");
                event[7] = in.f64("a phase in ppm");
                event[8] = in.f64("a frequency offset");
                event[9] = in.f64("a phase offset");
                out.rf.emplace(id, event);
                out.rf_use.emplace(id, in.bytes(1, "an RF use")[0]);
            }
        }

        void read_arbitrary(Cursor& in, Parsed& out)
        {
            const int64_t count = count_of(in, "gradients");
            for (int64_t i = 0; i < count; ++i)
            {
                const int id = in.i32("a gradient id");
                std::array<double, ARB_WIDTH> event{};
                event[0] = in.f64("an amplitude");
                event[1] = in.f64("the first sample");
                event[2] = in.f64("the last sample");
                event[3] = in.i32("an amplitude shape");
                event[4] = in.i32("a time shape");
                event[5] = seconds(in.i64("a delay"));
                out.arbitrary.emplace(id, event);
            }
        }

        void read_trapezoids(Cursor& in, Parsed& out)
        {
            const int64_t count = count_of(in, "trapezoids");
            for (int64_t i = 0; i < count; ++i)
            {
                const int id = in.i32("a gradient id");
                std::array<double, TRAP_WIDTH> event{};
                event[0] = in.f64("an amplitude");
                event[1] = seconds(in.i64("a rise time"));
                event[2] = seconds(in.i64("a flat time"));
                event[3] = seconds(in.i64("a fall time"));
                event[4] = seconds(in.i64("a delay"));
                out.trapezoid.emplace(id, event);
            }
        }

        void read_adc(Cursor& in, Parsed& out)
        {
            const int64_t count = count_of(in, "ADC events");
            for (int64_t i = 0; i < count; ++i)
            {
                const int id = in.i32("an ADC id");
                std::array<double, ADC_WIDTH> event{};
                event[0] = static_cast<double>(in.i64("a sample count"));
                event[1] = seconds(in.i64("a dwell time"));
                event[2] = seconds(in.i64("a delay"));
                event[3] = in.f64("a frequency in ppm");
                event[4] = in.f64("a phase in ppm");
                event[5] = in.f64("a frequency offset");
                event[6] = in.f64("a phase offset");
                event[7] = in.i32("a phase shape");
                out.adc.emplace(id, event);
            }
        }

        void read_shapes(Cursor& in, Parsed& out)
        {
            const int64_t count = count_of(in, "shapes");
            for (int64_t i = 0; i < count; ++i)
            {
                const int id = in.i32("a shape id");
                const int64_t uncompressed = count_of(in, "shape samples");
                const int64_t compressed = count_of(in, "shape samples");
                std::vector<double> samples;
                samples.reserve(static_cast<size_t>(compressed));
                for (int64_t s = 0; s < compressed; ++s)
                    samples.push_back(in.f32("a shape sample"));
                out.shapes.emplace(
                    id, std::make_pair(static_cast<int>(uncompressed), std::move(samples)));
            }
        }

        void read_chain(Cursor& in, Parsed& out)
        {
            const int64_t count = count_of(in, "extension links");
            for (int64_t i = 0; i < count; ++i)
            {
                const int id = in.i32("an extension id");
                std::array<int32_t, EXTENSION_WIDTH> chain{};
                chain[0] = in.i32("an extension type");
                chain[1] = in.i32("an extension reference");
                chain[2] = in.i32("the next link");
                out.chains.emplace(id, chain);
            }
        }

        /** The type id every specification section names itself with. */
        void declare(Cursor& in, Parsed& out, const char* name)
        {
            out.extension_types[name] = in.i32("an extension type id");
        }

        void read_triggers(Cursor& in, Parsed& out)
        {
            declare(in, out, "TRIGGERS");
            const int64_t count = count_of(in, "triggers");
            for (int64_t i = 0; i < count; ++i)
            {
                const int id = in.i32("a trigger id");
                std::array<double, TRIGGER_WIDTH> trigger{};
                trigger[0] = in.i32("a trigger type");
                trigger[1] = in.i32("a channel");
                trigger[2] = seconds(in.i64("a delay"));
                trigger[3] = seconds(in.i64("a duration"));
                out.triggers.emplace(id, trigger);
            }
        }

        void read_labels(Cursor& in, Parsed& out, bool increment)
        {
            declare(in, out, increment ? "LABELINC" : "LABELSET");
            const int64_t count = count_of(in, "labels");
            for (int64_t i = 0; i < count; ++i)
            {
                const int id = in.i32("a label id");
                ParsedLabel label;
                label.value = in.i32("a label value");
                label.label_id = in.i32("a label");
                (increment ? out.label_inc : out.label_set).emplace(id, label);
            }
        }

        void read_soft_delays(Cursor& in, Parsed& out)
        {
            declare(in, out, "DELAYS");
            const int64_t count = count_of(in, "soft delays");
            for (int64_t i = 0; i < count; ++i)
            {
                const int id = in.i32("a soft delay id");
                SoftDelay delay;
                delay.num = in.i32("a delay number");
                delay.offset = seconds(in.i64("an offset"));
                delay.factor = in.f64("a factor");
                const int32_t length = in.i32("a hint length");
                if (length < 0)
                    in.fail("a negative hint length");
                delay.hint = in.bytes(static_cast<size_t>(length), "a hint");
                out.soft_delays.emplace(id, std::move(delay));
            }
        }

        void read_rf_shims(Cursor& in, Parsed& out)
        {
            declare(in, out, "RF_SHIMS");
            const int64_t count = count_of(in, "RF shims");
            for (int64_t i = 0; i < count; ++i)
            {
                const int id = in.i32("an RF shim id");
                const int32_t channels = in.i32("a channel count");
                if (channels < 0)
                    in.fail("a negative channel count");
                std::vector<double> values;
                values.reserve(static_cast<size_t>(channels) * 2);
                // A magnitude and a phase per channel.
                for (int32_t v = 0; v < channels * 2; ++v)
                    values.push_back(in.f64("a shim value"));
                out.shims.emplace(id, std::move(values));
            }
        }

        void read_rotations(Cursor& in, Parsed& out)
        {
            declare(in, out, "ROTATIONS");
            const int64_t count = count_of(in, "rotations");
            for (int64_t i = 0; i < count; ++i)
            {
                const int id = in.i32("a rotation id");
                std::array<double, ROTATION_WIDTH> quaternion{};
                for (int component = 0; component < ROTATION_WIDTH; ++component)
                    quaternion[static_cast<size_t>(component)] = in.f64("a quaternion component");
                out.rotations.emplace(id, quaternion);
            }
        }

    } // namespace

    Parsed parse_binary(const std::string& contents)
    {
        Cursor in(contents.data(), contents.data() + contents.size());
        in.skip(sizeof(BINARY_MAGIC), "the file header");

        Parsed out;
        out.major = static_cast<int>(in.i64("the major version"));
        out.minor = static_cast<int>(in.i64("the minor version"));
        out.revision = static_cast<int>(in.i64("the revision"));
        if (out.major < 1 || out.major > 999)
            throw std::runtime_error(
                "read(): this file's version reads as " + std::to_string(out.major) +
                ", so its numbers were written the other way round; only little-endian "
                "sequence files can be read");
        if (out.combined() < 1005000)
            throw std::runtime_error(
                "read(): this is a Pulseq " + std::to_string(out.major) + "." +
                std::to_string(out.minor) + "." + std::to_string(out.revision) +
                " file, and 1.5.0 is the oldest that can be read");

        while (!in.done())
        {
            const uint64_t code = in.u64("a section code");
            if ((code >> 32) != 0xFFFFFFFFULL)
                in.fail("expected a section code");
            switch (static_cast<Section>(code & 0xFFFFFFFFULL))
            {
            case SEC_DEFINITIONS: read_definitions(in, out); break;
            case SEC_BLOCKS: read_blocks(in, out); break;
            case SEC_RF: read_rf(in, out); break;
            case SEC_GRADIENTS: read_arbitrary(in, out); break;
            case SEC_TRAPEZOIDS: read_trapezoids(in, out); break;
            case SEC_ADC: read_adc(in, out); break;
            case SEC_DELAYS:
                // Pre-1.4 delays, which nothing writes any more: the rows are
                // a fixed twelve bytes and say nothing a 1.5 file needs.
                in.skip(static_cast<size_t>(count_of(in, "delays")) * 12, "the delays");
                break;
            case SEC_SHAPES: read_shapes(in, out); break;
            case SEC_EXTENSIONS: read_chain(in, out); break;
            case SEC_TRIGGERS: read_triggers(in, out); break;
            case SEC_LABELSET: read_labels(in, out, false); break;
            case SEC_LABELINC: read_labels(in, out, true); break;
            case SEC_SOFTDELAYS: read_soft_delays(in, out); break;
            case SEC_RFSHIMS: read_rf_shims(in, out); break;
            case SEC_ROTATIONS: read_rotations(in, out); break;
            case SEC_SIGNATURE:
            {
                // Written by one toolbox and not by this one. The bytes are
                // skipped rather than checked: what they cover is the file up
                // to the section, which is not a claim this reader makes.
                const int32_t type_length = in.i32("a signature type length");
                if (type_length < 0)
                    in.fail("a negative signature type length");
                in.skip(static_cast<size_t>(type_length), "the signature type");
                const int32_t digest_length = in.i32("a signature length");
                if (digest_length < 0)
                    in.fail("a negative signature length");
                in.skip(static_cast<size_t>(digest_length), "the signature");
                in.i64("the signed length");
                break;
            }
            default: in.fail("unknown section " + std::to_string(code & 0xFFFFFFFFULL));
            }
        }

        return out;
    }

} // namespace pulseq
