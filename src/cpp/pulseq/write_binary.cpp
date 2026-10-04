/**
 * @file write_binary.cpp
 * @brief Little-endian Pulseq binary serialisation.
 *
 * Times use integer picoseconds, rounded to even for reference byte parity.
 * Shape samples use float32. An optional signature stores the raw MD5 digest
 * of the preceding bytes and the byte count covered.
 */

#include "pulseq/binary.hpp"

#include "pulseq/md5.hpp"
#include "pulseq/parallel.hpp"
#include "pulseq/sequence.hpp"
#include "pulseq/write.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace pulseq
{

    namespace
    {
        /* ============================================================== */
        /*  Putting bytes down                                            */
        /* ============================================================== */

        /** The low @p bytes bytes of @p value at @p at, least significant first. */
        inline void put_le(char* at, uint64_t value, int bytes)
        {
            for (int i = 0; i < bytes; ++i)
                at[i] = static_cast<char>((value >> (8 * i)) & 0xFF);
        }

        void put_u64(std::string& out, uint64_t value)
        {
            char bytes[8];
            for (int i = 0; i < 8; ++i)
                bytes[i] = static_cast<char>((value >> (8 * i)) & 0xFF);
            out.append(bytes, 8);
        }

        void put_i64(std::string& out, int64_t value)
        {
            put_u64(out, static_cast<uint64_t>(value));
        }

        void put_i32(std::string& out, int32_t value)
        {
            const uint32_t bits = static_cast<uint32_t>(value);
            char bytes[4];
            for (int i = 0; i < 4; ++i)
                bytes[i] = static_cast<char>((bits >> (8 * i)) & 0xFF);
            out.append(bytes, 4);
        }

        void put_f64(std::string& out, double value)
        {
            uint64_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            put_u64(out, bits);
        }

        void put_f32(std::string& out, double value)
        {
            const float narrowed = static_cast<float>(value);
            uint32_t bits;
            std::memcpy(&bits, &narrowed, sizeof(bits));
            char bytes[4];
            for (int i = 0; i < 4; ++i)
                bytes[i] = static_cast<char>((bits >> (8 * i)) & 0xFF);
            out.append(bytes, 4);
        }

        /** A NUL-terminated string, as the definition and label sections carry them. */
        void put_string(std::string& out, const std::string& text)
        {
            out.append(text);
            out.push_back('\0');
        }

        void put_section(std::string& out, Section code)
        {
            put_u64(out, SECTION_PREFIX | static_cast<uint64_t>(code));
        }

        /** Seconds as whole picoseconds, halves to even. */
        int64_t picoseconds(double seconds)
        {
            return static_cast<int64_t>(std::nearbyint(seconds * 1e12));
        }

        int64_t whole(double value)
        {
            return static_cast<int64_t>(std::nearbyint(value));
        }

        /* ============================================================== */
        /*  Sections                                                      */
        /* ============================================================== */

        void write_definitions(std::string& out, const Sequence& seq)
        {
            if (seq.definitions().empty())
                return;
            put_section(out, SEC_DEFINITIONS);
            put_i64(out, static_cast<int64_t>(seq.definitions().size()));
            for (const auto& entry : seq.definitions())
            {
                // The name's length in front of it, not a terminator after
                // it, and the value count is an int32 -- so a definition with
                // one value per slice is as writable as any other.
                put_i32(out, static_cast<int32_t>(entry.first.size()));
                out.append(entry.first);

                const Definition& value = entry.second;
                if (value.kind() == Definition::Kind::Text)
                {
                    // For text the count is the byte length, and the bytes
                    // that follow are not terminated.
                    put_i32(out, static_cast<int32_t>(value.text().size()));
                    out.push_back('c');
                    out.append(value.text());
                    continue;
                }

                put_i32(out, static_cast<int32_t>(value.numbers().size()));
                const bool integral = value.kind() == Definition::Kind::Int;
                out.push_back(integral ? 'i' : 'f');
                for (const double number : value.numbers())
                {
                    if (integral)
                        put_i32(out, static_cast<int32_t>(whole(number)));
                    else
                        put_f64(out, number);
                }
            }
        }

        void write_blocks(std::string& out, const Sequence& seq)
        {
            // Written even when there are none, because a sequence with no
            // blocks is still a sequence and a reader should see that said.
            const int count = seq.num_blocks();
            put_section(out, SEC_BLOCKS);
            put_i64(out, static_cast<int64_t>(count));

            const double raster = seq.block_duration_raster();
            const int32_t* events = seq.block_events();
            const double* durations = seq.block_durations();
            constexpr size_t record = sizeof(int64_t) + BLOCK_FILE_COLUMNS * sizeof(int32_t);
            const size_t start = out.size();
            out.resize(start + static_cast<size_t>(count) * record);
            char* const base = &out[start];
            parallel_ranges(
                static_cast<size_t>(count),
                worker_count(static_cast<size_t>(count)),
                [&](size_t first, size_t last)
                {
                    for (size_t i = first; i < last; ++i)
                    {
                        const double exact = durations[i] / raster;
                        const double rounded = std::rint(exact);
                        if (std::fabs(rounded - exact) >= 1e-6)
                            throw std::runtime_error(
                                "write_binary(): block " + std::to_string(i + 1) +
                                " duration is not a multiple of the block duration raster");
                        char* at = base + i * record;
                        put_le(at, static_cast<uint64_t>(static_cast<int64_t>(rounded)), 8);
                        for (int column = 0; column < BLOCK_FILE_COLUMNS; ++column)
                            put_le(
                                at + 8 + 4 * column,
                                static_cast<uint32_t>(events[i * BLOCK_WIDTH + column]),
                                4);
                    }
                });
        }

        void write_rf(std::string& out, const Sequence& seq)
        {
            const Table& library = seq.rf_library();
            if (library.empty())
                return;
            put_section(out, SEC_RF);
            put_i64(out, static_cast<int64_t>(library.size()));
            for (int id = 1; id <= library.size(); ++id)
            {
                const double* d = library.row(id);
                put_i32(out, id);
                put_f64(out, d[0]);
                put_i32(out, static_cast<int32_t>(whole(d[1])));
                put_i32(out, static_cast<int32_t>(whole(d[2])));
                put_i32(out, static_cast<int32_t>(whole(d[3])));
                put_i64(out, picoseconds(d[4]));
                put_i64(out, picoseconds(d[5]));
                put_f64(out, d[6]);
                put_f64(out, d[7]);
                put_f64(out, d[8]);
                put_f64(out, d[9]);
                const char use = seq.rf_uses()[static_cast<size_t>(id) - 1];
                out.push_back(use ? use : 'u');
            }
        }

        void write_gradients(std::string& out, const Sequence& seq)
        {
            // The two kinds share one numbering, so each section holds an
            // ascending subset of the ids rather than a run from one.
            std::vector<int> arbitrary, trapezoids;
            for (int id = 1; id <= seq.num_gradients(); ++id)
                (seq.grad_kind(id) == GradKind::Arbitrary ? arbitrary : trapezoids).push_back(id);

            if (!arbitrary.empty())
            {
                put_section(out, SEC_GRADIENTS);
                put_i64(out, static_cast<int64_t>(arbitrary.size()));
                for (const int id : arbitrary)
                {
                    const double* d = seq.arb_library().row(seq.grad_row(id));
                    put_i32(out, id);
                    put_f64(out, d[0]);
                    put_f64(out, d[1]);
                    put_f64(out, d[2]);
                    put_i32(out, static_cast<int32_t>(whole(d[3])));
                    put_i32(out, static_cast<int32_t>(whole(d[4])));
                    put_i64(out, picoseconds(d[5]));
                }
            }

            if (!trapezoids.empty())
            {
                put_section(out, SEC_TRAPEZOIDS);
                put_i64(out, static_cast<int64_t>(trapezoids.size()));
                for (const int id : trapezoids)
                {
                    const double* d = seq.trap_library().row(seq.grad_row(id));
                    put_i32(out, id);
                    put_f64(out, d[0]);
                    for (int column = 1; column < TRAP_WIDTH; ++column)
                        put_i64(out, picoseconds(d[column]));
                }
            }
        }

        void write_adc(std::string& out, const Sequence& seq)
        {
            const Table& library = seq.adc_library();
            if (library.empty())
                return;
            put_section(out, SEC_ADC);
            put_i64(out, static_cast<int64_t>(library.size()));
            for (int id = 1; id <= library.size(); ++id)
            {
                const double* d = library.row(id);
                put_i32(out, id);
                put_i64(out, whole(d[0]));
                put_i64(out, picoseconds(d[1]));
                put_i64(out, picoseconds(d[2]));
                put_f64(out, d[3]);
                put_f64(out, d[4]);
                put_f64(out, d[5]);
                put_f64(out, d[6]);
                put_i32(out, static_cast<int32_t>(whole(d[7])));
            }
        }

        void write_shapes(std::string& out, const Sequence& seq)
        {
            const ShapeLibrary& shapes = seq.shape_library();
            if (shapes.empty())
                return;
            put_section(out, SEC_SHAPES);
            put_i64(out, static_cast<int64_t>(shapes.size()));
            for (int id = 1; id <= shapes.size(); ++id)
            {
                const int count = shapes.num_compressed(id);
                const double* samples = shapes.samples(id);
                put_i32(out, id);
                put_i64(out, static_cast<int64_t>(shapes.num_uncompressed(id)));
                put_i64(out, static_cast<int64_t>(count));
                for (int i = 0; i < count; ++i)
                    put_f32(out, samples[i]);
            }
        }

        void write_extension_chain(std::string& out, const Sequence& seq)
        {
            const IntTable& chains = seq.extensions_library();
            if (chains.empty())
                return;
            put_section(out, SEC_EXTENSIONS);
            put_i64(out, static_cast<int64_t>(chains.size()));
            for (int id = 1; id <= chains.size(); ++id)
            {
                const int32_t* row = chains.row(id);
                put_i32(out, id);
                put_i32(out, row[0]);
                put_i32(out, row[1]);
                put_i32(out, row[2]);
            }
        }

        void write_triggers(std::string& out, Sequence& seq)
        {
            const Table& library = seq.trigger_library();
            if (library.empty())
                return;
            put_section(out, SEC_TRIGGERS);
            put_i32(out, static_cast<int32_t>(seq.extension_type_id("TRIGGERS")));
            put_i64(out, static_cast<int64_t>(library.size()));
            for (int id = 1; id <= library.size(); ++id)
            {
                const double* d = library.row(id);
                put_i32(out, id);
                put_i32(out, static_cast<int32_t>(whole(d[0])));
                put_i32(out, static_cast<int32_t>(whole(d[1])));
                put_i64(out, picoseconds(d[2]));
                put_i64(out, picoseconds(d[3]));
            }
        }

        void write_labels(std::string& out, Sequence& seq)
        {
            const std::pair<const char*, const IntTable*> sections[2] = {
                {"LABELSET", &seq.label_set_library()},
                {"LABELINC", &seq.label_inc_library()},
            };
            const Section codes[2] = {SEC_LABELSET, SEC_LABELINC};
            for (int which = 0; which < 2; ++which)
            {
                const IntTable& library = *sections[which].second;
                if (library.empty())
                    continue;
                put_section(out, codes[which]);
                put_i32(out, static_cast<int32_t>(seq.extension_type_id(sections[which].first)));
                put_i64(out, static_cast<int64_t>(library.size()));
                for (int id = 1; id <= library.size(); ++id)
                {
                    const int32_t* row = library.row(id);
                    put_i32(out, id);
                    put_i32(out, row[0]);
                    put_i32(out, row[1]);
                }
            }
        }

        void write_soft_delays(std::string& out, Sequence& seq)
        {
            if (seq.soft_delay_library().empty())
                return;
            put_section(out, SEC_SOFTDELAYS);
            put_i32(out, static_cast<int32_t>(seq.extension_type_id("DELAYS")));
            put_i64(out, static_cast<int64_t>(seq.soft_delay_library().size()));
            int id = 1;
            for (const SoftDelay& row : seq.soft_delay_library())
            {
                put_i32(out, id++);
                put_i32(out, row.num);
                put_i64(out, picoseconds(row.offset));
                put_f64(out, row.factor);
                put_i32(out, static_cast<int32_t>(row.hint.size()));
                out.append(row.hint);
            }
        }

        void write_rf_shims(std::string& out, Sequence& seq)
        {
            const RaggedTable& library = seq.rf_shim_library();
            if (library.empty())
                return;
            put_section(out, SEC_RFSHIMS);
            put_i32(out, static_cast<int32_t>(seq.extension_type_id("RF_SHIMS")));
            put_i64(out, static_cast<int64_t>(library.size()));
            for (int id = 1; id <= library.size(); ++id)
            {
                const int length = library.length(id);
                const double* values = library.row(id);
                put_i32(out, id);
                // A magnitude and a phase per channel, so the count is half.
                put_i32(out, length / 2);
                for (int i = 0; i < length; ++i)
                    put_f64(out, values[i]);
            }
        }

        void write_rotations(std::string& out, Sequence& seq)
        {
            const Table& library = seq.rotation_library();
            if (library.empty())
                return;
            put_section(out, SEC_ROTATIONS);
            put_i32(out, static_cast<int32_t>(seq.extension_type_id("ROTATIONS")));
            put_i64(out, static_cast<int64_t>(library.size()));
            for (int id = 1; id <= library.size(); ++id)
            {
                const double* d = library.row(id);
                put_i32(out, id);
                for (int column = 0; column < ROTATION_WIDTH; ++column)
                    put_f64(out, d[column]);
            }
        }
    } // namespace

    /* ================================================================== */
    /*  Entry points                                                      */
    /* ================================================================== */

    bool is_binary(std::string_view contents)
    {
        return contents.size() >= sizeof(BINARY_MAGIC) &&
               std::memcmp(contents.data(), BINARY_MAGIC, sizeof(BINARY_MAGIC)) == 0;
    }

    namespace
    {
        /** Everything a signature covers. */
        std::string binary_body(Sequence& seq)
        {
        seq.compress_shapes();
        seq.publish_rasters();
        declare_custom_labels(seq);

        // The sections after the blocks are the event vocabulary, encoded
        // first so that the whole body is allocated once.
        const Sequence& reading = seq;
        std::string rest;
        write_rf(rest, reading);
        write_gradients(rest, reading);
        write_adc(rest, reading);
        write_shapes(rest, reading);
        write_extension_chain(rest, reading);
        write_triggers(rest, seq);
        write_labels(rest, seq);
        write_soft_delays(rest, seq);
        write_rf_shims(rest, seq);
        write_rotations(rest, seq);

        std::string out;
        // A block is 32 bytes; the definitions and the signature are small.
        out.reserve(static_cast<size_t>(seq.num_blocks()) * 32 + rest.size() + 65536);
        out.append(reinterpret_cast<const char*>(BINARY_MAGIC), sizeof(BINARY_MAGIC));
        put_i64(out, seq.version_major());
        put_i64(out, seq.version_minor());
        put_i64(out, required_revision(seq));
        write_definitions(out, reading);
        write_blocks(out, reading);
        out += rest;

        return out;
    }

        /** The signature section over the first @p signed_length bytes, whose digest is @p hex. */
        std::string signature_section(const std::string& hex, size_t signed_length)
        {
            std::string out;
            put_section(out, SEC_SIGNATURE);
            const std::string type = "md5";
            put_i32(out, static_cast<int32_t>(type.size()));
            out.append(type);
            put_i32(out, static_cast<int32_t>(hex.size() / 2));
            for (size_t i = 0; i + 1 < hex.size(); i += 2)
                out.push_back(static_cast<char>(std::stoul(hex.substr(i, 2), nullptr, 16)));
            put_i64(out, static_cast<int64_t>(signed_length));
            return out;
        }
    } // namespace

    std::string write_binary(Sequence& seq, bool create_signature)
    {
        std::string out = binary_body(seq);
        if (create_signature)
            out += signature_section(md5_hex(out.data(), out.size()), out.size());
        return out;
    }

    std::string write_binary_file(Sequence& seq, const std::string& path, bool create_signature)
    {
        const std::string body = binary_body(seq);
        std::ofstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::trunc);
        if (!file)
            throw std::runtime_error("write_binary(): cannot open " + path + " for writing");
        // The body goes to the file while it is digested.
        std::thread writer([&] { file.write(body.data(), static_cast<std::streamsize>(body.size())); });
        std::string hex;
        if (create_signature)
            hex = md5_hex(body.data(), body.size());
        writer.join();
        if (create_signature)
        {
            const std::string section = signature_section(hex, body.size());
            file.write(section.data(), static_cast<std::streamsize>(section.size()));
        }
        file.close();
        if (!file)
            throw std::runtime_error("write_binary(): cannot write " + path);
        return hex;
    }

    bool binary_signature(std::string_view contents, std::string& type, std::string& value,
                          bool check)
    {
        type.clear();
        value.clear();

        /* The section ends with how much of the file it covers, so it is read
         * from the end backwards: the last eight bytes say where the payload
         * stops, and the digest and its name are what lie between there and
         * them. */
        if (contents.size() < sizeof(int64_t))
            return false;
        int64_t signed_length = 0;
        std::memcpy(
            &signed_length, contents.data() + contents.size() - sizeof(int64_t),
            sizeof(int64_t));
        if (signed_length <= 0 ||
            static_cast<size_t>(signed_length) + sizeof(int64_t) > contents.size())
            return false;

        size_t at = static_cast<size_t>(signed_length);
        const auto room = [&](size_t bytes) { return at + bytes <= contents.size(); };

        if (!room(sizeof(uint64_t)))
            return false;
        uint64_t section = 0;
        std::memcpy(&section, contents.data() + at, sizeof(section));
        at += sizeof(section);
        if (section != (SECTION_PREFIX | static_cast<uint64_t>(SEC_SIGNATURE)))
            return false;

        if (!room(sizeof(int32_t)))
            return false;
        int32_t type_length = 0;
        std::memcpy(&type_length, contents.data() + at, sizeof(type_length));
        at += sizeof(type_length);
        if (type_length < 0 || !room(static_cast<size_t>(type_length)))
            return false;
        type.assign(contents.substr(at, static_cast<size_t>(type_length)));
        at += static_cast<size_t>(type_length);

        if (!room(sizeof(int32_t)))
            return false;
        int32_t digest_length = 0;
        std::memcpy(&digest_length, contents.data() + at, sizeof(digest_length));
        at += sizeof(digest_length);
        if (digest_length < 0 || !room(static_cast<size_t>(digest_length)))
        {
            type.clear();
            return false;
        }

        static const char kHex[] = "0123456789abcdef";
        for (int32_t i = 0; i < digest_length; ++i)
        {
            const unsigned char byte =
                static_cast<unsigned char>(contents[at + static_cast<size_t>(i)]);
            value.push_back(kHex[byte >> 4]);
            value.push_back(kHex[byte & 0x0F]);
        }

        if (!check)
            return true;
        return value == md5_hex(contents.data(), static_cast<size_t>(signed_length));
    }

} // namespace pulseq
