/**
 * @file write_binary.cpp
 * @brief The Pulseq binary sequence writer.
 *
 * Every field goes out little-endian byte by byte, so the file a machine
 * writes does not depend on the machine.
 *
 * ### Times are integer picoseconds, rounded half to even
 *
 * The reference writer is MATLAB and Python, whose `round` breaks a tie
 * towards the even number; C's `llround` breaks it away from zero. A delay
 * landing exactly halfway between two picoseconds is rare and the difference
 * is a picosecond, but it is a difference in the bytes, so `nearbyint` under
 * the default rounding mode is what is used here.
 *
 * ### What the format cannot carry
 *
 * There is no signature section, so a binary file is not signed. Shape
 * samples are single precision, where the text form writes nine significant
 * digits -- so for a shape the text file is the more faithful container, and
 * for everything else the binary one is.
 */

#include "pulseq/binary.hpp"

#include "pulseq/sequence.hpp"
#include "pulseq/write.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulseq
{

    namespace
    {
        /* ============================================================== */
        /*  Putting bytes down                                            */
        /* ============================================================== */

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
            for (int i = 0; i < count; ++i)
            {
                const double exact = durations[i] / raster;
                const double rounded = std::rint(exact);
                if (std::fabs(rounded - exact) >= 1e-6)
                    throw std::runtime_error(
                        "write_binary(): block " + std::to_string(i + 1) +
                        " duration is not a multiple of the block duration raster");
                put_i64(out, static_cast<int64_t>(rounded));
                for (int column = 0; column < BLOCK_WIDTH; ++column)
                    put_i32(out, events[static_cast<size_t>(i) * BLOCK_WIDTH + column]);
            }
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

        /**
         * The names behind the label ids the label rows use.
         *
         * A label id means nothing outside the sequence that handed it out.
         * For a label Pulseq defines that costs nothing -- every toolbox
         * numbers those the same way -- so the section is written only when
         * the sequence uses a name Pulseq does not define, which is the case
         * where the number alone would not survive the trip. Writing it
         * always would put a section in every labelled file that a reader
         * predating it refuses.
         */
        void write_label_names(std::string& out, const Sequence& seq)
        {
            std::map<int32_t, std::string> used;
            bool any_custom = false;
            for (const IntTable* library : {&seq.label_set_library(), &seq.label_inc_library()})
                for (int id = 1; id <= library->size(); ++id)
                {
                    const int32_t label = library->row(id)[1];
                    const std::string& name = seq.label_name(label);
                    used[label] = name.empty() ? "UNKNOWN" : name;
                    any_custom = any_custom || seq.is_custom_label(label);
                }
            if (used.empty() || !any_custom)
                return;

            put_section(out, SEC_LABELNAMES);
            put_i64(out, static_cast<int64_t>(used.size()));
            for (const auto& entry : used)
            {
                put_i32(out, entry.first);
                put_string(out, entry.second);
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

    bool is_binary(const std::string& contents)
    {
        return contents.size() >= sizeof(BINARY_MAGIC) &&
               std::memcmp(contents.data(), BINARY_MAGIC, sizeof(BINARY_MAGIC)) == 0;
    }

    std::string write_binary(Sequence& seq)
    {
        seq.compress_shapes();
        seq.set_definition("TotalDuration", Definition(seq.duration()));
        seq.publish_rasters();

        std::string out;
        // A block is 32 bytes and dominates a large file; the rest is the
        // event vocabulary, which is small beside it.
        out.reserve(static_cast<size_t>(seq.num_blocks()) * 32 + 8192);

        out.append(reinterpret_cast<const char*>(BINARY_MAGIC), sizeof(BINARY_MAGIC));
        put_i64(out, seq.version_major());
        put_i64(out, seq.version_minor());
        // At least 1.5.1, whatever the sequence needs: the binary form is
        // itself a 1.5.1 addition, so a binary file declaring 1.5.0 claims a
        // revision that could not have written it.
        put_i64(out, std::max(1, required_revision(seq)));

        const Sequence& reading = seq;
        write_definitions(out, reading);
        write_blocks(out, reading);
        write_rf(out, reading);
        write_gradients(out, reading);
        write_adc(out, reading);
        write_shapes(out, reading);
        write_extension_chain(out, reading);
        write_triggers(out, seq);
        write_label_names(out, reading);
        write_labels(out, seq);
        write_soft_delays(out, seq);
        write_rf_shims(out, seq);
        write_rotations(out, seq);
        return out;
    }

} // namespace pulseq
