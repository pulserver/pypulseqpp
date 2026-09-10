/**
 * @file write_text.cpp
 * @brief The Pulseq `.seq` text writer.
 *
 * The layout is PyPulseq's, with the two 1.5.1 extension sections added.
 * Every format string below is character for character the one PyPulseq
 * uses, and that is not stylistic: the test that proves this file correct
 * diffs its output byte for byte against PyPulseq's for the same sequence,
 * so a `%g` that should have been a `%12g` shows up as a failed comparison
 * rather than as a subtly different file.
 *
 * (C's printf and Python's `%` agree on every conversion used here -- checked
 * across denormals, -0.0, and the exponent boundaries -- so matching the
 * format string is enough to match the bytes.)
 *
 * Two sections are as long as the scan rather than as long as the design:
 * `[BLOCKS]`, which has a row per block, and `[EXTENSIONS]`, which has one per
 * distinct chain and therefore one per labelled readout before deduplication.
 * Those two go through `render_rows` below, which lays digits
 * straight into a preallocated buffer, in parallel; everything else is per
 * distinct event and goes through snprintf.
 */

#include "pulseq/sequence.hpp"
#include "pulseq/write.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "pulseq/md5.hpp"

namespace pulseq
{

    namespace
    {
        /** snprintf into the output string. */
        void appendf(std::string& out, const char* format, ...)
#if defined(__GNUC__)
            __attribute__((format(printf, 2, 3)))
#endif
            ;

        void appendf(std::string& out, const char* format, ...)
        {
            char stack[512];
            va_list args;
            va_start(args, format);
            const int needed = std::vsnprintf(stack, sizeof(stack), format, args);
            va_end(args);

            if (needed < 0)
                throw std::runtime_error("write(): formatting failed");
            if (static_cast<size_t>(needed) < sizeof(stack))
            {
                out.append(stack, static_cast<size_t>(needed));
                return;
            }

            std::vector<char> heap(static_cast<size_t>(needed) + 1);
            va_start(args, format);
            std::vsnprintf(heap.data(), heap.size(), format, args);
            va_end(args);
            out.append(heap.data(), static_cast<size_t>(needed));
        }

        /** Decimal width of a non-negative count, for the block-number column. */
        int decimal_width(long value)
        {
            int width = 1;
            while (value >= 10)
            {
                value /= 10;
                ++width;
            }
            return width;
        }

        /* ============================================================== */
        /*  The two sections whose size is the scan                       */
        /* ============================================================== */
        //
        // `[BLOCKS]` has a row per block and `[EXTENSIONS]` one per distinct
        // chain, so on a large 3D protocol these two are most of the file and
        // rendering them is most of the cost of writing one.  Appending a
        // character at a time -- a bounds check and a possible reallocation per
        // digit -- is what makes that expensive, so instead each row's maximum
        // length is known up front, a buffer is allocated for the worst case,
        // and the digits go in through a raw cursor.
        //
        // Chunks are rendered in parallel and concatenated in order, so the
        // output does not depend on how many threads the machine offered.

        /** Write @p value right-aligned in @p width; returns the new cursor. */
        inline char* put_int_field(char* out, long value, int width)
        {
            char digits[24];
            int len = 0;
            unsigned long magnitude;

            if (value < 0)
                magnitude = static_cast<unsigned long>(-(value + 1)) + 1UL;
            else
                magnitude = static_cast<unsigned long>(value);

            do
            {
                digits[len++] = static_cast<char>('0' + (magnitude % 10));
                magnitude /= 10;
            } while (magnitude != 0);

            if (value < 0)
                digits[len++] = '-';

            for (int pad = len; pad < width; ++pad)
                *out++ = ' ';
            while (len > 0)
                *out++ = digits[--len];
            return out;
        }

        unsigned worker_count(size_t rows)
        {
            // Below this the threads cost more than the work they save.
            if (rows < (1u << 16))
                return 1;
            unsigned available = std::thread::hardware_concurrency();
            if (available == 0)
                available = 1;
            return std::min(available, 8u);
        }

        /**
         * Render @p rows rows of integer fields, in parallel, onto @p out.
         *
         * @p emit writes one row and returns the cursor it left off at; it must
         * never write more than @p row_bound bytes.
         *
         * The pieces land in @p out with a single copy each.  Going through an
         * intermediate string first would cost a second pass over what, on a
         * large scan, is eighty megabytes.
         */
        template <typename Emit>
        void render_rows(std::string& out, size_t rows, size_t row_bound, const Emit& emit)
        {
            if (rows == 0)
                return;

            const unsigned workers = worker_count(rows);
            const size_t chunk = (rows + workers - 1) / workers;

            std::vector<std::string> pieces(workers);
            auto run = [&](unsigned worker)
            {
                const size_t first = std::min(rows, static_cast<size_t>(worker) * chunk);
                const size_t last = std::min(rows, first + chunk);
                if (first >= last)
                    return;
                std::string& piece = pieces[worker];
                piece.resize((last - first) * row_bound);
                char* cursor = &piece[0];
                for (size_t row = first; row < last; ++row)
                    cursor = emit(cursor, row);
                piece.resize(static_cast<size_t>(cursor - piece.data()));
            };

            if (workers == 1)
            {
                run(0);
            }
            else
            {
                std::vector<std::thread> threads;
                threads.reserve(workers - 1);
                for (unsigned worker = 1; worker < workers; ++worker)
                    threads.emplace_back(run, worker);
                run(0);
                for (std::thread& thread : threads)
                    thread.join();
            }

            size_t total = 0;
            for (const std::string& piece : pieces)
                total += piece.size();

            // In worker order, so the file does not depend on how many threads
            // the machine offered.
            const size_t base = out.size();
            out.resize(base + total);
            char* destination = &out[base];
            for (const std::string& piece : pieces)
            {
                std::memcpy(destination, piece.data(), piece.size());
                destination += piece.size();
            }
        }

        /** Block durations as raster ticks, refusing any that is not a multiple. */
        std::vector<long> duration_ticks(const Sequence& seq)
        {
            const int n = seq.num_blocks();
            const double raster = seq.block_duration_raster();
            std::vector<long> ticks(static_cast<size_t>(n));
            const double* durations = seq.block_durations();

            for (int i = 0; i < n; ++i)
            {
                const double exact = durations[i] / raster;
                const double rounded = std::rint(exact);
                if (std::fabs(rounded - exact) >= 1e-6)
                {
                    throw std::runtime_error(
                        "block " + std::to_string(i + 1) +
                        " duration is not a multiple of the block duration raster (" +
                        std::to_string(exact * raster) + " s)");
                }
                ticks[static_cast<size_t>(i)] = static_cast<long>(rounded);
            }
            return ticks;
        }

        void write_definition_value(std::string& out, const Definition& def)
        {
            switch (def.kind())
            {
            case Definition::Kind::Text:
                out.append(def.text());
                out.push_back(' ');
                break;
            case Definition::Kind::Int:
            case Definition::Kind::Real:
                for (double value : def.numbers())
                    appendf(out, "%0.9g ", value);
                break;
            }
        }

        /**
         * Name the labels the builtin table does not, in `[DEFINITIONS]`.
         *
         * A label is written by name in the text form and by number in the
         * binary one, and a number means something only against a table. The
         * builtin table is shared, so a builtin label needs nothing said
         * about it; a name a sequence invented is minted past the end of that
         * table and its number would mean nothing anywhere else.
         *
         * So the names past the table are listed, in the order they were
         * minted, and a number above the table's length resolves by position
         * in that list. It goes in `[DEFINITIONS]` because both forms carry
         * definitions already and a reader is obliged to tolerate a key it
         * does not know -- a new section would make every file using a custom
         * label unreadable by anything that predates it.
         */
    } // namespace

    void declare_custom_labels(Sequence& seq)
    {
            int highest = 0;
            for (const IntTable* library : {&seq.label_set_library(), &seq.label_inc_library()})
                for (int id = 1; id <= library->size(); ++id)
                {
                    const int32_t label = library->row(id)[1];
                    if (seq.is_custom_label(label) && label > highest)
                        highest = label;
                }
            if (highest == 0)
                return;

            // From the first id past the table up to the highest one used:
            // a gap would shift every name after it out of position.
            const int builtin = static_cast<int>(Sequence::builtin_labels().size());
            std::string names;
            for (int id = builtin + 1; id <= highest; ++id)
            {
                if (!names.empty())
                    names.push_back(' ');
                const std::string& name = seq.label_name(id);
                names += name.empty() ? "UNKNOWN" : name;
            }
            seq.set_definition("CustomLabels", Definition(names));
    }


    namespace
    {
        /**
         * Name in `RequiredExtensions` anything a reader must understand.
         *
         * A rotation changes where the gradients point, so a reader that
         * skips the extension it does not know plays a different sequence
         * rather than an approximation of the right one. The definition says
         * so, and is added to whatever it already lists.
         */
        void declare_required_extensions(Sequence& seq)
        {
            if (seq.rotation_library().empty())
                return;

            std::string required;
            auto existing = seq.definitions().find("RequiredExtensions");
            if (existing != seq.definitions().end() &&
                existing->second.kind() == Definition::Kind::Text)
                required = existing->second.text();

            std::string word;
            std::istringstream words(required);
            while (words >> word)
                if (word == "ROTATIONS")
                    return;

            if (!required.empty())
                required.push_back(' ');
            required += "ROTATIONS";
            seq.set_definition("RequiredExtensions", Definition(required));
        }
    } // namespace

    /* ================================================================== */
    /*  Revision                                                          */
    /* ================================================================== */

    int required_revision(const Sequence& seq)
    {
        // The revision this package implements, whatever the sequence came
        // in declaring.  A writer says which revision of the format it wrote,
        // not which subset of it the sequence happened to use: a reader has
        // to understand 1.5.1 to be sure of reading what came out of here,
        // and a file that claims less than the writer implements is claiming
        // something nobody checked.
        (void)seq;
        return WRITTEN_REVISION;
    }

    /* ================================================================== */
    /*  The writer                                                        */
    /* ================================================================== */

    /**
     * The library rows a set of blocks plays, by id: nonzero where a row is
     * referred to. Slot 0 stands for "none" and is never written.
     *
     * An excerpt of a scan is written with only these, because a per-playout
     * value -- a phase-encode amplitude, a spoiling phase -- is a library row
     * of its own, so the libraries grow with the scan as the blocks do.
     */
    struct Referenced
    {
        std::vector<int32_t> rf, grad, adc, chain, trigger, label_set, label_inc, shim,
            rotation, soft_delay, shape;
    };

    /** Each extension specification the sequence declares, by the member numbering it. */
    static std::vector<std::pair<int, std::vector<int32_t> Referenced::*>> specifications(
        const Sequence& seq)
    {
        const std::pair<const char*, std::vector<int32_t> Referenced::*> kinds[] = {
            {"TRIGGERS", &Referenced::trigger},   {"LABELSET", &Referenced::label_set},
            {"LABELINC", &Referenced::label_inc}, {"RF_SHIMS", &Referenced::shim},
            {"ROTATIONS", &Referenced::rotation}, {"DELAYS", &Referenced::soft_delay},
        };
        std::vector<std::pair<int, std::vector<int32_t> Referenced::*>> declared;
        for (const auto& kind : kinds)
        {
            const int type = seq.find_extension_type_id(kind.first);
            if (type != 0)
                declared.emplace_back(type, kind.second);
        }
        return declared;
    }

    /**
     * What row @p id is numbered in the file: its own id when every row is
     * written, its place among the rows kept otherwise.
     */
    static double renumbered(const std::vector<int32_t>& numbering, double id)
    {
        if (numbering.empty() || id <= 0)
            return id;
        const long at = std::lround(id);
        return static_cast<size_t>(at) < numbering.size() ? numbering[static_cast<size_t>(at)]
                                                           : 0;
    }

    static void mark(std::vector<int32_t>& used, double id)
    {
        const long at = std::lround(id);
        if (at > 0 && static_cast<size_t>(at) < used.size())
            used[static_cast<size_t>(at)] = 1;
    }

    static Referenced referenced_by(const Sequence& seq, const std::vector<int32_t>& rows)
    {
        Referenced used;
        auto sized = [](std::vector<int32_t>& mask, int count)
        { mask.assign(static_cast<size_t>(count) + 1, 0); };
        sized(used.rf, seq.rf_library().size());
        sized(used.grad, seq.num_gradients());
        sized(used.adc, seq.adc_library().size());
        sized(used.chain, seq.extensions_library().size());
        sized(used.trigger, seq.trigger_library().size());
        sized(used.label_set, seq.label_set_library().size());
        sized(used.label_inc, seq.label_inc_library().size());
        sized(used.shim, seq.rf_shim_library().size());
        sized(used.rotation, seq.rotation_library().size());
        sized(used.soft_delay, static_cast<int>(seq.soft_delay_library().size()));
        sized(used.shape, seq.shape_library().size());

        const IntTable& chains = seq.extensions_library();
        const int32_t* events = seq.block_events();
        for (const int32_t block : rows)
        {
            const int32_t* row = events + static_cast<size_t>(block - 1) * BLOCK_WIDTH;
            mark(used.rf, row[0]);
            for (int axis = 1; axis <= 3; ++axis)
                mark(used.grad, row[axis]);
            mark(used.adc, row[4]);
            // Chains share their tails, so a link already marked has the rest
            // of its chain marked too.
            for (int32_t link = row[5]; link > 0 && !used.chain[static_cast<size_t>(link)];
                 link = chains.row(link)[2])
                used.chain[static_cast<size_t>(link)] = 1;
        }

        const auto declared = specifications(seq);
        for (int id = 1; id <= chains.size(); ++id)
        {
            if (!used.chain[static_cast<size_t>(id)])
                continue;
            const int32_t* link = chains.row(id);
            for (const auto& kind : declared)
            {
                if (kind.first == link[0])
                    mark(used.*kind.second, link[1]);
            }
        }

        for (int id = 1; id <= seq.rf_library().size(); ++id)
        {
            if (!used.rf[static_cast<size_t>(id)])
                continue;
            const double* d = seq.rf_library().row(id);
            for (int column = 1; column <= 3; ++column)
                mark(used.shape, d[column]);
        }
        for (int id = 1; id <= seq.num_gradients(); ++id)
        {
            if (!used.grad[static_cast<size_t>(id)] || seq.grad_kind(id) != GradKind::Arbitrary)
                continue;
            const double* d = seq.arb_library().row(seq.grad_row(id));
            mark(used.shape, d[3]);
            mark(used.shape, d[4]);
        }
        for (int id = 1; id <= seq.adc_library().size(); ++id)
        {
            if (used.adc[static_cast<size_t>(id)])
                mark(used.shape, seq.adc_library().row(id)[7]);
        }
        // A reader holds a file's ids to running without gaps, so the rows
        // kept are numbered afresh, in the order the library holds them.
        for (std::vector<int32_t>* numbering :
             {&used.rf, &used.grad, &used.adc, &used.chain, &used.trigger, &used.label_set,
              &used.label_inc, &used.shim, &used.rotation, &used.soft_delay, &used.shape})
        {
            int32_t next = 0;
            for (size_t id = 1; id < numbering->size(); ++id)
            {
                if ((*numbering)[id])
                    (*numbering)[id] = ++next;
            }
        }
        return used;
    }

    /** Whether row @p id is written: every row is when @p used is empty. */
    static bool kept(const std::vector<int32_t>& used, int id)
    {
        return used.empty() || used[static_cast<size_t>(id)] != 0;
    }

    /** Whether any of a library's @p count rows is written. */
    static bool keeps_any(const std::vector<int32_t>& used, int count)
    {
        if (used.empty())
            return count > 0;
        for (size_t id = 1; id < used.size(); ++id)
        {
            if (used[id])
                return true;
        }
        return false;
    }

    /** Write the blocks @p rows names, or every block if it is null. */
    static std::string write_text_of(
        Sequence& seq, bool create_signature, const std::vector<int32_t>* rows)
    {
        // Prewrite: a sequence built block by block registers its waveforms as
        // it was given them, because compressing a candidate that
        // deduplication is about to drop is work spent on a shape that will
        // not be in the file.  This is where the survivors are encoded.
        seq.compress_shapes();

        const int n_blocks = seq.num_blocks();
        const size_t n_written = rows ? rows->size() : static_cast<size_t>(n_blocks);
        const std::vector<long> ticks = duration_ticks(seq);

        // The rasters go in because a reader cannot recover a block duration
        // without them: durations are stored as raster ticks. `TotalDuration`
        // deliberately does not, because the reference toolbox records it
        // when it reports on a sequence rather than when it writes one, and a
        // line here that it does not write is a file that does not match.
        seq.publish_rasters();
        declare_required_extensions(seq);
        declare_custom_labels(seq);

        /* Every read below goes through a const reference, so the writer does
         * not look like a mutation to the sequence.  Taking `Table&` from a
         * non-const Sequence is what gives up its deduplicated claim, and a
         * writer that only reads must not cost a caller that pass. */
        const Sequence& reading = seq;

        std::string out;
        // Blocks dominate a large file; ~35 characters each is close enough
        // that the buffer grows a couple of times rather than a couple of
        // dozen.
        out.reserve(n_written * 36 + 4096);

        out.append("# Pulseq sequence file\n# Created by PyPulseq\n\n");

        out.append("[VERSION]\n");
        appendf(out, "major %d\n", reading.version_major());
        appendf(out, "minor %d\n", reading.version_minor());
        appendf(out, "revision %d\n", required_revision(seq));
        out.push_back('\n');

        if (!reading.definitions().empty())
        {
            out.append("[DEFINITIONS]\n");
            for (const auto& entry : reading.definitions())
            {
                out.append(entry.first);
                out.push_back(' ');
                write_definition_value(out, entry.second);
                out.push_back('\n');
            }
            out.push_back('\n');
        }

        /* -- blocks ---------------------------------------------------- */

        out.append("# Format of blocks:\n");
        out.append("# NUM DUR RF  GX  GY  GZ  ADC  EXT\n");
        // Every library row under its own id, or for an excerpt only the rows
        // its blocks play, numbered afresh.
        Referenced used;
        if (rows)
            used = referenced_by(reading, *rows);
        static_assert(BLOCK_FILE_COLUMNS == 6, "a block row is RF, three gradients, ADC, chain");
        const std::vector<int32_t>* const numbering[BLOCK_FILE_COLUMNS] = {
            &used.rf, &used.grad, &used.grad, &used.grad, &used.adc, &used.chain};

        out.append("[BLOCKS]\n");
        if (n_written > 0)
        {
            const int number_width = decimal_width(static_cast<long>(n_written));
            const int widths[8] = {number_width, 3, 3, 3, 3, 3, 2, 2};
            const int32_t* events = reading.block_events();
            const long* tick = ticks.data();

            // The worst a row can be: every field at its widest, a separator
            // between each, and the newline.
            size_t row_bound = 1; // newline
            for (int column = 0; column < 8; ++column)
                row_bound += 1 + std::max(widths[column], 12);

            render_rows(
                out,
                n_written,
                row_bound,
                [&](char* cursor, size_t i)
                {
                    const size_t block = rows ? static_cast<size_t>((*rows)[i] - 1) : i;
                    const int32_t* row = events + block * BLOCK_WIDTH;
                    cursor = put_int_field(cursor, static_cast<long>(i) + 1, widths[0]);
                    *cursor++ = ' ';
                    cursor = put_int_field(cursor, tick[block], widths[1]);
                    for (int column = 0; column < BLOCK_FILE_COLUMNS; ++column)
                    {
                        *cursor++ = ' ';
                        cursor = put_int_field(
                            cursor,
                            static_cast<long>(renumbered(*numbering[column], row[column])),
                            widths[column + 2]);
                    }
                    *cursor++ = '\n';
                    return cursor;
                });
        }
        out.push_back('\n');

        /* -- RF -------------------------------------------------------- */

        if (keeps_any(used.rf, reading.rf_library().size()))
        {
            out.append("# Format of RF events:\n");
            out.append("# id ampl. mag_id phase_id time_shape_id center delay freqPPM phasePPM "
                       "freq phase use\n");
            out.append("# ..   Hz      ..       ..            ..     us    us     ppm  rad/MHz   "
                       "Hz   rad  ..\n");
            out.append("# Field 'use' is the initial of: \n#   excitation refocusing inversion "
                       "saturation preparation other undefined\n");
            out.append("[RF]\n");

            const double raster = reading.rf_raster_time();
            for (int id = 1; id <= reading.rf_library().size(); ++id)
            {
                if (!kept(used.rf, id))
                    continue;
                const double* d = reading.rf_library().row(id);
                const double center = d[4] * 1e6;
                const double delay = std::rint(d[5] / raster) * raster * 1e6;
                const char use = reading.rf_uses()[static_cast<size_t>(id) - 1];
                appendf(
                    out,
                    "%.0f %12g %.0f %.0f %.0f %g %g %g %g %g %g %c\n",
                    renumbered(used.rf, id),
                    d[0],
                    renumbered(used.shape, d[1]),
                    renumbered(used.shape, d[2]),
                    renumbered(used.shape, d[3]),
                    center,
                    delay,
                    d[6],
                    d[7],
                    d[8],
                    d[9],
                    use ? use : 'u');
            }
            out.push_back('\n');
        }

        /* -- gradients ------------------------------------------------- */
        //
        // The two kinds share one numbering, so they are collected by walking
        // that numbering once rather than by iterating the tables they are
        // stored in -- which is what keeps the ids in the file the ids the
        // sequence handed out.

        std::vector<int> arbitrary, trapezoids;
        for (int id = 1; id <= reading.num_gradients(); ++id)
        {
            if (!kept(used.grad, id))
                continue;
            if (reading.grad_kind(id) == GradKind::Arbitrary)
                arbitrary.push_back(id);
            else
                trapezoids.push_back(id);
        }

        if (!arbitrary.empty())
        {
            out.append("# Format of arbitrary gradients:\n");
            out.append("#   time_shape_id of 0 means default timing (stepping with grad_raster "
                       "starting at 1/2 of grad_raster)\n");
            out.append("# id amplitude first last amp_shape_id time_shape_id delay\n");
            out.append("# ..      Hz/m  Hz/m Hz/m        ..         ..          us\n");
            out.append("[GRADIENTS]\n");
            for (int id : arbitrary)
            {
                const double* d = reading.arb_library().row(reading.grad_row(id));
                appendf(
                    out,
                    "%.0f %12g %12g %12g %.0f %.0f %.0f\n",
                    renumbered(used.grad, id),
                    d[0],
                    d[1],
                    d[2],
                    renumbered(used.shape, d[3]),
                    renumbered(used.shape, d[4]),
                    std::rint(d[5] * 1e6));
            }
            out.push_back('\n');
        }

        if (!trapezoids.empty())
        {
            out.append("# Format of trapezoid gradients:\n");
            out.append("# id amplitude rise flat fall delay\n");
            out.append("# ..      Hz/m   us   us   us    us\n");
            out.append("[TRAP]\n");
            for (int id : trapezoids)
            {
                const double* d = reading.trap_library().row(reading.grad_row(id));
                appendf(
                    out,
                    "%2.0f %12g %3.0f %4.0f %3.0f %3.0f\n",
                    renumbered(used.grad, id),
                    d[0],
                    1e6 * d[1],
                    1e6 * d[2],
                    1e6 * d[3],
                    1e6 * d[4]);
            }
            out.push_back('\n');
        }

        /* -- ADC ------------------------------------------------------- */

        if (keeps_any(used.adc, reading.adc_library().size()))
        {
            out.append("# Format of ADC events:\n");
            out.append("# id num dwell delay freqPPM phasePPM freq phase phase_id\n");
            out.append("# ..  ..    ns    us     ppm  rad/MHz   Hz   rad       ..\n");
            out.append("[ADC]\n");
            for (int id = 1; id <= reading.adc_library().size(); ++id)
            {
                if (!kept(used.adc, id))
                    continue;
                const double* d = reading.adc_library().row(id);
                appendf(
                    out,
                    "%.0f %.0f %.0f %.0f %g %g %g %g %.0f\n",
                    renumbered(used.adc, id),
                    d[0],
                    1e9 * d[1],
                    1e6 * d[2],
                    d[3],
                    d[4],
                    d[5],
                    d[6],
                    renumbered(used.shape, d[7]));
            }
            out.push_back('\n');
        }

        /* -- extension chains ------------------------------------------ */

        if (keeps_any(used.chain, reading.extensions_library().size()))
        {
            out.append("# Format of extension lists:\n");
            out.append("# id type ref next_id\n");
            out.append("# next_id of 0 terminates the list\n");
            out.append("# Extension list is followed by extension specifications\n");
            out.append("[EXTENSIONS]\n");
            const IntTable& chains = reading.extensions_library();
            const auto declared = specifications(reading);
            const std::vector<int32_t> as_stored;
            std::vector<int> written;
            written.reserve(static_cast<size_t>(chains.size()));
            for (int id = 1; id <= chains.size(); ++id)
            {
                if (kept(used.chain, id))
                    written.push_back(id);
            }
            render_rows(
                out,
                written.size(),
                4 * 13 + 1,
                [&](char* cursor, size_t i)
                {
                    const int id = written[i];
                    const int32_t* row = chains.row(id);
                    // A type this writer does not number keeps its references.
                    const std::vector<int32_t>* refs = &as_stored;
                    for (const auto& kind : declared)
                    {
                        if (kind.first == row[0])
                            refs = &(used.*kind.second);
                    }
                    const long fields[1 + EXTENSION_WIDTH] = {
                        static_cast<long>(renumbered(used.chain, id)),
                        row[0],
                        static_cast<long>(renumbered(*refs, row[1])),
                        static_cast<long>(renumbered(used.chain, row[2]))};
                    cursor = put_int_field(cursor, fields[0], 1);
                    for (int column = 1; column <= EXTENSION_WIDTH; ++column)
                    {
                        *cursor++ = ' ';
                        cursor = put_int_field(cursor, fields[column], 1);
                    }
                    *cursor++ = '\n';
                    return cursor;
                });
            out.push_back('\n');
        }

        /* -- extension specifications ---------------------------------- */

        if (keeps_any(used.trigger, reading.trigger_library().size()))
        {
            out.append("# Extension specification for digital output and input triggers:\n");
            out.append("# id type channel delay (us) duration (us)\n");
            appendf(out, "extension TRIGGERS %d\n", seq.extension_type_id("TRIGGERS"));
            for (int id = 1; id <= reading.trigger_library().size(); ++id)
            {
                if (!kept(used.trigger, id))
                    continue;
                const double* d = reading.trigger_library().row(id);
                appendf(
                    out,
                    "%.0f %.0f %.0f %.0f %.0f\n",
                    renumbered(used.trigger, id),
                    d[0],
                    d[1],
                    1e6 * d[2],
                    1e6 * d[3]);
            }
            out.push_back('\n');
        }

        {
            struct LabelSection
            {
                const char* name;
                const IntTable* library;
                const std::vector<int32_t>* used;
            };
            const LabelSection label_sections[2] = {
                {"LABELSET", &reading.label_set_library(), &used.label_set},
                {"LABELINC", &reading.label_inc_library(), &used.label_inc},
            };
            for (const LabelSection& section : label_sections)
            {
                if (!keeps_any(*section.used, section.library->size()))
                    continue;
                const bool increment = section.name[5] == 'I';
                out.append(
                    increment ? "# Extension specification for increasing labels:\n"
                              : "# Extension specification for setting labels:\n");
                out.append(increment ? "# id inc labelstring\n" : "# id set labelstring\n");
                appendf(
                    out,
                    "extension %s %d\n",
                    section.name,
                    seq.extension_type_id(section.name));
                for (int id = 1; id <= section.library->size(); ++id)
                {
                    if (!kept(*section.used, id))
                        continue;
                    const int32_t* row = section.library->row(id);
                    // The name is what the file carries; the number was only
                    // ever how this sequence indexed it.
                    const std::string& name = reading.label_name(row[1]);
                    appendf(
                        out,
                        "%.0f %.0f %s\n",
                        renumbered(*section.used, id),
                        static_cast<double>(row[0]),
                        name.empty() ? "UNKNOWN" : name.c_str());
                }
                out.push_back('\n');
            }
        }

        if (keeps_any(used.shim, reading.rf_shim_library().size()))
        {
            out.append("# Extension specification for RF shimming:\n");
            out.append("# id num_chan magn_c1 phase_c1 magn_c2 phase_c2 ...\n");
            appendf(out, "extension RF_SHIMS %d\n", seq.extension_type_id("RF_SHIMS"));
            for (int id = 1; id <= reading.rf_shim_library().size(); ++id)
            {
                if (!kept(used.shim, id))
                    continue;
                const int length = reading.rf_shim_library().length(id);
                const double* values = reading.rf_shim_library().row(id);
                appendf(
                    out, "%d %d", static_cast<int>(renumbered(used.shim, id)), length / 2);
                for (int i = 0; i < length; ++i)
                    appendf(out, " %g", values[i]);
                out.push_back('\n');
            }
            out.push_back('\n');
        }

        if (keeps_any(used.rotation, reading.rotation_library().size()))
        {
            out.append("# Extension specification for rotation events:\n");
            out.append("# id RotQuat0 RotQuatX RotQuatY RotQuatZ\n");
            appendf(out, "extension ROTATIONS %d\n", seq.extension_type_id("ROTATIONS"));
            for (int id = 1; id <= reading.rotation_library().size(); ++id)
            {
                if (!kept(used.rotation, id))
                    continue;
                const double* d = reading.rotation_library().row(id);
                // Two spaces after the id: the reference writer prints the
                // id with a trailing space and then each component with a
                // leading one, and this section is compared against files it
                // wrote.
                appendf(
                    out,
                    "%.0f  %g %g %g %g\n",
                    renumbered(used.rotation, id),
                    d[0],
                    d[1],
                    d[2],
                    d[3]);
            }
            out.push_back('\n');
        }

        if (keeps_any(used.soft_delay, static_cast<int>(reading.soft_delay_library().size())))
        {
            out.append("# Extension specification for soft delays:\n");
            out.append("# id num offset factor hint\n");
            out.append("# ..  ..     us     ..   ..\n");
            appendf(out, "extension DELAYS %d\n", seq.extension_type_id("DELAYS"));
            int id = 0;
            for (const SoftDelay& row : reading.soft_delay_library())
            {
                ++id;
                if (!kept(used.soft_delay, id))
                    continue;
                appendf(
                    out,
                    "%.0f %.0f %g %g %s\n",
                    renumbered(used.soft_delay, id),
                    static_cast<double>(row.num),
                    row.offset * 1e6,
                    row.factor,
                    row.hint.c_str());
            }
            out.push_back('\n');
        }

        /* -- shapes ---------------------------------------------------- */

        if (keeps_any(used.shape, reading.shape_library().size()))
        {
            out.append("# Sequence Shapes\n");
            out.append("[SHAPES]\n\n");
            for (int id = 1; id <= reading.shape_library().size(); ++id)
            {
                if (!kept(used.shape, id))
                    continue;
                appendf(out, "shape_id %.0f\n", renumbered(used.shape, id));
                appendf(
                    out,
                    "num_samples %.0f\n",
                    static_cast<double>(reading.shape_library().num_uncompressed(id)));
                const int count = reading.shape_library().num_compressed(id);
                const double* samples = reading.shape_library().samples(id);
                for (int i = 0; i < count; ++i)
                    appendf(out, "%.9g\n", samples[i]);
                out.push_back('\n');
            }
        }

        /* -- signature -------------------------------------------------- */

        if (create_signature)
        {
            const std::string hex = md5_hex(out.data(), out.size());

            // Wrapped, and misspelled, exactly as the reference writer wraps
            // and misspells it: this block is inside no digest but it is
            // compared against files that toolbox wrote.
            out.append("\n[SIGNATURE]\n");
            out.append("# This is the hash of the Pulseq file, calculated right before the "
                       "[SIGNATURE]\n");
            out.append("# section was added. It can be reproduced/verified with md5sum if the "
                       "file\n");
            out.append("# trimmed to the position right above [SIGNATURE]. The new line "
                       "character\n");
            out.append("# preceding [SIGNATURE] BELONGS to the signature (and needs to be "
                       "sripped away\n");
            out.append("# for recalculating/verification)\n");
            out.append("Type md5\n");
            appendf(out, "Hash %s\n", hex.c_str());
        }

        return out;
    }


    /* ================================================================== */
    /*  Pulseq 1.4.1                                                      */
    /* ================================================================== */

    std::string write_text(Sequence& seq, bool create_signature)
    {
        return write_text_of(seq, create_signature, nullptr);
    }

    std::string write_text(
        Sequence& seq, bool create_signature, const std::vector<int32_t>& rows)
    {
        const int n_blocks = seq.num_blocks();
        for (const int32_t row : rows)
        {
            if (row < 1 || row > n_blocks)
                throw std::out_of_range(
                    "block " + std::to_string(row) + " is not one of the sequence's " +
                    std::to_string(n_blocks) + " blocks");
        }
        return write_text_of(seq, create_signature, &rows);
    }

    std::string write_text_v141(Sequence& seq, bool create_signature, double gamma, double field)
    {
        if (!seq.rotation_library().empty() || !seq.rf_shim_library().empty())
            throw std::runtime_error(
                "write_text_v141(): this sequence rotates or shims, and 1.4.1 has no way to "
                "say so. Dropping either would move every gradient it applies to, so the "
                "file is refused rather than written as a sequence that is not this one");

        seq.compress_shapes();
        seq.publish_rasters();

        const Sequence& reading = seq;
        const int n_blocks = seq.num_blocks();
        const std::vector<long> ticks = duration_ticks(seq);

        // The one thing 1.4 cannot express that can be folded rather than
        // dropped: an offset in parts per million is an offset in hertz once
        // the field and the gyromagnetic ratio are known.
        const double ppm_to_hz = 1e-6 * gamma * field;

        std::string out;
        out.reserve(static_cast<size_t>(n_blocks) * 36 + 4096);
        out.append("# Pulseq sequence file\n# Created by PyPulseq\n\n");

        out.append("[VERSION]\n");
        out.append("major 1\n");
        out.append("minor 4\n");
        out.append("revision 1\n");
        out.push_back('\n');

        if (!reading.definitions().empty())
        {
            out.append("[DEFINITIONS]\n");
            for (const auto& entry : reading.definitions())
            {
                out.append(entry.first);
                out.push_back(' ');
                write_definition_value(out, entry.second);
                out.push_back('\n');
            }
            out.push_back('\n');
        }

        out.append("# Format of blocks:\n");
        out.append("# NUM DUR RF  GX  GY  GZ  ADC  EXT\n");
        out.append("[BLOCKS]\n");
        if (n_blocks > 0)
        {
            const int number_width = decimal_width(n_blocks);
            const int widths[8] = {number_width, 3, 3, 3, 3, 3, 2, 2};
            const int32_t* events = reading.block_events();
            const long* tick = ticks.data();
            size_t row_bound = 1;
            for (int column = 0; column < 8; ++column)
                row_bound += 1 + std::max(widths[column], 12);
            render_rows(
                out,
                static_cast<size_t>(n_blocks),
                row_bound,
                [&](char* cursor, size_t i)
                {
                    const int32_t* row = events + i * BLOCK_WIDTH;
                    cursor = put_int_field(cursor, static_cast<long>(i) + 1, widths[0]);
                    *cursor++ = ' ';
                    cursor = put_int_field(cursor, tick[i], widths[1]);
                    for (int column = 0; column < BLOCK_FILE_COLUMNS; ++column)
                    {
                        *cursor++ = ' ';
                        cursor = put_int_field(cursor, row[column], widths[column + 2]);
                    }
                    *cursor++ = '\n';
                    return cursor;
                });
        }
        out.push_back('\n');

        if (!reading.rf_library().empty())
        {
            out.append("# Format of RF events:\n");
            out.append("# id amplitude mag_id phase_id time_shape_id delay freq phase\n");
            out.append("# ..        Hz   ....     ....          ....    us   Hz   rad\n");
            out.append("[RF]\n");
            const double raster = reading.rf_raster_time();
            for (int id = 1; id <= reading.rf_library().size(); ++id)
            {
                const double* d = reading.rf_library().row(id);
                appendf(
                    out,
                    "%.0f %12g %.0f %.0f %.0f %g %g %g\n",
                    static_cast<double>(id),
                    d[0],
                    d[1],
                    d[2],
                    d[3],
                    std::rint(d[5] / raster) * raster * 1e6,
                    d[8] + d[6] * ppm_to_hz,
                    d[9] + d[7] * ppm_to_hz);
            }
            out.push_back('\n');
        }

        std::vector<int> arbitrary, trapezoids;
        for (int id = 1; id <= reading.num_gradients(); ++id)
            (reading.grad_kind(id) == GradKind::Arbitrary ? arbitrary : trapezoids).push_back(id);

        if (!arbitrary.empty())
        {
            out.append("# Format of arbitrary gradients:\n");
            out.append("#   time_shape_id of 0 means default timing (stepping with grad_raster "
                       "starting at 1/2 of grad_raster)\n");
            out.append("# id amplitude amp_shape_id time_shape_id delay\n");
            out.append("# ..      Hz/m       ..         ..          us\n");
            out.append("[GRADIENTS]\n");
            for (const int id : arbitrary)
            {
                const double* d = reading.arb_library().row(reading.grad_row(id));
                appendf(
                    out,
                    "%.0f %12g %.0f %.0f %.0f\n",
                    static_cast<double>(id),
                    d[0],
                    d[3],
                    d[4],
                    std::rint(d[5] * 1e6));
            }
            out.push_back('\n');
        }

        if (!trapezoids.empty())
        {
            out.append("# Format of trapezoid gradients:\n");
            out.append("# id amplitude rise flat fall delay\n");
            out.append("# ..      Hz/m   us   us   us    us\n");
            out.append("[TRAP]\n");
            for (const int id : trapezoids)
            {
                const double* d = reading.trap_library().row(reading.grad_row(id));
                appendf(
                    out,
                    "%2.0f %12g %3.0f %4.0f %3.0f %3.0f\n",
                    static_cast<double>(id),
                    d[0],
                    1e6 * d[1],
                    1e6 * d[2],
                    1e6 * d[3],
                    1e6 * d[4]);
            }
            out.push_back('\n');
        }

        if (!reading.adc_library().empty())
        {
            out.append("# Format of ADC events:\n");
            out.append("# id num dwell delay freq phase\n");
            out.append("# ..  ..    ns    us   Hz   rad\n");
            out.append("[ADC]\n");
            for (int id = 1; id <= reading.adc_library().size(); ++id)
            {
                const double* d = reading.adc_library().row(id);
                appendf(
                    out,
                    "%.0f %.0f %.0f %.0f %g %g\n",
                    static_cast<double>(id),
                    d[0],
                    1e9 * d[1],
                    1e6 * d[2],
                    d[5] + d[3] * ppm_to_hz,
                    d[6] + d[4] * ppm_to_hz);
            }
            out.push_back('\n');
        }

        if (!reading.extensions_library().empty())
        {
            out.append("# Format of extension lists:\n");
            out.append("# id type ref next_id\n");
            out.append("# next_id of 0 terminates the list\n");
            out.append("# Extension list is followed by extension specifications\n");
            out.append("[EXTENSIONS]\n");
            const IntTable& chains = reading.extensions_library();
            for (int id = 1; id <= chains.size(); ++id)
            {
                const int32_t* row = chains.row(id);
                appendf(
                    out,
                    "%.0f %.0f %.0f %.0f\n",
                    static_cast<double>(id),
                    static_cast<double>(row[0]),
                    static_cast<double>(row[1]),
                    static_cast<double>(row[2]));
            }
            out.push_back('\n');
        }

        if (!reading.trigger_library().empty())
        {
            out.append("# Extension specification for digital output and input triggers:\n");
            out.append("# id type channel delay (us) duration (us)\n");
            appendf(out, "extension TRIGGERS %d\n", seq.extension_type_id("TRIGGERS"));
            for (int id = 1; id <= reading.trigger_library().size(); ++id)
            {
                const double* d = reading.trigger_library().row(id);
                appendf(
                    out,
                    "%.0f %.0f %.0f %.0f %.0f\n",
                    static_cast<double>(id),
                    d[0],
                    d[1],
                    std::rint(1e6 * d[2]),
                    std::rint(1e6 * d[3]));
            }
            out.push_back('\n');
        }

        {
            // 1.4.1 heads both label sections the same way, where 1.5 gives
            // the increasing one its own wording.
            const std::pair<const char*, const IntTable*> sections[2] = {
                {"LABELSET", &reading.label_set_library()},
                {"LABELINC", &reading.label_inc_library()},
            };
            for (const auto& section : sections)
            {
                if (section.second->empty())
                    continue;
                out.append("# Extension specification for setting labels:\n");
                out.append("# id set labelstring\n");
                appendf(
                    out, "extension %s %d\n", section.first,
                    seq.extension_type_id(section.first));
                for (int id = 1; id <= section.second->size(); ++id)
                {
                    const int32_t* row = section.second->row(id);
                    const std::string& name = reading.label_name(row[1]);
                    appendf(
                        out,
                        "%.0f %.0f %s\n",
                        static_cast<double>(id),
                        static_cast<double>(row[0]),
                        name.empty() ? "UNKNOWN" : name.c_str());
                }
                out.push_back('\n');
            }
        }

        if (!reading.shape_library().empty())
        {
            out.append("# Sequence Shapes\n");
            out.append("[SHAPES]\n\n");
            for (int id = 1; id <= reading.shape_library().size(); ++id)
            {
                appendf(out, "shape_id %.0f\n", static_cast<double>(id));
                appendf(
                    out,
                    "num_samples %.0f\n",
                    static_cast<double>(reading.shape_library().num_uncompressed(id)));
                const int count = reading.shape_library().num_compressed(id);
                const double* samples = reading.shape_library().samples(id);
                for (int i = 0; i < count; ++i)
                    appendf(out, "%.9g\n", samples[i]);
                out.push_back('\n');
            }
        }

        if (create_signature)
        {
            const std::string hex = md5_hex(out.data(), out.size());
            out.append("\n[SIGNATURE]\n");
            out.append("# This is the hash of the Pulseq file, calculated right before the "
                       "[SIGNATURE]\n");
            out.append("# section was added. It can be reproduced/verified with md5sum if the "
                       "file\n");
            out.append("# trimmed to the position right above [SIGNATURE]. The new line "
                       "character\n");
            out.append("# preceding [SIGNATURE] BELONGS to the signature (and needs to be "
                       "sripped away\n");
            out.append("# for recalculating/verification)\n");
            out.append("Type md5\n");
            appendf(out, "Hash %s\n", hex.c_str());
        }

        return out;
    }

} // namespace pulseq
