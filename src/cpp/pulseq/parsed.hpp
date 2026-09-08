/**
 * @file parsed.hpp
 * @brief What a sequence file said, before any of it is registered.
 *
 * Both readers fill this and hand it to build_sequence(), so the two forms of
 * the file differ only in how they are taken apart. The rules that turn a
 * file into a sequence -- what order the libraries are registered in, that
 * the ids handed out are the ids the file used, that blocks come last because
 * a block is forked as it is added -- are written once, here and in the
 * builder, rather than twice.
 *
 * Every library is keyed by the id the file gave it rather than collected in
 * the order the rows appear, so a file whose rows are out of order still
 * registers them in the numbering it declared.
 */

#ifndef PULSEQ_CXX_PARSED_HPP
#define PULSEQ_CXX_PARSED_HPP

#include "pulseq/sequence.hpp"

#include <array>
#include <map>
#include <string>
#include <vector>

namespace pulseq
{

    struct ParsedBlock
    {
        long ticks = 0;
        std::array<int32_t, BLOCK_WIDTH> events{};
    };

    /** A label row: the file may name its label or leave it as a number. */
    struct ParsedLabel
    {
        int32_t value = 0;
        int32_t label_id = 0;
        std::string name;
    };

    struct Parsed
    {
        int major = 0;
        int minor = 0;
        int revision = 0;

        /** major.minor.revision as one number, the way Pulseq compares them. */
        long combined() const
        {
            return 1000000L * major + 1000L * minor + revision;
        }

        std::map<std::string, Definition> definitions;
        std::map<int, ParsedBlock> blocks;
        std::map<int, std::array<double, RF_WIDTH>> rf;
        std::map<int, char> rf_use;
        std::map<int, std::array<double, ARB_WIDTH>> arbitrary;
        std::map<int, std::array<double, TRAP_WIDTH>> trapezoid;
        std::map<int, std::array<double, ADC_WIDTH>> adc;
        std::map<int, std::array<int32_t, EXTENSION_WIDTH>> chains;
        std::map<std::string, int> extension_types;
        std::map<int, std::array<double, TRIGGER_WIDTH>> triggers;
        std::map<int, std::array<double, ROTATION_WIDTH>> rotations;
        std::map<int, ParsedLabel> label_set;
        std::map<int, ParsedLabel> label_inc;
        std::map<int, std::vector<double>> shims;
        std::map<int, SoftDelay> soft_delays;
        std::map<int, std::pair<int, std::vector<double>>> shapes;

        /** Text only: the binary form carries no signature. */
        bool has_signature = false;
        std::string signature;
        /** Where the `[SIGNATURE]` header starts, so the digest knows its end. */
        size_t signature_offset = 0;
    };

    /** Register everything the file said, in the order that makes it a sequence. */
    Sequence build_sequence(const Parsed& parsed);

    /** Parse the binary form.  See binary.hpp for the layout. */
    Parsed parse_binary(const std::string& contents);

} // namespace pulseq

#endif /* PULSEQ_CXX_PARSED_HPP */
