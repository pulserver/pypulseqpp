/**
 * @file write.hpp
 * @brief Serialise Pulseq text, recording custom label definitions when needed.
 */

#ifndef PULSEQ_CXX_WRITE_HPP
#define PULSEQ_CXX_WRITE_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace pulseq
{

    class Sequence;

    /**
     * Record custom label names in assignment order in the CustomLabels definition.
     * Binary IDs beyond the builtin table resolve by position in this list.
     */
    void declare_custom_labels(Sequence& seq);

    /** The revision of the format this package writes: Pulseq 1.5.1. */
    constexpr int WRITTEN_REVISION = 1;

    /**
     * The revision a file written here declares, which is WRITTEN_REVISION.
     *
     * It does not depend on the sequence.  A writer says which revision of
     * the format it produced, not which subset of it a particular sequence
     * used, so a file that came in as 1.5.0 goes out as 1.5.1 -- which is
     * what the reference toolbox does and why its own round-trip test
     * compares against a canonical rewrite rather than against the source.
     */
    int required_revision(const Sequence& seq);

    /**
     * Serialise Pulseq 1.4.1 text.
     *
     * Fold ppm offsets into absolute offsets using @p gamma (Hz/T) and @p field
     * (tesla). Omit RF centre/use, gradient endpoints and soft-delay extensions;
     * soft-delay omission emits a warning.
     * @throws std::runtime_error for rotation or RF-shim extensions.
     */
    std::string write_text_v141(
        Sequence& seq, bool create_signature, double gamma = 42576000.0, double field = 1.5);

    /**
     * Serialize as a Pulseq `.seq` text file.
     *
     * @param create_signature  Append the `[SIGNATURE]` section: an MD5 of
     *                          everything above it, including the newline that
     *                          precedes the header.
     * @throws std::runtime_error if a block duration is not a whole number of
     *         block-duration rasters.
     */
    std::string write_text(Sequence& seq, bool create_signature);

    /**
     * Serialize the blocks @p rows names, in that order, as a `.seq` text file.
     *
     * @p rows holds 1-based block indices, and the file numbers the blocks it
     * holds from 1. Every library is written whole whichever blocks refer to
     * it, which the format allows.
     *
     * @throws std::out_of_range if a row is not one of the sequence's blocks.
     */
    std::string write_text(
        Sequence& seq, bool create_signature, const std::vector<int32_t>& rows);

} // namespace pulseq

#endif /* PULSEQ_CXX_WRITE_HPP */
