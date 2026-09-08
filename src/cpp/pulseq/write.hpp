/**
 * @file write.hpp
 * @brief Serializing a pulseq::Sequence as a Pulseq `.seq` text file.
 *
 * The writer takes the sequence by non-const reference because it records
 * `TotalDuration` in `[DEFINITIONS]` before it starts, which is what PyPulseq
 * does.
 */

#ifndef PULSEQ_CXX_WRITE_HPP
#define PULSEQ_CXX_WRITE_HPP

#include <string>

namespace pulseq
{

    class Sequence;

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
     * Serialize as a Pulseq `.seq` text file.
     *
     * @param create_signature  Append the `[SIGNATURE]` section: an MD5 of
     *                          everything above it, including the newline that
     *                          precedes the header.
     * @throws std::runtime_error if a block duration is not a whole number of
     *         block-duration rasters.
     */
    std::string write_text(Sequence& seq, bool create_signature);

} // namespace pulseq

#endif /* PULSEQ_CXX_WRITE_HPP */
