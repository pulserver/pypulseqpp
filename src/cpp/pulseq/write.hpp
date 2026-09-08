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

    /**
     * Record in `[DEFINITIONS]` the label names the builtin table does not
     * carry, so a label number above that table resolves by position.
     *
     * Both writers call it: the text form writes a label's name and the
     * binary form its number, but the number is what the binary form has and
     * this is what gives it a meaning.
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
     * Serialize as a Pulseq 1.4.1 `.seq` text file.
     *
     * For a scanner whose interpreter predates 1.5. What 1.5 added is folded
     * back or dropped: the ppm frequency and phase offsets become absolute
     * hertz at @p gamma and @p field, which is the only place those two are
     * needed and why they are arguments rather than read from the sequence;
     * an RF pulse's centre and use go, and so do an arbitrary gradient's
     * first and last sample, because 1.4 has no column for any of them.
     *
     * A soft delay is dropped, which the reference toolbox also does and
     * warns about -- the file is then only partly what the sequence said.
     *
     * @throws std::runtime_error if the sequence rotates or shims, neither of
     *         which 1.4 can express at all. Dropping a rotation would move
     *         every gradient it turns, so the file is refused rather than
     *         written wrong.
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

} // namespace pulseq

#endif /* PULSEQ_CXX_WRITE_HPP */
