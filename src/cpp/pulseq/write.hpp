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
     * The Pulseq revision @p seq actually needs.
     *
     * Rotations and RF shims are 1.5.1, the newest revision there is.  A
     * sequence using neither stays at whatever revision it declares, so the
     * ordinary case writes a file any interpreter reads.
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
