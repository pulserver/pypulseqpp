/**
 * @file read.hpp
 * @brief Read Pulseq text or binary contents into a Sequence.
 */

#ifndef PULSEQ_CXX_READ_HPP
#define PULSEQ_CXX_READ_HPP

#include <string>

namespace pulseq
{

    class Sequence;

    /**
     * Parse the contents of a `.seq` file.
     *
     * @param contents  The whole file.
     * @param verify    Check the `[SIGNATURE]` section against the bytes it
     *                  covers, and throw if they disagree. A file carrying no
     *                  signature passes either way.
     * @throws std::runtime_error on a malformed file, naming the line.
     */
    Sequence read(const std::string& contents, bool verify = false);

    /** As read(), for a file on disk. */
    Sequence read_file(const std::string& path, bool verify = false);

} // namespace pulseq

#endif /* PULSEQ_CXX_READ_HPP */
