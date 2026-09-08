/**
 * @file read.hpp
 * @brief Reading a Pulseq `.seq` file back into a pulseq::Sequence.
 *
 * The reader is the writer's inverse and is tested as one: a file written
 * here, read back and written again is byte for byte the file it started as.
 *
 * Reading fills the sequence through the same registration calls a design
 * loop makes, so a sequence that came off disk carries the same definitions
 * and instances as one that was built, and neither has to be told which it
 * is. That is also why the file is parsed whole before anything is
 * registered: a block names events the file lists after it, and a block can
 * only be forked once the events it names have been.
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
