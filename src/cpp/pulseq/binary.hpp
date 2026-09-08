/**
 * @file binary.hpp
 * @brief The Pulseq binary sequence file, as MATLAB Pulseq's writeBinary
 *        defines it.
 *
 * The same sequence as the text form and the same units; only the container
 * changes. What it buys is parsing: the scanner reads a block table of fixed
 * records instead of formatting and reparsing a decimal number per column,
 * and times cross as integer picoseconds rather than as nine significant
 * digits.
 *
 * A binary file always declares at least revision 1: the format arrived with
 * Pulseq 1.5.1, so a file claiming 1.5.0 claims a revision that had no way to
 * write it.
 *
 * Nothing here writes a signature section. One toolbox does, and a file
 * carrying one is read and its signature skipped.
 */

#ifndef PULSEQ_CXX_BINARY_HPP
#define PULSEQ_CXX_BINARY_HPP

#include <cstdint>
#include <string>

namespace pulseq
{

    class Sequence;

    /** The eight bytes every binary sequence file opens with. */
    constexpr unsigned char BINARY_MAGIC[8] = {0x01, 'p', 'u', 'l', 's', 'e', 'q', 0x02};

    /**
     * Section codes.
     *
     * Each section opens with one of these as an int64 whose high word is all
     * ones, so a code cannot be mistaken for a count.
     */
    enum Section : uint64_t
    {
        SEC_DEFINITIONS = 1,
        SEC_BLOCKS = 2,
        SEC_RF = 3,
        SEC_GRADIENTS = 4,
        SEC_TRAPEZOIDS = 5,
        SEC_ADC = 6,
        SEC_DELAYS = 7, /**< Pre-1.4 delays; never written, skipped on the way in. */
        SEC_SHAPES = 8,
        SEC_EXTENSIONS = 9,
        SEC_TRIGGERS = 10,
        SEC_LABELSET = 11,
        SEC_LABELINC = 12,
        SEC_SOFTDELAYS = 13,
        SEC_RFSHIMS = 14,
        SEC_ROTATIONS = 15,
        SEC_LABELNAMES = 16,
        /** Written by MATLAB Pulseq; read and skipped, never written here. */
        SEC_SIGNATURE = 0x00FFFFFF,
    };

    /** The high word that marks an int64 as a section code. */
    constexpr uint64_t SECTION_PREFIX = 0xFFFFFFFFULL << 32;

    /** Whether @p contents opens with the binary magic. */
    bool is_binary(const std::string& contents);

    /**
     * Serialize as a Pulseq binary sequence file.
     *
     * Written little-endian whatever the host is, so one machine's file is
     * another's.
     */
    std::string write_binary(Sequence& seq);

} // namespace pulseq

#endif /* PULSEQ_CXX_BINARY_HPP */
