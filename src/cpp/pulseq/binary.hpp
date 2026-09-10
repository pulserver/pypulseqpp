/**
 * @file binary.hpp
 * @brief Pulseq 1.5.1 binary format, using little-endian records and picosecond times.
 *
 * Shape samples are float32. Optional MD5 signatures are written and can be
 * checked with binary_signature(). Custom label names are stored in the
 * CustomLabels definition, in the order of their assigned IDs.
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
        /**
         * Optional MD5 signature section; checked by binary_signature().
         */
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
     *
     * @param create_signature  Append the signature section: the type, an MD5
     *                          of everything above it, and how many bytes that
     *                          is. The digest is stored raw where the text
     *                          form writes it as hex, which is the only place
     *                          the two forms differ about it.
     */
    std::string write_binary(Sequence& seq, bool create_signature = true);

    /**
     * The signature a binary file carries, or empty strings if it carries none.
     *
     * @param contents  The whole file.
     * @param type      Filled with the digest's name, `md5`.
     * @param value     Filled with the digest, as lowercase hex.
     * @return Whether the digest is the digest of what it covers. False when
     *         there is no signature to check, which @p type says apart.
     */
    bool binary_signature(
        const std::string& contents, std::string& type, std::string& value);

} // namespace pulseq

#endif /* PULSEQ_CXX_BINARY_HPP */
