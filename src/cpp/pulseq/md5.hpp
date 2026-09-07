/**
 * @file md5.hpp
 * @brief MD5 digest, for the `[SIGNATURE]` section of a written sequence.
 */

#ifndef PULSEQ_CXX_MD5_HPP
#define PULSEQ_CXX_MD5_HPP

#include <cstddef>
#include <string>

namespace pulseq
{

    /**
     * @brief Lowercase hex MD5 digest of a buffer, per RFC 1321.
     *
     * @param data    Start of the buffer.
     * @param length  Bytes to digest.
     * @return The 32-character hex digest.
     */
    std::string md5_hex(const void* data, std::size_t length);

} // namespace pulseq

#endif // PULSEQ_CXX_MD5_HPP
