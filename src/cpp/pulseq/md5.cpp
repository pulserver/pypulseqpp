/**
 * @file md5.cpp
 * @brief RFC 1321 MD5, one-shot over a buffer already in memory.
 *
 * A sequence is signed once, over the whole file body, so there is no
 * streaming interface here: the caller has the bytes and wants the digest.
 */

#include "pulseq/md5.hpp"

#include <cstdint>
#include <cstring>

namespace pulseq
{

    namespace
    {

        /** K[i] = floor(2^32 * |sin(i + 1)|), with i in radians. */
        const std::uint32_t K[64] = {
            0xd76aa478u, 0xe8c7b756u, 0x242070dbu, 0xc1bdceeeu, 0xf57c0fafu, 0x4787c62au,
            0xa8304613u, 0xfd469501u, 0x698098d8u, 0x8b44f7afu, 0xffff5bb1u, 0x895cd7beu,
            0x6b901122u, 0xfd987193u, 0xa679438eu, 0x49b40821u, 0xf61e2562u, 0xc040b340u,
            0x265e5a51u, 0xe9b6c7aau, 0xd62f105du, 0x02441453u, 0xd8a1e681u, 0xe7d3fbc8u,
            0x21e1cde6u, 0xc33707d6u, 0xf4d50d87u, 0x455a14edu, 0xa9e3e905u, 0xfcefa3f8u,
            0x676f02d9u, 0x8d2a4c8au, 0xfffa3942u, 0x8771f681u, 0x6d9d6122u, 0xfde5380cu,
            0xa4beea44u, 0x4bdecfa9u, 0xf6bb4b60u, 0xbebfbc70u, 0x289b7ec6u, 0xeaa127fau,
            0xd4ef3085u, 0x04881d05u, 0xd9d4d039u, 0xe6db99e5u, 0x1fa27cf8u, 0xc4ac5665u,
            0xf4292244u, 0x432aff97u, 0xab9423a7u, 0xfc93a039u, 0x655b59c3u, 0x8f0ccc92u,
            0xffeff47du, 0x85845dd1u, 0x6fa87e4fu, 0xfe2ce6e0u, 0xa3014314u, 0x4e0811a1u,
            0xf7537e82u, 0xbd3af235u, 0x2ad7d2bbu, 0xeb86d391u};

        /** Per-round left-rotation amounts. */
        const int S[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                           5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
                           4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                           6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

        inline std::uint32_t rotate_left(std::uint32_t value, int count)
        {
            return (value << count) | (value >> (32 - count));
        }

        void transform(std::uint32_t state[4], const unsigned char block[64])
        {
            std::uint32_t words[16];
            for (int i = 0; i < 16; ++i)
                words[i] = static_cast<std::uint32_t>(block[4 * i]) |
                           (static_cast<std::uint32_t>(block[4 * i + 1]) << 8) |
                           (static_cast<std::uint32_t>(block[4 * i + 2]) << 16) |
                           (static_cast<std::uint32_t>(block[4 * i + 3]) << 24);

            std::uint32_t a = state[0];
            std::uint32_t b = state[1];
            std::uint32_t c = state[2];
            std::uint32_t d = state[3];

            for (int i = 0; i < 64; ++i)
            {
                std::uint32_t mixed;
                int index;
                if (i < 16)
                {
                    mixed = (b & c) | (~b & d);
                    index = i;
                }
                else if (i < 32)
                {
                    mixed = (d & b) | (~d & c);
                    index = (5 * i + 1) % 16;
                }
                else if (i < 48)
                {
                    mixed = b ^ c ^ d;
                    index = (3 * i + 5) % 16;
                }
                else
                {
                    mixed = c ^ (b | ~d);
                    index = (7 * i) % 16;
                }

                const std::uint32_t rotated = a + mixed + K[i] + words[index];
                a = d;
                d = c;
                c = b;
                b = b + rotate_left(rotated, S[i]);
            }

            state[0] += a;
            state[1] += b;
            state[2] += c;
            state[3] += d;
        }

    } // namespace

    std::string md5_hex(const void* data, std::size_t length)
    {
        std::uint32_t state[4] = {0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u};

        const unsigned char* bytes = static_cast<const unsigned char*>(data);
        std::size_t consumed = 0;
        for (; consumed + 64 <= length; consumed += 64)
            transform(state, bytes + consumed);

        /* The tail, its 0x80 terminator, zero padding, and the bit count as a
         * little-endian 64-bit trailer: one block when the remainder leaves
         * room for the trailer, two when it does not. */
        const std::size_t remainder = length - consumed;
        const std::size_t total = (remainder < 56) ? 64 : 128;
        unsigned char tail[128];
        std::memcpy(tail, bytes + consumed, remainder);
        tail[remainder] = 0x80;
        std::memset(tail + remainder + 1, 0, total - remainder - 9);

        const std::uint64_t bits = static_cast<std::uint64_t>(length) * 8;
        for (int i = 0; i < 8; ++i)
            tail[total - 8 + i] = static_cast<unsigned char>((bits >> (8 * i)) & 0xffu);

        transform(state, tail);
        if (total == 128)
            transform(state, tail + 64);

        static const char digits[] = "0123456789abcdef";
        std::string out(32, '0');
        for (int word = 0; word < 4; ++word)
            for (int byte = 0; byte < 4; ++byte)
            {
                const unsigned value = (state[word] >> (8 * byte)) & 0xffu;
                out[2 * (4 * word + byte)] = digits[value >> 4];
                out[2 * (4 * word + byte) + 1] = digits[value & 0x0fu];
            }
        return out;
    }

} // namespace pulseq
