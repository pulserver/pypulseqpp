/**
 * @file shape.hpp
 * @brief Pulseq shape compression: run-length encoding of the derivative.
 *
 * Compression is deferred until the library is prepared for writing.
 */

#ifndef PULSEQ_CXX_SHAPE_HPP
#define PULSEQ_CXX_SHAPE_HPP

#include <vector>

namespace pulseq
{

    /**
     * @p count samples, run-length encoded on their derivative.
     *
     * Returns the compressed form, or a copy of the samples when compressing
     * them would not make them shorter -- which is Pulseq's own rule, and what
     * decides whether a `[SHAPES]` entry holds a code or a waveform.  Shapes of
     * four samples or fewer are never compressed.
     */
    std::vector<double> compress_shape(const double* samples, int count);

    /**
     * The inverse: @p count encoded samples back to @p num_uncompressed of
     * them.
     *
     * Run-length decoding of the `[SHAPES]` section's derivative form.  When
     * the two counts are equal the entry was never compressed -- that is
     * compress_shape's "keep the original if it is no longer" rule -- and the
     * samples are copied through unchanged.
     *
     * @p force decodes even when the counts are equal.  Before Pulseq 1.4
     * that coincidence could happen to a shape that really was encoded, so a
     * file from that era is decoded this way and re-encoded, which is what
     * makes "equal counts mean uncompressed" true of it afterwards.
     *
     * Throws std::runtime_error on a malformed run length, which is the only
     * way the encoding can be inconsistent with itself.
     */
    std::vector<double> decompress_shape(
        const double* samples, int count, int num_uncompressed, bool force = false);

} // namespace pulseq

#endif /* PULSEQ_CXX_SHAPE_HPP */
