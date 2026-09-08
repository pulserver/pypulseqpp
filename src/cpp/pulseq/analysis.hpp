/**
 * @file analysis.hpp
 * @brief What a sequence is, read off the libraries and the trajectory.
 *
 * A report on a sequence answers two questions that neither the block table
 * nor the waveforms answer directly: how hard each pulse tips the
 * magnetisation, and what the encoding covers.
 *
 * ### The flip angle is a property of the pulse, not of the block
 *
 * How far a pulse tips is the integral of its envelope, so it belongs to the
 * RF library row rather than to any block that plays it: a pulse played ten
 * thousand times is integrated once. What comes back is one angle per row,
 * in degrees.
 *
 * ### What the encoding covers is a property of the samples
 *
 * Where the ADC samples fall in k-space says how many distinct positions the
 * scan visits, how often each is revisited -- slices, averages, contrasts --
 * and whether the positions form a grid. All three fall out of binning the
 * sampled trajectory onto a lattice fine enough to separate neighbouring
 * positions and coarse enough to merge one position reached twice.
 */

#ifndef PULSEQ_ANALYSIS_HPP
#define PULSEQ_ANALYSIS_HPP

#include <vector>

#include "pulseq/sequence.hpp"

namespace pulseq
{

    /**
     * Every pulse's flip angle in degrees, one per RF library row, in id
     * order.
     *
     * The integral of the complex envelope over the pulse: an amplitude in
     * hertz over a time in seconds is a number of turns, and a turn is 360
     * degrees. A pulse carrying no time shape is sampled at the centre of
     * each RF raster interval, which is where the format puts it.
     */
    std::vector<double> flip_angles(const Sequence& sequence);

    /** What the sampled trajectory covers, and how often it goes back. */
    struct KspaceCoverage
    {
        /** Distinct positions along each axis, in the axis order given. */
        std::vector<double> unique_positions;
        /** How many times a position is visited, over the positions. */
        double repeats_min = 0.0;
        double repeats_max = 0.0;
        double repeats_median = 0.0;
        /** Whether the positions visited once fill the grid they span. */
        bool is_cartesian = false;
    };

    /**
     * Bin @p samples onto a lattice of @p threshold and report what it covers.
     *
     * @param samples  @p axes rows of @p count positions, row-major.
     * @param axes     How many axes the trajectory moves along.
     * @param count    How many samples were taken.
     * @param threshold  How far apart two positions must be to be two.
     *
     * A position along one axis is merged with a neighbouring lattice cell,
     * so a coordinate that lands either side of a cell boundary counts once.
     */
    KspaceCoverage kspace_coverage(
        const double* samples,
        int axes,
        int count,
        double threshold);

} // namespace pulseq

#endif /* PULSEQ_ANALYSIS_HPP */
