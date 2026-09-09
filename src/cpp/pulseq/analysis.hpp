/**
 * @file analysis.hpp
 * @brief What a sequence is, read off the libraries and the trajectory.
 *
 * A report on a sequence answers two questions that neither the block table
 * nor the waveforms answer directly: how hard each pulse tips the
 * magnetisation, and what the encoding covers.
 *
 * ### The flip angle is a property of the envelope, not of the playout
 *
 * How far a pulse tips is the integral of its envelope times the amplitude it
 * is played at. The envelope belongs to the RF *definition* and the amplitude
 * to the playout, so an inversion train sweeping one pulse over a thousand
 * flip angles is one integral and a thousand multiplies -- and playing each of
 * those a hundred times costs nothing further, since a block names a library
 * row rather than carrying one.
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

#include <string>
#include <utility>
#include <vector>

#include "pulseq/sequence.hpp"

namespace pulseq
{

    /**
     * Every distinct flip angle the sequence uses, in degrees, ascending.
     *
     * An amplitude in hertz over a time in seconds is a number of turns, and
     * a turn is 360 degrees. A pulse carrying no time shape is sampled at the
     * centre of each RF raster interval, which is where the format puts it.
     */
    std::vector<double> flip_angles(const Sequence& sequence);

    /**
     * What a label is set to, block by block.
     *
     * A label is running state: set or incremented where a block says so, and
     * in force until another block says otherwise. So reading one back is a
     * walk over the blocks in order, applying what each carries -- which is
     * a pass over the extension chains rather than over decoded blocks, and
     * costs nothing for the blocks that carry none, which is most of them.
     */
    struct LabelEvolution
    {
        /** The labels the sequence touches, in the order they first appear. */
        std::vector<std::string> names;
        /** Per label, its value at each point recorded.  See `evaluate_labels`. */
        std::vector<std::vector<int32_t>> values;
    };

    /** Where a label evolution is recorded. */
    enum class LabelEvolutionAt
    {
        End,     /**< Once, at the end: what the labels finish at. */
        Blocks,  /**< Every block. */
        Adc,     /**< Every block that acquires. */
        Label,   /**< Every block that sets or increments one. */
    };

    /**
     * Follow every label the sequence uses.
     *
     * @param seq          The sequence to walk.
     * @param at           Where to record a value.
     * @param first_block  First block to walk, 1-based.
     * @param last_block   Last block, or 0 for the end of the sequence.
     * @param start        What each label is before the walk begins, by name.
     *                     A label named here is reported whether or not the
     *                     blocks touch it.
     */
    LabelEvolution evaluate_labels(
        const Sequence& seq,
        LabelEvolutionAt at,
        int first_block,
        int last_block,
        const std::vector<std::pair<std::string, int32_t>>& start);

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
