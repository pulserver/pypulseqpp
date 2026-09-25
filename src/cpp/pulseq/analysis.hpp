/**
 * @file analysis.hpp
 * @brief RF flip angles, label evolution and sampled k-space coverage.
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
     * The flip angle of every RF library row, in degrees: entry i is id i + 1.
     *
     * The rule `flip_angles` lists the distinct values of: the magnitude of
     * the complex envelope's integral times the amplitude, a dynamic pTx
     * pulse integrated channel by channel on its shared time base and the
     * channels summed.
     */
    std::vector<double> rf_flip_angles(const Sequence& sequence);

    /**
     * The transmit channels every RF library row holds: entry i is id i + 1,
     * counted by `rf_channels` from the row's sample times.
     */
    std::vector<int> rf_channel_counts(const Sequence& sequence);

    /**
     * Label values carried across blocks until set or incremented again.
     */
    struct LabelEvolution
    {
        /** The labels the sequence touches, in the order they first appear. */
        std::vector<std::string> names;
        /** Per label, its value at each point recorded.  See `evaluate_labels`. */
        std::vector<std::vector<int32_t>> values;
    };

    /** Points at which a label evolution is recorded. */
    enum class LabelEvolutionAt
    {
        End,     /**< Once, at the end: the labels' final values. */
        Blocks,  /**< Every block. */
        Adc,     /**< Every block that acquires. */
        Label,   /**< Every block that sets or increments one. */
    };

    /**
     * Follow every label the sequence uses.
     *
     * @param seq          The sequence to evaluate.
     * @param at           Where to record a value.
     * @param first_block  First block to evaluate, 1-based.
     * @param last_block   Last block, or 0 for the end of the sequence.
     * @param start        Initial value of each label, by name. A label named
     *                     here is reported whether or not the blocks touch it.
     */
    LabelEvolution evaluate_labels(
        const Sequence& seq,
        LabelEvolutionAt at,
        int first_block,
        int last_block,
        const std::vector<std::pair<std::string, int32_t>>& start);

    /** k-space extent of the sampled trajectory and how often positions repeat. */
    struct KspaceCoverage
    {
        /** Distinct positions along each axis, in the axis order given. */
        std::vector<double> unique_positions;
        /** Number of visits per position, summarised over the positions. */
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
