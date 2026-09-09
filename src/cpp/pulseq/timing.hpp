/**
 * @file timing.hpp
 * @brief Whether every event time in a sequence lands where a scanner can put it.
 *
 * A sequencer can only start an event on one of its clock ticks, and it needs
 * a settling window around RF and digitisation. A design that asks for a pulse
 * 3.7 microseconds in is not played 3.7 microseconds in; it is played wherever
 * the interpreter rounds it to, and the sequence that comes back off the
 * scanner is not the one that was designed. This answers, before the file
 * leaves the bench, which times cannot be honoured and by how far each misses.
 */

#ifndef PULSEQ_TIMING_HPP
#define PULSEQ_TIMING_HPP

#include <string>
#include <vector>

#include "pulseq/sequence.hpp"

namespace pulseq
{

    /**
     * What the sequence is judged against: the system's, not its own.
     *
     * A file records the rasters it was laid out on, but the question is
     * whether the machine that will play it can address those times -- so the
     * rasters come from the system, alongside the dead times, the ringdown and
     * the sample-count divisor, none of which a sequence carries.
     */
    struct TimingLimits
    {
        double rf_raster_time = 1e-6;
        double grad_raster_time = 10e-6;
        double adc_raster_time = 100e-9;
        double block_duration_raster = 10e-6;

        double rf_dead_time = 0.0;
        double rf_ringdown_time = 0.0;
        double adc_dead_time = 0.0;

        /** ADC sample counts must be a multiple of this; 0 or 1 asks nothing. */
        double adc_samples_divisor = 1.0;

        /**
         * How far off resonance a pulse or a window may be asked to sit, in
         * hertz. Zero asks nothing, which is what a scanner that does not say
         * gets.
         *
         * An offset is recorded twice over: in hertz, and as a shift in parts
         * per million of the Larmor frequency. Either alone can be within the
         * limit while the two together are not, so all three are weighed --
         * which is what the reference toolbox does.
         */
        double max_freq_offset = 0.0;
        /** What a ppm shift is a shift *of*: the Larmor frequency, in hertz. */
        double larmor = 42576000.0 * 1.5;
    };

    /**
     * One problem with one block.
     *
     * `error_type` names the kind, and the kind says which of the fields below
     * carry anything: a RASTER finding fills `value`, `value_rounded`, `error`
     * and `raster`; an RF_RINGDOWN_TIME finding fills `value`, `duration` and
     * `ringdown_time`; and so on. `event` and `field` name where in the block
     * it is -- `rf.delay`, `gx.rise_time`, `block.duration`.
     */
    struct TimingFinding
    {
        int block = 0;
        std::string event;
        std::string field;
        std::string error_type;
        std::string raster;
        std::string hint;
        std::string prev_hint;
        double num_id = 0.0;
        double value = 0.0;
        double value_rounded = 0.0;
        double error = 0.0;
        double duration = 0.0;
        double dead_time = 0.0;
        double ringdown_time = 0.0;
        double divisor = 0.0;
        /** A gradient amplitude, in Hz/m, where the finding is about one. */
        double amplitude = 0.0;
        /** A frequency offset and the limit it is over, in hertz. */
        double offset = 0.0;
        double limit = 0.0;
    };

    /**
     * Every timing problem in @p seq, in block order.
     *
     * Raster alignment is a property of a library row rather than of a block,
     * so it is decided once per distinct event and attributed to each block
     * that plays it; what is genuinely per block -- the stored duration
     * against the events it holds, the dead-time margins, and whether soft
     * delays sharing a numeric id agree -- is arithmetic over precomputed
     * extents. No block is decoded.
     *
     * @param seq     The sequence to check.
     * @param limits  Rasters, dead times and ringdown, which @p seq does not carry.
     * @return One finding per problem, empty when the sequence is clean.
     */
    std::vector<TimingFinding> check_timing(const Sequence& seq, const TimingLimits& limits);

} // namespace pulseq

#endif /* PULSEQ_TIMING_HPP */
