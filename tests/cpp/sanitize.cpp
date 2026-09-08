/**
 * @file sanitize.cpp
 * @brief Drive the core over the corpus with the sanitisers watching.
 *
 * A read past the end of an array is not a wrong answer; it is whatever
 * happened to be there. The ordinary tests catch it only when the memory it
 * lands on differs enough to change a number, which is a coincidence -- one
 * such bug reached CI reporting the same waveform on five platforms and a
 * different one on the sixth, and gave two different answers in one process.
 *
 * So this runs the whole core -- reading, writing, deduplication, timing,
 * waveform expansion -- over every file in the corpus and over a sequence
 * built here to reach what the corpus does not, with AddressSanitizer and
 * UndefinedBehaviorSanitizer on. It asserts nothing. The sanitisers do.
 */

#include "pulseq/binary.hpp"
#include "pulseq/read.hpp"
#include "pulseq/sequence.hpp"
#include "pulseq/timing.hpp"
#include "pulseq/waveforms.hpp"
#include "pulseq/write.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace
{

    /** Somewhere for the results to go, so nothing is optimised away. */
    double sink = 0.0;

    void touch(const std::vector<double>& values)
    {
        for (size_t i = 0; i < values.size(); ++i)
            sink += values[i];
    }

    /** Everything a caller can ask of a sequence, asked. */
    void exercise(pulseq::Sequence& seq)
    {
        pulseq::TimingLimits limits;
        limits.rf_raster_time = seq.rf_raster_time();
        limits.grad_raster_time = seq.grad_raster_time();
        limits.adc_raster_time = seq.adc_raster_time();
        limits.block_duration_raster = seq.block_duration_raster();
        limits.rf_dead_time = 100e-6;
        limits.rf_ringdown_time = 30e-6;
        limits.adc_dead_time = 10e-6;
        const std::vector<pulseq::TimingFinding> found = pulseq::check_timing(seq, limits);
        sink += static_cast<double>(found.size());

        for (int append = 0; append < 2; ++append)
        {
            pulseq::WaveformOptions options;
            options.append_rf = append != 0;
            const pulseq::Waveforms made = pulseq::waveforms_and_times(seq, options);
            for (int axis = 0; axis < 3; ++axis)
            {
                touch(made.times[static_cast<size_t>(axis)]);
                touch(made.amplitudes[static_cast<size_t>(axis)]);
            }
            touch(made.rf_times);
            touch(made.adc_times);
            touch(made.adc_phase);
            touch(made.adc_modulation);
            sink += made.duration;
        }

        /* Part of a sequence as well as all of it: the range is where an
         * index is most likely to be one past where it should stop. */
        if (seq.num_blocks() > 3)
        {
            pulseq::WaveformOptions window;
            window.first_block = 2;
            window.last_block = seq.num_blocks() - 1;
            const pulseq::Waveforms part = pulseq::waveforms_and_times(seq, window);
            for (int axis = 0; axis < 3; ++axis)
                touch(part.times[static_cast<size_t>(axis)]);
        }

        const pulseq::Repetition repeat = seq.repetition();
        sink += repeat.size + repeat.start;
        sink += seq.locate_repetition(repeat.size > 0 ? repeat.size : 1).size;
        sink += seq.detect_rf_uses(1.5, 42576000.0);
        sink += seq.duration();

        std::map<std::string, double> values;
        values["TE"] = 5e-3;
        values["TR"] = 100e-3;
        const pulseq::SoftDelayReport applied = seq.apply_soft_delays(values);
        sink += static_cast<double>(applied.hints.size() + applied.rounded.size());

        for (int axis = 0; axis < 3; ++axis)
        {
            try
            {
                seq.scale_gradient_axis(axis, 0.5);
                seq.scale_gradient_axis(axis, 2.0);
            }
            catch (const std::exception&)
            {
                // A gradient played on two axes cannot be scaled for one.
            }
        }

        const std::string text = pulseq::write_text(seq, true);
        const std::string binary = pulseq::write_binary(seq);
        sink += static_cast<double>(text.size() + binary.size());

        pulseq::Sequence again = pulseq::read(binary, false);
        again.remove_duplicates();
        sink += again.duration();
    }

    /**
     * A sequence built here rather than read, to reach what the corpus does
     * not: a rotation, which sends one axis's gradient onto three.
     */
    pulseq::Sequence rotated()
    {
        pulseq::Sequence seq;
        seq.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

        const double trap[pulseq::TRAP_WIDTH] = {1.0e5, 1e-4, 2e-3, 1e-4, 0.0};
        const int readout = seq.register_trap(trap);

        std::vector<double> wave(64);
        for (size_t i = 0; i < wave.size(); ++i)
            wave[i] = std::sin(3.14159265358979323846 * static_cast<double>(i) /
                               static_cast<double>(wave.size() - 1));
        const int shape = seq.register_raw_shape(wave.data(), static_cast<int>(wave.size()));
        const double arb[pulseq::ARB_WIDTH] = {
            5.0e4, 0.0, 0.0, static_cast<double>(shape), 0.0, 0.0};
        const int wobble = seq.register_arbitrary(arb);

        const double delayed_trap[pulseq::TRAP_WIDTH] = {7.3e4, 1.3e-4, 1.9e-3, 1.3e-4, 1.7e-4};
        const int delayed = seq.register_trap(delayed_trap);

        const double adc[pulseq::ADC_WIDTH] = {64.0, 1e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        const int window = seq.register_adc(adc);

        const int rotations = seq.extension_type_id("ROTATIONS");
        for (int shot = 0; shot < 16; ++shot)
        {
            const double angle = 3.14159265358979323846 * shot / 16.0;
            const double quaternion[pulseq::ROTATION_WIDTH] = {
                std::cos(angle / 2.0), 0.0, 0.0, std::sin(angle / 2.0)};
            const int turned = seq.register_rotation(quaternion);

            /* A single gradient rotated onto two axes is the case that puts a
             * corner time and the union's last time at exactly the same
             * moment, which is where an index runs one past the end; a
             * delayed one makes the offset something the arithmetic has to
             * carry, and durations that do not land on the raster keep the
             * running total from being a round number. */
            pulseq::Block alone;
            alone.gx = delayed;
            alone.adc = window;
            alone.ext = seq.append_extension(rotations, turned, 0);
            alone.duration = 2.53e-3;
            seq.add_block(alone);

            pulseq::Block together;
            together.gx = readout;
            together.gy = wobble;
            together.ext = seq.append_extension(rotations, turned, 0);
            together.duration = 2.53e-3;
            seq.add_block(together);
        }
        return seq;
    }

} // namespace

int main(int argc, char** argv)
{
    int checked = 0;
    for (int i = 1; i < argc; ++i)
    {
        std::ifstream file(argv[i], std::ios::binary);
        if (!file)
        {
            std::printf("cannot open %s\n", argv[i]);
            return 1;
        }
        std::ostringstream held;
        held << file.rdbuf();

        try
        {
            pulseq::Sequence seq = pulseq::read(held.str(), false);
            exercise(seq);
            ++checked;
        }
        catch (const std::exception& refused)
        {
            std::printf("%s: %s\n", argv[i], refused.what());
            return 1;
        }
    }

    pulseq::Sequence turned = rotated();
    exercise(turned);

    std::printf("%d files and one rotated sequence, sink %g\n", checked, sink);
    return 0;
}
