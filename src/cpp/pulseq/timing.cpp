/**
 * @file timing.cpp
 * @brief Event and block timing checks; see timing.hpp.
 */

#include "pulseq/timing.hpp"

#include "pulseq/channels.hpp"
#include "pulseq/shape.hpp"

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace pulseq
{

    namespace
    {

        /** The gradient axes, in block-table column order. */
        const char* const kAxis[3] = {"gx", "gy", "gz"};

        /** A nanosecond: the grid a Pulseq file is written on. */
        constexpr double kEps = 1e-9;

        /** What a delay, duration or dwell is measured against. */
        struct Raster
        {
            double time;
            const char* name;
        };

        /**
         * Whether @p value is an integer multiple of @p raster, and by how far
         * it misses.
         *
         * The ratio must land within a nanosecond of an integer, which is a
         * relative test and so independent of how long the event is.  Rounding
         * is to even, so the value a report prints is the one the file writes.
         */
        bool on_raster(double value, double raster, double* rounded, double* error)
        {
            const double ratio = value / raster;
            const double nearest = std::nearbyint(ratio);
            *rounded = nearest * raster;
            *error = value - *rounded;
            return std::fabs(ratio - nearest) < kEps;
        }

        TimingFinding raster_finding(const char* field, double value, const Raster& raster)
        {
            TimingFinding f;
            f.field = field;
            f.error_type = "RASTER";
            f.raster = raster.name;
            f.value = value;
            on_raster(value, raster.time, &f.value_rounded, &f.error);
            return f;
        }

        /** Record a RASTER finding for @p value unless it lands on @p raster. */
        void judge(
            std::vector<TimingFinding>& into,
            double value,
            const char* field,
            const Raster& raster)
        {
            double rounded = 0.0;
            double error = 0.0;
            if (!on_raster(value, raster.time, &rounded, &error))
                into.push_back(raster_finding(field, value, raster));
        }

        void judge_delay(
            std::vector<TimingFinding>& into,
            double delay,
            const Raster& raster)
        {
            if (delay < -kEps)
            {
                TimingFinding f;
                f.field = "delay";
                f.error_type = "NEGATIVE_DELAY";
                f.value = delay;
                into.push_back(f);
            }
            judge(into, delay, "delay", raster);
        }

        /**
         * What one library row contributes: its problems, and where it ends
         * relative to the start of the block that plays it.
         */
        struct EventVerdict
        {
            std::vector<TimingFinding> findings;
            double extent = 0.0;
            /** What a gradient is left at when it stops; zero for anything else. */
            double last = 0.0;
        };

        std::vector<double> decompressed(const ShapeLibrary& shapes, int id)
        {
            return decompress_shape(
                shapes.samples(id), shapes.num_compressed(id), shapes.num_uncompressed(id));
        }

        /**
         * Every RF row's verdict, indexed by row id.
         *
         * A pulse laid out on the RF raster carries no time shape, and its
         * samples sit at the centre of each raster interval; one that carries
         * a time shape says where every sample is, and lasts until the raster
         * boundary at or past its last one.
         */
        std::vector<EventVerdict> judge_rf(const Sequence& seq, const TimingLimits& limits)
        {
            const Raster rf_raster = {limits.rf_raster_time, "rf_raster_time"};
            const Raster sample_raster = {
                std::min(limits.adc_raster_time, limits.rf_raster_time),
                "min(adc_raster_time,rf_raster_time)"};

            const Table& lib = seq.rf_library();
            const ShapeLibrary& shapes = seq.shape_library();
            std::vector<EventVerdict> out(static_cast<size_t>(lib.size()) + 1);

            for (int id = 1; id <= lib.size(); ++id)
            {
                const double* row = lib.row(id);
                const int mag_shape = static_cast<int>(row[1]);
                const int time_shape = static_cast<int>(row[3]);
                const double delay = row[5];
                EventVerdict& v = out[static_cast<size_t>(id)];

                judge_delay(v.findings, delay, rf_raster);

                /* Sample times in units of the RF raster, which is how a time
                 * shape stores them and how the raster test wants them. */
                std::vector<double> rt;
                double shape_dur = 0.0;
                if (time_shape > 0 && time_shape <= shapes.size())
                {
                    rt = decompressed(shapes, time_shape);
                    const double last = rt.empty() ? 0.0 : rt.back() * limits.rf_raster_time;
                    shape_dur =
                        std::ceil((last - kEps) / limits.rf_raster_time) * limits.rf_raster_time;
                }
                else if (mag_shape > 0 && mag_shape <= shapes.size())
                {
                    const int samples = shapes.num_uncompressed(mag_shape);
                    rt.resize(static_cast<size_t>(samples));
                    for (int i = 0; i < samples; ++i)
                        rt[static_cast<size_t>(i)] = (i + 1) - 0.5;
                    shape_dur = samples * limits.rf_raster_time;
                }

                judge(v.findings, shape_dur, "shape_dur", rf_raster);

                // A dynamic pTx pulse repeats one time base per channel, and
                // each channel is judged as the pulse it is.
                const size_t per_channel = rt.size() / rf_channels(rt);
                if (per_channel >= 4)
                {
                    const double step = rt[1] - rt[0];
                    bool uniform = true;
                    for (size_t i = 2; i < per_channel; ++i)
                    {
                        if (std::fabs((rt[i] - rt[i - 1]) - step) >= kEps / limits.rf_raster_time)
                        {
                            uniform = false;
                            break;
                        }
                    }
                    if (uniform)
                    {
                        judge(v.findings, step * limits.rf_raster_time, "dwell", sample_raster);
                    }
                    else
                    {
                        double worst = 0.0;
                        for (size_t i = 0; i < per_channel; ++i)
                            worst = std::max(worst, std::fabs(rt[i] - std::nearbyint(rt[i])));
                        if (worst > 1e-6)
                        {
                            TimingFinding f;
                            f.field = "t";
                            f.error_type = "RASTER";
                            f.raster = "rf_raster_time";
                            f.value = worst * limits.rf_raster_time;
                            f.value_rounded = 0.0;
                            f.error = f.value;
                            v.findings.push_back(f);
                        }
                    }
                }

                v.extent = delay + shape_dur + limits.rf_ringdown_time;
            }
            return out;
        }

        /** Every gradient id's verdict, indexed by gradient id. */
        std::vector<EventVerdict> judge_gradients(const Sequence& seq, const TimingLimits& limits)
        {
            const Raster grad_raster = {limits.grad_raster_time, "grad_raster_time"};
            const ShapeLibrary& shapes = seq.shape_library();
            const int count = seq.num_gradients();
            std::vector<EventVerdict> out(static_cast<size_t>(count) + 1);

            for (int id = 1; id <= count; ++id)
            {
                const GradKind kind = seq.grad_kind(id);
                const int rid = seq.grad_row(id);
                EventVerdict& v = out[static_cast<size_t>(id)];

                if (kind == GradKind::Trap)
                {
                    const double* row = seq.trap_library().row(rid);
                    judge_delay(v.findings, row[4], grad_raster);
                    judge(v.findings, row[1], "rise_time", grad_raster);
                    judge(v.findings, row[2], "flat_time", grad_raster);
                    judge(v.findings, row[3], "fall_time", grad_raster);
                    v.extent = row[4] + row[1] + row[2] + row[3];
                }
                else if (kind == GradKind::Arbitrary)
                {
                    const double* row = seq.arb_library().row(rid);
                    const int amp_shape = static_cast<int>(row[3]);
                    const int time_shape = static_cast<int>(row[4]);
                    judge_delay(v.findings, row[5], grad_raster);
                    v.last = row[2];

                    /* A waveform that starts away from zero has to be picked
                     * up from where the axis already is, and a delay puts the
                     * axis at zero for as long as it lasts. */
                    if (std::fabs(row[1]) > kEps && std::fabs(row[5]) > kEps)
                    {
                        TimingFinding f;
                        f.field = "delay";
                        f.error_type = "GRADIENT_START_DELAY";
                        f.value = row[5];
                        f.amplitude = row[1];
                        v.findings.push_back(f);
                    }

                    /* A time shape says where the waveform ends; without one
                     * the samples sit on the gradient raster, and a time id of
                     * -1 marks a waveform oversampled by two. */
                    double shape_dur = 0.0;
                    if (time_shape > 0 && time_shape <= shapes.size())
                    {
                        const std::vector<double> tt = decompressed(shapes, time_shape);
                        shape_dur = tt.empty() ? 0.0 : tt.back() * limits.grad_raster_time;
                    }
                    else if (amp_shape > 0 && amp_shape <= shapes.size())
                    {
                        const int samples = shapes.num_uncompressed(amp_shape);
                        shape_dur = time_shape == -1
                            ? (samples + 1) * 0.5 * limits.grad_raster_time
                            : samples * limits.grad_raster_time;
                    }
                    v.extent = row[5] + shape_dur;
                }
            }
            return out;
        }

        /**
         * Every ADC row's verdict.
         *
         * The start time is judged against the RF raster and the dwell against
         * the ADC raster: the digitiser counts samples on its own clock, but
         * the sequencer has to address the moment it opens.
         */
        std::vector<EventVerdict> judge_adc(const Sequence& seq, const TimingLimits& limits)
        {
            const Raster rf_raster = {limits.rf_raster_time, "rf_raster_time"};
            const Raster adc_raster = {limits.adc_raster_time, "adc_raster_time"};

            const Table& lib = seq.adc_library();
            std::vector<EventVerdict> out(static_cast<size_t>(lib.size()) + 1);

            for (int id = 1; id <= lib.size(); ++id)
            {
                const double* row = lib.row(id);
                const double samples = row[0];
                const double dwell = row[1];
                const double delay = row[2];
                EventVerdict& v = out[static_cast<size_t>(id)];

                judge_delay(v.findings, delay, rf_raster);

                if (dwell < limits.adc_raster_time - kEps)
                {
                    TimingFinding f;
                    f.field = "dwell";
                    f.error_type = "RASTER";
                    f.raster = "adc_raster_time";
                    f.value = dwell;
                    f.value_rounded = limits.adc_raster_time;
                    f.error = dwell - limits.adc_raster_time;
                    v.findings.push_back(f);
                }
                judge(v.findings, dwell, "dwell", adc_raster);

                const double divisor = limits.adc_samples_divisor;
                if (divisor != 0.0 &&
                    std::fabs(samples / divisor - std::nearbyint(samples / divisor)) > kEps)
                {
                    TimingFinding f;
                    f.field = "num_samples";
                    f.error_type = "ADC_SAMPLES_DIVISOR";
                    f.value = samples;
                    f.divisor = divisor;
                    v.findings.push_back(f);
                }

                v.extent = delay + samples * dwell + limits.adc_dead_time;
            }
            return out;
        }

        /** Every trigger row's verdict, indexed by row id. */
        std::vector<EventVerdict> judge_triggers(const Sequence& seq, const TimingLimits& limits)
        {
            const Raster rf_raster = {limits.rf_raster_time, "rf_raster_time"};
            const Raster grad_raster = {limits.grad_raster_time, "grad_raster_time"};

            const Table& lib = seq.trigger_library();
            std::vector<EventVerdict> out(static_cast<size_t>(lib.size()) + 1);

            for (int id = 1; id <= lib.size(); ++id)
            {
                const double* row = lib.row(id);
                /* A digital output is addressed on the RF raster, an input
                 * trigger only to a gradient tick. */
                const Raster& raster = row[0] == 1.0 ? rf_raster : grad_raster;
                EventVerdict& v = out[static_cast<size_t>(id)];
                judge_delay(v.findings, row[2], raster);
                judge(v.findings, row[3], "duration", raster);
                v.extent = row[2] + row[3];
            }
            return out;
        }

        /** What a block's extension chain names, decoded once. */
        struct ChainContents
        {
            int32_t soft_delay = 0;
            int32_t trigger = 0;
            int triggers = 0;
        };

        ChainContents walk_chain(
            const Sequence& seq,
            int32_t head,
            int delay_type,
            int trigger_type)
        {
            ChainContents found;
            const IntTable& links = seq.extensions_library();
            int32_t node = head;
            while (node > 0 && node <= links.size())
            {
                const int32_t* link = links.row(node);
                if (link[0] == delay_type)
                    found.soft_delay = link[1];
                else if (link[0] == trigger_type)
                {
                    found.trigger = link[1];
                    ++found.triggers;
                }
                node = link[2];
            }
            return found;
        }

        /**
         * Report a frequency offset the scanner will not play.
         *
         * An offset is recorded twice: in hertz, and as a shift in parts per
         * million of the Larmor frequency. Either can be within the limit
         * while the two together are not, so all three are weighed -- which
         * is what the reference toolbox does.
         */
        void note_offset(
            std::vector<TimingFinding>& report,
            const TimingLimits& limits,
            const char* event,
            double hertz,
            double ppm)
        {
            if (limits.max_freq_offset <= 0.0)
                return;
            const double shifted = ppm * 1e-6 * limits.larmor;
            const double worst = std::max(
                {std::fabs(hertz), std::fabs(shifted), std::fabs(hertz + shifted)});
            if (worst <= limits.max_freq_offset)
                return;
            TimingFinding f;
            f.event = event;
            f.field = "freq_offset";
            f.error_type = "FREQ_OFFSET";
            f.value = hertz;
            f.offset = worst;
            f.limit = limits.max_freq_offset;
            report.push_back(f);
        }

    } // namespace

    std::vector<TimingFinding> check_timing(const Sequence& seq, const TimingLimits& limits)
    {
        const Raster block_raster = {limits.block_duration_raster, "block_duration_raster"};

        const std::vector<EventVerdict> rf = judge_rf(seq, limits);
        const std::vector<EventVerdict> grad = judge_gradients(seq, limits);
        const std::vector<EventVerdict> adc = judge_adc(seq, limits);
        const std::vector<EventVerdict> trig = judge_triggers(seq, limits);

        const int32_t* events = seq.block_events();
        const double* durations = seq.block_durations();
        const int blocks = seq.num_blocks();

        const int delay_type = seq.find_extension_type_id("DELAYS");
        const int trigger_type = seq.find_extension_type_id("TRIGGERS");
        const std::vector<SoftDelay>& soft = seq.soft_delay_library();
        std::map<int32_t, double> soft_default;
        std::map<int32_t, std::string> soft_hint;

        std::vector<TimingFinding> report;

        for (int b = 0; b < blocks; ++b)
        {
            const int32_t* row = events + static_cast<size_t>(b) * BLOCK_WIDTH;
            const double stored = durations[b];
            const size_t first = report.size();

            const int32_t rf_id =
                (row[0] > 0 && row[0] < static_cast<int32_t>(rf.size())) ? row[0] : 0;
            const int32_t adc_id =
                (row[4] > 0 && row[4] < static_cast<int32_t>(adc.size())) ? row[4] : 0;

            ChainContents chain;
            if (row[5] > 0)
                chain = walk_chain(seq, row[5], delay_type, trigger_type);
            /* A block naming more than one trigger has no single event to
             * attribute a finding to, so none of them is judged. */
            const int32_t trig_id = (chain.triggers == 1 &&
                                     chain.trigger > 0 &&
                                     chain.trigger < static_cast<int32_t>(trig.size()))
                ? chain.trigger
                : 0;

            /* A block lasts as long as the longest thing in it, or as long as
             * the duration it stored if that is longer.  Only an event running
             * past the stored duration disagrees with it. */
            double computed = stored;
            if (rf_id && rf[rf_id].extent > computed)
                computed = rf[rf_id].extent;
            for (int axis = 0; axis < 3; ++axis)
            {
                const int32_t id = row[1 + axis];
                if (id > 0 && id < static_cast<int32_t>(grad.size()) && grad[id].extent > computed)
                    computed = grad[id].extent;
            }
            if (adc_id && adc[adc_id].extent > computed)
                computed = adc[adc_id].extent;
            if (trig_id && trig[trig_id].extent > computed)
                computed = trig[trig_id].extent;

            double rounded = 0.0;
            double error = 0.0;
            if (!on_raster(computed, block_raster.time, &rounded, &error))
            {
                TimingFinding f = raster_finding("duration", computed, block_raster);
                f.event = "block";
                report.push_back(f);
            }

            double duration = computed;
            if (std::fabs(computed - stored) > kEps)
            {
                TimingFinding f;
                f.event = "block";
                f.field = "duration";
                f.error_type = "BLOCK_DURATION_MISMATCH";
                f.value = computed;
                f.duration = stored;
                report.push_back(f);
                duration = stored;
            }

            /* The event walk, in the order a decoded block presents them. */
            const size_t named_rf = report.size();
            if (rf_id)
                report.insert(report.end(), rf[rf_id].findings.begin(), rf[rf_id].findings.end());
            for (size_t i = named_rf; i < report.size(); ++i)
                report[i].event = "rf";

            /* One gradient row can be played on any axis, so which axis a
             * finding belongs to is known here and not in the verdict. */
            for (int axis = 0; axis < 3; ++axis)
            {
                const int32_t id = row[1 + axis];
                if (id <= 0 || id >= static_cast<int32_t>(grad.size()))
                    continue;
                const size_t named = report.size();
                report.insert(report.end(), grad[id].findings.begin(), grad[id].findings.end());
                for (size_t i = named; i < report.size(); ++i)
                    report[i].event = kAxis[axis];

                /* An axis is at zero wherever nothing is playing on it, so a
                 * waveform left away from zero before the block ends is a
                 * step down to zero in no time at all. */
                if (std::fabs(grad[id].last) > kEps &&
                    std::fabs(grad[id].extent - computed) > kEps)
                {
                    TimingFinding f;
                    f.event = kAxis[axis];
                    f.field = "duration";
                    f.error_type = "GRADIENT_END_NONZERO";
                    f.value = grad[id].extent;
                    f.duration = computed;
                    f.amplitude = grad[id].last;
                    report.push_back(f);
                }
            }

            const size_t named_adc = report.size();
            if (adc_id)
                report.insert(
                    report.end(), adc[adc_id].findings.begin(), adc[adc_id].findings.end());
            for (size_t i = named_adc; i < report.size(); ++i)
                report[i].event = "adc";

            const size_t named_trig = report.size();
            if (trig_id)
                report.insert(
                    report.end(), trig[trig_id].findings.begin(), trig[trig_id].findings.end());
            for (size_t i = named_trig; i < report.size(); ++i)
                report[i].event = "trig";

            if (rf_id)
            {
                const double* rrow = seq.rf_library().row(rf_id);
                const double delay = rrow[5];
                if (delay - limits.rf_dead_time < -kEps)
                {
                    TimingFinding f;
                    f.event = "rf";
                    f.field = "delay";
                    f.error_type = "RF_DEAD_TIME";
                    f.value = delay;
                    f.dead_time = limits.rf_dead_time;
                    report.push_back(f);
                }
                /* The pulse ends at its last sample, which is half a raster
                 * short of the shape's duration when it carries no time
                 * shape; the ringdown is measured from there. */
                const double ends = rf[rf_id].extent - limits.rf_ringdown_time;
                const int time_shape = static_cast<int>(rrow[3]);
                const double last =
                    time_shape > 0 ? ends : ends - 0.5 * limits.rf_raster_time;
                if (last + limits.rf_ringdown_time - duration > kEps)
                {
                    TimingFinding f;
                    f.event = "rf";
                    f.field = "duration";
                    f.error_type = "RF_RINGDOWN_TIME";
                    f.value = last;
                    f.duration = duration;
                    f.ringdown_time = limits.rf_ringdown_time;
                    report.push_back(f);
                }
                note_offset(report, limits, "rf", rrow[8], rrow[6]);
            }

            if (adc_id)
            {
                const double* arow = seq.adc_library().row(adc_id);
                if (arow[2] - limits.adc_dead_time < -kEps)
                {
                    TimingFinding f;
                    f.event = "adc";
                    f.field = "delay";
                    f.error_type = "ADC_DEAD_TIME";
                    f.value = arow[2];
                    f.dead_time = limits.adc_dead_time;
                    report.push_back(f);
                }
                if (adc[adc_id].extent > duration + kEps)
                {
                    TimingFinding f;
                    f.event = "adc";
                    f.field = "duration";
                    f.error_type = "POST_ADC_DEAD_TIME";
                    f.value = arow[2] + arow[0] * arow[1];
                    f.duration = duration;
                    f.dead_time = limits.adc_dead_time;
                    report.push_back(f);
                }
                note_offset(report, limits, "adc", arow[5], arow[3]);
            }

            if (chain.soft_delay >= 1 &&
                chain.soft_delay <= static_cast<int32_t>(soft.size()))
            {
                const SoftDelay& sd = soft[static_cast<size_t>(chain.soft_delay) - 1];

                /* A soft delay is addressed by number at run time, and the
                 * numbering starts at zero -- which is what the toolbox
                 * writes for the first one.  A negative number names nothing,
                 * and nothing else about such a delay is worth reporting. */
                if (sd.num < 0)
                {
                    TimingFinding f;
                    f.event = "soft_delay";
                    f.field = "delay";
                    f.error_type = "SOFT_DELAY_INVALID_NUMID";
                    f.value = sd.num;
                    f.hint = sd.hint;
                    f.num_id = sd.num;
                    report.push_back(f);
                }
                else
                {
                    if (sd.factor == 0.0)
                    {
                        TimingFinding f;
                        f.event = "soft_delay";
                        f.field = "delay";
                        f.error_type = "SOFT_DELAY_FACTOR";
                        f.value = sd.factor;
                        f.hint = sd.hint;
                        f.num_id = sd.num;
                        report.push_back(f);
                    }

                    const double def = (stored - sd.offset) * sd.factor;
                    const std::map<int32_t, double>::iterator seen = soft_default.find(sd.num);
                    if (seen == soft_default.end())
                        soft_default[sd.num] = def;
                    else if (std::fabs(def - seen->second) > 1e-7)
                    {
                        TimingFinding f;
                        f.event = "soft_delay";
                        f.field = "delay";
                        f.error_type = "SOFT_DELAY_DUR_INCONSISTENCY";
                        f.value = def;
                        f.hint = sd.hint;
                        f.num_id = sd.num;
                        report.push_back(f);
                    }

                    const std::map<int32_t, std::string>::iterator known =
                        soft_hint.find(sd.num);
                    if (known == soft_hint.end())
                        soft_hint[sd.num] = sd.hint;
                    else if (known->second != sd.hint)
                    {
                        TimingFinding f;
                        f.event = "soft_delay";
                        f.field = "delay";
                        f.error_type = "SOFT_DELAY_HINT_INCONSISTENCY";
                        f.hint = sd.hint;
                        f.prev_hint = known->second;
                        f.num_id = sd.num;
                        report.push_back(f);
                    }
                }
            }

            for (size_t i = first; i < report.size(); ++i)
                report[i].block = b + 1;
        }

        return report;
    }

} // namespace pulseq
