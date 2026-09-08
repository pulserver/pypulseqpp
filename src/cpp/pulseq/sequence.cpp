/**
 * @file sequence.cpp
 * @brief Storage and block operations for pulseq::Sequence.
 *
 * See src/cpp/include/pulseq/sequence.hpp for the model.  Everything here is
 * bookkeeping: appending rows, handing out ids, and reading a block back.
 * The writers, the reader and deduplication live beside this file.
 */

#include "pulseq/sequence.hpp"

#include <numeric>

namespace pulseq
{

    namespace
    {
        const std::string kNoName;

        void require_block(int index, int count)
        {
            if (index < 1 || index > count)
            {
                throw std::out_of_range(
                    "block index " + std::to_string(index) + " is outside 1.." +
                    std::to_string(count));
            }
        }
    } // namespace

    const std::vector<std::string>& Sequence::builtin_labels()
    {
        // Pulseq's own list, in the order that numbers it.  The order is the
        // whole content of this table: a file carries a label's *name*, and
        // its position here is the number every other library row refers to
        // it by, so a list in a different order does not fail -- it renames
        // every label past the first difference.
        static const std::vector<std::string> names{
            "SLC", "SEG", "REP",   "AVG", "SET", "ECO",   "PHS", "LIN",
            "PAR", "ACQ", "TRID",  "NAV", "REV", "SMS",   "REF", "IMA",
            "OFF", "NOISE", "PMC", "NOROT", "NOPOS", "NOSCL", "ONCE"};
        return names;
    }

    Sequence::Sequence()
    {
        label_names_ = builtin_labels();
        for (size_t i = 0; i < label_names_.size(); ++i)
            label_ids_.emplace(label_names_[i], static_cast<int>(i) + 1);
    }

    int Sequence::find_label_id(const std::string& name) const
    {
        auto it = label_ids_.find(name);
        return it == label_ids_.end() ? 0 : it->second;
    }

    int Sequence::label_id(const std::string& name)
    {
        auto it = label_ids_.find(name);
        if (it != label_ids_.end())
            return it->second;

        label_names_.push_back(name);
        const int id = static_cast<int>(label_names_.size());
        label_ids_.emplace(name, id);
        return id;
    }

    const std::string& Sequence::label_name(int id) const
    {
        if (id < 1 || id > static_cast<int>(label_names_.size()))
            return kNoName;
        return label_names_[id - 1];
    }

    bool Sequence::is_custom_label(int id) const
    {
        return id > static_cast<int>(builtin_labels().size());
    }

    /* ================================================================== */
    /*  Version and rasters                                               */
    /* ================================================================== */

    void Sequence::set_version(int major, int minor, int revision)
    {
        version_major_ = major;
        version_minor_ = minor;
        version_revision_ = revision;
    }

    void Sequence::set_rasters(double rf, double grad, double adc, double block)
    {
        rf_raster_ = rf;
        grad_raster_ = grad;
        adc_raster_ = adc;
        block_raster_ = block;
    }

    void Sequence::publish_rasters()
    {
        const std::pair<const char*, double> rasters[4] = {
            {"GradientRasterTime", grad_raster_},
            {"RadiofrequencyRasterTime", rf_raster_},
            {"AdcRasterTime", adc_raster_},
            {"BlockDurationRaster", block_raster_},
        };
        for (const auto& entry : rasters)
        {
            // A file that already declares one was read with it, so
            // set_rasters already agrees; leaving it alone keeps the bytes
            // a round trip produces identical to the ones it read.
            if (definitions_.find(entry.first) == definitions_.end())
                definitions_[entry.first] = Definition(entry.second);
        }
    }

    /* ================================================================== */
    /*  Definitions                                                       */
    /* ================================================================== */

    void Sequence::set_definition(const std::string& key, Definition value)
    {
        definitions_[key] = std::move(value);
    }

    const Definition* Sequence::definition(const std::string& key) const
    {
        auto it = definitions_.find(key);
        return it == definitions_.end() ? nullptr : &it->second;
    }

    /* ================================================================== */
    /*  Extension type registry                                           */
    /* ================================================================== */

    int Sequence::find_extension_type_id(const std::string& name) const
    {
        auto it = extension_ids_.find(name);
        return it == extension_ids_.end() ? 0 : it->second;
    }

    int Sequence::extension_type_id(const std::string& name)
    {
        auto it = extension_ids_.find(name);
        if (it != extension_ids_.end())
            return it->second;

        // Ids are handed out in first-use order, which is what PyPulseq does
        // and therefore what keeps a written file identical to one it wrote.
        const int id = static_cast<int>(extension_ids_.size()) + 1;
        extension_ids_.emplace(name, id);
        extension_names_.emplace(id, name);
        if (name == "TRIGGERS")
            trigger_type_id_ = id;
        return id;
    }

    const std::string& Sequence::extension_type_name(int id) const
    {
        auto it = extension_names_.find(id);
        return it == extension_names_.end() ? kNoName : it->second;
    }

    void Sequence::set_extension_type_id(const std::string& name, int id)
    {
        // Reading a file is the case that has to force the mapping: the file
        // says what its own numbering is, and it need not match the order the
        // sections happen to appear in.
        const int was = trigger_type_id_;
        auto existing = extension_ids_.find(name);
        if (existing != extension_ids_.end())
            extension_names_.erase(existing->second);
        extension_ids_[name] = id;
        extension_names_[id] = name;
        trigger_type_id_ = find_extension_type_id("TRIGGERS");

        // Which chains carry a trigger is read off a type id, so forcing the
        // mapping can change the answer for chains that already exist -- and
        // with it which blocks are pure delays.
        if (trigger_type_id_ != was && !extensions_.empty())
        {
            recompute_chain_triggers();
            refork_blocks();
        }
    }

    /* ================================================================== */
    /*  Event registration                                                */
    /* ================================================================== */

    int Sequence::register_rf(const double* row, char use)
    {
        deduplicated_ = false;
        rf_use_.push_back(use);
        shapes_.mark(static_cast<int>(row[1]), SHAPE_ROLE_RF_MAGNITUDE);
        shapes_.mark(static_cast<int>(row[2]), SHAPE_ROLE_RF_PHASE);
        shapes_.mark(static_cast<int>(row[3]), SHAPE_ROLE_RF_TIME);
        rf_def_.push_back(rf_defs_.intern(rf_key(row, use)));
        return rf_.append(row);
    }

    int Sequence::register_trap(const double* row)
    {
        deduplicated_ = false;
        const int slot = trap_.append(row);
        grad_slot_.push_back(static_cast<int32_t>(slot));
        grad_def_.push_back(grad_defs_.intern(trap_key(row)));
        return static_cast<int>(grad_slot_.size());
    }

    int Sequence::register_arbitrary(const double* row)
    {
        deduplicated_ = false;
        shapes_.mark(static_cast<int>(row[3]), SHAPE_ROLE_GRADIENT);
        shapes_.mark(static_cast<int>(row[4]), SHAPE_ROLE_GRADIENT_TIME);
        const int slot = arb_.append(row);
        grad_slot_.push_back(-static_cast<int32_t>(slot));
        grad_def_.push_back(grad_defs_.intern(arb_key(row)));
        return static_cast<int>(grad_slot_.size());
    }

    int Sequence::register_adc(const double* row)
    {
        deduplicated_ = false;
        shapes_.mark(static_cast<int>(row[7]), SHAPE_ROLE_ADC_PHASE);
        adc_def_.push_back(adc_defs_.intern(adc_key(row)));
        return adc_.append(row);
    }

    int Sequence::register_trigger(const double* row)
    {
        deduplicated_ = false;
        return trigger_.append(row);
    }

    int Sequence::register_rotation(const double* row)
    {
        deduplicated_ = false;
        return rotation_.append(row);
    }

    int Sequence::register_label_set(int32_t value, int32_t label_id)
    {
        deduplicated_ = false;
        const int32_t row[LABEL_WIDTH] = {value, label_id};
        return label_set_.append(row);
    }

    int Sequence::register_label_inc(int32_t value, int32_t label_id)
    {
        deduplicated_ = false;
        const int32_t row[LABEL_WIDTH] = {value, label_id};
        return label_inc_.append(row);
    }

    int Sequence::register_rf_shim(const double* values, int count)
    {
        deduplicated_ = false;
        return rf_shim_.append(values, count);
    }

    int Sequence::register_soft_delay(const SoftDelay& row)
    {
        deduplicated_ = false;
        soft_delays_.push_back(row);
        return static_cast<int>(soft_delays_.size());
    }

    int Sequence::register_shape(int num_uncompressed, const double* samples, int count)
    {
        deduplicated_ = false;
        return shapes_.append(num_uncompressed, samples, count);
    }

    int Sequence::register_raw_shape(const double* samples, int count)
    {
        deduplicated_ = false;
        return shapes_.append_raw(samples, count);
    }

    int Sequence::register_raw_shape_divided(const double* samples, int count, double divisor)
    {
        deduplicated_ = false;
        return shapes_.append_raw_divided(samples, count, divisor);
    }

    void Sequence::compress_shapes()
    {
        /* Compressing can make two shapes that differed only below the codec's
         * quantum into one row, so what was distinct may not be any more --
         * but only if it actually compressed something.  A library already
         * encoded is left exactly as it was, and saying so is what lets a
         * sequence read from a file keep its deduplicated claim through a
         * writer that calls this on the way past. */
        if (shapes_.compress())
            deduplicated_ = false;
    }

    int Sequence::chain_extension(int32_t type_id, int32_t ref, int32_t next)
    {
        deduplicated_ = false;
        const std::array<int32_t, EXTENSION_WIDTH> key{type_id, ref, next};
        auto it = chain_index_.find(key);
        if (it != chain_index_.end())
            return it->second;

        const int id = extensions_.append(key.data());
        chain_index_.emplace(key, id);
        note_chain(type_id, next);
        return id;
    }

    int Sequence::append_extension(int32_t type_id, int32_t ref, int32_t next)
    {
        deduplicated_ = false;
        const std::array<int32_t, EXTENSION_WIDTH> row{type_id, ref, next};
        const int id = extensions_.append(row.data());
        note_chain(type_id, next);
        return id;
    }

    /* ================================================================== */
    /*  Gradients                                                         */
    /* ================================================================== */

    GradKind Sequence::grad_kind(int id) const
    {
        if (id < 1 || id > static_cast<int>(grad_slot_.size()))
            return GradKind::None;
        return grad_slot_[id - 1] > 0 ? GradKind::Trap : GradKind::Arbitrary;
    }

    int Sequence::grad_row(int id) const
    {
        if (id < 1 || id > static_cast<int>(grad_slot_.size()))
            return 0;
        const int32_t slot = grad_slot_[id - 1];
        return slot > 0 ? slot : -slot;
    }

    /* ================================================================== */
    /*  Blocks                                                            */
    /* ================================================================== */

    int Sequence::add_block(const Block& block)
    {
        deduplicated_ = false;
        detach_blocks_before_growth();
        blocks_->insert(
            blocks_->end(),
            {block.rf, block.gx, block.gy, block.gz, block.adc, block.ext});
        durations_->push_back(block.duration);
        int32_t def = 0;
        int32_t adc_def = 0;
        fork_instance(block, def, adc_def);
        instance_def_.push_back(def);
        instance_adc_def_.push_back(adc_def);
        return static_cast<int>(durations_->size());
    }

    void Sequence::fork_instance(const Block& block, int32_t& def, int32_t& adc_def) const
    {
        adc_def = definition_of(block.adc, adc_def_);
        Definitions& blocks = const_cast<Definitions&>(block_defs_);
        def = is_pure_delay(block)
                  ? blocks.intern(delay_key())
                  : blocks.intern(block_key(
                        definition_of(block.rf, rf_def_),
                        definition_of(block.gx, grad_def_),
                        definition_of(block.gy, grad_def_),
                        definition_of(block.gz, grad_def_),
                        block.duration));
    }

    void Sequence::instance_row(const Block& block, double* p) const
    {
        for (int i = 0; i < INSTANCE_WIDTH; ++i)
            p[i] = 0.0;

        // A trapezoid has no waveform to name, so its shape column stays 0.
        const auto gradient = [this](int32_t id, double* out)
        {
            if (id <= 0)
                return;
            const int32_t slot = grad_slot_[static_cast<size_t>(id) - 1];
            if (slot > 0)
            {
                out[0] = trap_.row(slot)[0];
            }
            else
            {
                const double* g = arb_.row(-slot);
                out[0] = g[0];
                out[1] = g[3];
            }
        };
        gradient(block.gx, p + 0);
        gradient(block.gy, p + 2);
        gradient(block.gz, p + 4);

        if (block.rf > 0)
        {
            const double* r = rf_.row(block.rf);
            p[6] = r[0];
            p[7] = r[8];
            p[8] = r[9];
            p[9] = r[6];
            p[10] = r[7];
        }
        if (block.adc > 0)
        {
            const double* a = adc_.row(block.adc);
            p[11] = a[5];
            p[12] = a[6];
            p[13] = a[3];
            p[14] = a[4];
            p[15] = a[7];
        }
    }

    std::vector<double> Sequence::instance_parameters() const
    {
        const int count = num_blocks();
        std::vector<double> out(static_cast<size_t>(count) * INSTANCE_WIDTH, 0.0);
        for (int i = 0; i < count; ++i)
        {
            const int32_t* e = blocks_->data() + static_cast<size_t>(i) * BLOCK_WIDTH;
            Block block;
            block.rf = e[0];
            block.gx = e[1];
            block.gy = e[2];
            block.gz = e[3];
            block.adc = e[4];
            block.ext = e[5];
            block.duration = (*durations_)[static_cast<size_t>(i)];
            instance_row(block, out.data() + static_cast<size_t>(i) * INSTANCE_WIDTH);
        }
        return out;
    }

    void Sequence::rebuild_definitions()
    {
        rf_defs_.clear();
        grad_defs_.clear();
        adc_defs_.clear();

        rf_def_.clear();
        rf_def_.reserve(static_cast<size_t>(rf_.size()));
        for (int id = 1; id <= rf_.size(); ++id)
            rf_def_.push_back(
                rf_defs_.intern(rf_key(rf_.row(id), rf_use_[static_cast<size_t>(id) - 1])));

        grad_def_.clear();
        grad_def_.reserve(grad_slot_.size());
        for (const int32_t slot : grad_slot_)
            grad_def_.push_back(grad_defs_.intern(
                slot > 0 ? trap_key(trap_.row(slot)) : arb_key(arb_.row(-slot))));

        adc_def_.clear();
        adc_def_.reserve(static_cast<size_t>(adc_.size()));
        for (int id = 1; id <= adc_.size(); ++id)
            adc_def_.push_back(adc_defs_.intern(adc_key(adc_.row(id))));

        recompute_chain_triggers();
        refork_blocks();
    }

    void Sequence::recompute_chain_triggers()
    {
        const int trigger = trigger_type_id_;
        chain_carries_trigger_.assign(static_cast<size_t>(extensions_.size()), 0);
        for (int node = 1; node <= extensions_.size(); ++node)
        {
            const int32_t* row = extensions_.row(node);
            const int32_t next = row[2];
            const bool carries =
                (trigger != 0 && row[0] == trigger) ||
                (next >= 1 && next < node &&
                 chain_carries_trigger_[static_cast<size_t>(next) - 1] != 0);
            chain_carries_trigger_[static_cast<size_t>(node) - 1] = carries ? 1 : 0;
        }
    }

    void Sequence::refork_blocks()
    {
        block_defs_.clear();
        const int count = num_blocks();
        instance_def_.assign(static_cast<size_t>(count), 0);
        instance_adc_def_.assign(static_cast<size_t>(count), 0);
        for (int i = 0; i < count; ++i)
        {
            const int32_t* e = blocks_->data() + static_cast<size_t>(i) * BLOCK_WIDTH;
            Block block;
            block.rf = e[0];
            block.gx = e[1];
            block.gy = e[2];
            block.gz = e[3];
            block.adc = e[4];
            block.ext = e[5];
            block.duration = (*durations_)[static_cast<size_t>(i)];
            fork_instance(
                block,
                instance_def_[static_cast<size_t>(i)],
                instance_adc_def_[static_cast<size_t>(i)]);
        }
    }

    void Sequence::set_block(int index, const Block& block)
    {
        deduplicated_ = false;
        require_block(index, num_blocks());
        detach_blocks();
        int32_t* row = blocks_->data() + static_cast<size_t>(index - 1) * BLOCK_WIDTH;
        row[0] = block.rf;
        row[1] = block.gx;
        row[2] = block.gy;
        row[3] = block.gz;
        row[4] = block.adc;
        row[5] = block.ext;
        (*durations_)[index - 1] = block.duration;
        const size_t at = static_cast<size_t>(index) - 1;
        fork_instance(block, instance_def_[at], instance_adc_def_[at]);
    }

    Block Sequence::get_block(int index) const
    {
        require_block(index, num_blocks());
        const int32_t* row = blocks_->data() + static_cast<size_t>(index - 1) * BLOCK_WIDTH;
        Block block;
        block.rf = row[0];
        block.gx = row[1];
        block.gy = row[2];
        block.gz = row[3];
        block.adc = row[4];
        block.ext = row[5];
        block.duration = (*durations_)[index - 1];
        return block;
    }

    void Sequence::set_blocks(const int32_t* events, const double* durations, int count)
    {
        deduplicated_ = false;
        detach_blocks();
        blocks_->assign(events, events + static_cast<size_t>(count) * BLOCK_WIDTH);
        durations_->assign(durations, durations + count);
        rebuild_definitions();
    }

    void Sequence::set_grad_slots(const int32_t* slots, int count)
    {
        deduplicated_ = false;
        grad_slot_.assign(slots, slots + count);
    }

    void Sequence::set_shapes(
        const int32_t* num_uncompressed,
        int count,
        const int32_t* starts,
        const double* samples)
    {
        deduplicated_ = false;
        shapes_.assign(num_uncompressed, count, starts, samples);
    }

    void Sequence::set_rf_shims(const int32_t* starts, int count, const double* values)
    {
        deduplicated_ = false;
        rf_shim_.assign(starts, count, values);
    }

    namespace
    {
        /**
         * Pairwise summation, as NumPy's `.sum()` does it.
         *
         * Adding a couple of million block durations left to right lets the
         * rounding error grow with the number of terms; summing in a balanced
         * tree makes it grow with the logarithm instead, which on a scan of two
         * million blocks is the difference between an error in the last few
         * bits and one that is visible.
         *
         * The blocking is NumPy's exactly -- eight accumulators up to 128
         * elements, splitting on a multiple of eight above that -- and that is
         * deliberate rather than incidental: `TotalDuration` goes into the
         * binary file as a full float64, so a sequence written here and the
         * same one written through NumPy have to agree to the bit, not just to
         * nine significant figures.
         */
        double pairwise_sum(const double* values, size_t n)
        {
            constexpr size_t BLOCKSIZE = 128;

            if (n < 8)
            {
                double result = 0.0;
                for (size_t i = 0; i < n; ++i)
                    result += values[i];
                return result;
            }

            if (n <= BLOCKSIZE)
            {
                double r[8];
                for (int k = 0; k < 8; ++k)
                    r[k] = values[k];

                size_t i = 8;
                for (; i < n - (n % 8); i += 8)
                {
                    for (int k = 0; k < 8; ++k)
                        r[k] += values[i + k];
                }

                double result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
                for (; i < n; ++i)
                    result += values[i];
                return result;
            }

            // Halve, but keep the split on a multiple of the unroll factor so
            // both halves take the branch above in the same shape.
            size_t half = n / 2;
            half -= half % 8;
            return pairwise_sum(values, half) + pairwise_sum(values + half, n - half);
        }
    } // namespace

    double Sequence::duration() const
    {
        return pairwise_sum(durations_->data(), durations_->size());
    }

} // namespace pulseq
