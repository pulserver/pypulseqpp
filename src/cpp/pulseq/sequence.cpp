/**
 * @file sequence.cpp
 * @brief Storage and block operations for pulseq::Sequence; see sequence.hpp.
 */

#include "pulseq/shape.hpp"
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
        // Seed for the numbering, not a definition of what a label may be.
        //
        // A label is named, not numbered: the text format writes the name,
        // `label_id` mints one for a name it has not seen, and a sequence is
        // free to use a name nothing here lists.  This table only fixes where
        // the numbering starts, and the order matters in exactly one place --
        // the binary format writes the number rather than the name, so two
        // implementations reading each other's binary files agree about a
        // label only as far as they agree about this list.  For a name
        // outside it they cannot agree at all, which is what `LABELNAMES` is
        // for.  The order is the reference toolbox's.
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
        for (Promoted& column : promoted_)
        {
            if (name == column.name)
                column.type_id = id;
        }
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
        int32_t promoted_was[2];
        for (size_t which = 0; which < promoted_.size(); ++which)
            promoted_was[which] = promoted_[which].type_id;
        auto existing = extension_ids_.find(name);
        if (existing != extension_ids_.end())
            extension_names_.erase(existing->second);
        extension_ids_[name] = id;
        extension_names_[id] = name;
        trigger_type_id_ = find_extension_type_id("TRIGGERS");
        bool promotion_moved = false;
        for (size_t which = 0; which < promoted_.size(); ++which)
        {
            promoted_[which].type_id = find_extension_type_id(promoted_[which].name);
            promotion_moved |= promoted_[which].type_id != promoted_was[which];
        }

        // Which chains carry a trigger is read off a type id, so forcing the
        // mapping can change the answer for chains that already exist -- and
        // with it which blocks are pure delays.
        if (promotion_moved && !extensions_.empty())
        {
            recompute_chain_promotions();
            refill_block_promotions();
        }
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
        changed();
        rf_use_.push_back(use);
        shapes_.mark(static_cast<int>(row[1]), SHAPE_ROLE_RF_MAGNITUDE);
        shapes_.mark(static_cast<int>(row[2]), SHAPE_ROLE_RF_PHASE);
        shapes_.mark(static_cast<int>(row[3]), SHAPE_ROLE_RF_TIME);
        rf_def_.push_back(rf_defs_.intern(rf_key(row, use)));
        return rf_.append(row);
    }

    int Sequence::register_trap(const double* row)
    {
        changed();
        const int slot = trap_.append(row);
        grad_slot_.push_back(static_cast<int32_t>(slot));
        grad_def_.push_back(grad_defs_.intern(trap_key(row)));
        return static_cast<int>(grad_slot_.size());
    }

    int Sequence::register_arbitrary(const double* row)
    {
        changed();
        shapes_.mark(static_cast<int>(row[3]), SHAPE_ROLE_GRADIENT);
        shapes_.mark(static_cast<int>(row[4]), SHAPE_ROLE_GRADIENT_TIME);
        const int slot = arb_.append(row);
        grad_slot_.push_back(-static_cast<int32_t>(slot));
        grad_def_.push_back(grad_defs_.intern(arb_key(row)));
        return static_cast<int>(grad_slot_.size());
    }

    int Sequence::register_adc(const double* row)
    {
        changed();
        shapes_.mark(static_cast<int>(row[7]), SHAPE_ROLE_ADC_PHASE);
        adc_def_.push_back(adc_defs_.intern(adc_key(row)));
        return adc_.append(row);
    }

    int Sequence::register_trigger(const double* row)
    {
        changed();
        return trigger_.append(row);
    }

    int Sequence::register_rotation(const double* row)
    {
        changed();
        return rotation_.append(row);
    }

    int Sequence::register_label_set(int32_t value, int32_t label_id)
    {
        changed();
        const int32_t row[LABEL_WIDTH] = {value, label_id};
        return label_set_.append(row);
    }

    int Sequence::register_label_inc(int32_t value, int32_t label_id)
    {
        changed();
        const int32_t row[LABEL_WIDTH] = {value, label_id};
        return label_inc_.append(row);
    }

    int Sequence::register_rf_shim(const double* values, int count)
    {
        changed();
        return rf_shim_.append(values, count);
    }

    int Sequence::register_soft_delay(const SoftDelay& row)
    {
        changed();
        soft_delays_.push_back(row);
        return static_cast<int>(soft_delays_.size());
    }

    int32_t Sequence::soft_delay_number(const std::string& hint, int32_t requested)
    {
        const std::map<std::string, int32_t>::const_iterator known =
            soft_delay_hints_.find(hint);

        if (known != soft_delay_hints_.end())
        {
            if (requested >= 0 && requested != known->second)
                throw std::invalid_argument(
                    "Soft delay hint '" + hint + "' is already assigned to numID " +
                    std::to_string(known->second) + ". Cannot use numID " +
                    std::to_string(requested) +
                    ". Consider using a different hint or omitting numID.");
            return known->second;
        }

        int32_t number = requested;
        if (number < 0)
        {
            // The next one nothing is called by, counting from zero.
            number = 0;
            for (std::map<std::string, int32_t>::const_iterator taken =
                     soft_delay_hints_.begin();
                 taken != soft_delay_hints_.end();
                 ++taken)
            {
                if (taken->second >= number)
                    number = taken->second + 1;
            }
        }
        else
        {
            for (std::map<std::string, int32_t>::const_iterator taken =
                     soft_delay_hints_.begin();
                 taken != soft_delay_hints_.end();
                 ++taken)
            {
                if (taken->second == number)
                    throw std::invalid_argument(
                        "numID " + std::to_string(number) +
                        " is already used by soft delay '" + taken->first +
                        "'. Use a different numID or omit it for auto-assignment.");
            }
        }

        soft_delay_hints_[hint] = number;
        return number;
    }

    int Sequence::register_shape(int num_uncompressed, const double* samples, int count)
    {
        changed();
        return shapes_.append(num_uncompressed, samples, count);
    }

    int Sequence::register_raw_shape(const double* samples, int count)
    {
        changed();
        return shapes_.append_raw(samples, count);
    }

    int Sequence::register_raw_shape_divided(const double* samples, int count, double divisor)
    {
        changed();
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
            changed();
    }

    int Sequence::chain_extension(int32_t type_id, int32_t ref, int32_t next)
    {
        changed();
        const std::array<int32_t, EXTENSION_WIDTH> key{type_id, ref, next};
        auto it = chain_index_.find(key);
        if (it != chain_index_.end())
            return it->second;

        const int id = extensions_.append(key.data());
        chain_index_.emplace(key, id);
        note_chain(type_id, ref, next);
        return id;
    }

    int Sequence::append_extension(int32_t type_id, int32_t ref, int32_t next)
    {
        changed();
        const std::array<int32_t, EXTENSION_WIDTH> row{type_id, ref, next};
        const int id = extensions_.append(row.data());
        note_chain(type_id, ref, next);
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
        changed();
        repetition_known_ = false;
        detach_blocks_before_growth();
        blocks_->insert(
            blocks_->end(),
            {block.rf, block.gx, block.gy, block.gz, block.adc, block.ext,
             promoted_in_chain(0, block.ext), promoted_in_chain(1, block.ext)});
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
            block.rot = e[BLOCK_ROTATION_COLUMN];
            block.shim = e[BLOCK_SHIM_COLUMN];
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

        /* The chains have moved or been rebuilt, so what each one names has
         * to be read off again -- and with it the block table's own column. */
        recompute_chain_promotions();
        refill_block_promotions();
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

    void Sequence::recompute_chain_promotions()
    {
        for (Promoted& column : promoted_)
        {
            column.named.assign(static_cast<size_t>(extensions_.size()), 0);
            for (int node = 1; node <= extensions_.size(); ++node)
            {
                const int32_t* row = extensions_.row(node);
                const int32_t next = row[2];
                column.named[static_cast<size_t>(node) - 1] =
                    (column.type_id != 0 && row[0] == column.type_id)
                    ? row[1]
                    : (next >= 1 && next < node
                           ? column.named[static_cast<size_t>(next) - 1]
                           : 0);
            }
        }
    }

    void Sequence::refill_block_promotions()
    {
        detach_blocks();
        int32_t* row = blocks_->data();
        const size_t rows = blocks_->size() / BLOCK_WIDTH;
        for (size_t i = 0; i < rows; ++i, row += BLOCK_WIDTH)
        {
            for (size_t which = 0; which < promoted_.size(); ++which)
                row[promoted_[which].column] = promoted_in_chain(which, row[5]);
        }
    }

    void Sequence::refork_blocks()
    {
        // Definition ids change here, and the repeating unit is read off
        // them -- so whatever was found before this is about the old ones.
        repetition_known_ = false;

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
            block.rot = e[BLOCK_ROTATION_COLUMN];
            block.shim = e[BLOCK_SHIM_COLUMN];
            block.duration = (*durations_)[static_cast<size_t>(i)];
            fork_instance(
                block,
                instance_def_[static_cast<size_t>(i)],
                instance_adc_def_[static_cast<size_t>(i)]);
        }
    }

    void Sequence::set_block(int index, const Block& block)
    {
        changed();
        repetition_known_ = false;
        require_block(index, num_blocks());
        detach_blocks();
        int32_t* row = blocks_->data() + static_cast<size_t>(index - 1) * BLOCK_WIDTH;
        row[0] = block.rf;
        row[1] = block.gx;
        row[2] = block.gy;
        row[3] = block.gz;
        row[4] = block.adc;
        row[5] = block.ext;
        row[BLOCK_ROTATION_COLUMN] = promoted_in_chain(0, block.ext);
        row[BLOCK_SHIM_COLUMN] = promoted_in_chain(1, block.ext);
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
        block.rot = row[BLOCK_ROTATION_COLUMN];
        block.shim = row[BLOCK_SHIM_COLUMN];
        block.duration = (*durations_)[index - 1];
        return block;
    }

    namespace
    {
        /**
         * Match NumPy's pairwise summation and blocking order.
         *
         * TotalDuration is serialised as float64 in binary, so reference parity
         * requires identical rounding, not merely agreement to text precision.
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

    void Sequence::scale_gradient_axis(int axis, double modifier)
    {
        if (axis < 0 || axis > 2)
            throw std::invalid_argument("axis must be 0, 1 or 2");

        /* A library row rather than a gradient id is what gets scaled, and
         * deduplication can leave two ids sharing one -- so the question is
         * whether a *row* is played on this axis and on another. */
        const size_t rows = static_cast<size_t>(trap_.size() + arb_.size()) + 1;
        std::vector<uint8_t> on_axis(rows, 0);
        std::vector<uint8_t> elsewhere(rows, 0);

        const auto slot_of = [&](int32_t id) -> size_t {
            const size_t row = static_cast<size_t>(grad_row(id));
            return grad_kind(id) == GradKind::Trap
                ? row
                : row + static_cast<size_t>(trap_.size());
        };

        const int32_t* block = blocks_->data();
        const size_t count = blocks_->size() / BLOCK_WIDTH;
        for (size_t b = 0; b < count; ++b, block += BLOCK_WIDTH)
        {
            for (int played = 0; played < 3; ++played)
            {
                const int32_t id = block[1 + played];
                if (id <= 0 || id > num_gradients())
                    continue;
                (played == axis ? on_axis : elsewhere)[slot_of(id)] = 1;
            }
        }

        for (size_t slot = 1; slot < rows; ++slot)
        {
            if (on_axis[slot] && elsewhere[slot])
                throw std::runtime_error(
                    "mod_grad_axis does not yet support the same gradient event "
                    "used on multiple axes.");
        }

        for (int id = 1; id <= num_gradients(); ++id)
        {
            // Two ids can share a row, so a row is scaled the first time it
            // is reached and marked spent.
            if (on_axis[slot_of(id)] != 1)
                continue;
            on_axis[slot_of(id)] = 2;
            const int row = grad_row(id);
            if (grad_kind(id) == GradKind::Trap)
            {
                trap_.row(row)[0] *= modifier;
            }
            else
            {
                double* values = arb_.row(row);
                values[0] *= modifier;
                values[1] *= modifier;
                values[2] *= modifier;
            }
        }

        changed();
    }

    Repetition Sequence::locate_repetition(int size) const
    {
        Repetition found;
        const int blocks = static_cast<int>(instance_def_.size());
        if (size < 1 || size * 2 > blocks)
            return found;

        int start = blocks - size;
        while (start > 0 &&
               instance_def_[static_cast<size_t>(start) - 1] ==
                   instance_def_[static_cast<size_t>(start) - 1 + size])
            --start;

        if (blocks - start >= 2 * size)
        {
            found.size = size;
            found.start = start;
        }
        return found;
    }

    Repetition Sequence::repetition()
    {
        if (repetition_known_)
            return repetition_;

        repetition_known_ = true;
        repetition_ = Repetition();

        const int blocks = static_cast<int>(instance_def_.size());
        if (blocks < 2)
            return repetition_;

        /* The last block's definition is played once per repetition, so the
         * gaps between the places it appears are the only periods worth
         * trying -- smallest first, which is the fundamental one. Fifty is
         * far more than a real scan needs and stops a sequence whose last
         * block is also its commonest from being walked to death. */
        constexpr int kCandidates = 50;
        const int32_t last = instance_def_[static_cast<size_t>(blocks) - 1];

        int tried = 0;
        for (int before = blocks - 2; before >= 0 && tried < kCandidates; --before)
        {
            if (instance_def_[static_cast<size_t>(before)] != last)
                continue;
            ++tried;

            const int period = blocks - 1 - before;
            if (period * 2 > blocks)
                break; // no room for the two repeats it would take to say so

            /* How far back the period holds. Everything before that is the
             * prologue: dummy shots, preparation, a noise scan. */
            const Repetition holds = locate_repetition(period);
            if (holds.size != 0)
            {
                repetition_ = holds;
                return repetition_;
            }
        }

        return repetition_;
    }

    int Sequence::detect_rf_uses(double b0, double gamma)
    {
        /** One shape, as the samples it stands for. */
        const auto samples_of = [&](int id) -> std::vector<double> {
            if (id < 1 || id > shapes_.size())
                return {};
            return decompress_shape(
                shapes_.samples(id), shapes_.num_compressed(id), shapes_.num_uncompressed(id));
        };

        /* Where fat sits relative to water, in parts per million. A
         * saturation pulse is put there on purpose and nothing else is. */
        constexpr double kFatPpmLow = -3.5;
        constexpr double kFatPpmHigh = -3.4;
        /* Long enough to be selective in frequency rather than in space. */
        constexpr double kSaturationDuration = 6e-3;
        /* Ninety degrees, with room for a short pulse's rounding. */
        constexpr double kExcitationDegrees = 90.01;

        int labelled = 0;
        for (int id = 1; id <= rf_.size(); ++id)
        {
            if (id <= static_cast<int>(rf_use_.size()) && rf_use_[static_cast<size_t>(id) - 1] != 'u')
                continue;

            const double* row = rf_.row(id);
            const std::vector<double> magnitude = samples_of(static_cast<int>(row[1]));
            const std::vector<double> phase = samples_of(static_cast<int>(row[2]));
            const int time_shape = static_cast<int>(row[3]);

            std::vector<double> t;
            if (time_shape > 0)
            {
                t = samples_of(time_shape);
                for (size_t i = 0; i < t.size(); ++i)
                    t[i] *= rf_raster_;
            }
            else
            {
                t.resize(magnitude.size());
                for (size_t i = 0; i < t.size(); ++i)
                    t[i] = (static_cast<double>(i) + 0.5) * rf_raster_;
            }

            /* The flip angle is the envelope integrated over its own times,
             * in turns; the magnitude of that, in degrees. */
            double real = 0.0;
            double imaginary = 0.0;
            for (size_t i = 0; i + 1 < magnitude.size() && i + 1 < t.size(); ++i)
            {
                const double turns =
                    6.283185307179586476925286766559 * (i < phase.size() ? phase[i] : 0.0);
                const double weight = row[0] * magnitude[i] * (t[i + 1] - t[i]);
                real += weight * std::cos(turns);
                imaginary += weight * std::sin(turns);
            }
            const double degrees = std::sqrt(real * real + imaginary * imaginary) * 360.0;

            const double shape_dur = time_shape > 0
                ? (t.empty() ? 0.0 : t.back())
                : static_cast<double>(magnitude.size()) * rf_raster_;
            const double ppm =
                (b0 != 0.0 && gamma != 0.0) ? 1e6 * row[8] / b0 / gamma : 0.0;

            char use = 'r';
            if (degrees < kExcitationDegrees)
                use = 'e';
            else if (shape_dur > kSaturationDuration && ppm >= kFatPpmLow && ppm <= kFatPpmHigh)
                use = 's';

            if (static_cast<int>(rf_use_.size()) < id)
                rf_use_.resize(static_cast<size_t>(id), 'u');
            rf_use_[static_cast<size_t>(id) - 1] = use;
            ++labelled;
        }

        if (labelled != 0)
        {
            /* What a pulse is for is part of what makes it the definition it
             * is, so labelling one can split a definition two pulses shared:
             * an excitation and a refocusing off the same shape are one
             * definition while both are unlabelled and two once they are
             * not. The tables that say so are re-derived here rather than
             * left saying what was true before.
             *
             * Two rows the libraries could not tell apart may now differ too,
             * so a collapse already done no longer covers them. */
            changed();
            rebuild_definitions();
        }
        return labelled;
    }

    SoftDelayReport Sequence::apply_soft_delays(const std::map<std::string, double>& values)
    {
        SoftDelayReport report;

        const int delay_type = find_extension_type_id("DELAYS");
        if (delay_type <= 0)
            return report;

        const double raster = block_duration_raster();
        /* Durations are written in place below, so the table is taken away
         * from any view still reading it before the walk starts. */
        double* durations = block_durations();
        std::map<std::string, int32_t> number_of;
        std::map<int32_t, std::string> hint_of;
        std::map<int32_t, bool> warned;

        const int32_t* row = blocks_->data();
        const size_t count = blocks_->size() / BLOCK_WIDTH;

        for (size_t index = 0; index < count; ++index, row += BLOCK_WIDTH)
        {
            /* At most one soft delay per block, the last its chain names. */
            int32_t found = 0;
            int32_t node = row[5];
            while (node > 0 && node <= extensions_.size())
            {
                const int32_t* link = extensions_.row(node);
                if (link[0] == delay_type)
                    found = link[1];
                node = link[2];
            }
            if (found < 1 || found > static_cast<int32_t>(soft_delays_.size()))
                continue;

            const SoftDelay& delay = soft_delays_[static_cast<size_t>(found) - 1];
            const int block = static_cast<int>(index) + 1;

            /* A hint and a number name the same delay, so each has to name
             * the other everywhere it appears. */
            const std::map<std::string, int32_t>::const_iterator numbered =
                number_of.find(delay.hint);
            if (numbered == number_of.end())
                number_of[delay.hint] = delay.num;
            else if (numbered->second != delay.num)
            {
                report.problem = SoftDelayReport::Problem::HintRenumbered;
                report.block = block;
                report.hint = delay.hint;
                report.num = delay.num;
                return report;
            }

            const std::map<int32_t, std::string>::const_iterator named =
                hint_of.find(delay.num);
            if (named == hint_of.end())
            {
                hint_of[delay.num] = delay.hint;
                report.hints.push_back(delay.hint);
            }
            else if (named->second != delay.hint)
            {
                report.problem = SoftDelayReport::Problem::NumberRenamed;
                report.block = block;
                report.hint = delay.hint;
                report.num = delay.num;
                return report;
            }

            const std::map<std::string, double>::const_iterator asked =
                values.find(delay.hint);
            if (asked == values.end())
                continue;

            const double wanted = asked->second / delay.factor + delay.offset;
            const double rounded = std::nearbyint(wanted / raster) * raster;

            /* Half a microsecond: below that the move is the raster doing
             * its job, above it the caller did not get what they asked for. */
            const double missed = std::fabs(rounded - wanted);
            if (missed > 0.5e-6 && !warned[delay.num])
            {
                warned[delay.num] = true;
                SoftDelayReport::Rounding note;
                note.block = block;
                note.hint = delay.hint;
                note.num = delay.num;
                note.error = missed;
                report.rounded.push_back(note);
            }

            if (rounded < 0.0)
            {
                report.problem = SoftDelayReport::Problem::Negative;
                report.block = block;
                report.hint = delay.hint;
                report.num = delay.num;
                report.duration = rounded;
                report.offset = delay.offset;
                report.factor = delay.factor;
                return report;
            }

            durations[index] = rounded;
        }

        return report;
    }

    std::array<int64_t, BLOCK_FILE_COLUMNS> Sequence::event_counts() const
    {
        std::array<int64_t, BLOCK_FILE_COLUMNS> counts{};
        const int32_t* row = blocks_->data();
        const size_t rows = blocks_->size() / BLOCK_WIDTH;
        for (size_t block = 0; block < rows; ++block, row += BLOCK_WIDTH)
        {
            for (int column = 0; column < BLOCK_FILE_COLUMNS; ++column)
                counts[static_cast<size_t>(column)] += row[column] > 0 ? 1 : 0;
        }
        return counts;
    }

} // namespace pulseq
