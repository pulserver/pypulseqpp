/**
 * @file expand.cpp
 * @brief A scan played more than once, resolved into its block table.
 *
 * The block table and the per-block definition arrays are built in one pass
 * over the source table: a repeated block is its source row, and only a chain
 * that loses its `ONCE` links or gains the repetition's label is renamed, once
 * per distinct chain.
 */

#include "pulseq/sequence.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace pulseq
{

    ExpandResult Sequence::expand_repeats(int repeats, const std::string& label, bool strip_once)
    {
        if (repeats < 1)
            throw std::invalid_argument(
                "expand_repeats: repeats is at least 1, not " + std::to_string(repeats));

        const int n = num_blocks();
        const int labelset = find_extension_type_id("LABELSET");
        const int labelinc = find_extension_type_id("LABELINC");
        const int once_id = find_label_id("ONCE");
        const int counter_id = (!label.empty() && repeats > 1) ? find_label_id(label) : 0;

        // The ONCE value in force at each block, and whether the scan writes the
        // counter already: one walk of each block's chain, ONCE being sticky.
        std::vector<int8_t> once(static_cast<size_t>(n), 0);
        bool counter_written = false;
        {
            int current = 0;
            const int32_t* events = blocks_->data();
            for (int b = 0; b < n; ++b)
            {
                for (int32_t link = events[static_cast<size_t>(b) * BLOCK_WIDTH + 5]; link != 0;)
                {
                    const int32_t* row = extensions_.row(link);
                    if (row[1] != 0 && labelset != 0 && row[0] == labelset)
                    {
                        const int32_t* entry = label_set_.row(row[1]);
                        if (once_id != 0 && entry[1] == once_id)
                            current = entry[0];
                        counter_written |= counter_id != 0 && entry[1] == counter_id;
                    }
                    else if (row[1] != 0 && labelinc != 0 && row[0] == labelinc)
                    {
                        const int32_t* entry = label_inc_.row(row[1]);
                        if (once_id != 0 && entry[1] == once_id)
                            current += entry[0];
                        counter_written |= counter_id != 0 && entry[1] == counter_id;
                    }
                    link = row[2];
                }
                if (current < 0 || current > 2)
                    throw std::runtime_error(
                        "expand_repeats: block " + std::to_string(b + 1) + " plays with ONCE=" +
                        std::to_string(current) + ", where ONCE is 0, 1 or 2");
                once[static_cast<size_t>(b)] = static_cast<int8_t>(current);
            }
        }
        if (counter_written)
            throw std::runtime_error(
                "expand_repeats: the scan already writes " + label +
                ", which would then count two things; label the repetitions with another counter, "
                "or with none");

        ExpandResult result;
        result.repeats = repeats;
        result.blocks_before = n;
        for (int8_t state : once)
        {
            if (state == 1)
                ++result.prep_blocks;
            else if (state == 2)
                ++result.cooldown_blocks;
            else
                ++result.body_blocks;
        }
        if (repeats > 1 && result.body_blocks == 0)
            throw std::runtime_error(
                "expand_repeats: every block plays with ONCE=1 or ONCE=2, so nothing plays on "
                "every repetition");

        // Each distinct chain without its ONCE links, rebuilt in its order.
        std::unordered_map<int32_t, int32_t> stripped;
        const auto without_once = [&](int32_t head) -> int32_t
        {
            if (!strip_once || once_id == 0 || head == 0)
                return head;
            const auto found = stripped.find(head);
            if (found != stripped.end())
                return found->second;
            std::vector<std::pair<int32_t, int32_t>> kept;
            for (int32_t link = head; link != 0;)
            {
                const int32_t* row = extensions_.row(link);
                const int32_t type = row[0], ref = row[1], next = row[2];
                const bool flag =
                    ref != 0 &&
                    ((labelset != 0 && type == labelset && label_set_.row(ref)[1] == once_id) ||
                     (labelinc != 0 && type == labelinc && label_inc_.row(ref)[1] == once_id));
                if (!flag)
                    kept.emplace_back(type, ref);
                link = next;
            }
            int32_t chain = 0;
            for (size_t k = kept.size(); k-- > 0;)
                chain = chain_extension(kept[k].first, kept[k].second, chain);
            stripped.emplace(head, chain);
            return chain;
        };

        // Each repetition's label in front of a chain, once per chain and value.
        const int32_t counter = (!label.empty() && repeats > 1) ? label_id(label) : 0;
        const int32_t labelset_out = counter != 0 ? extension_type_id("LABELSET") : 0;
        std::unordered_map<int64_t, int32_t> stamped_chains;
        const auto stamped = [&](int32_t pass, int32_t chain) -> int32_t
        {
            const int64_t key = (static_cast<int64_t>(pass) << 32) | static_cast<uint32_t>(chain);
            const auto found = stamped_chains.find(key);
            if (found != stamped_chains.end())
                return found->second;
            const int32_t row = intern_label_set(pass, counter);
            const int32_t out = chain_extension(labelset_out, row, chain);
            stamped_chains.emplace(key, out);
            return out;
        };

        const size_t total = static_cast<size_t>(repeats) * result.body_blocks +
                             result.prep_blocks + result.cooldown_blocks;
        std::vector<int32_t> table;
        std::vector<double> durations;
        std::vector<int32_t> defs, adc_defs;
        table.reserve(total * BLOCK_WIDTH);
        durations.reserve(total);
        defs.reserve(total);
        adc_defs.reserve(total);

        for (int pass = 0; pass < repeats; ++pass)
        {
            bool labelled = counter == 0 || pass == 0;
            for (int b = 0; b < n; ++b)
            {
                const int8_t state = once[static_cast<size_t>(b)];
                if ((state == 1 && pass != 0) || (state == 2 && pass != repeats - 1))
                    continue;
                // The source table is read through blocks_ every block: the
                // chains registered above grow only the libraries, never it.
                const int32_t* row = blocks_->data() + static_cast<size_t>(b) * BLOCK_WIDTH;
                const int32_t source_ext = row[5];
                int32_t ext = without_once(source_ext);
                if (!labelled)
                {
                    ext = stamped(pass, ext);
                    labelled = true;
                }
                table.insert(table.end(), row, row + BLOCK_WIDTH);
                int32_t* out = table.data() + table.size() - BLOCK_WIDTH;
                int32_t def = instance_def_[static_cast<size_t>(b)];
                int32_t adc_def = instance_adc_def_[static_cast<size_t>(b)];
                if (ext != source_ext)
                {
                    out[5] = ext;
                    out[BLOCK_ROTATION_COLUMN] = promoted_in_chain(0, ext);
                    out[BLOCK_SHIM_COLUMN] = promoted_in_chain(1, ext);
                    if (out[0] == 0 && out[1] == 0 && out[2] == 0 && out[3] == 0 && out[4] == 0)
                    {
                        Block block;
                        block.ext = ext;
                        block.duration = (*durations_)[static_cast<size_t>(b)];
                        fork_instance(block, def, adc_def);
                    }
                }
                durations.push_back((*durations_)[static_cast<size_t>(b)]);
                defs.push_back(def);
                adc_defs.push_back(adc_def);
            }
        }

        changed();
        repetition_known_ = false;
        detach_blocks();
        blocks_->swap(table);
        durations_->swap(durations);
        instance_def_.swap(defs);
        instance_adc_def_.swap(adc_defs);
        result.blocks_after = num_blocks();
        return result;
    }

} // namespace pulseq
