/**
 * @file expand.cpp
 * @brief A scan played more than once, resolved into its block table.
 *
 * Every repetition starts as a scan starts, its counters at zero, the
 * repetition's own counter aside: the first block each repetition past the
 * first plays sets back to zero the counters the repetition before left
 * elsewhere. A block's `LABELSET` directives apply before its `LABELINC`
 * directives, as an interpreter applies them. The block table and the
 * per-block definition arrays are built in one pass over the source table: a
 * repeated block is its source row, and only a chain that loses its `ONCE`
 * links or starts a repetition is renamed.
 */

#include "pulseq/sequence.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pulseq
{

    namespace
    {

        /** The label directives of one chain: the counters it sets and those it increments. */
        struct Directives
        {
            std::vector<std::pair<int32_t, int32_t>> sets;
            std::vector<std::pair<int32_t, int32_t>> incs;

            bool sets_label(int32_t label) const
            {
                for (const auto& [id, value] : sets)
                    if (id == label)
                        return true;
                return false;
            }

            bool writes(int32_t label) const
            {
                if (sets_label(label))
                    return true;
                for (const auto& [id, value] : incs)
                    if (id == label)
                        return true;
                return false;
            }

            /** Apply to @p state, sets first, as a block applies them. */
            void apply(std::vector<int32_t>& state) const
            {
                for (const auto& [id, value] : sets)
                    state[static_cast<size_t>(id)] = value;
                for (const auto& [id, value] : incs)
                    state[static_cast<size_t>(id)] += value;
            }
        };

    } // namespace

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

        // The directives of each distinct chain, and of each block through it.
        std::vector<Directives> directives(1);  // 0: none
        std::unordered_map<int32_t, int32_t> directives_of_chain;
        std::vector<int32_t> directive(static_cast<size_t>(n), 0);
        {
            const int32_t* events = blocks_->data();
            for (int b = 0; b < n; ++b)
            {
                const int32_t head = events[static_cast<size_t>(b) * BLOCK_WIDTH + 5];
                if (head == 0)
                    continue;
                const auto found = directives_of_chain.find(head);
                if (found != directives_of_chain.end())
                {
                    directive[static_cast<size_t>(b)] = found->second;
                    continue;
                }
                Directives d;
                for (int32_t link = head; link != 0;)
                {
                    const int32_t* row = extensions_.row(link);
                    if (row[1] != 0 && labelset != 0 && row[0] == labelset)
                    {
                        const int32_t* entry = label_set_.row(row[1]);
                        d.sets.emplace_back(entry[1], entry[0]);
                    }
                    else if (row[1] != 0 && labelinc != 0 && row[0] == labelinc)
                    {
                        const int32_t* entry = label_inc_.row(row[1]);
                        d.incs.emplace_back(entry[1], entry[0]);
                    }
                    link = row[2];
                }
                int32_t index = 0;
                if (!d.sets.empty() || !d.incs.empty())
                {
                    index = static_cast<int32_t>(directives.size());
                    directives.push_back(std::move(d));
                }
                directives_of_chain.emplace(head, index);
                directive[static_cast<size_t>(b)] = index;
            }
        }

        const size_t labels = label_names_.size() + 1;
        bool counter_written = false;
        for (const Directives& d : directives)
            counter_written |= counter_id != 0 && d.writes(counter_id);
        if (counter_written)
            throw std::runtime_error(
                "expand_repeats: the scan already writes " + label +
                ", which would then count two things; label the repetitions with another counter, "
                "or with none");

        // The ONCE value in force at each block, ONCE being sticky.
        std::vector<int8_t> once(static_cast<size_t>(n), 0);
        if (once_id != 0)
        {
            std::vector<int32_t> state(labels, 0);
            for (int b = 0; b < n; ++b)
            {
                directives[static_cast<size_t>(directive[static_cast<size_t>(b)])].apply(state);
                const int32_t current = state[static_cast<size_t>(once_id)];
                if (current < 0 || current > 2)
                    throw std::runtime_error(
                        "expand_repeats: block " + std::to_string(b + 1) + " plays with ONCE=" +
                        std::to_string(current) + ", where ONCE is 0, 1 or 2");
                once[static_cast<size_t>(b)] = static_cast<int8_t>(current);
            }
        }

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
        result.blocks_after = n;
        if (repeats == 1)
            return result;
        if (result.body_blocks == 0)
            throw std::runtime_error(
                "expand_repeats: every block plays with ONCE=1 or ONCE=2, so nothing plays on "
                "every repetition");

        // The counters a repetition leaves behind, from zero: the first plays
        // its ONCE=1 blocks, a later one only its ONCE=0 blocks. A stripped
        // ONCE is not a counter of the expanded scan.
        const auto left_by = [&](bool first) -> std::vector<int32_t>
        {
            std::vector<int32_t> state(labels, 0);
            for (int b = 0; b < n; ++b)
            {
                const int8_t state_of_block = once[static_cast<size_t>(b)];
                if (state_of_block == 2 || (state_of_block == 1 && !first))
                    continue;
                directives[static_cast<size_t>(directive[static_cast<size_t>(b)])].apply(state);
            }
            if (strip_once && once_id != 0)
                state[static_cast<size_t>(once_id)] = 0;
            return state;
        };
        const std::vector<int32_t> after_first = left_by(true);
        const std::vector<int32_t> after_later = repeats > 2 ? left_by(false) : after_first;

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

        const int32_t counter = label.empty() ? 0 : label_id(label);
        const int32_t labelset_out = extension_type_id("LABELSET");

        /* The chain of the first block a repetition past the first plays: the
         * counters the repetition before left elsewhere than zero set back to
         * zero, those the block sets itself aside, and the repetition's own
         * counter set to its index. */
        const auto starting = [&](int pass, int32_t chain, const Directives& own) -> int32_t
        {
            const std::vector<int32_t>& left = pass == 1 ? after_first : after_later;
            for (size_t id = 1; id < labels; ++id)
            {
                const int32_t name = static_cast<int32_t>(id);
                if (left[id] == 0 || name == counter || own.sets_label(name))
                    continue;
                chain = chain_extension(labelset_out, intern_label_set(0, name), chain);
            }
            if (counter != 0)
                chain = chain_extension(labelset_out, intern_label_set(pass, counter), chain);
            return chain;
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
            bool started = pass == 0;
            for (int b = 0; b < n; ++b)
            {
                const int8_t state = once[static_cast<size_t>(b)];
                if ((state == 1 && pass != 0) || (state == 2 && pass != repeats - 1))
                    continue;
                // The source table is read through blocks_ every block: the
                // chains registered here grow only the libraries, never it.
                const int32_t* row = blocks_->data() + static_cast<size_t>(b) * BLOCK_WIDTH;
                const int32_t source_ext = row[5];
                int32_t ext = without_once(source_ext);
                if (!started)
                {
                    ext = starting(
                        pass, ext, directives[static_cast<size_t>(directive[static_cast<size_t>(b)])]);
                    started = true;
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
