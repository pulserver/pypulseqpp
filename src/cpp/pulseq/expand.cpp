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

        /** The rows a repetition writes: block table, durations and definitions, in step. */
        struct RepeatedRows
        {
            std::vector<int32_t> rows;
            std::vector<double> durations;
            std::vector<int32_t> defs;
            std::vector<int32_t> adc_defs;

            void reserve(size_t blocks)
            {
                rows.reserve(blocks * BLOCK_WIDTH);
                durations.reserve(blocks);
                defs.reserve(blocks);
                adc_defs.reserve(blocks);
            }
        };

    } // namespace

    /** The ONCE reading, label directives and chain rewrites of one call to Sequence::expand_repeats. */
    class RepeatExpansion
    {
    public:
        RepeatExpansion(Sequence& seq, const std::string& label, bool strip_once)
            : seq_(seq), label_(label), strip_once_(strip_once), n_(seq.num_blocks()),
              labelset_(seq.find_extension_type_id("LABELSET")),
              labelinc_(seq.find_extension_type_id("LABELINC")),
              once_id_(seq.find_label_id("ONCE")), labels_(seq.label_names_.size() + 1)
        {
        }

        /** Read every block's directives and ONCE; count the blocks of each ONCE. */
        ExpandResult survey(int repeats)
        {
            read_directives();
            refuse_written_counter(repeats);
            read_once();
            ExpandResult result;
            result.repeats = repeats;
            result.blocks_before = n_;
            result.blocks_after = n_;
            for (int8_t state : once_)
            {
                if (state == 1)
                    ++result.prep_blocks;
                else if (state == 2)
                    ++result.cooldown_blocks;
                else
                    ++result.body_blocks;
            }
            return result;
        }

        /** Replace the block table with @p repeats repetitions of it. */
        void write(int repeats, ExpandResult& result)
        {
            if (result.body_blocks == 0)
                throw std::runtime_error(
                    "expand_repeats: every block plays with ONCE=1 or ONCE=2, so nothing plays on "
                    "every repetition");
            after_first_ = left_by(true);
            after_later_ = repeats > 2 ? left_by(false) : after_first_;
            counter_ = label_.empty() ? 0 : seq_.label_id(label_);
            labelset_out_ = seq_.extension_type_id("LABELSET");

            RepeatedRows table;
            table.reserve(static_cast<size_t>(repeats) * result.body_blocks + result.prep_blocks +
                          result.cooldown_blocks);
            for (int pass = 0; pass < repeats; ++pass)
                write_pass(pass, repeats, table);

            seq_.changed();
            seq_.repetition_known_ = false;
            seq_.detach_blocks();
            seq_.blocks_->swap(table.rows);
            seq_.durations_->swap(table.durations);
            seq_.instance_def_.swap(table.defs);
            seq_.instance_adc_def_.swap(table.adc_defs);
            result.blocks_after = seq_.num_blocks();
        }

    private:
        Directives read_chain(int32_t head) const
        {
            Directives d;
            for (int32_t link = head; link != 0;)
            {
                const int32_t* row = seq_.extensions_.row(link);
                if (row[1] != 0 && labelset_ != 0 && row[0] == labelset_)
                {
                    const int32_t* entry = seq_.label_set_.row(row[1]);
                    d.sets.emplace_back(entry[1], entry[0]);
                }
                else if (row[1] != 0 && labelinc_ != 0 && row[0] == labelinc_)
                {
                    const int32_t* entry = seq_.label_inc_.row(row[1]);
                    d.incs.emplace_back(entry[1], entry[0]);
                }
                link = row[2];
            }
            return d;
        }

        /** The directives of each distinct chain, and of each block through it. */
        void read_directives()
        {
            std::unordered_map<int32_t, int32_t> of_chain;
            directive_.assign(static_cast<size_t>(n_), 0);
            const int32_t* events = seq_.blocks_->data();
            for (int b = 0; b < n_; ++b)
            {
                const int32_t head = events[static_cast<size_t>(b) * BLOCK_WIDTH + 5];
                if (head == 0)
                    continue;
                auto found = of_chain.find(head);
                if (found == of_chain.end())
                {
                    Directives d = read_chain(head);
                    int32_t index = 0;
                    if (!d.sets.empty() || !d.incs.empty())
                    {
                        index = static_cast<int32_t>(directives_.size());
                        directives_.push_back(std::move(d));
                    }
                    found = of_chain.emplace(head, index).first;
                }
                directive_[static_cast<size_t>(b)] = found->second;
            }
        }

        void refuse_written_counter(int repeats) const
        {
            const int counter = (!label_.empty() && repeats > 1) ? seq_.find_label_id(label_) : 0;
            if (counter == 0)
                return;
            for (const Directives& d : directives_)
                if (d.writes(counter))
                    throw std::runtime_error(
                        "expand_repeats: the scan already writes " + label_ +
                        ", which would then count two things; label the repetitions with another "
                        "counter, or with none");
        }

        const Directives& of_block(int b) const
        {
            return directives_[static_cast<size_t>(directive_[static_cast<size_t>(b)])];
        }

        /** The ONCE value in force at each block, ONCE being sticky. */
        void read_once()
        {
            once_.assign(static_cast<size_t>(n_), 0);
            if (once_id_ == 0)
                return;
            std::vector<int32_t> state(labels_, 0);
            for (int b = 0; b < n_; ++b)
            {
                of_block(b).apply(state);
                const int32_t current = state[static_cast<size_t>(once_id_)];
                if (current < 0 || current > 2)
                    throw std::runtime_error(
                        "expand_repeats: block " + std::to_string(b + 1) + " plays with ONCE=" +
                        std::to_string(current) + ", where ONCE is 0, 1 or 2");
                once_[static_cast<size_t>(b)] = static_cast<int8_t>(current);
            }
        }

        /** Whether block @p b plays on @p pass of @p repeats. */
        bool plays(int b, int pass, int repeats) const
        {
            const int8_t state = once_[static_cast<size_t>(b)];
            return !((state == 1 && pass != 0) || (state == 2 && pass != repeats - 1));
        }

        /* The counters a repetition leaves behind, from zero: the first plays
         * its ONCE=1 blocks, a later one only its ONCE=0 blocks. A stripped
         * ONCE is not a counter of the expanded scan. */
        std::vector<int32_t> left_by(bool first) const
        {
            std::vector<int32_t> state(labels_, 0);
            for (int b = 0; b < n_; ++b)
            {
                const int8_t state_of_block = once_[static_cast<size_t>(b)];
                if (state_of_block == 2 || (state_of_block == 1 && !first))
                    continue;
                of_block(b).apply(state);
            }
            if (strip_once_ && once_id_ != 0)
                state[static_cast<size_t>(once_id_)] = 0;
            return state;
        }

        bool is_once_link(int32_t type, int32_t ref) const
        {
            if (ref == 0)
                return false;
            if (labelset_ != 0 && type == labelset_)
                return seq_.label_set_.row(ref)[1] == once_id_;
            if (labelinc_ != 0 && type == labelinc_)
                return seq_.label_inc_.row(ref)[1] == once_id_;
            return false;
        }

        /** The chain @p head without its ONCE links, rebuilt in its order, once per distinct chain. */
        int32_t without_once(int32_t head)
        {
            if (!strip_once_ || once_id_ == 0 || head == 0)
                return head;
            const auto found = stripped_.find(head);
            if (found != stripped_.end())
                return found->second;
            std::vector<std::pair<int32_t, int32_t>> kept;
            for (int32_t link = head; link != 0;)
            {
                const int32_t* row = seq_.extensions_.row(link);
                if (!is_once_link(row[0], row[1]))
                    kept.emplace_back(row[0], row[1]);
                link = row[2];
            }
            int32_t chain = 0;
            for (size_t k = kept.size(); k-- > 0;)
                chain = seq_.chain_extension(kept[k].first, kept[k].second, chain);
            stripped_.emplace(head, chain);
            return chain;
        }

        /* The chain of the first block a repetition past the first plays: the
         * counters the repetition before left elsewhere than zero set back to
         * zero, those the block sets itself aside, and the repetition's own
         * counter set to its index. */
        int32_t starting(int pass, int32_t chain, const Directives& own)
        {
            const std::vector<int32_t>& left = pass == 1 ? after_first_ : after_later_;
            for (size_t id = 1; id < labels_; ++id)
            {
                const int32_t name = static_cast<int32_t>(id);
                if (left[id] == 0 || name == counter_ || own.sets_label(name))
                    continue;
                chain = seq_.chain_extension(labelset_out_, seq_.intern_label_set(0, name), chain);
            }
            if (counter_ != 0)
                chain = seq_.chain_extension(labelset_out_, seq_.intern_label_set(pass, counter_), chain);
            return chain;
        }

        /** Append block @p b as chain @p ext, forking its definitions when the chain changed. */
        void append(int b, int32_t ext, RepeatedRows& table) const
        {
            // The source table is read through blocks_ every block: the chains
            // registered here grow only the libraries, never it.
            const int32_t* row = seq_.blocks_->data() + static_cast<size_t>(b) * BLOCK_WIDTH;
            table.rows.insert(table.rows.end(), row, row + BLOCK_WIDTH);
            int32_t* out = table.rows.data() + table.rows.size() - BLOCK_WIDTH;
            int32_t def = seq_.instance_def_[static_cast<size_t>(b)];
            int32_t adc_def = seq_.instance_adc_def_[static_cast<size_t>(b)];
            const double duration = (*seq_.durations_)[static_cast<size_t>(b)];
            if (ext != row[5])
            {
                out[5] = ext;
                out[BLOCK_ROTATION_COLUMN] = seq_.promoted_in_chain(0, ext);
                out[BLOCK_SHIM_COLUMN] = seq_.promoted_in_chain(1, ext);
                if (out[0] == 0 && out[1] == 0 && out[2] == 0 && out[3] == 0 && out[4] == 0)
                {
                    Block block;
                    block.ext = ext;
                    block.duration = duration;
                    seq_.fork_instance(block, def, adc_def);
                }
            }
            table.durations.push_back(duration);
            table.defs.push_back(def);
            table.adc_defs.push_back(adc_def);
        }

        void write_pass(int pass, int repeats, RepeatedRows& table)
        {
            bool started = pass == 0;
            for (int b = 0; b < n_; ++b)
            {
                if (!plays(b, pass, repeats))
                    continue;
                int32_t ext = without_once(seq_.blocks_->data()[static_cast<size_t>(b) * BLOCK_WIDTH + 5]);
                if (!started)
                {
                    ext = starting(pass, ext, of_block(b));
                    started = true;
                }
                append(b, ext, table);
            }
        }

        Sequence& seq_;
        const std::string& label_;
        const bool strip_once_;
        const int n_;
        const int labelset_;
        const int labelinc_;
        const int once_id_;
        const size_t labels_;

        std::vector<Directives> directives_ = std::vector<Directives>(1); // 0: none
        std::vector<int32_t> directive_;
        std::vector<int8_t> once_;
        std::vector<int32_t> after_first_;
        std::vector<int32_t> after_later_;
        std::unordered_map<int32_t, int32_t> stripped_;
        int32_t counter_ = 0;
        int32_t labelset_out_ = 0;
    };

    ExpandResult Sequence::expand_repeats(int repeats, const std::string& label, bool strip_once)
    {
        if (repeats < 1)
            throw std::invalid_argument(
                "expand_repeats: repeats is at least 1, not " + std::to_string(repeats));
        RepeatExpansion expansion(*this, label, strip_once);
        ExpandResult result = expansion.survey(repeats);
        if (repeats > 1)
            expansion.write(repeats, result);
        return result;
    }

} // namespace pulseq
