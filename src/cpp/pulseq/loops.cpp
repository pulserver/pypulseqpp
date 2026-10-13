#include "loops.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>

namespace pulseq
{

    namespace
    {

        using Stream = std::vector<int32_t>;

        class Folder
        {
          public:
            Folder(int32_t wait, int max_period) : wait_(wait), max_period_(max_period) {}

            /** Fold until no tandem repeat of at most max_period tokens is left. */
            Stream fold(Stream s)
            {
                for (bool again = true; again;)
                {
                    again = false;
                    const int longest = std::min(max_period_, static_cast<int>(s.size()) / 2);
                    for (int period = 1; period <= longest; ++period)
                        if (sweep(s, period))
                        {
                            again = true;
                            break;
                        }
                }
                return s;
            }

            /** The token of @p body played @p count times, one per distinct pair. */
            int32_t loop(const Stream& body, int count)
            {
                auto key = std::make_pair(body, count);
                const auto found = interned_.find(key);
                if (found != interned_.end())
                    return found->second;
                const int32_t token = -static_cast<int32_t>(loops_.bodies.size()) - 1;
                loops_.bodies.push_back(body);
                loops_.counts.push_back(count);
                interned_.emplace(std::move(key), token);
                return token;
            }

            Loops& loops() { return loops_; }

          private:
            /* Replace every tandem repeat of @p period, left to right, by its loop. */
            bool sweep(Stream& s, int period)
            {
                const int n = static_cast<int>(s.size());
                Stream out;
                out.reserve(s.size());
                bool changed = false;
                int i = 0;
                while (i < n)
                {
                    int matched = i;
                    while (matched + period < n && s[matched] == s[matched + period])
                        ++matched;
                    const int copies = (matched - i) / period + 1;
                    if (copies >= 2 && s[i] != wait_)
                    {
                        out.push_back(loop(Stream(s.begin() + i, s.begin() + i + period), copies));
                        i += copies * period;
                        changed = true;
                    }
                    else
                        out.push_back(s[i++]);
                }
                if (changed)
                    s.swap(out);
                return changed;
            }

            int32_t wait_;
            int max_period_;
            Loops loops_;
            std::map<std::pair<Stream, int>, int32_t> interned_;
        };

        void play(const Loops& loops, const Stream& tokens, Stream& out)
        {
            for (const int32_t token : tokens)
            {
                if (token > 0)
                {
                    out.push_back(token);
                    continue;
                }
                const size_t k = static_cast<size_t>(-token - 1);
                for (int c = 0; c < loops.counts[k]; ++c)
                    play(loops, loops.bodies[k], out);
            }
        }

        /** Whole copies of the @p size blocks from @p start that follow one another. */
        int copies_from(const Stream& keys, int size, int start)
        {
            const int n = static_cast<int>(keys.size());
            if (size <= 0 || start < 0 || start + size > n)
                return 0;
            int copies = 1;
            while (start + (copies + 1) * size <= n)
            {
                const int from = start + copies * size;
                for (int i = 0; i < size; ++i)
                    if (keys[static_cast<size_t>(from + i)] != keys[static_cast<size_t>(start + i)])
                        return copies;
                ++copies;
            }
            return copies;
        }

        void append(Stream& to, const Stream& from) { to.insert(to.end(), from.begin(), from.end()); }

    } // namespace

    Loops fold_loops(
        const std::vector<int32_t>& keys, int size, int start, int32_t wait, int max_period)
    {
        Folder folder(wait, max_period);
        const int copies = copies_from(keys, size, start);
        Stream top;
        if (copies >= 2)
        {
            const auto at = [&](int block) { return keys.begin() + block; };
            const int end = start + copies * size;
            append(top, folder.fold(Stream(at(0), at(start))));
            const Stream body = folder.fold(Stream(at(start), at(start + size)));
            top.push_back(folder.loop(body, copies));
            append(top, folder.fold(Stream(at(end), keys.end())));
        }
        else
            top = folder.fold(keys);

        Loops loops = std::move(folder.loops());
        loops.top = std::move(top);

        Stream played;
        played.reserve(keys.size());
        play(loops, loops.top, played);
        if (played != keys)
            throw std::logic_error("folded loops do not play the block definitions back");
        return loops;
    }

} // namespace pulseq
