#pragma once

#include <cstdint>
#include <vector>

namespace pulseq
{

    /**
     * A stream of block definition ids folded into nested loops.
     *
     * A token above zero is a block definition id. A token `-(k + 1)` is loop
     * `k`: its body, itself a token stream, played `counts[k]` times.
     * Playing `top` in order, each loop body as often as its count, plays the
     * stream that was folded.
     */
    struct Loops
    {
        std::vector<int32_t> top;
        std::vector<std::vector<int32_t>> bodies;
        std::vector<int32_t> counts;
    };

    /**
     * Fold @p keys into nested loops, its outer loop anchored at a known period.
     *
     * When the @p size blocks from @p start repeat whole at least twice, the
     * copies are the outer loop, the blocks before them a prologue and those
     * after them an epilogue. Each part is then folded on its own: every
     * tandem repeat of the shortest period, up to @p max_period tokens, becomes
     * a loop and one token, and the next pass folds runs of those tokens in
     * turn, so loops nest as deep as the stream does. Without an anchor the
     * whole stream is folded that way. No loop starts with the pure delay
     * @p wait (0 for none): a wait ends the iteration it follows, so a run of
     * waits plays as one longer wait rather than a loop, and the wait that
     * closes one train is not taken as the start of the next.
     *
     * @throws std::logic_error if the loops do not play @p keys back.
     */
    Loops fold_loops(
        const std::vector<int32_t>& keys, int size, int start, int32_t wait = 0, int max_period = 64);

} // namespace pulseq
