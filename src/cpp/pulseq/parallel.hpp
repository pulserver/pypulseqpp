/**
 * @file parallel.hpp
 * @brief Contiguous ranges of independent items split across threads.
 */

#ifndef PULSEQ_PARALLEL_HPP
#define PULSEQ_PARALLEL_HPP

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

namespace pulseq
{

    /** Threads for @p items independent items: one below @p threshold, else up to eight. */
    inline unsigned worker_count(size_t items, size_t threshold = size_t(1) << 16)
    {
        if (items < threshold)
            return 1;
        const unsigned available = std::thread::hardware_concurrency();
        return std::max(1u, std::min(available, 8u));
    }

    /**
     * Call @p run(first, last) on @p workers contiguous ranges covering
     * [0, @p count), the calling thread taking the first.
     */
    template <typename Run>
    void parallel_ranges(size_t count, unsigned workers, const Run& run)
    {
        if (workers <= 1 || count < 2)
        {
            run(size_t(0), count);
            return;
        }
        const size_t chunk = (count + workers - 1) / workers;
        std::vector<std::thread> pool;
        for (size_t first = chunk; first < count; first += chunk)
            pool.emplace_back([&run, first, count, chunk] { run(first, std::min(count, first + chunk)); });
        run(size_t(0), std::min(count, chunk));
        for (std::thread& thread : pool)
            thread.join();
    }

} // namespace pulseq

#endif
