/**
 * @file parallel.hpp
 * @brief Contiguous ranges of independent items split across threads.
 */

#ifndef PULSEQ_PARALLEL_HPP
#define PULSEQ_PARALLEL_HPP

#include <algorithm>
#include <cstddef>
#include <exception>
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
     * [0, @p count), the calling thread taking the first. Ranges start at
     * multiples of ceil(count / workers). An exception thrown in a range is
     * rethrown once every range has finished, the earliest range's first.
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
        std::vector<std::exception_ptr> failed((count + chunk - 1) / chunk);
        const auto guarded = [&run, &failed, chunk](size_t first, size_t last)
        {
            try
            {
                run(first, last);
            }
            catch (...)
            {
                failed[first / chunk] = std::current_exception();
            }
        };
        std::vector<std::thread> pool;
        for (size_t first = chunk; first < count; first += chunk)
            pool.emplace_back([&guarded, first, count, chunk] { guarded(first, std::min(count, first + chunk)); });
        guarded(size_t(0), std::min(count, chunk));
        for (std::thread& thread : pool)
            thread.join();
        for (const std::exception_ptr& failure : failed)
            if (failure)
                std::rethrow_exception(failure);
    }

} // namespace pulseq

#endif
