/**
 * @file channels.hpp
 * @brief Transmit channels of a dynamic pTx RF pulse.
 */

#pragma once

#include <cstddef>
#include <vector>

namespace pulseq
{

    /**
     * Transmit channels an RF pulse's sample times describe.
     *
     * A dynamic pTx pulse (Roos et al., Magn Reson Med 2025,
     * doi:10.1002/mrm.30601) stores its channels one after another over one
     * shared time base, so the times restart once per channel. The count is
     * the number of samples at the first sample time, as the reference
     * interpreter reads it, and is accepted only when the times are that many
     * identical copies; anything else is one channel.
     */
    inline size_t rf_channels(const std::vector<double>& times)
    {
        if (times.empty())
            return 1;
        size_t count = 0;
        for (const double t : times)
            if (t == times[0])
                ++count;
        if (count < 2 || times.size() % count != 0)
            return 1;
        const size_t per_channel = times.size() / count;
        for (size_t i = per_channel; i < times.size(); ++i)
            if (times[i] != times[i - per_channel])
                return 1;
        return count;
    }

} // namespace pulseq
