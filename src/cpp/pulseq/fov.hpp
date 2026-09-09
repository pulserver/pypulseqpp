/**
 * @file fov.hpp
 * @brief Moving the imaging volume of a sequence that is already designed.
 *
 * Shifting the field of view is a phase, and the phase is `dr . k`: where the
 * trajectory stands times how far the volume moved. So everything here is
 * built on knowing where k stands, which is what `block_k_origins` says.
 *
 * ### The frame the shift is written in
 *
 * A prescribed offset is written in the *logical* frame -- `(1, 0, 0)` moves
 * the reconstructed image along its own x, whichever physical axis that turns
 * out to be. That is not a limitation to work around; it is what makes this
 * cheap. Rotate `dr` and `k` together and `dr . k` does not change, so a shift
 * written in the frame the gradients are designed in needs to know nothing
 * about the rotation the scanner applies -- the prescribed orientation, and
 * PMC composed into it. An offset that arrives in the physical frame is
 * turned once by the caller, `dr_logical = R^T dr_physical`, and never again.
 *
 * A block's own `ROTATIONS` extension is a different object and does not
 * vanish: it turns the gradients *inside* the logical frame, so that block's
 * phase is `dr . (R k)`. Everything here works in the unrotated logical
 * frame, which is why a turned readout is handed on rather than baked.
 *
 * ### Units
 *
 * k is in 1/m and a shift in m, so `dr . k` is in **cycles**, with no factor
 * of two pi anywhere -- which is also the unit an RF phase shape is stored
 * in, so a phase profile goes in unscaled.
 */

#ifndef PULSEQ_FOV_HPP
#define PULSEQ_FOV_HPP

#include <array>
#include <vector>

#include "pulseq/sequence.hpp"

namespace pulseq
{

    /**
     * Where a pulse leaves the trajectory.
     *
     * An excitation starts a new one, so k is zero after it. A refocusing
     * turns it around: what was `k` becomes `-k`, which is what walks a spin
     * echo back towards the origin. Everything else -- an inversion, a
     * saturation, a pulse doing something a scan does not encode against --
     * leaves it where it was.
     *
     * @param use     The pulse's recorded use, as its first letter.
     * @param at      Where k stands at the pulse's centre.
     * @param origin  The running origin, updated in place.
     * @return Whether the pulse moved it.
     */
    bool advance_origin(char use, const double at[3], double origin[3]);

    /**
     * Where the trajectory stands at the start of each block.
     *
     * The running total of what every gradient before it has swept, reset by
     * each excitation and turned around by each refocusing -- at the pulse's
     * *centre*, not at its block boundary, so a refocusing between two
     * crushers has each crusher counted on the right side of the flip.
     *
     * Answered in the logical frame: a block's `ROTATIONS` extension is not
     * applied, because `dr . k` does not care and the consumer that does can
     * read the rotation off the block.
     *
     * @param seq     The sequence to walk.
     * @param first   First block to report, 1-based.
     * @param last    Last block, or 0 for the end of the sequence.
     * @param carry   Where k stands entering @p first, updated in place to
     *                where it stands leaving @p last. A caller walking a scan
     *                in chunks hands the same array back in; one walking from
     *                the beginning starts it at zero.
     * @return One origin per block in the range, in order.
     *
     * A pulse that records no use is read as an excitation, which is what
     * `calculate_kspace` reads it as: one sequence cannot have two stories
     * about where its trajectory restarts. Before revision 1.5.0 the format
     * had nowhere to write a use, so a file older than that arrives with
     * every pulse undefined, and `detect_rf_use` is what fills them in.
     */
    std::vector<std::array<double, 3>> block_k_origins(
        const Sequence& seq, int first, int last, double carry[3]);

} // namespace pulseq

#endif /* PULSEQ_FOV_HPP */
