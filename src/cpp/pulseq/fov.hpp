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

    /** What a shift is allowed to write on. */
    enum class FovShiftScope
    {
        /**
         * The RF side only. Every readout is left to a consumer that will
         * apply the shift itself from the trajectory -- which is what a
         * reconstructor wants anyway, and what lets it re-apply the shift for
         * a new prescription without the sequence being touched again.
         */
        RfOnly,
        /**
         * Both sides, so the file needs nothing downstream. What a `.seq`
         * handed to another toolbox has to be.
         */
        RfAndAdc,
    };

    /**
     * Move the field of view by @p shift_m, written in logical metres.
     *
     * ### What each event gets
     *
     * **A pulse** is always handled: nothing downstream could do it instead.
     * Under a gradient that does not change across the pulse -- a slice
     * select's flat top, which is nearly every pulse -- the shift is a
     * frequency and a phase on the RF row, and no shape is registered.
     * Otherwise the phase shape gains `dr . k(t)`, referenced to the pulse's
     * own centre so what the pulse does is unchanged.
     *
     * **A readout** gets the same split under `RfAndAdc`: a frequency, a
     * phase, and whatever will not fit in those into `phase_modulation`.
     * Under a constant gradient the residual is identically zero, so a
     * Cartesian readout costs two numbers.
     *
     * ### Where the phase comes from, and why not from the trajectory
     *
     * `dr` against everything the gradients have swept -- unbroken, and
     * deliberately *not* `block_k_origins`. The trajectory restarts at every
     * excitation, and it is right to; a phase does not. What a readout is
     * measured by is its phase against the phase its own excitation was
     * given, so the two have to be counted from the same place, and resetting
     * between them references them to different zeros and leaves a phase on
     * the signal that is not the shift. Two identical repetitions would then
     * read their echoes at different phases, which is how this was caught.
     *
     * The fractional part is taken per segment rather than at the end: the
     * total reaches thousands of turns across a scan and only the fraction is
     * a phase, so summing first spends the precision on what is thrown away.
     *
     * @param seq      The sequence to move.
     * @param shift_m  The offset, in logical metres.
     * @param scope    Which sides to write on.
     * @param first    First block to move, 1-based.
     * @param last     Last block, or 0 for the end.
     * @param carry    Where k stands entering @p first, updated in place.
     */
    void apply_fov_shift(
        Sequence& seq,
        const double shift_m[3],
        FovShiftScope scope,
        int first,
        int last,
        double carry[3]);

} // namespace pulseq

#endif /* PULSEQ_FOV_HPP */
