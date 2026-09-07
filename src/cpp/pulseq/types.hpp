/**
 * @file types.hpp
 * @brief The enumerated values the Pulseq file format numbers.
 *
 * Every value here is the number that appears in a `.seq` file, so a cast to
 * `int` is what the format carries and a cast back is how it is read. They
 * are collected in one place because several of them are written in more
 * than one section.
 */

#ifndef PULSEQ_CXX_TYPES_HPP
#define PULSEQ_CXX_TYPES_HPP

#include <vector>

namespace pulseq
{

    /** Decompressed `[SHAPES]` entry (owning; see decompress_shape()). */
    struct Shape
    {
        int num_uncompressed_samples = 0;
        std::vector<float> samples;
    };

    /** Design-time raster times, in microseconds. */
    struct Raster
    {
        float rf_us = 0.0f;
        float grad_us = 0.0f;
        float block_us = 0.0f;
    };

    /** RF use tag: the trailing e/r/i/s code on an RF library row. */
    enum class RfUse : int
    {
        Unknown = 0,
        Excitation = 1,
        Refocusing = 2,
        Inversion = 3,
        Saturation = 4,
        Preparation = 5,
        Other = 6,
    };

    /** Block extension-chain entry type. */
    enum class ExtensionType : int
    {
        List = 0,
        Trigger = 1,
        Rotation = 2,
        LabelSet = 3,
        LabelInc = 4,
        RfShim = 5,
        Delay = 6,
        Unknown = 7,
    };

    /** Numeric id for a LABELSET/LABELINC name. */
    enum class LabelId : int
    {
        Slc = 1,
        Seg = 2,
        Rep = 3,
        Avg = 4,
        Set = 5,
        Eco = 6,
        Phs = 7,
        Lin = 8,
        Par = 9,
        Acq = 10,
        Nav = 11,
        Rev = 12,
        Sms = 13,
        Ref = 14,
        Ima = 15,
        Noise = 16,
        Pmc = 17,
        NoRot = 18,
        NoPos = 19,
        NoScl = 20,
        Once = 21,
        Trid = 22,
        Off = 23,
    };

    /** One past the last built-in label id; custom labels are numbered above. */
    constexpr int LABEL_BUILTIN_COUNT = 24;

    /** Numeric id for a soft-delay hint name. */
    enum class HintId : int
    {
        Te = 1,
        Tr = 2,
        Ti = 3,
        Esp = 4,
        RecTime = 5,
        T2Prep = 6,
        Te2 = 7,
    };

    /** Gradient library row discriminator. */
    enum class GradientType : int
    {
        Trapezoid = 1,
        Arbitrary = 2,
    };

    /** Digital trigger / physio event type. */
    enum class TriggerType : int
    {
        Output = 1, ///< TTL / digital output
        Input = 2,  ///< ECG / cardiac gating
    };

    /** Trigger channel for TriggerType::Input events. */
    enum class TriggerChannelInput : int
    {
        Physio1 = 1,
        Physio2 = 2,
    };

    /** Trigger channel for TriggerType::Output events. */
    enum class TriggerChannelOutput : int
    {
        Osc0 = 1,
        Osc1 = 2,
        Ext1 = 3,
    };

} // namespace pulseq

#endif // PULSEQ_CXX_TYPES_HPP
