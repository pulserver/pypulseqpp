/**
 * @file rfft.hpp
 * @brief Forward real FFT of one length, over MKL when a runtime is supplied.
 *
 * MKL is loaded at run time from a path the caller names, so the extension
 * links nothing beyond the standard library and a wheel carries no MKL. Where
 * no runtime is named, or it does not load, the transform is pocketfft, the
 * FFT NumPy itself runs (`third_party/pocketfft_hdronly.h`, BSD 3-Clause).
 */

#ifndef PULSEQ_RFFT_HPP
#define PULSEQ_RFFT_HPP

#include <complex>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace pulseq
{

    /** A planned forward transform of real samples of one length. */
    class RealFft
    {
    public:
        /**
         * @param length       Samples per transform.
         * @param mkl_runtime  Path of an MKL runtime library (`libmkl_rt`,
         *                     `mkl_rt.N.dll`); empty for pocketfft.
         */
        RealFft(size_t length, const std::string& mkl_runtime);
        ~RealFft();
        RealFft(const RealFft&) = delete;
        RealFft& operator=(const RealFft&) = delete;

        /**
         * Unnormalised forward transform, X_k = sum_n x_n exp(-2 pi i k n / length).
         *
         * @param in   `length` samples; not modified.
         * @param out  `length / 2 + 1` bins, DC to Nyquist.
         */
        void forward(const double* in, std::complex<double>* out);

        /** "mkl" or "pocketfft". */
        const char* backend() const;

    private:
        struct Mkl;
        struct Pocket;
        size_t length_;
        std::unique_ptr<Mkl> mkl_;
        std::unique_ptr<Pocket> pocket_;
        std::vector<double> scratch_;
    };

} // namespace pulseq

#endif /* PULSEQ_RFFT_HPP */
