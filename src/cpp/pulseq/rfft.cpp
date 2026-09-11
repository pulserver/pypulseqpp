/**
 * @file rfft.cpp
 * @brief Forward real FFT over MKL or pocketfft.  See rfft.hpp.
 */

#include "pulseq/rfft.hpp"

#define POCKETFFT_NO_MULTITHREADING
#include "third_party/pocketfft_hdronly.h"

#include <algorithm>
#include <map>
#include <mutex>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace pulseq
{

    namespace
    {

        /* The DFTI entry points and configuration values, as `mkl_dfti.h`
         * declares them. They are restated rather than included because the
         * header is not a build dependency. MKL_LONG is `long` in the LP64
         * interface `mkl_rt` defaults to, on every platform. */
        using DftiHandle = void*;
        using CreateFn = long (*)(DftiHandle*, int, ...);
        using SetFn = long (*)(DftiHandle, int, ...);
        using CommitFn = long (*)(DftiHandle);
        using ForwardFn = long (*)(DftiHandle, void*, ...);
        using FreeFn = long (*)(DftiHandle*);

        constexpr int DFTI_CONJUGATE_EVEN_STORAGE = 10;
        constexpr int DFTI_PLACEMENT = 11;
        constexpr int DFTI_THREAD_LIMIT = 27;
        constexpr int DFTI_REAL = 33;
        constexpr int DFTI_COMPLEX_COMPLEX = 39;
        constexpr int DFTI_NOT_INPLACE = 44;

        struct DftiApi
        {
            CreateFn create = nullptr;
            SetFn set = nullptr;
            CommitFn commit = nullptr;
            ForwardFn forward = nullptr;
            FreeFn release = nullptr;

            bool complete() const
            {
                return create && set && commit && forward && release;
            }
        };

        void* open_library(const std::string& path)
        {
#ifdef _WIN32
            const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
            if (wide <= 0)
                return nullptr;
            std::wstring name(static_cast<size_t>(wide), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &name[0], wide);
            /* MKL's DLLs load their siblings from their own directory. */
            return reinterpret_cast<void*>(
                LoadLibraryExW(name.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH));
#else
            return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        }

        template <typename Fn>
        Fn symbol(void* library, const char* name)
        {
#ifdef _WIN32
            return reinterpret_cast<Fn>(
                GetProcAddress(reinterpret_cast<HMODULE>(library), name));
#else
            return reinterpret_cast<Fn>(dlsym(library, name));
#endif
        }

        /**
         * The DFTI entry points of the runtime at @p path, resolved once per
         * path. A runtime is never unloaded: descriptors outlive any one call,
         * and NumPy may hold the same library.
         */
        const DftiApi* dfti(const std::string& path)
        {
            static std::mutex guard;
            static std::map<std::string, DftiApi> loaded;
            std::lock_guard<std::mutex> hold(guard);

            const auto found = loaded.find(path);
            if (found != loaded.end())
                return found->second.complete() ? &found->second : nullptr;

            DftiApi api;
            if (void* library = open_library(path))
            {
                api.create = symbol<CreateFn>(library, "DftiCreateDescriptor_d_1d");
                api.set = symbol<SetFn>(library, "DftiSetValue");
                api.commit = symbol<CommitFn>(library, "DftiCommitDescriptor");
                api.forward = symbol<ForwardFn>(library, "DftiComputeForward");
                api.release = symbol<FreeFn>(library, "DftiFreeDescriptor");
            }
            const DftiApi& kept = loaded.emplace(path, api).first->second;
            return kept.complete() ? &kept : nullptr;
        }

    } // namespace

    struct RealFft::Mkl
    {
        const DftiApi* api = nullptr;
        DftiHandle handle = nullptr;

        ~Mkl()
        {
            if (handle != nullptr)
                api->release(&handle);
        }
    };

    struct RealFft::Pocket
    {
        explicit Pocket(size_t length) : plan(length)
        {
        }
        pocketfft::detail::pocketfft_r<double> plan;
    };

    RealFft::RealFft(size_t length, const std::string& mkl_runtime) : length_(length)
    {
        if (!mkl_runtime.empty())
        {
            if (const DftiApi* api = dfti(mkl_runtime))
            {
                std::unique_ptr<Mkl> mkl(new Mkl);
                mkl->api = api;
                bool ok = api->create(&mkl->handle, DFTI_REAL, static_cast<long>(length)) == 0;
                ok = ok && api->set(mkl->handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE) == 0;
                ok = ok &&
                    api->set(mkl->handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX) == 0;
                /* One transform is one window: MKL's own threads would cost
                 * more to wake than the transform takes. */
                ok = ok && api->set(mkl->handle, DFTI_THREAD_LIMIT, 1L) == 0;
                ok = ok && api->commit(mkl->handle) == 0;
                if (ok)
                    mkl_ = std::move(mkl);
            }
        }
        if (!mkl_)
        {
            pocket_.reset(new Pocket(length));
            scratch_.resize(length);
        }
    }

    RealFft::~RealFft() = default;

    void RealFft::forward(const double* in, std::complex<double>* out)
    {
        if (mkl_)
        {
            mkl_->api->forward(mkl_->handle, const_cast<double*>(in), static_cast<void*>(out));
            return;
        }

        /* pocketfft's real plan works in place and leaves the spectrum
         * packed as r0, r1, i1, r2, i2, ... with the Nyquist bin's real part
         * last when the length is even. */
        double* packed = scratch_.data();
        std::copy(in, in + length_, packed);
        pocket_->plan.exec(packed, 1.0, true);
        out[0] = std::complex<double>(packed[0], 0.0);
        size_t bin = 1;
        for (size_t i = 1; i + 1 < length_; i += 2, ++bin)
            out[bin] = std::complex<double>(packed[i], packed[i + 1]);
        if (length_ % 2 == 0 && length_ > 1)
            out[bin] = std::complex<double>(packed[length_ - 1], 0.0);
    }

    const char* RealFft::backend() const
    {
        return mkl_ ? "mkl" : "pocketfft";
    }

} // namespace pulseq
