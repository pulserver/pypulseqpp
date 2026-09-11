/**
 * @file sequence.hpp
 * @brief Pulseq event libraries, block storage and sequence operations.
 *
 * Library IDs are 1-based; a block's zero ID means no event. Trapezoid and
 * arbitrary gradients share file IDs but occupy separate fixed-width tables.
 * The signed grad_slot_ mapping selects the corresponding table.
 */

#ifndef PULSEQ_CXX_SEQUENCE_HPP
#define PULSEQ_CXX_SEQUENCE_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <unordered_map>
#include "pulseq/write.hpp"

#include "pulseq/shape.hpp"

namespace pulseq
{

    /* ================================================================== */
    /*  Column counts                                                     */
    /* ================================================================== */

    /** amplitude, mag_shape, phase_shape, time_shape, center, delay,
     *  freq_ppm, phase_ppm, freq, phase */
    constexpr int RF_WIDTH = 10;
    /** amplitude, rise, flat, fall, delay */
    constexpr int TRAP_WIDTH = 5;
    /** amplitude, first, last, amp_shape, time_shape, delay */
    constexpr int ARB_WIDTH = 6;
    /** num, dwell, delay, freq_ppm, phase_ppm, freq, phase, phase_shape */
    constexpr int ADC_WIDTH = 8;
    /** type, channel, delay, duration */
    constexpr int TRIGGER_WIDTH = 4;
    /** quaternion, scalar first, canonical */
    constexpr int ROTATION_WIDTH = 4;
    /** type, ref, next */
    constexpr int EXTENSION_WIDTH = 3;
    /** value, label id */
    constexpr int LABEL_WIDTH = 2;
    /**
     * rf, gx, gy, gz, adc, ext -- the block columns a file row carries.
     *
     * The first six of the block table's columns, in the order the format
     * writes them, so a writer walks a row without knowing what else is
     * stored beside it.
     */
    constexpr int BLOCK_FILE_COLUMNS = 6;

    /**
     * The block table's columns: the six a file carries, then the promoted.
     *
     * A rotation and an RF shim are extensions, and they stay extensions --
     * they are written and read as ones, and the chain still names them, so
     * nothing about the format changes. They are *also* columns here because
     * of how often they are asked for, and how early: expanding a waveform,
     * weighing a slew rate and following a trajectory all need to know
     * whether a block turns its gradients before they can do anything with
     * them, and anything that materialises a pulse needs its shim the same
     * way. Walking an extension chain to find out costs a pointer chase per
     * block. One column answers it.
     */
    constexpr int BLOCK_ROTATION_COLUMN = 6;
    constexpr int BLOCK_SHIM_COLUMN = 7;
    constexpr int BLOCK_WIDTH = 8;

    /**
     * The rotation a quaternion stands for, as a matrix.
     *
     * A rotation is stored as a unit quaternion, scalar first, because that
     * is what the format writes and what composes cleanly. What anything
     * using it wants is the matrix: a rotated block plays a sum of its three
     * gradients on each axis, and that sum is a row of this.
     *
     * @param q     Four doubles, w then x, y, z.
     * @param into  Filled with the matrix.
     */
    inline void rotation_matrix(const double* q, double into[3][3])
    {
        const double w = q[0];
        const double x = q[1];
        const double y = q[2];
        const double z = q[3];
        into[0][0] = 1.0 - 2.0 * (y * y + z * z);
        into[0][1] = 2.0 * (x * y - w * z);
        into[0][2] = 2.0 * (x * z + w * y);
        into[1][0] = 2.0 * (x * y + w * z);
        into[1][1] = 1.0 - 2.0 * (x * x + z * z);
        into[1][2] = 2.0 * (y * z - w * x);
        into[2][0] = 2.0 * (x * z - w * y);
        into[2][1] = 2.0 * (y * z + w * x);
        into[2][2] = 1.0 - 2.0 * (x * x + y * y);
    }

    /** Turn @p vector by @p matrix, in place. */
    inline void rotate(const double matrix[3][3], double vector[3])
    {
        const double x = vector[0];
        const double y = vector[1];
        const double z = vector[2];
        for (int axis = 0; axis < 3; ++axis)
            vector[axis] =
                matrix[axis][0] * x + matrix[axis][1] * y + matrix[axis][2] * z;
    }

    /* ================================================================== */
    /*  Tables                                                            */
    /* ================================================================== */

    /**
     * A library of fixed-width rows, stored as one flat array.
     *
     * Ids are 1-based and handed out in append order, which is what makes a
     * written file reproducible: PyPulseq numbers by first appearance too, so
     * the same sequence built the same way comes out with the same ids.
     */
    template <typename T> class BasicTable
    {
    public:
        explicit BasicTable(int width) : width_(width)
        {
        }

        int width() const
        {
            return width_;
        }
        int size() const
        {
            return width_ ? static_cast<int>(values_.size()) / width_ : 0;
        }
        bool empty() const
        {
            return values_.empty();
        }

        /** Row @p id, 1-based.  Unchecked; callers hold ids they were given. */
        const T* row(int id) const
        {
            return values_.data() + static_cast<size_t>(id - 1) * width_;
        }
        T* row(int id)
        {
            return values_.data() + static_cast<size_t>(id - 1) * width_;
        }

        const T* data() const
        {
            return values_.data();
        }
        T* data()
        {
            return values_.data();
        }

        /** Append one row.  @return its 1-based id. */
        int append(const T* values)
        {
            values_.insert(values_.end(), values, values + width_);
            return size();
        }

        void reserve(int rows)
        {
            values_.reserve(static_cast<size_t>(rows) * width_);
        }

        /** Grow or shrink to @p rows, zero-filling anything new.  Bulk loading
         *  (a composed scan handed over whole) resizes once and writes in. */
        void resize(int rows)
        {
            values_.resize(static_cast<size_t>(rows) * width_, T{});
        }

        void clear()
        {
            values_.clear();
        }

    private:
        std::vector<T> values_;
        int width_;
    };

    using Table = BasicTable<double>;
    using IntTable = BasicTable<int32_t>;

    /**
     * A vector that is handed out as a view and copied before it is written.
     *
     * The block table is read back as a NumPy array pointing straight into it,
     * which is what keeps a million-row table free to read a column out of.
     * The array holds a share of the buffer, so growing or rewriting the table
     * copies it first and the array keeps reading what it was given: a
     * snapshot, never a pointer into memory that has been freed.
     *
     * Copying one of these copies the buffer, so two sequences never share a
     * table; the only thing that ever shares one is a view.
     */
    template <typename T> class CowVector
    {
    public:
        CowVector() : data_(std::make_shared<std::vector<T>>())
        {
        }
        CowVector(const CowVector& other)
            : data_(std::make_shared<std::vector<T>>(*other.data_))
        {
        }
        CowVector& operator=(const CowVector& other)
        {
            if (this != &other)
                data_ = std::make_shared<std::vector<T>>(*other.data_);
            return *this;
        }
        CowVector(CowVector&&) noexcept = default;
        CowVector& operator=(CowVector&&) noexcept = default;

        std::vector<T>* operator->()
        {
            return data_.get();
        }
        const std::vector<T>* operator->() const
        {
            return data_.get();
        }
        std::vector<T>& operator*()
        {
            return *data_;
        }
        const std::vector<T>& operator*() const
        {
            return *data_;
        }

        /** The buffer, for a caller handing out a view that must outlive us. */
        std::shared_ptr<const std::vector<T>> buffer() const
        {
            return data_;
        }

        /** Take a private copy if anything else holds this buffer. */
        void detach()
        {
            if (data_.use_count() > 1)
                data_ = std::make_shared<std::vector<T>>(*data_);
        }

    private:
        std::shared_ptr<std::vector<T>> data_;
    };

    /**
     * A library whose rows differ in length: shapes, and pTx shim vectors.
     *
     * Kept as one sample array plus a row-start index, so a row is a pointer
     * and a length rather than a vector of its own.
     */
    class RaggedTable
    {
    public:
        RaggedTable() = default;
        RaggedTable(const RaggedTable& other)
        {
            copy_from(other);
        }
        RaggedTable& operator=(const RaggedTable& other)
        {
            if (this != &other)
                copy_from(other);
            return *this;
        }
        RaggedTable(RaggedTable&&) noexcept = default;
        RaggedTable& operator=(RaggedTable&&) noexcept = default;

        int size() const
        {
            return static_cast<int>(len_.size());
        }
        bool empty() const
        {
            return size() == 0;
        }

        int length(int id) const
        {
            return len_[static_cast<size_t>(id) - 1];
        }
        const double* row(int id) const
        {
            return at(start_[static_cast<size_t>(id) - 1]);
        }
        double* row(int id)
        {
            return at(start_[static_cast<size_t>(id) - 1]);
        }

        /** Make room for @p rows rows; the values need no reservation, a
         *  chunk is allocated once and never moved. */
        void reserve(int rows, size_t /*values*/)
        {
            start_.reserve(static_cast<size_t>(rows));
            len_.reserve(static_cast<size_t>(rows));
        }

        /** Append a row of @p count values each divided by @p divisor (1 when
         *  it is 0).  @return its 1-based id. */
        int append_divided(const double* values, int count, double divisor)
        {
            const int64_t where = place(count);
            double* dst = at(where);
            if (divisor != 0.0)
                for (int i = 0; i < count; ++i)
                    dst[i] = values[i] / divisor;
            else if (count > 0)
                std::memcpy(dst, values, static_cast<size_t>(count) * sizeof(double));
            start_.push_back(where);
            len_.push_back(count);
            return size();
        }
        /** Append a row of @p count values.  @return its 1-based id. */
        int append(const double* values, int count)
        {
            const int64_t where = place(count);
            if (count > 0)
                std::memcpy(at(where), values, static_cast<size_t>(count) * sizeof(double));
            start_.push_back(where);
            len_.push_back(count);
            return size();
        }

        void clear()
        {
            chunks_.clear();
            bases_.clear();
            caps_.clear();
            start_.clear();
            len_.clear();
            cursor_ = 0;
        }

        /**
         * Keep the rows flagged in @p keep (indexed by id - 1), in order;
         * @p new_id (indexed by id) receives every row's new id, 0 for one
         * dropped. Rows are reached through their offsets, so nothing moves:
         * a dropped row's bytes stay in their chunk until the table is
         * cleared, and the survivors keep their places.
         */
        void compact(const uint8_t* keep, int32_t* new_id)
        {
            const int total = size();
            int kept = 0;
            for (int id = 1; id <= total; ++id)
            {
                if (!keep[id - 1])
                {
                    new_id[id] = 0;
                    continue;
                }
                start_[static_cast<size_t>(kept)] = start_[static_cast<size_t>(id) - 1];
                len_[static_cast<size_t>(kept)] = len_[static_cast<size_t>(id) - 1];
                new_id[id] = ++kept;
            }
            start_.resize(static_cast<size_t>(kept));
            len_.resize(static_cast<size_t>(kept));
        }


    private:
        /** Rows live in chunks allocated once: a chunk fills until the next
         *  row would not fit, and a row longer than a chunk gets one of its
         *  own. Offsets are into one virtual span the chunks tile. */
        static constexpr int64_t kChunkValues = int64_t{1} << 22;

        double* at(int64_t offset)
        {
            const size_t i = chunk_of(offset);
            return chunks_[i].get() + (offset - bases_[i]);
        }
        const double* at(int64_t offset) const
        {
            const size_t i = chunk_of(offset);
            return chunks_[i].get() + (offset - bases_[i]);
        }
        size_t chunk_of(int64_t offset) const
        {
            const auto it = std::upper_bound(bases_.begin(), bases_.end(), offset);
            return static_cast<size_t>(it - bases_.begin()) - 1;
        }
        int64_t place(int count)
        {
            if (!chunks_.empty() && cursor_ - bases_.back() + count <= caps_.back())
            {
                const int64_t where = cursor_;
                cursor_ += count;
                return where;
            }
            const int64_t cap = count > kChunkValues ? count : kChunkValues;
            chunks_.push_back(make_chunk(cap));
            bases_.push_back(cursor_);
            caps_.push_back(cap);
            const int64_t where = cursor_;
            cursor_ += count;
            return where;
        }
        void copy_from(const RaggedTable& other)
        {
            clear();
            reserve(other.size(), 0);
            for (int id = 1; id <= other.size(); ++id)
                append(other.row(id), other.length(id));
        }

        /** A chunk's storage: an anonymous mapping advised onto huge pages
         *  where the platform has them (a 32 MB chunk is then a handful of
         *  page faults instead of eight thousand), plain heap otherwise. */
        struct ChunkDeleter
        {
            size_t bytes = 0;
            bool mapped = false;
            void operator()(double* p) const;
        };
        using Chunk = std::unique_ptr<double[], ChunkDeleter>;
        static Chunk make_chunk(int64_t cap);

        std::vector<Chunk> chunks_;
        std::vector<int64_t> bases_;
        std::vector<int64_t> caps_;
        std::vector<int64_t> start_;
        std::vector<int32_t> len_;
        int64_t cursor_ = 0;
    };

    /**
     * What a shape is played as, as a bitmask.
     *
     * A `[SHAPES]` entry does not say what it is. The file says so only where
     * an event refers to it, so answering "every gradient waveform" or "every
     * RF envelope" means walking the event libraries and following their shape
     * columns. Recording it where the reference is made costs one OR per
     * reference and answers the question directly, and a file read back fills
     * it in on the way past because reading registers its events too.
     *
     * It is a mask rather than a tag because deduplication merges shapes that
     * hold the same numbers: a gradient waveform and an RF envelope can be one
     * entry, and after the merge it is both.
     */
    enum ShapeRole : uint32_t
    {
        SHAPE_ROLE_NONE = 0u,
        SHAPE_ROLE_RF_MAGNITUDE = 1u << 0,
        SHAPE_ROLE_RF_PHASE = 1u << 1,
        SHAPE_ROLE_RF_TIME = 1u << 2,
        SHAPE_ROLE_GRADIENT = 1u << 3,
        SHAPE_ROLE_GRADIENT_TIME = 1u << 4,
        SHAPE_ROLE_ADC_PHASE = 1u << 5,
        /** Either time array, for a caller that does not care which. */
        SHAPE_ROLE_TIME = SHAPE_ROLE_RF_TIME | SHAPE_ROLE_GRADIENT_TIME,
    };

    class ShapeLibrary
    {
    public:
        int size() const
        {
            return data_.size();
        }
        bool empty() const
        {
            return size() == 0;
        }

        int num_uncompressed(int id) const
        {
            return num_uncompressed_[id - 1];
        }
        int num_compressed(int id) const
        {
            return data_.length(id);
        }
        const double* samples(int id) const
        {
            return data_.row(id);
        }
        /** What @p id is played as: a mask of ShapeRole. */
        uint32_t roles(int id) const
        {
            return roles_[id - 1];
        }
        /** Record that @p id is played as @p role too.  Id 0 means no shape. */
        void mark(int id, uint32_t role)
        {
            if (id > 0)
                roles_[static_cast<size_t>(id) - 1] |= role;
        }

        /** Append a shape already in its compressed form.  @return its id. */
        int append(int num_uncompressed, const double* samples, int count);

        /**
         * Append a shape as it stands, to be compressed later.
         *
         * The registration path a scan takes when it is going to deduplicate
         * anyway: compressing is a pass over every sample, and a sequence that
         * registers a waveform per shot registers far more of them than it
         * keeps.  See `compress`.
         */
        int append_raw(const double* samples, int count);
        /** As append_raw for samples that are @p divisor times the row to
         *  store, @p divisor their signed peak: the row's peak is 1 without
         *  a scan. A divisor of 1 stores the samples as they are, scanned. */
        int append_raw_divided(const double* samples, int count, double divisor);

        /** Encode every shape still held raw.  @return whether any row
         *  changed: a shape the encoding would not shorten is kept as it is,
         *  so a library of such shapes is untouched.  Idempotent. */
        bool compress();

        /**
         * Keep only the shapes @p first maps onto themselves (first[id] ==
         * id), in id order, renumbering densely.  @return for every old id
         * the id its first appearance now has.
         */
        std::vector<int32_t> keep_first_appearances(const std::vector<int32_t>& first);

        void clear()
        {
            num_uncompressed_.clear();
            is_compressed_.clear();
            roles_.clear();
            data_.clear();
        }

    private:
        std::vector<int32_t> num_uncompressed_;
        std::vector<uint8_t> is_compressed_;
        /** Per shape, a mask of ShapeRole; filled where a reference is made. */
        std::vector<uint32_t> roles_;
        RaggedTable data_;
    };

    /**
     * Every shape decompressed at most once.
     *
     * A shape is stored run-length encoded, and anything that wants the
     * samples themselves -- expanding a waveform, weighing a slew rate --
     * wants them once per shape however many events name it. A readout
     * played a hundred thousand times names one shape, and decoding it per
     * block is the whole cost of the pass.
     *
     * Held beside the library rather than in it: what the library keeps is
     * what a file holds, and the decoded samples are several times larger.
     */
    class ShapeCache
    {
    public:
        explicit ShapeCache(const ShapeLibrary& library)
            : library_(library), held_(static_cast<size_t>(library.size()) + 1)
        {
        }

        /** The samples of shape @p id, decoded on first asking.  Id 0, and
         *  any id the library does not have, is empty. */
        const std::vector<double>& operator[](int id)
        {
            if (id < 1 || id > library_.size())
                return empty_;
            std::vector<double>& samples = held_[static_cast<size_t>(id)];
            if (samples.empty())
                samples = decompress_shape(
                    library_.samples(id),
                    library_.num_compressed(id),
                    library_.num_uncompressed(id));
            return samples;
        }

    private:
        const ShapeLibrary& library_;
        std::vector<std::vector<double>> held_;
        std::vector<double> empty_;
    };

    /**
     * What applying soft delay values found.
     *
     * The pass is over the block table rather than over decoded blocks, so
     * what it reports is the little that a caller has to be told about: which
     * delays the sequence carries, where a duration had to be moved onto the
     * raster, and the first thing that was wrong. The wording is left to the
     * caller, which is where the toolbox's own messages live.
     */
    struct SoftDelayReport
    {
        /** What a duration was rounded by to reach the block raster. */
        struct Rounding
        {
            int block = 0;
            std::string hint;
            int32_t num = 0;
            double error = 0.0;
        };

        /** What stopped the pass, if anything did. */
        enum class Problem
        {
            None,
            /** One hint under two numbers. */
            HintRenumbered,
            /** One number under two hints. */
            NumberRenamed,
            /** The value asked for makes the block last less than nothing. */
            Negative
        };

        /** Every hint the sequence carries, in the order first seen. */
        std::vector<std::string> hints;
        std::vector<Rounding> rounded;

        Problem problem = Problem::None;
        int block = 0;
        std::string hint;
        int32_t num = 0;
        double duration = 0.0;
        double offset = 0.0;
        double factor = 0.0;
    };

    /**
     * The repeating unit of a scan, in blocks.
     *
     * A scan is a handful of things played over and over with different
     * numbers in them, and the stream of block definition ids is where that
     * shows: a gradient echo reads 1 2 3 4 1 2 3 4 whatever its phase encode
     * is doing. The scan is the longest stretch ending at the last block that
     * repeats at least twice, and this is its smallest period and where it
     * starts; the blocks before `start` are the prologue. An outer loop whose
     * iterations each carry their own preparation -- a slice with its dummy
     * shots -- is therefore one repetition, not a prologue followed by the
     * lines of the last slice.
     *
     * A `size` of zero means no repetition was found, which is the honest
     * answer for a sequence that plays each position once.
     */
    struct Repetition
    {
        int size = 0;
        int start = 0;
    };

    /** One soft-delay row: a numeric id, an offset, a factor, and a hint name. */
    struct SoftDelay
    {
        int32_t num = 0;
        double offset = 0.0;
        double factor = 0.0;
        std::string hint;
    };

    /* ================================================================== */
    /*  Definitions                                                       */
    /* ================================================================== */

    /**
     * One `[DEFINITIONS]` entry.
     *
     * Whether a value is text, whole numbers or reals is not decoration: the
     * text writer formats each differently and the binary format tags them,
     * so a definition read as an integer has to be written back as one.
     */
    class Definition
    {
    public:
        enum class Kind
        {
            Text,
            Int,
            Real
        };

        Definition() = default;
        explicit Definition(std::string text) : kind_(Kind::Text), text_(std::move(text))
        {
        }
        explicit Definition(double value) : kind_(Kind::Real), numbers_{value}
        {
        }
        explicit Definition(std::vector<double> values)
            : kind_(Kind::Real), numbers_(std::move(values))
        {
        }
        static Definition integers(std::vector<double> values)
        {
            Definition d;
            d.kind_ = Kind::Int;
            d.numbers_ = std::move(values);
            return d;
        }

        Kind kind() const
        {
            return kind_;
        }
        const std::string& text() const
        {
            return text_;
        }
        const std::vector<double>& numbers() const
        {
            return numbers_;
        }

    private:
        Kind kind_ = Kind::Real;
        std::string text_;
        std::vector<double> numbers_;
    };

    /* ================================================================== */
    /*  Blocks                                                            */
    /* ================================================================== */

    /** One block's event ids and its duration in seconds.  0 means absent. */
    struct Block
    {
        int32_t rf = 0;
        int32_t gx = 0;
        int32_t gy = 0;
        int32_t gz = 0;
        int32_t adc = 0;
        int32_t ext = 0;
        /** The rotation row this block turns its gradients by, and the shim
         *  row its pulse is played through; 0 for none. Both are also in the
         *  extension chain, which is what the file carries. */
        int32_t rot = 0;
        int32_t shim = 0;
        double duration = 0.0;
    };

    /** What a gradient id resolves to. */
    enum class GradKind
    {
        None,
        Trap,
        Arbitrary
    };

    /* ================================================================== */
    /*  Definitions and instances                                         */
    /* ================================================================== */

    /**
     * Interned key for the event parameters fixed across playouts.
     *
     * RF shape IDs belong to the definition; gradient waveform IDs belong to
     * the instance. Each event kind has a separate table and key layout.
     */
    struct DefKey
    {
        std::array<uint64_t, 4> w{};
        bool operator==(const DefKey& other) const
        {
            return w == other.w;
        }
    };

    struct DefKeyHash
    {
        size_t operator()(const DefKey& key) const noexcept
        {
            uint64_t h = 1469598103934665603ull;
            for (uint64_t x : key.w)
            {
                h ^= x;
                h *= 1099511628211ull;
                h ^= h >> 29;
            }
            return static_cast<size_t>(h);
        }
    };

    /** A time, in nanoseconds: finer than any raster, and exact on one. */
    inline uint64_t nanos(double seconds)
    {
        return static_cast<uint64_t>(std::llround(seconds * 1e9));
    }
    /** A count, as itself. */
    inline uint64_t whole(double value)
    {
        return static_cast<uint64_t>(std::llround(value));
    }
    inline uint64_t pack(uint64_t high, uint64_t low)
    {
        return (high << 32) | (low & 0xffffffffull);
    }

    /** magnitude, phase and time shapes; delay; center; use. */
    inline DefKey rf_key(const double* row, char use)
    {
        DefKey key;
        key.w[0] = pack(whole(row[1]), whole(row[2]));
        key.w[1] = pack(whole(row[3]), static_cast<uint64_t>(use));
        key.w[2] = nanos(row[5]);
        key.w[3] = nanos(row[4]);
        return key;
    }
    /** rise, flat and fall times; delay. */
    inline DefKey trap_key(const double* row)
    {
        DefKey key;
        key.w[0] = 1;
        key.w[1] = pack(nanos(row[1]), nanos(row[2]));
        key.w[2] = pack(nanos(row[3]), nanos(row[4]));
        return key;
    }
    /** The time shape and the delay.  The waveform belongs to the instance. */
    inline DefKey arb_key(const double* row)
    {
        DefKey key;
        key.w[0] = 2;
        key.w[1] = whole(row[4]);
        key.w[2] = nanos(row[5]);
        return key;
    }
    /** Sample count, dwell and delay.  The phase modulation is per instance. */
    inline DefKey adc_key(const double* row)
    {
        DefKey key;
        key.w[0] = whole(row[0]);
        key.w[1] = nanos(row[1]);
        key.w[2] = nanos(row[2]);
        return key;
    }
    /**
     * The definitions a block plays, and how long it lasts.
     *
     * A block that plays something lasts as long as its longest event, or as
     * long as the duration it was given if that is longer and it is padded
     * out; either way the duration follows from what is in the block, so it
     * belongs to the definition.
     *
     * The ADC is left out, so a preparation shot playing the imaging shot's
     * gradients with the digitiser off is the same definition as the shot it
     * stands in for, and a position digitised two ways still repeats every
     * shot rather than every pair. So are labels and rotations: those are
     * things one playout does, not a different block.
     */
    inline DefKey block_key(int32_t rf, int32_t gx, int32_t gy, int32_t gz, double duration)
    {
        DefKey key;
        key.w[0] = pack(static_cast<uint64_t>(rf), static_cast<uint64_t>(gx));
        key.w[1] = pack(static_cast<uint64_t>(gy), static_cast<uint64_t>(gz));
        key.w[2] = nanos(duration);
        return key;
    }

    /**
     * Every pure delay, which is one definition.
     *
     * A block with no RF, no gradient, no ADC and no trigger or digital
     * output plays nothing, and an interpreter sets how long it waits there
     * at run time. Its duration is therefore a per-playout parameter and not
     * part of what it is: a TI fill and a TR pad that vary shot to shot are
     * one position waited at, not a sequence that changes. Labels, flags and
     * a rotation may be present; none of them makes the block play anything.
     *
     * `block_key` leaves the last word at zero, so setting it here is what
     * keeps a delay from ever colliding with a block that plays something.
     */
    inline DefKey delay_key()
    {
        DefKey key;
        key.w[3] = 1;
        return key;
    }

    /**
     * Interns keys, handing out dense 1-based ids in order of first appearance.
     *
     * A design loop cycles through a handful of definitions and asks about
     * them millions of times, so a small direct-mapped cache stands in front
     * of the map: the common answer is a comparison rather than a hash lookup.
     */
    class Definitions
    {
    public:
        int32_t intern(const DefKey& key)
        {
            Slot& slot = cache_[DefKeyHash{}(key) & (CACHE - 1)];
            if (slot.id != 0 && slot.key == key)
                return slot.id;
            const int32_t id =
                index_.try_emplace(key, static_cast<int32_t>(index_.size() + 1)).first->second;
            slot.key = key;
            slot.id = id;
            return id;
        }
        int size() const
        {
            return static_cast<int>(index_.size());
        }
        void clear()
        {
            index_.clear();
            cache_.fill(Slot{});
        }

    private:
        static constexpr size_t CACHE = 64;
        struct Slot
        {
            DefKey key;
            int32_t id = 0;
        };
        std::array<Slot, CACHE> cache_{};
        std::unordered_map<DefKey, int32_t, DefKeyHash> index_;
    };

    /**
     * One block's per-playout parameters, by column:
     *
     *   0,1   gx amplitude, gx waveform shape (0 for a trapezoid)
     *   2,3   gy amplitude, gy waveform shape
     *   4,5   gz amplitude, gz waveform shape
     *   6-10  rf amplitude, freq, phase, freq_ppm, phase_ppm
     *  11-15  adc freq, phase, freq_ppm, phase_ppm, phase modulation shape
     */
    constexpr int INSTANCE_WIDTH = 16;

    /* ================================================================== */
    /*  The sequence                                                      */
    /* ================================================================== */

    class Sequence
    {
    public:
        Sequence();

        /**
         * Copyable by value, and deliberately so.
         *
         * Every member below is a plain container, so the compiler's copy is
         * both correct and about as cheap as a copy of this much data can
         * be -- one allocation and one memcpy per library.  That is what lets
         * a transform offer "give me the result, leave the original alone"
         * without a bespoke clone that would have to be kept in step with
         * every field added here.  Anything added to this class that owns a
         * resource by raw pointer would silently break that guarantee, which
         * is why these are spelled out rather than left implicit.
         */
        Sequence(const Sequence&) = default;
        Sequence& operator=(const Sequence&) = default;
        Sequence(Sequence&&) noexcept = default;
        Sequence& operator=(Sequence&&) noexcept = default;

        /* -- version ---------------------------------------------------- */

        int version_major() const
        {
            return version_major_;
        }
        int version_minor() const
        {
            return version_minor_;
        }
        int version_revision() const
        {
            return version_revision_;
        }
        void set_version(int major, int minor, int revision);

        /* -- rasters ---------------------------------------------------- */

        double rf_raster_time() const
        {
            return rf_raster_;
        }
        double grad_raster_time() const
        {
            return grad_raster_;
        }
        double adc_raster_time() const
        {
            return adc_raster_;
        }
        double block_duration_raster() const
        {
            return block_raster_;
        }
        void set_rasters(double rf, double grad, double adc, double block);

        /**
         * Record the four rasters in `[DEFINITIONS]`, unless already recorded.
         *
         * Block durations are serialised as raster ticks, so a reader that
         * cannot see the raster cannot recover the seconds: it falls back to
         * a default, and every time in the file comes out scaled. The writers
         * call this.
         */
        void publish_rasters();

        /* -- definitions ------------------------------------------------ */

        /** Sorted by key, which is the order the writers emit them in. */
        const std::map<std::string, Definition>& definitions() const
        {
            return definitions_;
        }
        void set_definition(const std::string& key, Definition value);
        const Definition* definition(const std::string& key) const;

        /* -- extension type registry ------------------------------------ */

        /**
         * The numeric id an extension name is written under.
         *
         * Ids are per sequence and assigned on first use, exactly as PyPulseq
         * assigns them, because the file declares the mapping itself: an
         * `extension ROTATIONS 3` line says what 3 means in *this* file.
         */
        int extension_type_id(const std::string& name);
        /** Look up without assigning; 0 if the name is not registered. */
        int find_extension_type_id(const std::string& name) const;
        /** The name an id was registered under, or empty. */
        const std::string& extension_type_name(int id) const;
        /** Force a name to a given id, as reading a file does. */
        void set_extension_type_id(const std::string& name, int id);

        /* -- label registry ---------------------------------------------- */
        /*
         * A label id means nothing on its own -- the file writes the *name*,
         * and the number is only how this sequence happens to have indexed it.
         * That matters because the numbering is not agreed anywhere: PyPulseq
         * knows 22 labels, the C reader's table knows 24, and a research
         * sequence may use a name neither has heard of.  So the sequence keeps
         * its own table, seeded with Pulseq's built-ins in Pulseq's order, and
         * a name outside them is appended.  Reading and writing both go
         * through the name, so the two never have to agree on a number.
         */

        /** Pulseq's built-in label names, in the order that numbers them. */
        static const std::vector<std::string>& builtin_labels();

        /** Id for @p name, appending it to this sequence's table if new. */
        int label_id(const std::string& name);
        /**
         * Id for @p name without appending; 0 if this sequence has no such
         * label.
         *
         * What `label_id` cannot answer: a name nothing in the sequence
         * carries has no id, and asking for one should not invent it.
         */
        int find_label_id(const std::string& name) const;
        /** The name id @p id was registered under, or empty. */
        const std::string& label_name(int id) const;
        /** Whether @p id names something outside Pulseq's own table. */
        bool is_custom_label(int id) const;

        /* -- event registration ----------------------------------------- */
        /*
         * These append unconditionally rather than searching for an equal row.
         * A scan is built once and deduplicated once, at the end, over whole
         * columns -- searching per event would make building quadratic in the
         * thing that is already the largest.
         */

        int register_rf(const double* row, char use);
        int register_trap(const double* row);
        int register_arbitrary(const double* row);
        int register_adc(const double* row);
        int register_trigger(const double* row);
        int register_rotation(const double* row);
        int register_label_set(int32_t value, int32_t label_id);
        int register_label_inc(int32_t value, int32_t label_id);
        int register_rf_shim(const double* values, int count);
        int register_soft_delay(const SoftDelay& row);

        /**
         * The number @p hint is addressed by, assigning one if it is new.
         *
         * A soft delay is named in a design script and numbered in the file,
         * and the number is the sequence's to hand out: every block naming
         * the same hint has to get the same one, and no two hints may share.
         *
         * Asked where an event is registered, not where a row is stored: a
         * file being read carries the numbers it was written with, and
         * nothing may renumber those.
         *
         * @param hint       What the delay is called.
         * @param requested  The number asked for, or a negative one to be
         *                   given the next free.
         * @throws std::invalid_argument if @p hint already has a different
         *         number, or @p requested already belongs to another hint.
         */
        int32_t soft_delay_number(const std::string& hint, int32_t requested);

        /**
         * Set soft-delay block durations from values keyed by hint.
         *
         * Duration is value / factor + offset, rounded to the block raster.
         * Updates are incremental: an error leaves earlier blocks modified.
         * @return Validation findings; see SoftDelayReport.
         */
        SoftDelayReport apply_soft_delays(const std::map<std::string, double>& values);

        /**
         * Infer undefined RF uses from flip angle, duration and frequency offset.
         * Existing uses are preserved. Legacy files have no RF-use column.
         *
         * @param b0     Field strength in tesla, for the fat offset.
         * @param gamma  Gyromagnetic ratio in Hz/T.
         * @return Number of pulses assigned a use.
         */
        int detect_rf_uses(double b0, double gamma);

        /**
         * The repeating unit of the scan, found once and remembered.
         *
         * Cheap because the structural fork has already done the hard part:
         * two blocks playing the same things for the same length share a
         * definition id whatever their amplitudes, so finding the repeat is
         * finding the period of an array of integers rather than comparing
         * blocks event by event.
         *
         * Adding or rewriting a block makes the answer stale, and the next
         * call works it out again.
         */
        Repetition repetition();

        /**
         * Where a repeating unit of @p size starts, if it repeats at all.
         *
         * For a caller who already knows the period -- a file that records
         * it, a protocol that fixes it -- and wants to know how much of the
         * sequence is prologue. Returns a size of zero if the stream does not
         * in fact repeat with that period.
         */
        Repetition locate_repetition(int size) const;
        int register_shape(int num_uncompressed, const double* samples, int count);

        /**
         * Register a waveform as it stands, to be compressed before writing.
         *
         * See ShapeLibrary::append_raw.  A sequence built this way must reach
         * a writer (or `compress_shapes`) before its shapes mean what the file
         * format says they mean.
         */
        int register_raw_shape(const double* samples, int count);
        /** See ShapeLibrary::append_raw_divided. */
        int register_raw_shape_divided(const double* samples, int count, double divisor);

        /** Compress every shape still held raw.  Idempotent; writers call it. */
        void compress_shapes();

        /**
         * Append an extension-chain link.
         *
         * A chain is a singly-linked list through this library, so a block
         * carrying two extensions owns two rows of it.  Unlike the event
         * libraries this one is searched: chains are shared far more often
         * than they are distinct (every readout in a scan carries the same
         * pair of labels), and a scan-length library that deduplicates to
         * three rows is worth the lookup.
         */
        int chain_extension(int32_t type_id, int32_t ref, int32_t next);

        /**
         * Append a chain link without looking for an equal one.
         *
         * The counterpart to `chain_extension` for a caller that is going to
         * `remove_duplicates` anyway: the search is what makes chaining cost
         * more than appending, and a scan that labels every TR pays it a
         * million times to be told what the final pass would work out in one.
         *
         * Mixing the two is allowed and simply leaves duplicate links behind,
         * which deduplication collapses like any other repeated row.
         */
        int append_extension(int32_t type_id, int32_t ref, int32_t next);

        /* -- blocks ------------------------------------------------------ */

        int num_blocks() const
        {
            return static_cast<int>(durations_->size());
        }

        /** Append @p block.  @return its 1-based index. */
        int add_block(const Block& block);

        /* -- definitions and instances --------------------------------- */
        int num_block_definitions() const
        {
            return block_defs_.size();
        }
        int num_rf_definitions() const
        {
            return rf_defs_.size();
        }
        /** Per RF id, the id of the definition it plays. */
        const std::vector<int32_t>& rf_definitions() const
        {
            return rf_def_;
        }
        int num_grad_definitions() const
        {
            return grad_defs_.size();
        }
        int num_adc_definitions() const
        {
            return adc_defs_.size();
        }
        /** Per block, the id of the definition it plays. */
        const std::vector<int32_t>& instance_definitions() const
        {
            return instance_def_;
        }
        /** Per block, the ADC definition it digitises with; 0 if it does not. */
        const std::vector<int32_t>& instance_adc_definitions() const
        {
            return instance_adc_def_;
        }
        /**
         * Per block, INSTANCE_WIDTH per-playout parameters.
         *
         * Read out of the event libraries on demand rather than stored a
         * second time: an amplitude is already a column of the row the block
         * names, and a scan is not carried twice.
         */
        std::vector<double> instance_parameters() const;
        /**
         * Re-derive every definition from the libraries as they now stand.
         *
         * Deduplication merges library rows and renumbers them, so the ids a
         * key was built from move and the per-event definition arrays shrink.
         * Re-interning from the surviving rows is both the remap and the
         * shrink, and it cannot disagree with what registration would have
         * produced because it is the same code.
         */
        void rebuild_definitions();

        /** Overwrite block @p index (1-based).  Throws if out of range. */
        void set_block(int index, const Block& block);

        /** Block @p index (1-based).  Throws if out of range. */
        Block get_block(int index) const;

        /** Total playing time, the sum of the block durations. */
        double duration() const;

        /**
         * Scale every gradient played on @p axis by @p modifier.
         *
         * Only the amplitude moves: the ramps, the delay and the shape stay
         * as they are, so the definition a gradient belongs to is the one it
         * belonged to before and nothing has to be re-derived. An arbitrary
         * gradient's stored first and last samples scale with it, since those
         * are amplitudes too.
         *
         * @param axis      0, 1 or 2 for x, y or z.
         * @param modifier  What to multiply by; -1 inverts, 0 silences.
         * @throws std::runtime_error if a gradient row is played on this axis
         *         and on another, where there is no one answer.
         */
        void scale_gradient_axis(int axis, double modifier);

        /**
         * How many blocks carry an event in each column of the block table.
         *
         * One pass over the integer columns, which is what makes it worth
         * asking of a million-block scan at all: the same count taken in
         * Python builds a boolean array the size of the table first.
         */
        std::array<int64_t, BLOCK_FILE_COLUMNS> event_counts() const;

        /* -- deduplication ------------------------------------------------ */

        /**
         * Collapse every library onto its distinct rows and renumber what
         * points at them.  In place; the caller copies first if it wants both.
         *
         * `register_*` appends unconditionally, so a scan built block by block
         * arrives here holding one row per *use* of an event -- a readout
         * repeated a hundred thousand times is a hundred thousand identical
         * gradient rows.  This is the pass that makes it a file.
         *
         * Two rows are the same row when they agree at the precision the `.seq`
         * format writes that column at, which is why the surviving row is the
         * *rounded* one: keeping more precision than the file records would
         * mean the sequence read back differs from the one written.  See
         * dedup.cpp for the per-library profiles.
         */
        void remove_duplicates();

        /**
         * Whether every library is already down to its distinct rows.
         *
         * Set by @ref remove_duplicates and cleared by anything that could
         * undo it -- registering a row, editing the block table, or taking a
         * mutable reference to a library.  It is deliberately pessimistic: a
         * caller that asks for write access and then writes nothing loses the
         * claim, because the alternative is a claim that can be wrong, and a
         * wrong one is a file full of rows the writer was told it need not
         * collapse.
         *
         * Worth stating rather than re-deriving because deduplication is a
         * pass over every row of every library, and the ordinary path
         * deduplicates once and then writes.
         */
        bool deduplicated() const
        {
            return deduplicated_;
        }

        /**
         * How many times the sequence has been edited.
         *
         * Rises on every mutation, in step with the claim above being
         * dropped, and never falls.  Anything worked out about a sequence and
         * kept -- what the gradients slew at, how long the whole thing lasts
         * -- can record the number it was worked out at and tell, in one
         * comparison, whether it is still about the sequence in hand.
         *
         * Nothing to do with @ref version_revision, which is the revision of
         * the file format.
         */
        uint64_t edits() const
        {
            return edits_;
        }

        /**
         * Hand over a whole block table at once.
         *
         * A composed scan arrives as columns already -- that is what it was
         * built as -- so this takes them as they are rather than making the
         * caller replay millions of `add_block` calls to arrive back at the
         * same arrays.  @p events is row-major, BLOCK_WIDTH per block.
         */

        /** Raw block table, row-major, BLOCK_WIDTH per block. */
        const int32_t* block_events() const
        {
            return blocks_->data();
        }
        int32_t* block_events()
        {
            changed();
            detach_blocks();
            return blocks_->data();
        }
        const double* block_durations() const
        {
            return durations_->data();
        }
        double* block_durations()
        {
            changed();
            detach_blocks();
            return durations_->data();
        }

        /** The block table's buffers, for a caller that hands out a view over
         *  them and needs them to outlive this sequence. */
        std::shared_ptr<const std::vector<int32_t>> block_events_buffer() const
        {
            return blocks_.buffer();
        }
        std::shared_ptr<const std::vector<double>> block_durations_buffer() const
        {
            return durations_.buffer();
        }

        /* -- gradients --------------------------------------------------- */

        /** What gradient id @p id is, and which row of which table holds it. */
        GradKind grad_kind(int id) const;
        /** Row index into the trapezoid or arbitrary table, 1-based. */
        int grad_row(int id) const;
        int num_gradients() const
        {
            return static_cast<int>(grad_slot_.size());
        }

        /* -- libraries --------------------------------------------------- */

        const Table& rf_library() const
        {
            return rf_;
        }
        Table& rf_library()
        {
            changed();
            return rf_;
        }
        const std::vector<char>& rf_uses() const
        {
            return rf_use_;
        }
        std::vector<char>& rf_uses()
        {
            changed();
            return rf_use_;
        }

        const Table& trap_library() const
        {
            return trap_;
        }
        Table& trap_library()
        {
            changed();
            return trap_;
        }
        const Table& arb_library() const
        {
            return arb_;
        }
        Table& arb_library()
        {
            changed();
            return arb_;
        }
        const Table& adc_library() const
        {
            return adc_;
        }
        Table& adc_library()
        {
            changed();
            return adc_;
        }
        const Table& trigger_library() const
        {
            return trigger_;
        }
        Table& trigger_library()
        {
            changed();
            return trigger_;
        }
        const Table& rotation_library() const
        {
            return rotation_;
        }
        Table& rotation_library()
        {
            changed();
            return rotation_;
        }

        const IntTable& extensions_library() const
        {
            return extensions_;
        }
        IntTable& extensions_library()
        {
            changed();
            return extensions_;
        }
        const IntTable& label_set_library() const
        {
            return label_set_;
        }
        IntTable& label_set_library()
        {
            changed();
            return label_set_;
        }
        const IntTable& label_inc_library() const
        {
            return label_inc_;
        }
        IntTable& label_inc_library()
        {
            changed();
            return label_inc_;
        }

        const RaggedTable& rf_shim_library() const
        {
            return rf_shim_;
        }
        RaggedTable& rf_shim_library()
        {
            changed();
            return rf_shim_;
        }
        const ShapeLibrary& shape_library() const
        {
            return shapes_;
        }
        ShapeLibrary& shape_library()
        {
            changed();
            return shapes_;
        }
        const std::vector<SoftDelay>& soft_delay_library() const
        {
            return soft_delays_;
        }
        std::vector<SoftDelay>& soft_delay_library()
        {
            changed();
            return soft_delays_;
        }

    private:
        /* What a sequence built here is, until a file it is read from says
         * otherwise: the revision this package writes. */
        int version_major_ = 1;
        int version_minor_ = 5;
        int version_revision_ = WRITTEN_REVISION;

        double rf_raster_ = 1e-6;
        double grad_raster_ = 10e-6;
        double adc_raster_ = 100e-9;
        double block_raster_ = 10e-6;

        std::map<std::string, Definition> definitions_;

        std::map<std::string, int> extension_ids_;
        std::map<int, std::string> extension_names_;

        /** Label names by id-1; seeded with builtin_labels() and appended to. */
        std::vector<std::string> label_names_;
        std::map<std::string, int> label_ids_;

        Table rf_{RF_WIDTH};
        std::vector<char> rf_use_;
        Table trap_{TRAP_WIDTH};
        Table arb_{ARB_WIDTH};
        Table adc_{ADC_WIDTH};
        Table trigger_{TRIGGER_WIDTH};
        Table rotation_{ROTATION_WIDTH};
        IntTable extensions_{EXTENSION_WIDTH};
        IntTable label_set_{LABEL_WIDTH};
        IntTable label_inc_{LABEL_WIDTH};
        RaggedTable rf_shim_;
        ShapeLibrary shapes_;
        std::vector<SoftDelay> soft_delays_;
        /** What each soft delay hint is numbered as. */
        std::map<std::string, int32_t> soft_delay_hints_;

        /** grad id (1-based) -> +trap row / -arb row.  See the file comment. */
        std::vector<int32_t> grad_slot_;

        /** Extension chain rows by value, so a repeated chain costs one row. */
        std::map<std::array<int32_t, EXTENSION_WIDTH>, int> chain_index_;

        CowVector<int32_t> blocks_;
        CowVector<double> durations_;

        /** Take a private copy of the block table if anything else holds it. */
        void detach_blocks()
        {
            blocks_.detach();
            durations_.detach();
        }

        /** Detach only if appending one block would move the table.
         *
         * Appending cannot reach a view unless it reallocates, and a capacity
         * comparison is a plain load where checking for a shared buffer is an
         * atomic one. This is on the per-block path, so it is the comparison
         * that runs every time and the atomic that runs only when the table
         * grows.
         */
        void detach_blocks_before_growth()
        {
            if (blocks_->capacity() - blocks_->size() < static_cast<size_t>(BLOCK_WIDTH) ||
                durations_->capacity() == durations_->size())
                detach_blocks();
        }

        /** See deduplicated(); false until remove_duplicates() says otherwise. */
        /** Note a change: what was worked out about the sequence is stale. */
        void changed()
        {
            deduplicated_ = false;
            ++edits_;
        }

        bool deduplicated_ = false;
        uint64_t edits_ = 0;

        /* -- definitions and instances --------------------------------- */
        Definitions rf_defs_, grad_defs_, adc_defs_, block_defs_;
        std::vector<int32_t> rf_def_;           /**< by RF id - 1 */
        std::vector<int32_t> grad_def_;         /**< by gradient id - 1 */
        std::vector<int32_t> adc_def_;          /**< by ADC id - 1 */
        std::vector<int32_t> instance_def_;     /**< by block - 1 */
        /** The repeating unit, once someone has asked for it. */
        Repetition repetition_;
        bool repetition_known_ = false;
        std::vector<int32_t> instance_adc_def_; /**< by block - 1 */

        /**
         * Per extension chain node, whether it or anything below it is a
         * trigger or a digital output.  Both travel as `TRIGGERS`, so one
         * flag answers for both, and a chain is built tail first, so a node's
         * answer is its own type or the answer already recorded for `next`.
         */
        std::vector<uint8_t> chain_carries_trigger_;

        /** Refill the block table's promoted columns from the chains. */
        void refill_block_promotions();

        /** Refill chain_carries_trigger_ from the chains as they stand. */
        void recompute_chain_triggers();
        /** Re-derive the block definitions, and only those. */
        void refork_blocks();

        /**
         * The id of the `TRIGGERS` type, or 0 while nothing has claimed it.
         *
         * Held rather than looked up: a scan that labels every TR appends a
         * chain node per block, and a lookup by name is a string comparison
         * down a map on a path that runs a million times.
         */
        int trigger_type_id_ = 0;

        /**
         * An extension type that is also a column of the block table.
         *
         * What a chain names is read off as the chain is built, so storing a
         * block costs a lookup rather than a walk and a block asked later
         * which way it turns, or what shim it plays through, reads a column.
         * Held per type rather than named one by one so that promoting a
         * third is this list and the width.
         */
        struct Promoted
        {
            const char* name;   /**< the type's name in the file */
            int column;         /**< the block table column it fills */
            int type_id = 0;    /**< its id here; 0 while nothing claims it */
            /** Per chain node, the row the chain from there names; 0 none. */
            std::vector<int32_t> named;
        };

        std::array<Promoted, 2> promoted_{
            {{"ROTATIONS", BLOCK_ROTATION_COLUMN},
             {"RF_SHIMS", BLOCK_SHIM_COLUMN}}};

        /** Refill every promoted column's chain cache as the chains stand. */
        void recompute_chain_promotions();

        /** The row promoted type @p which is named at by chain head @p ext. */
        int32_t promoted_in_chain(size_t which, int32_t ext) const
        {
            const std::vector<int32_t>& named = promoted_[which].named;
            const size_t node = static_cast<size_t>(ext) - 1;
            return ext >= 1 && node < named.size() ? named[node] : 0;
        }

        /** Note a chain node's trigger flag and promotions as it is appended. */
        void note_chain(int32_t type_id, int32_t ref, int32_t next)
        {
            const size_t behind = static_cast<size_t>(next) - 1;
            for (Promoted& column : promoted_)
            {
                column.named.push_back(
                    column.type_id != 0 && type_id == column.type_id
                        ? ref
                        : (next >= 1 && behind < column.named.size()
                               ? column.named[behind]
                               : 0));
            }

            // A tail this node cannot see is read as carrying one, on the
            // same grounds as is_pure_delay: too coarse is the answer that
            // merges blocks, and too fine is the one that merely splits them.
            const size_t tail = static_cast<size_t>(next) - 1;
            const bool carries =
                (trigger_type_id_ != 0 && type_id == trigger_type_id_) ||
                (next >= 1 &&
                 (tail >= chain_carries_trigger_.size() || chain_carries_trigger_[tail] != 0));
            chain_carries_trigger_.push_back(carries ? 1 : 0);
        }

        /** The definition @p id was interned as, or 0 where there is no event. */
        static int32_t definition_of(int32_t id, const std::vector<int32_t>& defs)
        {
            return id > 0 && static_cast<size_t>(id) <= defs.size()
                       ? defs[static_cast<size_t>(id) - 1]
                       : 0;
        }
        /**
         * Whether @p block plays nothing at all.
         *
         * Reads the answer for its extension chain out of
         * `chain_carries_trigger_` rather than walking it, so this stays a
         * handful of comparisons on the per-block path.
         */
        bool is_pure_delay(const Block& block) const
        {
            if (block.rf || block.gx || block.gy || block.gz || block.adc)
                return false;
            if (block.ext <= 0)
                return true;
            // A chain that has not been registered is a corrupt sequence,
            // which deduplication says so about. Until then it is read as
            // playing something, because that keeps the duration in the key
            // and so cannot merge two blocks that are not one.
            const size_t node = static_cast<size_t>(block.ext) - 1;
            return node < chain_carries_trigger_.size() && !chain_carries_trigger_[node];
        }
        /** One block's definition id and the ADC definition it plays with. */
        void fork_instance(const Block& block, int32_t& def, int32_t& adc_def) const;
        /** One block's per-playout parameters, INSTANCE_WIDTH of them. */
        void instance_row(const Block& block, double* params) const;
    };

} // namespace pulseq

#endif /* PULSEQ_CXX_SEQUENCE_HPP */
