/*
 * kube_storage.c - native packed-cell storage engine for Kube
 *
 * This file exists to keep the core packed-cell runtime in native code even
 * when the renderer/editor is not involved.
 *
 * The Python layer owns:
 * - file-format rules
 * - shape/state validation
 * - higher-level API semantics
 *
 * This native layer owns the hot loops that benefit from crossing the Python/C
 * boundary once per batch instead of once per cell or face:
 * - packed value reads and writes
 * - face extraction and update within one packed cell
 * - batched face reads and writes
 * - contiguous packed-value batch writes
 * - zeroing byte ranges during clear/import flows
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(_WIN32)
#define KUBE_STORAGE_EXPORT __declspec(dllexport)
#else
#define KUBE_STORAGE_EXPORT __attribute__((visibility("default")))
#endif

/* Byte-offset helpers shared by all packed-width specializations. */
static const void *kube_storage_ptr_const(const void *base, int itemsize, size_t index) {
    return (const unsigned char *)base + ((size_t)itemsize * index);
}

static void *kube_storage_ptr(void *base, int itemsize, size_t index) {
    return (unsigned char *)base + ((size_t)itemsize * index);
}

/* Read one packed cell regardless of whether storage uses 8, 16, 32, or 64 bits. */
static uint64_t kube_storage_read_value(const void *base, int itemsize, size_t index) {
    switch (itemsize) {
        case 1:
            return *(const uint8_t *)kube_storage_ptr_const(base, itemsize, index);
        case 2:
            return *(const uint16_t *)kube_storage_ptr_const(base, itemsize, index);
        case 4:
            return *(const uint32_t *)kube_storage_ptr_const(base, itemsize, index);
        case 8:
            return *(const uint64_t *)kube_storage_ptr_const(base, itemsize, index);
        default:
            return 0;
    }
}

/* Write one packed cell regardless of packed storage width. */
static void kube_storage_write_value(void *base, int itemsize, size_t index, uint64_t value) {
    switch (itemsize) {
        case 1:
            *(uint8_t *)kube_storage_ptr(base, itemsize, index) = (uint8_t)value;
            return;
        case 2:
            *(uint16_t *)kube_storage_ptr(base, itemsize, index) = (uint16_t)value;
            return;
        case 4:
            *(uint32_t *)kube_storage_ptr(base, itemsize, index) = (uint32_t)value;
            return;
        case 8:
            *(uint64_t *)kube_storage_ptr(base, itemsize, index) = (uint64_t)value;
            return;
        default:
            return;
    }
}

/* Exported single-value accessors used by the Python runtime for direct cell IO. */
KUBE_STORAGE_EXPORT uint64_t kube_storage_get_packed(const void *base, int itemsize, size_t index) {
    return kube_storage_read_value(base, itemsize, index);
}

KUBE_STORAGE_EXPORT void kube_storage_set_packed(
    void *base,
    int itemsize,
    size_t index,
    uint64_t value
) {
    kube_storage_write_value(base, itemsize, index, value);
}

/*
 * Specialized per-itemsize batch loop bodies.
 *
 * The generic helpers above re-dispatch on `itemsize` inside every loop
 * iteration via `kube_storage_read_value`/`write_value`'s switch statement.
 * For large batches that per-element branch is a real cost (branch
 * misprediction risk, and it blocks the compiler from auto-vectorizing the
 * loop since the element type isn't known at compile time within the loop
 * body). This macro generates one concretely-typed loop per itemsize so the
 * itemsize branch happens once, outside the loop, and each generated loop
 * works directly with its native integer width.
 */
#define KUBE_STORAGE_DEFINE_SET_PACKED_BATCH(SUFFIX, TYPE)                    \
    static void kube_storage_set_packed_batch_##SUFFIX(                      \
        void *base, size_t start_index, const uint64_t *values, size_t count \
    ) {                                                                      \
        TYPE *typed = (TYPE *)base + start_index;                            \
        size_t i;                                                            \
        for (i = 0; i < count; ++i) {                                        \
            typed[i] = (TYPE)values[i];                                      \
        }                                                                     \
    }

KUBE_STORAGE_DEFINE_SET_PACKED_BATCH(u8, uint8_t)
KUBE_STORAGE_DEFINE_SET_PACKED_BATCH(u16, uint16_t)
KUBE_STORAGE_DEFINE_SET_PACKED_BATCH(u32, uint32_t)
KUBE_STORAGE_DEFINE_SET_PACKED_BATCH(u64, uint64_t)

#undef KUBE_STORAGE_DEFINE_SET_PACKED_BATCH

/* Contiguous packed writes are the fast path for encode(), stream import, and shard ingest. */
KUBE_STORAGE_EXPORT void kube_storage_set_packed_batch(
    void *base,
    int itemsize,
    size_t start_index,
    const uint64_t *values,
    size_t count
) {
    switch (itemsize) {
        case 1:
            kube_storage_set_packed_batch_u8(base, start_index, values, count);
            return;
        case 2:
            kube_storage_set_packed_batch_u16(base, start_index, values, count);
            return;
        case 4:
            kube_storage_set_packed_batch_u32(base, start_index, values, count);
            return;
        case 8:
            kube_storage_set_packed_batch_u64(base, start_index, values, count);
            return;
        default:
            return;
    }
}

/* Face access decodes one measurement lane from the packed cell integer. */
KUBE_STORAGE_EXPORT uint32_t kube_storage_get_face(
    const void *base,
    int itemsize,
    size_t index,
    int bits_per_face,
    int face_idx
) {
    uint64_t packed = kube_storage_read_value(base, itemsize, index);
    uint64_t mask = ((uint64_t)1 << bits_per_face) - 1U;
    return (uint32_t)((packed >> (face_idx * bits_per_face)) & mask);
}

/* Specialized per-itemsize batch face-read loop bodies (see the batch-write comment above). */
#define KUBE_STORAGE_DEFINE_GET_FACES_BATCH(SUFFIX, TYPE)                      \
    static void kube_storage_get_faces_batch_##SUFFIX(                        \
        const void *base, const size_t *indices, const uint8_t *face_indices, \
        size_t count, int bits_per_face, uint32_t *out_values                 \
    ) {                                                                       \
        const TYPE *typed = (const TYPE *)base;                              \
        uint64_t mask = ((uint64_t)1 << bits_per_face) - 1U;                  \
        size_t i;                                                             \
        for (i = 0; i < count; ++i) {                                         \
            uint64_t packed = (uint64_t)typed[indices[i]];                    \
            out_values[i] = (uint32_t)((packed >> (face_indices[i] * bits_per_face)) & mask); \
        }                                                                      \
    }

KUBE_STORAGE_DEFINE_GET_FACES_BATCH(u8, uint8_t)
KUBE_STORAGE_DEFINE_GET_FACES_BATCH(u16, uint16_t)
KUBE_STORAGE_DEFINE_GET_FACES_BATCH(u32, uint32_t)
KUBE_STORAGE_DEFINE_GET_FACES_BATCH(u64, uint64_t)

#undef KUBE_STORAGE_DEFINE_GET_FACES_BATCH

/* Batch face reads amortize bridge overhead for sparse random-inspection workloads. */
KUBE_STORAGE_EXPORT void kube_storage_get_faces_batch(
    const void *base,
    int itemsize,
    const size_t *indices,
    const uint8_t *face_indices,
    size_t count,
    int bits_per_face,
    uint32_t *out_values
) {
    switch (itemsize) {
        case 1:
            kube_storage_get_faces_batch_u8(base, indices, face_indices, count, bits_per_face, out_values);
            return;
        case 2:
            kube_storage_get_faces_batch_u16(base, indices, face_indices, count, bits_per_face, out_values);
            return;
        case 4:
            kube_storage_get_faces_batch_u32(base, indices, face_indices, count, bits_per_face, out_values);
            return;
        case 8:
            kube_storage_get_faces_batch_u64(base, indices, face_indices, count, bits_per_face, out_values);
            return;
        default:
            return;
    }
}

/* Single-face updates rewrite only the requested measurement lane. */
KUBE_STORAGE_EXPORT void kube_storage_set_face(
    void *base,
    int itemsize,
    size_t index,
    int bits_per_face,
    int face_idx,
    uint32_t value
) {
    uint64_t packed = kube_storage_read_value(base, itemsize, index);
    uint64_t mask = ((uint64_t)1 << bits_per_face) - 1U;
    uint64_t shift = (uint64_t)(face_idx * bits_per_face);
    packed &= ~(mask << shift);
    packed |= ((uint64_t)value & mask) << shift;
    kube_storage_write_value(base, itemsize, index, packed);
}

/* Specialized per-itemsize batch face-write loop bodies (see the batch-write comment above). */
#define KUBE_STORAGE_DEFINE_SET_FACES_BATCH(SUFFIX, TYPE)                      \
    static void kube_storage_set_faces_batch_##SUFFIX(                        \
        void *base, const size_t *indices, const uint8_t *face_indices,       \
        const uint32_t *values, size_t count, int bits_per_face               \
    ) {                                                                       \
        TYPE *typed = (TYPE *)base;                                          \
        uint64_t mask = ((uint64_t)1 << bits_per_face) - 1U;                  \
        size_t i;                                                             \
        for (i = 0; i < count; ++i) {                                         \
            uint64_t packed = (uint64_t)typed[indices[i]];                    \
            uint64_t shift = (uint64_t)(face_indices[i] * bits_per_face);     \
            packed &= ~(mask << shift);                                      \
            packed |= ((uint64_t)values[i] & mask) << shift;                  \
            typed[indices[i]] = (TYPE)packed;                                \
        }                                                                      \
    }

KUBE_STORAGE_DEFINE_SET_FACES_BATCH(u8, uint8_t)
KUBE_STORAGE_DEFINE_SET_FACES_BATCH(u16, uint16_t)
KUBE_STORAGE_DEFINE_SET_FACES_BATCH(u32, uint32_t)
KUBE_STORAGE_DEFINE_SET_FACES_BATCH(u64, uint64_t)

#undef KUBE_STORAGE_DEFINE_SET_FACES_BATCH

/* Batch face updates are the write-side counterpart to batch face reads. */
KUBE_STORAGE_EXPORT void kube_storage_set_faces_batch(
    void *base,
    int itemsize,
    const size_t *indices,
    const uint8_t *face_indices,
    const uint32_t *values,
    size_t count,
    int bits_per_face
) {
    switch (itemsize) {
        case 1:
            kube_storage_set_faces_batch_u8(base, indices, face_indices, values, count, bits_per_face);
            return;
        case 2:
            kube_storage_set_faces_batch_u16(base, indices, face_indices, values, count, bits_per_face);
            return;
        case 4:
            kube_storage_set_faces_batch_u32(base, indices, face_indices, values, count, bits_per_face);
            return;
        case 8:
            kube_storage_set_faces_batch_u64(base, indices, face_indices, values, count, bits_per_face);
            return;
        default:
            return;
    }
}

/* Byte-range clearing is used by full clears and trailing unused-capacity cleanup. */
KUBE_STORAGE_EXPORT void kube_storage_zero_bytes(void *base, size_t byte_offset, size_t byte_count) {
    memset((unsigned char *)base + byte_offset, 0, byte_count);
}
