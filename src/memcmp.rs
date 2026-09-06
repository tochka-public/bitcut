//! Length of the common prefix of two byte slices, SIMD-accelerated.
//!
//! This is the single hottest primitive in the differ: the tiered matcher
//! spends most of its time extending a candidate alignment forward.

#[inline]
pub fn simd_memcmp(a: &[u8], b: &[u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 availability checked at runtime.
            return unsafe { simd_memcmp_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: NEON availability checked at runtime.
            return unsafe { simd_memcmp_neon(a, b) };
        }
    }

    simd_memcmp_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn simd_memcmp_avx2(a: &[u8], b: &[u8]) -> usize {
    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8};

    let len = a.len().min(b.len());
    let mut i: usize = 0;
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    // SAFETY (all intrinsics below): AVX2 is enabled via `target_feature`.
    // pa/pb come from slices of length ≥ len, and `i + 32 ≤ len` is checked
    // by the `while let` guard, so all loads are in-bounds. The `*const u8`
    // → `*const __m256i` cast is sound because `_mm256_loadu_si256` performs
    // unaligned loads.
    while let Some(next) = i.checked_add(32) {
        if next > len {
            break;
        }
        let chunk_a = _mm256_loadu_si256(pa.add(i).cast::<__m256i>());
        let chunk_b = _mm256_loadu_si256(pb.add(i).cast::<__m256i>());
        let cmp = _mm256_cmpeq_epi8(chunk_a, chunk_b);
        let mask_signed = _mm256_movemask_epi8(cmp);
        // i32 → u32 bit-cast (no `as`).
        let mask = u32::from_ne_bytes(mask_signed.to_ne_bytes());

        if mask != u32::MAX {
            let inverted = !mask;
            let diff_index = usize::try_from(inverted.trailing_zeros()).unwrap_or(usize::MAX);
            return i.checked_add(diff_index).unwrap_or(len);
        }
        i = next;
    }

    simd_memcmp_tail(a, b, i, len)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_memcmp_neon(a: &[u8], b: &[u8]) -> usize {
    use std::arch::aarch64::{vceqq_u8, vld1q_u8, vminvq_u8};

    let len = a.len().min(b.len());
    let mut i: usize = 0;
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    // SAFETY (all intrinsics below): NEON is enabled via `target_feature`.
    // pa/pb come from slices of length ≥ len, and `i + 16 ≤ len` is checked
    // by the `while let` guard, so all loads are in-bounds.
    while let Some(next) = i.checked_add(16) {
        if next > len {
            break;
        }
        let chunk_a = vld1q_u8(pa.add(i));
        let chunk_b = vld1q_u8(pb.add(i));
        let cmp = vceqq_u8(chunk_a, chunk_b);
        let min = vminvq_u8(cmp);

        if min != 0xFF_u8 {
            // Find the exact mismatch position via bounded scalar walk.
            let mut j: usize = 0;
            while j < 16 {
                let pos = i.checked_add(j).unwrap_or(len);
                let av = a.get(pos).copied().unwrap_or(0);
                let bv = b.get(pos).copied().unwrap_or(0);
                if av != bv || pos >= len {
                    return pos;
                }
                j = j.checked_add(1).unwrap_or(16);
            }
        }
        i = next;
    }

    simd_memcmp_tail(a, b, i, len)
}

/// Tail of a vectorised comparison. Only the SIMD paths have one; a target
/// with neither reaches the scalar routine directly.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
fn simd_memcmp_tail(a: &[u8], b: &[u8], start: usize, len: usize) -> usize {
    let mut i = start;
    while i < len {
        let av = a.get(i).copied().unwrap_or(0);
        let bv = b.get(i).copied().unwrap_or(0);
        if av != bv {
            return i;
        }
        i = i.checked_add(1).unwrap_or(len);
    }
    i
}

#[inline]
fn simd_memcmp_scalar(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}
