//! `bitcut` — create and apply binary patches.
//!
//! The library exposes a small, allocation-friendly API:
//!
//! * [`make_patch`] / [`make_patch_into`] — produce a patch describing how to
//!   reconstruct `new` from `old`.
//! * [`apply_patch`] / [`apply_patch_into`] — apply a patch to `old` and
//!   recover `new`.
//! * [`Op`] / [`OpIter`] — low-level access to the patch opcode stream.
//!
//! ## Patch format
//!
//! A patch is a sequence of opcodes:
//!
//! ```text
//! Copy: 0x00 offset:u32_le len:u32_le
//! Add : 0x01 len:u32_le bytes...
//! ```
//!
//! All integers are little-endian. The format limits both `old` and `new` to
//! `u32::MAX` (≈ 4 GiB) bytes.

use rustc_hash::{FxBuildHasher, FxHashMap};
use std::fmt;

const WINDOW_SIZE: usize = 10;
const HASH_BASE: u64 = 1_934_123_457;

const TAG_COPY: u8 = 0x00;
const TAG_ADD: u8 = 0x01;
const ADD_HEADER_LEN: usize = 5;

/// Errors produced by the patch deserializer / applier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchError {
    /// Patch ended in the middle of an opcode.
    UnexpectedEof,
    /// Opcode tag is neither `Copy` nor `Add`.
    InvalidOpcode(u8),
    /// `Copy` references a range outside `old`.
    CopyOutOfBounds {
        /// Requested offset.
        offset: u32,
        /// Requested length.
        len: u32,
        /// Actual length of `old`.
        old_len: usize,
    },
    /// Arithmetic overflow while computing a copy/add range. Indicates a
    /// crafted or corrupted patch.
    Overflow,
    /// `make_patch` was called with an input larger than `u32::MAX` bytes.
    InputTooLarge {
        /// The offending length.
        len: usize,
    },
}

impl fmt::Display for PatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            PatchError::UnexpectedEof => f.write_str("unexpected end of patch"),
            PatchError::InvalidOpcode(tag) => write!(f, "invalid opcode tag: 0x{tag:02X}"),
            PatchError::CopyOutOfBounds {
                offset,
                len,
                old_len,
            } => write!(
                f,
                "copy out of bounds: offset={offset} len={len} old_len={old_len}"
            ),
            PatchError::Overflow => f.write_str("arithmetic overflow in patch range"),
            PatchError::InputTooLarge { len } => {
                write!(f, "input too large for patch format: {len} bytes")
            }
        }
    }
}

impl std::error::Error for PatchError {}

/// A single patch opcode.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op<'a> {
    /// Copy `len` bytes from `old` starting at `offset`.
    Copy {
        /// Offset within `old`.
        offset: u32,
        /// Number of bytes to copy.
        len: u32,
    },
    /// Append a literal byte run.
    Add(&'a [u8]),
}

impl fmt::Debug for Op<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        enum Content<'a> {
            Text(&'a str),
            Bytes(&'a [u8]),
        }

        impl<'a> From<&'a [u8]> for Content<'a> {
            fn from(value: &'a [u8]) -> Self {
                match std::str::from_utf8(value) {
                    Ok(s) => Content::Text(s),
                    Err(_) => Content::Bytes(value),
                }
            }
        }

        impl fmt::Debug for Content<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    Content::Text(s) => write!(f, "Text({s:?})"),
                    Content::Bytes(b) => {
                        write!(f, "Bytes(")?;
                        for (i, byte) in b.iter().enumerate() {
                            if i > 0 {
                                write!(f, " ")?;
                            }
                            write!(f, "{byte:02X}")?;
                        }
                        write!(f, ")")
                    }
                }
            }
        }

        match *self {
            Self::Copy { offset, len } => f.debug_tuple("Copy").field(&offset).field(&len).finish(),
            Self::Add(content) => f.debug_tuple("Add").field(&Content::from(content)).finish(),
        }
    }
}

impl<'a> Op<'a> {
    /// Serialize this op into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`PatchError::InputTooLarge`] if an `Add` payload exceeds
    /// `u32::MAX` bytes.
    pub fn serialize_to(&self, out: &mut Vec<u8>) -> Result<(), PatchError> {
        match *self {
            Op::Copy { offset, len } => {
                out.push(TAG_COPY);
                out.extend_from_slice(&offset.to_le_bytes());
                out.extend_from_slice(&len.to_le_bytes());
                Ok(())
            }
            Op::Add(bytes) => {
                let len = u32::try_from(bytes.len())
                    .map_err(|_| PatchError::InputTooLarge { len: bytes.len() })?;
                out.push(TAG_ADD);
                out.extend_from_slice(&len.to_le_bytes());
                out.extend_from_slice(bytes);
                Ok(())
            }
        }
    }

    /// Parse the next opcode from `input`, returning the op and the remaining
    /// bytes.
    ///
    /// # Errors
    ///
    /// See [`PatchError`] variants.
    pub fn deserialize(input: &'a [u8]) -> Result<(Self, &'a [u8]), PatchError> {
        let (&tag, rest) = input.split_first().ok_or(PatchError::UnexpectedEof)?;
        match tag {
            TAG_COPY => {
                let (offset_bytes, rest) = rest
                    .split_first_chunk::<4>()
                    .ok_or(PatchError::UnexpectedEof)?;
                let (len_bytes, rest) = rest
                    .split_first_chunk::<4>()
                    .ok_or(PatchError::UnexpectedEof)?;
                Ok((
                    Op::Copy {
                        offset: u32::from_le_bytes(*offset_bytes),
                        len: u32::from_le_bytes(*len_bytes),
                    },
                    rest,
                ))
            }
            TAG_ADD => {
                let (len_bytes, rest) = rest
                    .split_first_chunk::<4>()
                    .ok_or(PatchError::UnexpectedEof)?;
                let declared = u32::from_le_bytes(*len_bytes);
                let len = usize::try_from(declared).map_err(|_| PatchError::Overflow)?;
                let (payload, rest) = rest
                    .split_at_checked(len)
                    .ok_or(PatchError::UnexpectedEof)?;
                Ok((Op::Add(payload), rest))
            }
            other => Err(PatchError::InvalidOpcode(other)),
        }
    }

    /// Iterate the opcodes in `patch` lazily, without allocating a `Vec`.
    #[must_use]
    pub fn iter(patch: &'a [u8]) -> OpIter<'a> {
        OpIter { input: patch }
    }
}

/// Iterator over the opcodes of a patch.
///
/// Yields `Err` once and then stops on the first malformed opcode.
#[derive(Debug, Clone)]
pub struct OpIter<'a> {
    input: &'a [u8],
}

impl<'a> Iterator for OpIter<'a> {
    type Item = Result<Op<'a>, PatchError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.input.is_empty() {
            return None;
        }
        match Op::deserialize(self.input) {
            Ok((op, rest)) => {
                self.input = rest;
                Some(Ok(op))
            }
            Err(e) => {
                // Stop iteration after the first error.
                self.input = &[];
                Some(Err(e))
            }
        }
    }
}

/// Build a patch describing how to reconstruct `new` from `old`.
///
/// # Errors
///
/// Returns [`PatchError::InputTooLarge`] if either `old` or `new` exceeds
/// `u32::MAX` bytes.
pub fn make_patch(old: &[u8], new: &[u8]) -> Result<Vec<u8>, PatchError> {
    let mut out = Vec::with_capacity(estimate_patch_capacity(new.len()));
    make_patch_into(old, new, &mut out)?;
    Ok(out)
}

/// Build a patch into a caller-supplied buffer. The buffer is **not**
/// cleared first; ops are appended.
///
/// # Errors
///
/// Returns [`PatchError::InputTooLarge`] if either input exceeds `u32::MAX`
/// bytes.
pub fn make_patch_into(old: &[u8], new: &[u8], out: &mut Vec<u8>) -> Result<(), PatchError> {
    if u32::try_from(old.len()).is_err() {
        return Err(PatchError::InputTooLarge { len: old.len() });
    }
    if u32::try_from(new.len()).is_err() {
        return Err(PatchError::InputTooLarge { len: new.len() });
    }

    // Inputs too small to roll a window: emit `new` verbatim.
    if old.len() < WINDOW_SIZE || new.len() < WINDOW_SIZE {
        return Op::Add(new).serialize_to(out);
    }

    let map = build_hash_map(old);

    // SAFETY (length): pre-checked above.
    let Some(mut rolling) = RollingHash::new(new, WINDOW_SIZE, HASH_BASE) else {
        return Op::Add(new).serialize_to(out);
    };

    let mut last_emitted: usize = 0;
    let mut idx: usize = 0;

    while let Some(hash) = rolling.next() {
        if let Some(&match_pos) = map.get(&hash) {
            let new_tail = new.get(idx..).unwrap_or(&[]);
            let old_tail = old.get(match_pos..).unwrap_or(&[]);
            let match_len = simd_memcmp(new_tail, old_tail);
            if match_len >= WINDOW_SIZE {
                if let Some(skipped) = new.get(last_emitted..idx) {
                    if !skipped.is_empty() {
                        Op::Add(skipped).serialize_to(out)?;
                    }
                }
                let copy_offset = u32::try_from(match_pos).map_err(|_| PatchError::Overflow)?;
                let copy_len = u32::try_from(match_len).map_err(|_| PatchError::Overflow)?;
                Op::Copy {
                    offset: copy_offset,
                    len: copy_len,
                }
                .serialize_to(out)?;
                idx = idx.checked_add(match_len).ok_or(PatchError::Overflow)?;
                last_emitted = idx;

                // Advance rolling hash past the match (we already consumed 1
                // step via .next() above, so skip `match_len - 1` more).
                // `nth(n)` consumes `n + 1` elements, hence `checked_sub(2)`.
                if let Some(skip) = match_len.checked_sub(2) {
                    let _ = rolling.nth(skip);
                }
                continue;
            }
        }
        idx = idx.checked_add(1).ok_or(PatchError::Overflow)?;
    }

    if let Some(tail) = new.get(last_emitted..) {
        if !tail.is_empty() {
            Op::Add(tail).serialize_to(out)?;
        }
    }
    Ok(())
}

#[inline]
fn estimate_patch_capacity(new_len: usize) -> usize {
    // Heuristic: a perfectly-matching patch is ~9 bytes, a fully-literal one
    // is `new_len + 5`. Pick a small floor.
    new_len.saturating_add(ADD_HEADER_LEN).clamp(64, 4096)
}

/// Apply `patch` to `old` and return the reconstructed bytes.
///
/// # Errors
///
/// See [`PatchError`] variants. The applier rejects out-of-bounds copies
/// and overflows; it never panics.
pub fn apply_patch(old: &[u8], patch: &[u8]) -> Result<Vec<u8>, PatchError> {
    let mut out = Vec::with_capacity(estimate_apply_capacity(patch.len()));
    apply_patch_into(old, patch, &mut out)?;
    Ok(out)
}

/// Apply `patch` into a caller-supplied buffer. The buffer is **not**
/// cleared first; reconstructed bytes are appended.
///
/// # Errors
///
/// See [`PatchError`] variants.
pub fn apply_patch_into(old: &[u8], patch: &[u8], out: &mut Vec<u8>) -> Result<(), PatchError> {
    for op in Op::iter(patch) {
        match op? {
            Op::Copy { offset, len } => {
                let start = usize::try_from(offset).map_err(|_| PatchError::Overflow)?;
                let len_usize = usize::try_from(len).map_err(|_| PatchError::Overflow)?;
                let end = start.checked_add(len_usize).ok_or(PatchError::Overflow)?;
                let slice = old.get(start..end).ok_or(PatchError::CopyOutOfBounds {
                    offset,
                    len,
                    old_len: old.len(),
                })?;
                out.extend_from_slice(slice);
            }
            Op::Add(bytes) => out.extend_from_slice(bytes),
        }
    }
    Ok(())
}

#[inline]
fn estimate_apply_capacity(patch_len: usize) -> usize {
    patch_len.saturating_mul(2).clamp(64, 1 << 20)
}

#[inline]
fn simd_memcmp(a: &[u8], b: &[u8]) -> usize {
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

#[inline]
fn window_hash(data: &[u8], base: u64) -> (u64, u64) {
    let mut hash: u64 = 0;
    let mut base_pow: u64 = 1;
    let last = data.len().saturating_sub(1);
    for (i, &byte) in data.iter().enumerate() {
        hash = hash.wrapping_mul(base).wrapping_add(u64::from(byte));
        if i < last {
            base_pow = base_pow.wrapping_mul(base);
        }
    }
    (hash, base_pow)
}

fn build_hash_map(data: &[u8]) -> FxHashMap<u64, usize> {
    let Some(rh) = RollingHash::new(data, WINDOW_SIZE, HASH_BASE) else {
        return FxHashMap::default();
    };
    let cap = data
        .len()
        .saturating_sub(WINDOW_SIZE)
        .saturating_add(1_usize);
    let mut map: FxHashMap<u64, usize> = FxHashMap::with_capacity_and_hasher(cap, FxBuildHasher);
    for (i, h) in rh.enumerate() {
        // Keep the *first* occurrence — earlier positions tend to yield
        // longer matches and more compact patches.
        map.entry(h).or_insert(i);
    }
    map
}

/// Iterator over rolling hashes of fixed-size windows.
pub struct RollingHash<'a> {
    data: &'a [u8],
    pos: usize,
    window_size: usize,
    base_pow: u64,
    hash: u64,
    base: u64,
}

impl<'a> RollingHash<'a> {
    /// Create a rolling hash iterator over `data`.
    ///
    /// Returns `None` if `data` is shorter than `window_size`.
    #[must_use]
    pub fn new(data: &'a [u8], window_size: usize, base: u64) -> Option<Self> {
        let initial = data.get(..window_size)?;
        let (hash, base_pow) = window_hash(initial, base);
        Some(Self {
            data,
            pos: 0,
            window_size,
            base_pow,
            hash,
            base,
        })
    }
}

impl Iterator for RollingHash<'_> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let end = self.pos.checked_add(self.window_size)?;
        if end > self.data.len() {
            return None;
        }

        let result = self.hash;

        // Roll the hash forward to the next window, if any.
        if let (Some(&out_byte), Some(&in_byte)) = (self.data.get(self.pos), self.data.get(end)) {
            self.hash = self
                .hash
                .wrapping_sub(u64::from(out_byte).wrapping_mul(self.base_pow))
                .wrapping_mul(self.base)
                .wrapping_add(u64::from(in_byte));
        }

        self.pos = self.pos.checked_add(1)?;
        Some(result)
    }
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::arithmetic_side_effects,
    clippy::default_numeric_fallback,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::as_conversions,
    clippy::missing_panics_doc,
    clippy::panic,
    clippy::type_complexity
)]
mod tests {
    use super::*;
    use rstest::rstest;
    use std::collections::HashSet;

    #[rstest]
    #[case::identical_digits(
        "identical_digits",
        (0_u8..100).collect::<Vec<_>>(),
        (0_u8..100).collect::<Vec<_>>(),
    )]
    #[case::shifted_digits(
        "shifted_digits",
        (0_u8..100).collect::<Vec<_>>(),
        (0_u8..100).skip(2).chain([0_u8, 1]).collect::<Vec<_>>(),
    )]
    #[case::empty_to_long("empty_to_long", vec![], vec![1, 2, 3, 4, 5, 6, 7, 8])]
    #[case::empty_to_short("empty_to_short", vec![], vec![1, 2, 3])]
    #[case::empty_to_empty("empty_to_empty", vec![], vec![])]
    #[case::nonempty_to_empty("nonempty_to_empty", vec![1, 2, 3, 4, 5, 6, 7, 8], vec![])]
    #[case::reorder_with_add(
        "reorder_with_add",
        b"-foo-baar-hello-world".to_vec(),
        b"hello-world-foo-baar-baz".to_vec(),
    )]
    #[case::reorder_no_add(
        "reorder_no_add",
        b"just-swaps-with-no-adds".to_vec(),
        b"-with-no-addsjust-swaps".to_vec(),
    )]
    fn roundtrip_table(#[case] name: &str, #[case] old: Vec<u8>, #[case] new: Vec<u8>) {
        let patch = make_patch(&old, &new).unwrap();
        let ops = Op::iter(&patch).collect::<Result<Vec<_>, _>>().unwrap();
        insta::assert_debug_snapshot!(name, ops);
        let patched = apply_patch(&old, &patch).unwrap();
        assert_eq!(patched, new);
    }

    #[test]
    fn rolling_hash_collision_check() {
        const WND: usize = 3;
        let data = vec![1_u8, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 8];
        let rh = RollingHash::new(&data, WND, 1_934_123_457).unwrap();
        let hashes: Vec<_> = data
            .as_slice()
            .windows(WND)
            .enumerate()
            .zip(rh)
            .filter(|((_, wnd), _)| wnd == &[2, 3, 4])
            .map(|((_, _), hash)| hash)
            .collect();
        assert!(hashes.len() > 1);
        assert_eq!(hashes.into_iter().collect::<HashSet<_>>().len(), 1);
    }

    #[rstest]
    #[case::long_matching_block(
        "long_matching_block",
        b"abcdefghij12345abcdefghij".to_vec(),
        b"abcdefghij12345abcdefghij".to_vec(),
    )]
    #[case::insertion_inside_long_block(
        "insertion_inside_long_block",
        b"abcdefghij12345abcdefghij".to_vec(),
        {
            let mut v = b"abcdefghij12345abcdefghij".to_vec();
            v.insert(WINDOW_SIZE + 2, b'X');
            v
        },
    )]
    #[case::deletion_inside_long_block(
        "deletion_inside_long_block",
        b"abcdefghij12345abcdefghij".to_vec(),
        {
            let mut v = b"abcdefghij12345abcdefghij".to_vec();
            v.remove(WINDOW_SIZE + 2);
            v
        },
    )]
    #[case::replacement_inside_long_block(
        "replacement_inside_long_block",
        b"abcdefghij12345abcdefghij".to_vec(),
        {
            let mut v = b"abcdefghij12345abcdefghij".to_vec();
            v[WINDOW_SIZE + 2] = b'Z';
            v
        },
    )]
    #[case::repeated_windows(
        "repeated_windows",
        b"abcdefghij".repeat(5),
        b"abcdefghij".repeat(5),
    )]
    #[case::add_at_start_copy_after_window(
        "add_at_start_copy_after_window",
        b"abcdefghij12345".to_vec(),
        b"ZZZabcdefghij12345".to_vec(),
    )]
    #[case::add_at_end_copy_at_start(
        "add_at_end_copy_at_start",
        b"abcdefghij12345".to_vec(),
        b"abcdefghij12345YYY".to_vec(),
    )]
    #[case::add_at_window_boundary(
        "add_at_window_boundary",
        b"abcdefghij12345abcdefghij".to_vec(),
        b"abcdefghij12345Xabcdefghij".to_vec(),
    )]
    fn edge_cases_table(#[case] name: &str, #[case] old: Vec<u8>, #[case] new: Vec<u8>) {
        let patch = make_patch(&old, &new).unwrap();
        let ops = Op::iter(&patch).collect::<Result<Vec<_>, _>>().unwrap();
        insta::assert_debug_snapshot!(name, ops);
        let patched = apply_patch(&old, &patch).unwrap();
        assert_eq!(patched, new);
    }

    #[rstest]
    #[case(&[] as &[u8], &[], 0)]
    #[case(b"Hello, world!", b"", 0)]
    #[case(b"", b"Hello, world!", 0)]
    #[case(b"Hello, world!", b"Hello, world!", 13)]
    #[case(b"abc", b"xbc", 0)]
    #[case(b"abc", b"axc", 1)]
    #[case(b"abc", b"abd", 2)]
    #[case(b"abcdef", b"abc", 3)]
    #[case(
        b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
        b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        32
    )]
    #[case(
        b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
        b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        31
    )]
    #[case(&vec![0u8; 1000][..], &vec![0u8; 1000][..], 1000)]
    #[case(&vec![0u8; 1000][..], &{
        let mut b = vec![0u8; 1000];
        b[500] = 1;
        b
    }[..], 500)]
    fn simd_memcmp_table(#[case] a: &[u8], #[case] b: &[u8], #[case] expected: usize) {
        assert_eq!(simd_memcmp(a, b), expected);
    }

    #[test]
    fn simd_memcmp_matches_naive_oracle() {
        // Exhaustive coverage of every mismatch position for every length
        // 0..=160. Catches NEON 16-byte / AVX2 32-byte chunk-boundary bugs.
        for n in 0_usize..=160 {
            let a = vec![0xAA_u8; n];
            assert_eq!(simd_memcmp(&a, &a), n, "full match n={n}");
            for pos in 0..n {
                let mut b = a.clone();
                b[pos] = 0x55;
                assert_eq!(simd_memcmp(&a, &b), pos, "mismatch pos={pos} n={n}");
                assert_eq!(simd_memcmp(&b, &a), pos, "reverse mismatch pos={pos} n={n}");
            }
        }
    }

    #[rstest]
    #[case::oob_copy(
        "oob_copy",
        b"short",
        {
            let mut p = Vec::new();
            Op::Copy { offset: 0, len: 100 }.serialize_to(&mut p).unwrap();
            p
        },
    )]
    #[case::invalid_opcode("invalid_opcode", b"old", vec![0xFF])]
    #[case::truncated_copy("truncated_copy", b"old", vec![TAG_COPY, 0, 0, 0])]
    fn apply_patch_rejects(#[case] name: &str, #[case] old: &[u8], #[case] patch: Vec<u8>) {
        let err = apply_patch(old, &patch).unwrap_err();
        insta::assert_debug_snapshot!(name, err);
    }

    #[test]
    fn op_iter_stops_after_error() {
        // Valid Add followed by garbage tag.
        let mut patch = Vec::new();
        Op::Add(b"hi").serialize_to(&mut patch).unwrap();
        patch.push(0xEE);
        let collected: Vec<_> = Op::iter(&patch).collect();
        insta::assert_debug_snapshot!(collected);
    }
}
