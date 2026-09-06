//! `bitcut` — create and apply binary patches.
//!
//! The library exposes a small, allocation-friendly API:
//!
//! * [`make_patch`] / [`make_patch_into`] — produce a patch describing how to
//!   reconstruct `new` from `old`.
//! * [`make_patch_stats`] / [`make_patch_into_stats`] — the same, also
//!   returning [`PatchStats`] describing how the patch was found.
//! * [`apply_patch`] / [`apply_patch_into`] — apply a patch to `old` and
//!   recover `new`.
//! * [`Op`] / [`OpIter`] — low-level access to the patch opcode stream.
//! * [`inspect`] — read a patch's header without applying it.
//!
//! Both `old` and `new` are limited to `u32::MAX` (≈ 4 GiB) bytes.
//!
//! ## Patch format
//!
//! Patches are written in the sectioned v2 layout, which names the base it
//! was built from; patches in the older opcode-stream layout are still read.
//! Both are described in the `wire` module.
//!
//! A patch built from one base **cannot** be applied to another: `old` is
//! fingerprinted at build time and re-checked at apply time, and a mismatch
//! is [`PatchError::WrongBase`] rather than a plausible-looking wrong
//! document. The fingerprint is xxh3 — it detects a wrong or stale base, not
//! a forged patch. If patches arrive from somewhere untrusted, authenticate
//! them at the transport or storage layer.
//!
//! ## Matching
//!
//! Matching is described in the `differ` module: the base is walked in
//! lockstep with the new document rather than indexed up front, so the search
//! cost tracks the number of edits rather than the size of the base. See
//! [`PatchStats`] for the counters that report when that assumption stops
//! holding.
//!
//! ## Cost
//!
//! Hashing the base is one linear pass over `old` on both the build and the
//! apply side, and on a base large enough for the differ to skip most of it
//! that pass dominates. On the benchmark corpus (1.6 MiB base, local edits)
//! it is roughly 40 µs of the ~120 µs build and of the ~70 µs apply. A caller
//! that cannot afford it can compare [`PatchHeader::base_len`] and
//! [`base_fingerprint`] itself, once, and cache the result alongside the
//! base.

mod apply;
#[cfg(feature = "demo")]
pub mod demo;
mod differ;
mod encode;
mod error;
mod index;
mod memcmp;
mod op;
mod vbyte;
mod wire;

pub use apply::{apply_patch, apply_patch_into};
pub use differ::PatchStats;
pub use error::PatchError;
pub use index::RollingHash;
pub use op::{Op, OpIter};

use op::ADD_HEADER_LEN;

/// What a patch states about itself, readable without applying it.
///
/// Only the v2 layout carries this; [`inspect`] returns `None` for a legacy
/// patch, which identifies neither its base nor its opcode count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PatchHeader {
    /// Length of the base the patch was built from.
    pub base_len: u64,
    /// xxh3 fingerprint of that base.
    pub base_hash: u64,
    /// Number of opcodes in the patch.
    pub ops: u32,
    /// Every copy reads forward of the previous one, so the patch can be
    /// applied while streaming the base rather than seeking within it.
    pub monotone_copies: bool,
}

/// Read a patch's header without applying it.
///
/// Useful for checking a stored patch against a base before paying to load
/// either, and for telemetry over a patch store.
///
/// # Errors
///
/// See [`PatchError`] variants.
pub fn inspect(patch: &[u8]) -> Result<Option<PatchHeader>, PatchError> {
    match wire::parse(patch)? {
        wire::Layout::V1(_) => Ok(None),
        wire::Layout::V2(header) => Ok(Some(PatchHeader {
            base_len: header.base_len,
            base_hash: header.base_hash,
            ops: u32::try_from(header.ops).map_err(|_| PatchError::Overflow)?,
            monotone_copies: header.flags & wire::FLAG_MONOTONE != 0,
        })),
    }
}

/// Fingerprint a base document the way a v2 patch header records it.
#[must_use]
pub fn base_fingerprint(base: &[u8]) -> (u64, u64) {
    wire::fingerprint(base)
}

/// Build a patch describing how to reconstruct `new` from `old`.
///
/// # Errors
///
/// Returns [`PatchError::InputTooLarge`] if either `old` or `new` exceeds
/// `u32::MAX` bytes.
pub fn make_patch(old: &[u8], new: &[u8]) -> Result<Vec<u8>, PatchError> {
    make_patch_stats(old, new).map(|(patch, _stats)| patch)
}

/// Build a patch into a caller-supplied buffer. The buffer is **not**
/// cleared first; ops are appended.
///
/// # Errors
///
/// Returns [`PatchError::InputTooLarge`] if either input exceeds `u32::MAX`
/// bytes.
pub fn make_patch_into(old: &[u8], new: &[u8], out: &mut Vec<u8>) -> Result<(), PatchError> {
    make_patch_into_stats(old, new, out).map(|_stats| ())
}

/// Build a patch and report how it was found.
///
/// # Errors
///
/// Returns [`PatchError::InputTooLarge`] if either input exceeds `u32::MAX`
/// bytes.
pub fn make_patch_stats(old: &[u8], new: &[u8]) -> Result<(Vec<u8>, PatchStats), PatchError> {
    let mut out = Vec::with_capacity(estimate_patch_capacity(new.len()));
    let stats = make_patch_into_stats(old, new, &mut out)?;
    Ok((out, stats))
}

/// Build a patch into a caller-supplied buffer and report how it was found.
/// The buffer is **not** cleared first; ops are appended.
///
/// # Errors
///
/// Returns [`PatchError::InputTooLarge`] if either input exceeds `u32::MAX`
/// bytes.
pub fn make_patch_into_stats(
    old: &[u8],
    new: &[u8],
    out: &mut Vec<u8>,
) -> Result<PatchStats, PatchError> {
    if u32::try_from(old.len()).is_err() {
        return Err(PatchError::InputTooLarge { len: old.len() });
    }
    if u32::try_from(new.len()).is_err() {
        return Err(PatchError::InputTooLarge { len: new.len() });
    }

    let mut chunks = Vec::new();
    let stats = differ::diff(old, new, &mut chunks);
    encode::v2(&chunks, old, new, out)?;
    Ok(stats)
}

#[inline]
fn estimate_patch_capacity(new_len: usize) -> usize {
    // Heuristic: a perfectly-matching patch is ~9 bytes, a fully-literal one
    // is `new_len + 5`. Pick a small floor and a ceiling that covers the
    // patch sizes this crate is tuned for without over-reserving.
    new_len.saturating_add(ADD_HEADER_LEN).clamp(64, 1 << 16)
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
    use crate::index::WINDOW_SIZE;
    use crate::memcmp::simd_memcmp;
    use crate::wire::TAG_COPY;
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

    /// `memchr::memmem` takes the haystack first. Swapping the arguments still
    /// type-checks and silently finds nothing, so pin the order down.
    #[test]
    fn memmem_argument_order() {
        assert_eq!(memchr::memmem::find(b"xxhello", b"hello"), Some(2));
    }

    /// A local edit must not touch the global index, and the drift cache must
    /// absorb repeated edits of the same shape.
    #[test]
    fn local_edits_do_not_escalate() {
        let old: Vec<u8> = (0..200_000_u32).flat_map(u32::to_le_bytes).collect();
        let mut new = old.clone();
        // Three length-changing edits at unrelated positions.
        for at in [10_000_usize, 300_000, 600_000] {
            new.splice(at..at, b"INSERTED-PAYLOAD".iter().copied());
        }

        let (patch, stats) = make_patch_stats(&old, &new).unwrap();
        assert_eq!(apply_patch(&old, &patch).unwrap(), new);
        assert_eq!(stats.escalations, 0, "a local edit must not need the index");
        assert!(
            stats.old_bytes_scanned < u64::try_from(old.len()).unwrap(),
            "window search scanned {} of {} base bytes",
            stats.old_bytes_scanned,
            old.len()
        );
    }

    /// Repeating the *same* length delta must be answered from the drift
    /// cache rather than by searching again.
    #[test]
    fn repeated_edit_shape_hits_the_shift_cache() {
        let unit: Vec<u8> = (0..2_000_u32).flat_map(u32::to_le_bytes).collect();
        let old = unit.repeat(4);
        let mut new = old.clone();
        for at in (5_000_usize..30_000).step_by(4_000) {
            new.splice(at..at, b"DELTA".iter().copied());
        }

        let (patch, stats) = make_patch_stats(&old, &new).unwrap();
        assert_eq!(apply_patch(&old, &patch).unwrap(), new);
        assert!(
            stats.shift_cache_hits > 0,
            "expected drift-cache hits, got {stats:?}"
        );
    }

    /// A wholesale reorder is exactly what local resync cannot see; the
    /// global index must catch it instead of producing one huge literal.
    #[test]
    fn reorder_escalates_and_still_copies() {
        let head: Vec<u8> = (0..30_000_u32).flat_map(u32::to_le_bytes).collect();
        let tail: Vec<u8> = (500_000..530_000_u32).flat_map(u32::to_le_bytes).collect();
        let old: Vec<u8> = head.iter().chain(tail.iter()).copied().collect();
        let new: Vec<u8> = tail.iter().chain(head.iter()).copied().collect();

        let (patch, stats) = make_patch_stats(&old, &new).unwrap();
        assert_eq!(apply_patch(&old, &patch).unwrap(), new);
        assert!(stats.escalations > 0, "expected escalation, got {stats:?}");
        assert!(
            patch.len() < new.len() / 2,
            "reorder degenerated into literals: patch {} for new {}",
            patch.len(),
            new.len()
        );
    }

    /// A 32-byte anchor that occurs many times must not align the differ to
    /// the wrong occurrence.
    #[test]
    fn repeated_anchor_does_not_misalign() {
        let filler = b"REPEATED-32-BYTE-ANCHOR-XXXXXXXX";
        assert_eq!(filler.len(), 32);
        let mut old = Vec::new();
        for i in 0..500_u32 {
            old.extend_from_slice(filler);
            old.extend_from_slice(&i.to_le_bytes());
        }
        let mut new = old.clone();
        new.splice(9_000..9_000, b"WEDGE".iter().copied());

        let patch = make_patch(&old, &new).unwrap();
        assert_eq!(apply_patch(&old, &patch).unwrap(), new);
    }

    /// Patches written before the v2 header existed must keep applying.
    #[test]
    fn v1_patches_still_apply() {
        let old: Vec<u8> = (0..20_000_u32).flat_map(u32::to_le_bytes).collect();
        let mut new = old.clone();
        new.splice(30_000..30_000, b"LEGACY".iter().copied());

        let mut chunks = Vec::new();
        differ::diff(&old, &new, &mut chunks);
        let mut legacy = Vec::new();
        encode::v1(&chunks, &new, &mut legacy).unwrap();

        assert!(
            !legacy.starts_with(&wire::MAGIC),
            "a v1 patch must not be mistaken for a v2 one"
        );
        assert_eq!(apply_patch(&old, &legacy).unwrap(), new);
        assert_eq!(inspect(&legacy).unwrap(), None);

        // Both layouts must decode to the same opcode sequence.
        let modern = make_patch(&old, &new).unwrap();
        let from_v1 = Op::iter(&legacy).collect::<Result<Vec<_>, _>>().unwrap();
        let from_v2 = Op::iter(&modern).collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(from_v1, from_v2);
    }

    /// The failure this format exists to prevent: a patch applied to a
    /// different version of the base must error, not silently produce a
    /// plausible document.
    #[test]
    fn wrong_base_is_rejected() {
        let old: Vec<u8> = (0..5_000_u32).flat_map(u32::to_le_bytes).collect();
        let mut new = old.clone();
        new.splice(1_000..1_000, b"CHANGE".iter().copied());
        let patch = make_patch(&old, &new).unwrap();

        // Same length, one byte different — the length check alone would miss it.
        let mut impostor = old.clone();
        impostor[4_096] ^= 0xFF;
        assert_eq!(impostor.len(), old.len());

        let err = apply_patch(&impostor, &patch).unwrap_err();
        assert!(
            matches!(err, PatchError::WrongBase { .. }),
            "expected WrongBase, got {err:?}"
        );

        // And a base of a different length.
        let err = apply_patch(&old[..old.len() - 1], &patch).unwrap_err();
        assert!(matches!(err, PatchError::WrongBase { .. }), "{err:?}");

        // The genuine base still applies.
        assert_eq!(apply_patch(&old, &patch).unwrap(), new);
    }

    #[test]
    fn inspect_reports_the_header() {
        let old: Vec<u8> = (0..5_000_u32).flat_map(u32::to_le_bytes).collect();
        let mut new = old.clone();
        new.extend_from_slice(b"TAIL");
        let patch = make_patch(&old, &new).unwrap();

        let header = inspect(&patch).unwrap().unwrap();
        let (len, hash) = base_fingerprint(&old);
        assert_eq!(header.base_len, len);
        assert_eq!(header.base_hash, hash);
        assert!(header.monotone_copies, "an append only copies forward");
        assert_eq!(
            usize::try_from(header.ops).unwrap(),
            Op::iter(&patch).count()
        );
    }

    #[test]
    fn future_versions_are_refused() {
        let old = b"the base document, long enough to matter";
        let mut patch = make_patch(old, b"the base document, long enough to matter!").unwrap();
        patch[4] = 99;
        assert_eq!(
            apply_patch(old, &patch),
            Err(PatchError::UnsupportedVersion(99))
        );
    }

    #[test]
    fn streamvbyte_roundtrips_every_width() {
        let values: Vec<u32> = (0..40_u32)
            .flat_map(|shift| {
                let base = 1_u32.checked_shl(shift).unwrap_or(0);
                [base.wrapping_sub(1), base, base.wrapping_add(1)]
            })
            .collect();
        let mut buffer = Vec::new();
        vbyte::encode(&values, &mut buffer);
        assert_eq!(buffer.len(), vbyte::encoded_len(&values));
        let decoded: Vec<u32> = vbyte::Reader::new(&buffer, values.len()).unwrap().collect();
        assert_eq!(decoded, values);
    }

    #[test]
    fn zigzag_roundtrips_and_keeps_small_steps_small() {
        for value in [0_i32, 1, -1, 2, -2, 127, -128, i32::MAX, i32::MIN] {
            assert_eq!(
                vbyte::unzigzag(vbyte::zigzag(value)),
                value,
                "value={value}"
            );
        }
        // The whole point: a small step in either direction stays one byte.
        for value in -63_i32..=63 {
            assert!(vbyte::zigzag(value) <= 0xFF, "value={value}");
        }
    }

    /// Two documents with nothing in common must still round-trip, and must
    /// not cost more than carrying `new` verbatim plus a header.
    #[test]
    fn dissimilar_inputs_roundtrip() {
        let old: Vec<u8> = (0..40_000_u32).flat_map(u32::to_le_bytes).collect();
        let new: Vec<u8> = (0..40_000_u32)
            .flat_map(|n| n.wrapping_mul(2_654_435_761).to_le_bytes())
            .collect();
        let patch = make_patch(&old, &new).unwrap();
        assert_eq!(apply_patch(&old, &patch).unwrap(), new);
        assert!(
            patch.len() < new.len() + 1024,
            "patch {} exceeds new {} by more than a header",
            patch.len(),
            new.len()
        );
    }

    /// The defaults must satisfy the invariants the matcher is happiest with.
    #[test]
    fn default_params_are_sound() {
        assert_eq!(differ::Params::for_input(1 << 20).check(), Ok(()));
        assert_eq!(differ::Params::for_input(16).check(), Ok(()));
        let p = differ::Params::for_input(1 << 20);
        assert_eq!(p.skip_ahead(), *p.gaps().last().unwrap());
    }

    #[rstest]
    #[case::ladder_must_start_at_zero(
        |p: &mut differ::Params| p.gaps[0] = 4,
        differ::ParamsError::GapLadder
    )]
    #[case::ladder_must_ascend(
        |p: &mut differ::Params| p.gaps.swap(1, 2),
        differ::ParamsError::GapOrder
    )]
    #[case::match_below_anchor(
        |p: &mut differ::Params| p.min_match = p.anchor + 1,
        differ::ParamsError::MinMatchAboveAnchor
    )]
    #[case::verify_above_anchor(
        |p: &mut differ::Params| p.anchor_verify = p.anchor - 1,
        differ::ParamsError::VerifyBelowAnchor
    )]
    #[case::window_holds_an_anchor(
        |p: &mut differ::Params| p.window_half_width = 1,
        differ::ParamsError::WindowBelowAnchor
    )]
    #[case::slots_are_bounded(
        |p: &mut differ::Params| p.slots = 999,
        differ::ParamsError::TooManySlots
    )]
    fn bad_params_are_named(
        #[case] break_it: fn(&mut differ::Params),
        #[case] expected: differ::ParamsError,
    ) {
        let mut params = differ::Params::for_input(1 << 20);
        break_it(&mut params);
        assert_eq!(params.check(), Err(expected));
    }

    /// Parameters the checker rejects must still produce a correct patch —
    /// only a worse one. Nothing about correctness may depend on tuning.
    #[test]
    fn rejected_params_still_roundtrip() {
        let old: Vec<u8> = (0..40_000_u32).flat_map(u32::to_le_bytes).collect();
        let mut new = old.clone();
        new.splice(60_000..60_000, b"WEDGE".iter().copied());
        new.truncate(120_000);

        // Every rule broken at once: a match threshold above the anchor, a
        // verification bar below it, and a window too small to hold one.
        let mut params = differ::Params::for_input(new.len());
        params.min_match = 64;
        params.anchor = 4;
        params.anchor_verify = 2;
        params.window_half_width = 1;
        params.gaps = [0, 1, 0, 0, 0, 0, 0, 0];
        params.gap_count = 2;
        params.slots = 0;
        assert_eq!(
            params.check(),
            Err(differ::ParamsError::MinMatchAboveAnchor)
        );

        let mut chunks = Vec::new();
        differ::diff_with(&old, &new, params, &mut chunks);
        let mut patch = Vec::new();
        encode::v2(&chunks, &old, &new, &mut patch).unwrap();
        assert_eq!(apply_patch(&old, &patch).unwrap(), new);
    }

    /// Recording must not change what the matcher decides. If it did, the
    /// walkthrough would be describing a different algorithm from the one that
    /// builds real patches.
    #[test]
    fn tracing_does_not_change_the_patch() {
        let old: Vec<u8> = (0..30_000_u32).flat_map(u32::to_le_bytes).collect();
        let mut new = old.clone();
        new.splice(40_000..40_000, b"INSERTED-PAYLOAD".iter().copied());
        new.splice(90_000..90_016, b"REPLACED".iter().copied());
        new.extend_from_slice(b"APPENDED TAIL");

        let params = differ::Params::for_input(new.len());
        let mut plain = Vec::new();
        let quiet = differ::diff_with(&old, &new, params, &mut plain);

        let mut traced = Vec::new();
        let (loud, recorder) =
            differ::diff_traced(&old, &new, params, CountingTrace::default(), &mut traced);

        assert_eq!(plain, traced, "tracing changed the opcodes");
        assert_eq!(quiet, loud, "tracing changed the statistics");
        assert!(recorder.events > 0, "nothing was recorded");
    }

    #[derive(Default)]
    struct CountingTrace {
        events: usize,
    }

    impl differ::Trace for CountingTrace {
        const ENABLED: bool = true;
        fn record(&mut self, _at: differ::At, _stats: &PatchStats, _event: differ::Event<'_>) {
            self.events += 1;
        }
    }

    /// The log must account for every byte of `new`. A recorder replaying only
    /// the emitted opcodes has to rebuild the document exactly — if the
    /// matcher writes an opcode without reporting it, this is what catches it.
    #[rstest]
    #[case::local_edits(
        b"cp:Acme,Borg,Cyan|01.03 pay 10|01.03 fee 20|01.03 pay 30|01.03 fee 40".to_vec(),
        b"cp:Acme,Beta,Borg,Cyan|01.03 payed 10|01.03 fee 20|01.03 payed 30|01.03 fee 40|01.03 fee 50".to_vec(),
    )]
    #[case::appended_tail(
        b"cp:Acme,Borg,Cyan|01.03 pay 10|01.03 fee 20".to_vec(),
        b"cp:Acme,Borg,Cyan|01.03 pay 10|01.03 fee 20|99.99 zzz 000111222333".to_vec(),
    )]
    #[case::swapped(
        b"cp:Acme,Borg,Cyan|01.03 done|09.09 audit|10.09 audit".to_vec(),
        b"|09.09 audit|10.09 auditcp:Acme,Borg,Cyan|01.03 done".to_vec(),
    )]
    #[case::unrelated(b"aaaaaaaaaaaaaaaaaaaaaaaa".to_vec(), b"zzzzzzzzzzzzzzzzzzzzzzzz".to_vec())]
    #[case::empty_new(b"cp:Acme,Borg,Cyan|01.03 pay 10".to_vec(), vec![])]
    fn the_log_accounts_for_every_byte(#[case] old: Vec<u8>, #[case] new: Vec<u8>) {
        let mut params = differ::Params::for_input(new.len());
        params.min_match = 6;
        params.anchor = 6;
        params.anchor_verify = 14;
        params.gaps = [0, 2, 6, 8, 0, 0, 0, 0];
        params.gap_count = 4;
        params.window_half_width = 8;
        params.slots = 4;

        let mut chunks = Vec::new();
        let replay = Replay {
            old: old.clone(),
            new: new.clone(),
            out: Vec::new(),
        };
        let (_stats, replay) = differ::diff_traced(&old, &new, params, replay, &mut chunks);
        assert_eq!(
            replay.out, new,
            "replaying only the recorded opcodes did not rebuild `new`"
        );
    }

    /// Rebuilds `new` from nothing but the recorded opcodes.
    struct Replay {
        old: Vec<u8>,
        new: Vec<u8>,
        out: Vec<u8>,
    }

    impl differ::Trace for Replay {
        const ENABLED: bool = true;
        fn record(&mut self, _at: differ::At, _stats: &PatchStats, event: differ::Event<'_>) {
            match event {
                differ::Event::EmittedCopy { offset, len } => {
                    self.out.extend_from_slice(&self.old[offset..offset + len]);
                }
                differ::Event::EmittedLiteral { start, len, .. } => {
                    self.out.extend_from_slice(&self.new[start..start + len]);
                }
                differ::Event::Restarted { .. } => self.out.clear(),
                _ => {}
            }
        }
    }

    /// The JSON log must be well formed and describe the same run.
    #[cfg(feature = "demo")]
    #[test]
    fn trace_json_is_parseable_and_complete() {
        let old = b"cp:Acme,Borg,Cyan|01.03 pay 10|01.03 fee 20|01.03 pay 30|01.03 fee 40";
        let new = b"cp:Acme,Beta,Borg,Cyan|01.03 payed 10|01.03 fee 20|01.03 payed 30|01.03 fee 40";
        let mut params = differ::Params::for_input(new.len());
        params.min_match = 6;
        params.anchor = 6;
        params.anchor_verify = 14;
        params.gaps = [0, 2, 6, 8, 0, 0, 0, 0];
        params.gap_count = 4;
        params.window_half_width = 16;
        params.slots = 4;

        let json = crate::demo::trace_json(old, new, params).unwrap();
        assert!(json.starts_with('[') && json.ends_with(']'));
        assert_eq!(
            json.matches("\"t\":").count(),
            json.matches("{\"t\":").count(),
            "every frame must open with its tag"
        );
        assert!(json.contains("\"t\":\"compared\""));
        assert!(json.contains("\"t\":\"aligned\""));
        // Braces must balance, which is the cheapest structural check there is.
        assert_eq!(json.matches('{').count(), json.matches('}').count());
    }

    #[test]
    fn stats_count_emitted_opcodes() {
        let old: Vec<u8> = (0..50_000_u32).flat_map(u32::to_le_bytes).collect();
        let mut new = old.clone();
        new.splice(50_000..50_000, b"XYZZY".iter().copied());

        let (patch, stats) = make_patch_stats(&old, &new).unwrap();
        let ops = Op::iter(&patch).collect::<Result<Vec<_>, _>>().unwrap();
        let copies = ops
            .iter()
            .filter(|op| matches!(op, Op::Copy { .. }))
            .count();
        let literals = ops.iter().filter(|op| matches!(op, Op::Add(_))).count();
        assert_eq!(usize::try_from(stats.copies).unwrap(), copies);
        assert_eq!(usize::try_from(stats.literals).unwrap(), literals);
    }
}
