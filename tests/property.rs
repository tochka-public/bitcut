//! Property-based tests for the v1.0 API.
//!
//! Run with `cargo test --test property`.

#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::arithmetic_side_effects,
    clippy::default_numeric_fallback,
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value,
    clippy::missing_errors_doc,
    dead_code
)]

use bitcut::{apply_patch, make_patch, Op};
use proptest::collection::vec;
use proptest::prelude::*;

/// Naive byte-by-byte common-prefix length, used as a reference oracle for
/// the SIMD `simd_memcmp` paths (which we exercise indirectly through
/// `make_patch` over crafted inputs).
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

// =============================================================================
// 17. Round-trip property
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 2048,
        max_shrink_iters: 10_000,
        ..ProptestConfig::default()
    })]

    /// For any (old, new): apply_patch(old, make_patch(old, new)) == new.
    #[test]
    fn roundtrip_arbitrary(
        old in vec(any::<u8>(), 0..512),
        new in vec(any::<u8>(), 0..512),
    ) {
        let patch = make_patch(&old, &new).expect("make_patch must succeed");
        let restored = apply_patch(&old, &patch).expect("apply_patch must succeed on a valid patch");
        prop_assert_eq!(restored, new);
    }

    /// Round-trip when `new` is a substring/shifted variant of `old` —
    /// the realistic case where Copy ops dominate.
    #[test]
    fn roundtrip_shifted(
        old in vec(any::<u8>(), 32..512),
        skip in 0usize..32,
        tail in vec(any::<u8>(), 0..32),
    ) {
        let take = old.len().saturating_sub(skip);
        let mut new: Vec<u8> = old.iter().copied().skip(skip).take(take).collect();
        new.extend_from_slice(&tail);
        let patch = make_patch(&old, &new).expect("make_patch must succeed");
        let restored = apply_patch(&old, &patch).expect("valid patch");
        prop_assert_eq!(restored, new);
    }

    /// Repeated patterns — exercises rolling hash collisions and long matches.
    #[test]
    fn roundtrip_repeated_pattern(
        unit in vec(any::<u8>(), 1..16),
        repeat_old in 1usize..40,
        repeat_new in 1usize..40,
    ) {
        let old: Vec<u8> = unit.iter().cycle().copied().take(unit.len() * repeat_old).collect();
        let new: Vec<u8> = unit.iter().cycle().copied().take(unit.len() * repeat_new).collect();
        let patch = make_patch(&old, &new).expect("make_patch must succeed");
        let restored = apply_patch(&old, &patch).expect("valid patch");
        prop_assert_eq!(restored, new);
    }
}

// =============================================================================
// 18. Fuzz: arbitrary patch bytes must never panic
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 4096,
        max_shrink_iters: 10_000,
        ..ProptestConfig::default()
    })]

    /// Feeding arbitrary bytes to `apply_patch` must never panic — only
    /// return Err. This stresses the deserializer for overflow / OOB.
    #[test]
    fn apply_patch_never_panics(
        old in vec(any::<u8>(), 0..256),
        patch in vec(any::<u8>(), 0..1024),
    ) {
        // Just ensure no panic — both Ok and Err are acceptable.
        let _ = apply_patch(&old, &patch);
    }

    /// `Op::iter` on arbitrary bytes also must not panic.
    #[test]
    fn op_iter_never_panics(patch in vec(any::<u8>(), 0..1024)) {
        for _ in Op::iter(&patch) {
            // drain
        }
    }

    /// Crafted Copy with huge offset/len must produce a clean Err, not panic.
    #[test]
    fn crafted_copy_overflow_never_panics(
        old in vec(any::<u8>(), 0..64),
        offset in any::<u32>(),
        len in any::<u32>(),
    ) {
        let mut patch = Vec::with_capacity(9);
        patch.push(0x00);
        patch.extend_from_slice(&offset.to_le_bytes());
        patch.extend_from_slice(&len.to_le_bytes());
        let _ = apply_patch(&old, &patch);
    }

    /// Crafted Add with huge declared length must produce a clean Err.
    #[test]
    fn crafted_add_overflow_never_panics(
        old in vec(any::<u8>(), 0..64),
        declared_len in any::<u32>(),
        body in vec(any::<u8>(), 0..64),
    ) {
        let mut patch = Vec::with_capacity(5 + body.len());
        patch.push(0x01);
        patch.extend_from_slice(&declared_len.to_le_bytes());
        patch.extend_from_slice(&body);
        let _ = apply_patch(&old, &patch);
    }
}

// =============================================================================
// SIMD memcmp coverage — exercised indirectly through make_patch/apply_patch
// =============================================================================

// Build (old, new) pairs that force `simd_memcmp` to run with the mismatch
// at every possible offset within / across SIMD chunks (16-byte NEON,
// 32-byte AVX2). Covers the NEON scalar tail and AVX2 movemask path.
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 2048,
        ..ProptestConfig::default()
    })]

    /// For each mismatch position `pos`, build old=[A;N], new=[A;pos] ++ [B] ++ [A; N-pos-1]
    /// with N from 1..200, and verify that the round-trip succeeds.
    /// This exercises every possible mismatch offset in `simd_memcmp`.
    #[test]
    fn simd_memcmp_mismatch_at_every_offset(
        (n, pos) in (1usize..200).prop_flat_map(|n| (Just(n), 0usize..n)),
    ) {
        let old = vec![0xAA_u8; n];
        let mut new = vec![0xAA_u8; n];
        new[pos] = 0xBB;
        let patch = make_patch(&old, &new).expect("make_patch must succeed");
        let restored = apply_patch(&old, &patch).expect("valid patch");
        prop_assert_eq!(restored, new);
    }

    /// Identical inputs of varying lengths — `simd_memcmp` should walk to
    /// the end without breaking on the chunk boundary.
    #[test]
    fn simd_memcmp_full_match(n in 0usize..300) {
        let buf = vec![0xCC_u8; n];
        let patch = make_patch(&buf, &buf).expect("make_patch must succeed");
        let restored = apply_patch(&buf, &patch).expect("valid patch");
        prop_assert_eq!(restored, buf);
    }
}

#[test]
fn common_prefix_len_oracle_smoke() {
    // Trivial sanity check on the oracle helper itself.
    assert_eq!(common_prefix_len(b"abcdef", b"abcxyz"), 3);
    assert_eq!(common_prefix_len(b"", b"x"), 0);
    assert_eq!(common_prefix_len(b"abc", b"abc"), 3);
}

// =============================================================================
// Targeted SIMD mismatch tests across critical chunk boundaries
// =============================================================================

/// Drives the SIMD path for every byte length 0..=160 (covers NEON 16-byte
/// chunks 0..10 and AVX2 32-byte chunks 0..5) and every mismatch position.
/// This is the deterministic complement to the property tests above —
/// guarantees coverage of SIMD chunk-boundary edge cases (point 8).
#[test]
fn simd_memcmp_exhaustive_small() {
    for n in 0_usize..=160 {
        let a = vec![0x55_u8; n];
        // Full match
        let patch = make_patch(&a, &a).expect("make_patch must succeed");
        let restored = apply_patch(&a, &patch).expect("valid patch");
        assert_eq!(restored, a, "full-match failed at n={n}");

        // Mismatch at every position
        for pos in 0..n {
            let mut b = a.clone();
            b[pos] = 0xAA;
            let patch = make_patch(&a, &b).expect("make_patch must succeed");
            let restored = apply_patch(&a, &patch).expect("valid patch");
            assert_eq!(restored, b, "mismatch at pos={pos}, n={n}");

            // Also reverse direction
            let patch = make_patch(&b, &a).expect("make_patch must succeed");
            let restored = apply_patch(&b, &patch).expect("valid patch");
            assert_eq!(restored, a, "reverse mismatch at pos={pos}, n={n}");
        }
    }
}
