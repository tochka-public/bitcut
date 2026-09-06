//! What an update that is *not* a local edit costs, and what the differ must
//! not do when it hits one.
//!
//! Run with `-- --nocapture` to see the measurements behind the assertions.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::default_numeric_fallback,
    clippy::cast_precision_loss,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::missing_panics_doc
)]

use bitcut::{apply_patch, make_patch_stats, Op, PatchStats};
use std::time::Instant;

/// xorshift64*, so the bytes carry no short period the differ could latch onto
/// and no structure a real match would be confused with.
fn noise(seed: u64, len: usize) -> Vec<u8> {
    let mut x = seed | 1;
    (0..len)
        .map(|_| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as u8
        })
        .collect()
}

struct Measured {
    patch_len: usize,
    literal_bytes: usize,
    stats: PatchStats,
}

fn measure(name: &str, old: &[u8], new: &[u8]) -> Measured {
    let start = Instant::now();
    let (patch, stats) = make_patch_stats(old, new).unwrap();
    let build = start.elapsed();
    assert_eq!(apply_patch(old, &patch).unwrap(), new, "{name}: roundtrip");

    let literal_bytes = Op::iter(&patch)
        .map(|op| match op.unwrap() {
            Op::Add(bytes) => bytes.len(),
            Op::Copy { .. } => 0,
        })
        .sum();

    println!(
        "{name}\n  {} KiB base, build {build:?}\n  patch {} B, literal {} B ({:.2}% of base)\n  {stats:?}\n",
        old.len() / 1024,
        patch.len(),
        literal_bytes,
        100.0 * literal_bytes as f64 / old.len() as f64,
    );
    Measured {
        patch_len: patch.len(),
        literal_bytes,
        stats,
    }
}

/// A local edit must never reach the index, whatever the base size.
#[test]
fn a_local_edit_never_escalates() {
    let base = noise(1, 2 << 20);
    let mut edited = base.clone();
    edited.splice(1_000_000..1_000_000, b"INSERTED".iter().copied());

    let m = measure("local edit", &base, &edited);
    assert_eq!(m.stats.escalations, 0);
    assert_eq!(m.stats.restarts, 0);
    assert!(m.patch_len < 256, "patch is {} B", m.patch_len);
    assert!(
        m.stats.old_bytes_scanned < 1 << 16,
        "scanned {} bytes of a 2 MiB base",
        m.stats.old_bytes_scanned
    );
}

/// A relocated block is exactly what local resync cannot see. Before the index
/// exists the differ carries skipped bytes as literals — and every one of them
/// is content the index can find. Once it is built, that work must be redone,
/// or an update whose entire content is present in the base still produces a
/// patch the size of the moved block.
#[test]
fn a_relocated_block_does_not_leak_into_literals() {
    let head = noise(1, 1 << 20);
    let tail = noise(999, 1 << 20);
    let old: Vec<u8> = head.iter().chain(tail.iter()).copied().collect();
    let new: Vec<u8> = tail.iter().chain(head.iter()).copied().collect();

    let m = measure("halves swapped", &old, &new);
    assert!(
        m.stats.escalations > 0,
        "expected escalation: {:?}",
        m.stats
    );
    assert_eq!(m.stats.restarts, 1, "the blind prefix must be redone");
    assert_eq!(
        m.literal_bytes, 0,
        "every byte of `new` exists in `old`, so nothing should be literal"
    );
    assert!(m.patch_len < 256, "patch is {} B", m.patch_len);
}

/// The same, with the move at the other end of the document.
#[test]
fn a_moved_prefix_does_not_leak_into_literals() {
    let base = noise(7, 2 << 20);
    let mut moved = base.clone();
    let prefix: Vec<u8> = moved.drain(..64 * 1024).collect();
    moved.extend_from_slice(&prefix);

    let m = measure("prefix moved to the end", &base, &moved);
    assert_eq!(m.stats.restarts, 1);
    assert_eq!(m.literal_bytes, 0);
    assert!(m.patch_len < 256, "patch is {} B", m.patch_len);
}

/// Two documents with nothing in common: the index is built, finds nothing,
/// and the whole of `new` is carried verbatim. The restart must not make that
/// case loop or double the patch.
#[test]
fn unrelated_documents_cost_one_literal() {
    let old = noise(3, 512 * 1024);
    let new = noise(4, 512 * 1024);

    let m = measure("unrelated documents", &old, &new);
    assert_eq!(m.literal_bytes, new.len());
    assert!(
        m.patch_len < new.len() + 1024,
        "patch {} for new {}",
        m.patch_len,
        new.len()
    );
}
