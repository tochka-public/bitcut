//! A document shaped like the production workload this crate is tuned for.
//!
//! `cargo test --release --test profile_workload -- --nocapture`
//!
//! One ordered activity log with head eviction, then several keyed sections
//! written in key order. Every section has its own retention limit, and one
//! business operation touches all of them at once.
//!
//! The question is not throughput. It is whether one operation produces a
//! patch that stays under the compressed threshold below which the base is
//! left alone — measured with the compression level the storage actually uses.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::default_numeric_fallback,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    clippy::unreadable_literal,
    clippy::format_push_string,
    clippy::manual_is_multiple_of,
    clippy::missing_panics_doc
)]

use bitcut::{apply_patch, make_patch_stats};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

/// What the storage layer compresses a patch with, and the size above which it
/// gives up and rewrites the base instead.
const COMPRESSION_LEVEL: i32 = 1;
const THRESHOLD: usize = 6 * 1024;

/// Retention limits, one per section.
const ENTRY_LIMIT: usize = 1_000;
const PARTY_LIMIT: usize = 5_000;

// ---------------------------------------------------------------------------
// Counting allocator. Not indexing the base is the point of the differ, and an
// index is a megabyte-scale allocation, so the heap is the one measurement
// that catches a regression the timings would not.
// ---------------------------------------------------------------------------

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: every method forwards to `System` unchanged; only counters are added.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::AcqRel) + layout.size();
            PEAK.fetch_max(live, Ordering::AcqRel);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::AcqRel);
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let out = System.realloc(ptr, layout, new_size);
        if !out.is_null() {
            let live = LIVE.fetch_add(new_size, Ordering::AcqRel) + new_size;
            PEAK.fetch_max(live, Ordering::AcqRel);
            LIVE.fetch_sub(layout.size(), Ordering::AcqRel);
        }
        out
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Held for the length of a run. The counters are process-wide, so a second
/// test allocating concurrently would be charged to this one; the harness runs
/// the tests in this binary on separate threads by default.
static METER: Mutex<()> = Mutex::new(());

fn meter() -> MutexGuard<'static, ()> {
    METER.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Run `f`, returning its value and the peak live heap above the level at
/// entry. The caller must hold [`METER`].
fn measure_peak<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.load(Ordering::Acquire);
    PEAK.store(base, Ordering::Release);
    let value = f();
    (value, PEAK.load(Ordering::Acquire).saturating_sub(base))
}

// ---------------------------------------------------------------------------
// Minimal CBOR, definite lengths only, keys in declaration order.
// ---------------------------------------------------------------------------

fn head(out: &mut Vec<u8>, major: u8, value: u64) {
    if let Ok(byte) = u8::try_from(value) {
        if byte < 0x18 {
            out.push(major | byte);
        } else {
            out.push(major | 0x18);
            out.push(byte);
        }
    } else if let Ok(short) = u16::try_from(value) {
        out.push(major | 0x19);
        out.extend_from_slice(&short.to_be_bytes());
    } else if let Ok(word) = u32::try_from(value) {
        out.push(major | 0x1A);
        out.extend_from_slice(&word.to_be_bytes());
    } else {
        out.push(major | 0x1B);
        out.extend_from_slice(&value.to_be_bytes());
    }
}
fn uint(out: &mut Vec<u8>, value: u64) {
    head(out, 0x00, value);
}
fn text(out: &mut Vec<u8>, value: &str) {
    head(out, 0x60, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}
fn array(out: &mut Vec<u8>, len: usize) {
    head(out, 0x80, len as u64);
}
fn map(out: &mut Vec<u8>, len: usize) {
    head(out, 0xA0, len as u64);
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

// ---------------------------------------------------------------------------
// The document
// ---------------------------------------------------------------------------

/// The kinds that actually occur, with the share of the stream each takes.
/// Two thirds of it moves money and carries three times the bytes of the rest;
/// most of the remainder is sign-in. A dozen further kinds exist and round to
/// nothing, which is why only one stands in for all of them here.
const KINDS: [(&str, u32); 5] = [
    ("wire_out", 3670),
    ("instant_out", 3020),
    ("signin", 3110),
    ("signout", 160),
    ("consent", 40),
];

/// The money-moving kinds, which are the ones the keyed sections annotate.
const PRICED: usize = 2;

fn pick_kind(rng: &mut Rng) -> usize {
    let total: u32 = KINDS.iter().map(|(_, share)| share).sum();
    let mut roll = rng.below(total as usize) as u32;
    for (index, (_, share)) in KINDS.iter().enumerate() {
        match roll.checked_sub(*share) {
            Some(rest) => roll = rest,
            None => return index,
        }
    }
    0
}

/// Field names are drawn from a fixed vocabulary, so the same name recurs
/// thousands of times across the document. This is what makes the real thing
/// compressible, and it is also what makes anchoring hard: a fragment of one
/// record reads much like a fragment of any other, so an anchor that is not
/// verified far enough aligns to the wrong record.
const STEMS: [&str; 16] = [
    "party", "route", "token", "batch", "origin", "target", "amount", "status", "channel",
    "device", "session", "region", "limit", "reason", "source", "handle",
];
const TAILS: [&str; 8] = [
    "", "_id", "_code", "_at", "_kind", "_flag", "_ref", "_value",
];

fn field(slot: usize) -> String {
    let stem = STEMS[slot % STEMS.len()];
    let tail = TAILS[(slot / STEMS.len()) % TAILS.len()];
    format!("{stem}{tail}")
}

/// Records of one kind share a schema: the same names, in the same order, with
/// the same value types. Only the values differ between two records.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Word,
    Number,
    Flag,
    Rate,
}

fn schema(kind: usize, leaves: usize) -> Vec<(String, Shape)> {
    let mut rng = Rng((kind as u64 * 0x9E37_79B9_7F4A_7C15) | 1);
    (0..leaves)
        .map(|slot| {
            let name = field((kind * 7 + slot * 3) % (STEMS.len() * TAILS.len()));
            let shape = match rng.below(100) {
                0..=81 => Shape::Word,
                82..=91 => Shape::Flag,
                92..=96 => Shape::Rate,
                _ => Shape::Number,
            };
            (name, shape)
        })
        .collect()
}

#[derive(Clone)]
struct Entry {
    id: u64,
    at: u64,
    kind: usize,
    body: Vec<u8>,
}

#[derive(Clone)]
struct Party {
    name: String,
    tax_id: String,
    touched_at: u64,
    operations: u64,
}

#[derive(Clone)]
struct Ledger {
    entries: Vec<Entry>,
    handsets: Vec<(String, String)>,
    parties: Vec<(String, Party)>,
    reviews: Vec<(String, String)>,
    context: Vec<(String, String)>,
    notes: Vec<(String, String)>,
}

fn party(rng: &mut Rng, index: u64, now: u64) -> (String, Party) {
    let key = format!(
        "{:09},{:020}",
        44525225 + index % 40,
        4081781000000000000_u64 + index * 7919
    );
    let party = Party {
        name: format!("Northwind Supply {}", rng.next() % 100_000),
        tax_id: format!("{:010}", 7_700_000_000_u64 + rng.next() % 99_999_999),
        touched_at: now - rng.next() % 10_000_000,
        operations: rng.next() % 4_000,
    };
    (key, party)
}

/// A short opaque value: the identifiers, codes and references that make up
/// most of a record. Median around ten characters, a long tail to forty.
fn word(rng: &mut Rng) -> String {
    let len = 6 + rng.below(34);
    let mut out = String::with_capacity(len);
    while out.len() < len {
        out.push_str(&format!("{:x}", rng.next()));
    }
    out.truncate(len);
    out
}

/// A money-moving record carries roughly three times the leaves of the rest,
/// which is what puts it near three kilobytes against one.
fn leaf_count(kind: usize) -> usize {
    if kind < PRICED {
        65
    } else {
        22
    }
}

fn entry(rng: &mut Rng, id: u64, at: u64, kind: usize) -> Entry {
    let fields = schema(kind, leaf_count(kind));
    let mut body = Vec::with_capacity(fields.len() * 48);
    map(&mut body, fields.len());
    for (name, shape) in &fields {
        text(&mut body, name);
        match *shape {
            Shape::Word => text(&mut body, &word(rng)),
            Shape::Number => uint(&mut body, rng.next() % 1_000_000_000_000),
            Shape::Flag => body.push(0xF4 | u8::from(rng.next() % 2 == 0)),
            Shape::Rate => text(
                &mut body,
                &format!("{}.{:02}", rng.next() % 100_000, rng.next() % 100),
            ),
        }
    }
    Entry { id, at, kind, body }
}

fn note(rng: &mut Rng) -> String {
    format!(
        "{{\"code\":{},\"channel\":\"{}\",\"ref\":\"{}\"}}",
        rng.next() % 10_000,
        if rng.next() % 2 == 0 { "fast" } else { "wire" },
        rng.next() % 1_000_000_000_000
    )
}

impl Ledger {
    fn new(seed: u64) -> Self {
        let mut rng = Rng(seed | 1);
        let now = 1_900_000_000_u64;

        let parties = (0..PARTY_LIMIT as u64)
            .map(|i| party(&mut rng, i, now))
            .collect();
        let entries: Vec<Entry> = (0..ENTRY_LIMIT)
            .map(|i| {
                let kind = pick_kind(&mut rng);
                entry(
                    &mut rng,
                    i as u64,
                    now - (ENTRY_LIMIT - i) as u64 * 600,
                    kind,
                )
            })
            .collect();
        let handsets = (0..12_u64)
            .map(|i| {
                (
                    format!("handset-{i:04}"),
                    format!(
                        "{{\"agent\":\"Mozilla/5.0 ({})\",\"seen\":{now}}}",
                        rng.next() % 999
                    ),
                )
            })
            .collect();

        let mut ledger = Self {
            entries,
            handsets,
            parties,
            reviews: Vec::new(),
            context: Vec::new(),
            notes: Vec::new(),
        };
        ledger.rebuild_derived(&mut rng);
        ledger
    }

    /// The keyed sections that hang off the log: they live and die with the
    /// entries they annotate.
    fn rebuild_derived(&mut self, rng: &mut Rng) {
        let live: Vec<u64> = self
            .entries
            .iter()
            .filter(|entry| entry.kind < PRICED)
            .map(|entry| entry.id)
            .collect();

        self.notes = live
            .iter()
            .map(|id| (format!("op-{id:012}"), note(rng)))
            .collect();
        self.context = live
            .iter()
            .take(live.len() / 2)
            .map(|id| {
                (
                    format!("op-{id:012}"),
                    format!("{{\"step\":\"{}\",\"at\":{}}}", rng.next() % 9, rng.next()),
                )
            })
            .collect();
        self.reviews = live
            .iter()
            .take(live.len() / 20)
            .map(|id| {
                (
                    format!("case-{id:012}"),
                    format!(
                        "{{\"status\":\"closed\",\"tag\":\"clean\",\"at\":{}}}",
                        rng.next()
                    ),
                )
            })
            .collect();
    }

    /// Field order matches the struct's declaration order; empty sections are
    /// not written at all; every keyed section is written in key order.
    fn encode(&self) -> Vec<u8> {
        let sections = 1
            + usize::from(!self.handsets.is_empty())
            + usize::from(!self.parties.is_empty())
            + usize::from(!self.reviews.is_empty())
            + usize::from(!self.context.is_empty())
            + usize::from(!self.notes.is_empty());

        let mut out = Vec::with_capacity(8 << 20);
        map(&mut out, sections);

        text(&mut out, "entries");
        array(&mut out, self.entries.len());
        for entry in &self.entries {
            map(&mut out, 4);
            text(&mut out, "id");
            uint(&mut out, entry.id);
            text(&mut out, "at");
            uint(&mut out, entry.at);
            text(&mut out, "kind");
            text(&mut out, KINDS[entry.kind].0);
            text(&mut out, "body");
            out.extend_from_slice(&entry.body);
        }

        write_pairs(&mut out, "handsets", &self.handsets);

        if !self.parties.is_empty() {
            let mut parties = self.parties.clone();
            parties.sort_by(|left, right| left.0.cmp(&right.0));
            text(&mut out, "parties");
            map(&mut out, parties.len());
            for (key, party) in &parties {
                text(&mut out, key);
                map(&mut out, 4);
                text(&mut out, "name");
                text(&mut out, &party.name);
                text(&mut out, "tax_id");
                text(&mut out, &party.tax_id);
                text(&mut out, "touched_at");
                uint(&mut out, party.touched_at);
                text(&mut out, "operations");
                uint(&mut out, party.operations);
            }
        }

        write_pairs(&mut out, "reviews", &self.reviews);
        write_pairs(&mut out, "context", &self.context);
        write_pairs(&mut out, "notes", &self.notes);
        out
    }

    /// One business operation, as the domain performs it: one party touched or
    /// created, one entry inserted by timestamp, annotations added, and every
    /// section trimmed back to its limit.
    ///
    /// Returns the bytes the operation invented, which no patch can avoid
    /// carrying however well it matches everything else.
    fn operation(&mut self, rng: &mut Rng, seq: u64, new_party_percent: usize) -> Vec<u8> {
        let now = 1_900_000_000_u64 + seq * 60;
        let mut invented = Vec::new();

        if rng.below(100) < new_party_percent {
            let fresh = party(rng, 900_000 + seq, now);
            invented.extend_from_slice(fresh.1.name.as_bytes());
            invented.extend_from_slice(fresh.1.tax_id.as_bytes());
            self.parties.push(fresh);
        } else {
            let which = rng.below(self.parties.len());
            self.parties[which].1.touched_at = now;
            self.parties[which].1.operations += 1;
        }
        while self.parties.len() > PARTY_LIMIT {
            let oldest = self
                .parties
                .iter()
                .enumerate()
                .min_by_key(|(_, (key, party))| (party.touched_at, key.clone()))
                .map(|(index, _)| index)
                .unwrap();
            self.parties.remove(oldest);
        }

        let id = 1_000_000 + seq;
        let kind = pick_kind(rng);
        let fresh = entry(rng, id, now, kind);
        invented.extend_from_slice(&fresh.body);
        let back = rng.below(3);
        let at = self.entries.len() - back.min(self.entries.len());
        self.entries.insert(at, fresh);
        while self.entries.len() > ENTRY_LIMIT {
            self.entries.remove(0);
        }

        if kind < PRICED {
            let key = format!("op-{id:012}");
            let note = note(rng);
            let context = format!("{{\"step\":\"{}\",\"at\":{}}}", rng.next() % 9, rng.next());
            invented.extend_from_slice(note.as_bytes());
            invented.extend_from_slice(context.as_bytes());
            self.notes.push((key.clone(), note));
            self.context.push((key, context));
        }

        let live: Vec<String> = self
            .entries
            .iter()
            .filter(|entry| entry.kind < PRICED)
            .map(|entry| format!("op-{:012}", entry.id))
            .collect();
        self.notes.retain(|(key, _)| live.contains(key));
        self.context.retain(|(key, _)| live.contains(key));

        invented
    }
}

fn write_pairs(out: &mut Vec<u8>, name: &str, pairs: &[(String, String)]) {
    if pairs.is_empty() {
        return;
    }
    let mut sorted = pairs.to_vec();
    sorted.sort_by(|left, right| left.0.cmp(&right.0));
    text(out, name);
    map(out, sorted.len());
    for (key, value) in &sorted {
        text(out, key);
        text(out, value);
    }
}

// ---------------------------------------------------------------------------

struct Outcome {
    base: usize,
    patches: usize,
    mean: f64,
    worst: usize,
    over: usize,
    escalated: usize,
    restarted: usize,
    build_ms: f64,
    scanned: u64,
    /// Worst ratio of patch to the compressed size of the content the batch
    /// invented. A ratio near 1 means every byte in the patch had to be there.
    overhead: f64,
    peak_heap: usize,
}

/// `batch` operations are applied between two saves, so one patch has to
/// describe all of them at once. That is what decides how far the head of the
/// log has shifted, and therefore whether the shift is still inside the
/// window the matcher searches before it gives up and indexes the base.
fn run(patches: usize, batch: usize, new_party_percent: usize) -> Outcome {
    let _meter = meter();
    let mut ledger = Ledger::new(20_260_906);
    let mut rng = Rng(0xC0FFEE);
    let base = ledger.encode();
    let mut current = base.clone();

    let mut worst = 0;
    let mut total = 0_usize;
    let mut over = 0;
    let mut escalated = 0;
    let mut restarted = 0;
    let mut build_us = 0_u128;
    let mut scanned = 0_u64;
    let mut overhead = 0.0_f64;
    let mut peak_heap = 0_usize;

    for round in 0..patches {
        let mut invented = Vec::new();
        for step in 0..batch {
            invented.extend_from_slice(&ledger.operation(
                &mut rng,
                (round * batch + step) as u64,
                new_party_percent,
            ));
        }
        let next = ledger.encode();

        let start = Instant::now();
        let ((patch, stats), heap) = measure_peak(|| make_patch_stats(&current, &next).unwrap());
        build_us += start.elapsed().as_micros();
        peak_heap = peak_heap.max(heap);
        assert_eq!(apply_patch(&current, &patch).unwrap(), next, "roundtrip");

        let compressed = zstd::encode_all(&patch[..], COMPRESSION_LEVEL)
            .unwrap()
            .len();
        let floor = zstd::encode_all(&invented[..], COMPRESSION_LEVEL)
            .unwrap()
            .len();
        overhead = overhead.max(compressed as f64 / floor as f64);
        total += compressed;
        worst = worst.max(compressed);
        over += usize::from(compressed > THRESHOLD);
        escalated += usize::from(stats.escalations > 0);
        restarted += usize::from(stats.restarts > 0);
        scanned = scanned.max(stats.old_bytes_scanned);
        current = next;
    }

    Outcome {
        base: base.len(),
        patches,
        mean: total as f64 / patches as f64,
        worst,
        over,
        escalated,
        restarted,
        build_ms: build_us as f64 / 1000.0 / patches as f64,
        scanned,
        overhead,
        peak_heap,
    }
}

fn report(label: &str, outcome: &Outcome) {
    println!(
        "| {label} | {:.0} B | {} B | {:.2}x | {}/{} | {}/{} | {}/{} | {:.2} ms | {:.0}% | {} KiB |",
        outcome.mean,
        outcome.worst,
        outcome.overhead,
        outcome.over,
        outcome.patches,
        outcome.escalated,
        outcome.patches,
        outcome.restarted,
        outcome.patches,
        outcome.build_ms,
        outcome.scanned as f64 * 100.0 / outcome.base as f64,
        outcome.peak_heap / 1024,
    );
}

const HEADER: &str = "| mean | worst | vs new content | over | escalated | restarted | build | peak base read | peak heap |";
const RULE: &str = "|---|---|---|---|---|---|---|---|---|---|";

#[test]
fn one_operation_per_patch_stays_under_the_threshold() {
    println!("\nOne operation per patch. zstd level {COMPRESSION_LEVEL}, threshold {THRESHOLD} B.");
    println!("\n| new parties {HEADER}");
    println!("{RULE}");

    let mut worst_overall = 0;
    for share in [0_usize, 20, 100] {
        let outcome = run(40, 1, share);
        report(&format!("{share}%"), &outcome);
        worst_overall = worst_overall.max(outcome.worst);

        assert_eq!(outcome.over, 0, "a patch exceeded the rewrite threshold");
        assert_eq!(
            outcome.escalated, 0,
            "a local edit reached the global index"
        );
        assert_eq!(outcome.restarted, 0);
        assert!(
            outcome.scanned < outcome.base as u64 / 2,
            "read {} bytes of a {}-byte base",
            outcome.scanned,
            outcome.base
        );
        assert!(
            outcome.overhead < 1.5,
            "{:.2}x the new content",
            outcome.overhead
        );
        // An index over this base would be tens of megabytes. Staying under a
        // megabyte is what says none was built.
        assert!(
            outcome.peak_heap < 1 << 20,
            "peak heap {} KiB on a {} KiB base",
            outcome.peak_heap / 1024,
            outcome.base / 1024
        );
    }
    println!("\nWorst patch: {worst_overall} B of {THRESHOLD} B allowed.");
}

/// Operations accumulate between saves. Each one evicts an entry from the head
/// of the log, so a batch of `n` shifts everything after it by roughly `n`
/// entries, and the whole tail of the document moves with it.
///
/// Past a certain batch the patch stops fitting under the threshold, and it
/// should: the operations invented more new content than the threshold allows.
/// What must not happen is the patch growing faster than that content, or the
/// matcher giving up on the shift and indexing the whole base to find it.
#[test]
fn batched_operations_survive_the_head_shift() {
    println!("\nOperations accumulated between saves. Same level and threshold.");
    println!("\n| batch {HEADER}");
    println!("{RULE}");

    for batch in [2_usize, 4, 8, 16, 32] {
        let outcome = run(16, batch, 20);
        report(&format!("{batch}"), &outcome);
        assert!(
            outcome.overhead < 1.5,
            "a batch of {batch} produced {:.2}x the content it invented",
            outcome.overhead
        );
        // Once a batch is over the threshold the base is rewritten anyway, so
        // what the matcher spends on it stops mattering. Below the threshold
        // it does, and there the shift has to be found locally.
        if outcome.over == 0 {
            assert_eq!(
                outcome.escalated, 0,
                "a batch of {batch} indexed the whole base to build a patch that fits"
            );
        }
    }
}
