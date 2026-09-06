//! Tiered resync differ.
//!
//! # Why not index the base
//!
//! The updates this crate is built for are local edits to a large document:
//! an entry inserted into a dictionary, a status flipped in the middle of a
//! log, records appended at the end. Every such edit shifts everything after
//! it by the same amount, so `drift = old_pos - new_pos` is a *piecewise
//! constant* function with one step per length-changing edit — typically
//! dozens of steps in a multi-megabyte document.
//!
//! Indexing the whole base to rediscover that costs `O(old.len())` time and
//! memory on every call. Instead the differ walks both documents in lockstep
//! and, at each divergence, spends a bounded search to find the next step:
//!
//! 1. **Remembered drift.** The same kind of edit produces the same length
//!    delta, so a handful of recent drifts closes most divergences with a
//!    single verified comparison and no search at all. Zero drift — an edit
//!    that did not change any length — is always tried first.
//! 2. **Windowed anchor search.** A 32-byte anchor taken from `new` is looked
//!    up with SIMD substring search in a window of `old` around the expected
//!    position, narrow first and then as wide as the document's own size
//!    warrants. Several anchor offsets are tried, because the bytes right
//!    after a divergence are the newly written ones and are not in `old` at
//!    all.
//! 3. **Skip ahead.** Newly written data has no counterpart to find. Rather
//!    than search harder, carry a bounded literal run and probe again; the
//!    runs merge, so an insertion of any size costs one literal and a few
//!    searches. For a pure insertion the drift never changed, so the very
//!    next probe realigns at zero drift.
//! 4. **Global index.** Built once, as soon as searching has walked a base's
//!    worth of `old` without a single hit. Searching that keeps finding
//!    alignments is searching that works, however much it has cost; a run of
//!    pure misses is the signal that the change is not local after all. This
//!    is what keeps a reorder, a bulk move, or a format change from
//!    degenerating into one document-sized literal.
//!
//! Search cost is therefore `steps × window`, independent of the size of the
//! base — as long as the update really is local. [`PatchStats::escalations`]
//! reports when it is not.

use crate::index::{build_hash_map, RollingHash, HASH_BASE, WINDOW_SIZE};
use crate::memcmp::simd_memcmp;
use memchr::memmem;
use rustc_hash::FxHashMap;

/// Wire size of a `Copy` opcode: tag, offset, length.
const COPY_WIRE_LEN: usize = 9;

/// Shortest match worth an opcode once the patch will be compressed before
/// storage, where the opcode *count* matters more than the opcode size: a
/// split interrupts the literal stream and can cost more after entropy coding
/// than the literal bytes it saved.
///
/// Swept over 10..=32 against zstd on the benchmark corpus. The curve is
/// almost flat — 1834 against 1849 bytes on the `combined` scenario between
/// the ends — because the tiered matcher already emits tens of opcodes rather
/// than hundreds, so there is little opcode pressure left to trade against. 16
/// is never worse than a larger value on any scenario and sits closest to the
/// raw wire break-even.
const MIN_MATCH_COMPRESSED: usize = 16;

/// Below this, a patch is too small for entropy coding to recover anything,
/// so the raw wire cost decides and the break-even is simply the size of a
/// `Copy`.
const COMPRESSION_MATTERS_ABOVE: usize = 4096;

/// Longest gap ladder the matcher will carry, and the width of the drift
/// cache. Both are fixed so neither allocates.
const MAX_GAPS: usize = 8;
const SHIFT_SLOTS: usize = 16;

/// How far around the expected position an anchor is looked for, as a fraction
/// of the document.
///
/// A fixed width cannot be right for both a config file and a multi-megabyte
/// record: an edit displaces the rest of the document by an amount that scales
/// with the records in it, not with any constant. The divisor is what keeps
/// one full round of probing an eighth of the base — past that the escalation
/// budget would rather pay for the index, and a round that overshoots the
/// budget by more than it saves is a round that should not have run.
const WINDOW_SHARE: usize = 128;
const WINDOW_MIN: usize = 4096;
const WINDOW_MAX: usize = 1 << 16;

/// What the matcher is allowed to spend, and how far it is allowed to look.
///
/// Every field is a measured default; they are grouped into a value only so
/// that a caller studying the algorithm can scale the whole ladder down far
/// enough to be visible on a document of a few dozen bytes. Nothing in the
/// crate's own paths ever passes anything but [`Params::for_input`].
///
/// Bad values cannot break the differ: the main loop refuses any step that
/// leaves both cursors where they were, so the diff always terminates and the
/// opcodes always reconstruct `new`. They can only produce a worse patch.
/// [`Params::check`] reports the combinations that would.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Params {
    /// Shortest match worth an opcode.
    pub min_match: usize,
    /// Length of the fragment handed to the substring search.
    pub anchor: usize,
    /// How far a candidate alignment must extend to be believed. Without it,
    /// a fragment that repeats across the document aligns the differ to the
    /// wrong place and every subsequent byte becomes a literal.
    pub anchor_verify: usize,
    /// Offsets the anchor is taken from, relative to the divergence point.
    /// The bytes immediately after a divergence are the newly written ones —
    /// they are not in `old` at all, so an anchor taken from there can never
    /// be found. Only the first gap that clears the edit matters, because
    /// everything past the resync point aligns at the same drift, so the
    /// ladder can be coarse. Ascending, starting at 0.
    pub gaps: [usize; MAX_GAPS],
    /// How many entries of `gaps` are in use.
    pub gap_count: usize,
    /// Half-width of the window of `old` searched for an anchor.
    pub window_half_width: usize,
    /// How many drift values are remembered.
    pub slots: usize,
}

/// A parameter combination that would produce a worse patch than it needs to.
///
/// Only the tuning path needs this: the crate's own entry points never build
/// anything but the defaults, so the check is not compiled into a normal
/// build at all.
#[cfg(any(test, feature = "demo"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamsError {
    /// `gaps` must hold at least one entry and start at 0.
    GapLadder,
    /// `gaps` must ascend, so that the widest is last.
    GapOrder,
    /// A match shorter than the anchor cannot be verified into a copy, so the
    /// matcher would find alignments it then refuses to use.
    MinMatchAboveAnchor,
    /// A verification bar below the anchor length verifies nothing: the
    /// anchor match would confirm itself.
    VerifyBelowAnchor,
    /// The window must be able to hold an anchor.
    WindowBelowAnchor,
    /// More remembered drifts than there are slots.
    TooManySlots,
}

impl Params {
    /// The measured defaults, with the match threshold chosen for this input
    /// size.
    #[must_use]
    pub fn for_input(new_len: usize) -> Self {
        Self {
            min_match: if new_len < COMPRESSION_MATTERS_ABOVE {
                COPY_WIRE_LEN.saturating_add(1)
            } else {
                MIN_MATCH_COMPRESSED
            },
            anchor: 32,
            anchor_verify: 64,
            gaps: [0, 32, 128, 512, 2048, 0, 0, 0],
            gap_count: 5,
            window_half_width: (new_len / WINDOW_SHARE).clamp(WINDOW_MIN, WINDOW_MAX),
            slots: SHIFT_SLOTS,
        }
    }

    /// The gap ladder actually in use.
    #[must_use]
    pub fn gaps(&self) -> &[usize] {
        self.gaps.get(..self.gap_count.min(MAX_GAPS)).unwrap_or(&[])
    }

    /// The ladder as a by-value iterator, so a caller holding `&mut self`
    /// elsewhere does not have to borrow — and does not have to allocate.
    fn ladder(self) -> impl Iterator<Item = usize> {
        self.gaps.into_iter().take(self.gap_count.min(MAX_GAPS))
    }

    /// Half-widths the anchor search tries, narrowest first. The narrow pass
    /// is dropped when it would be the wide one.
    fn window_widths(self) -> impl Iterator<Item = usize> {
        let narrow = WINDOW_MIN.min(self.window_half_width);
        let passes = if narrow < self.window_half_width {
            2
        } else {
            1
        };
        [narrow, self.window_half_width].into_iter().take(passes)
    }

    /// Literal run carried forward when no gap resyncs. It is the widest
    /// probed gap by construction: every gap up to it failed, so no resync
    /// point exists in the region they span and skipping exactly that far
    /// loses nothing.
    #[must_use]
    pub fn skip_ahead(&self) -> usize {
        self.gaps().last().copied().unwrap_or(0)
    }

    /// Bytes of `old` one full round of gap probing walks, at the widest
    /// window. The escalation budget adds it to what fruitless searching has
    /// already cost, so a round is only started if losing it would still leave
    /// the index the cheaper thing to have bought.
    #[must_use]
    pub fn search_round_cost(&self) -> usize {
        self.window_half_width
            .saturating_mul(2)
            .saturating_mul(self.gaps().len())
    }

    /// # Errors
    ///
    /// See [`ParamsError`]. A rejected combination still diffs correctly; it
    /// just wastes work or gives up matches it could have taken.
    #[cfg(any(test, feature = "demo"))]
    pub fn check(&self) -> Result<(), ParamsError> {
        let gaps = self.gaps();
        if gaps.first() != Some(&0) {
            return Err(ParamsError::GapLadder);
        }
        if gaps.windows(2).any(|w| w.first() >= w.last()) {
            return Err(ParamsError::GapOrder);
        }
        if self.min_match > self.anchor {
            return Err(ParamsError::MinMatchAboveAnchor);
        }
        if self.anchor_verify < self.anchor {
            return Err(ParamsError::VerifyBelowAnchor);
        }
        if self.window_half_width < self.anchor {
            return Err(ParamsError::WindowBelowAnchor);
        }
        if self.slots > SHIFT_SLOTS {
            return Err(ParamsError::TooManySlots);
        }
        Ok(())
    }
}

/// Counters describing how a patch was produced.
///
/// A rising [`Self::escalations`] share in production means updates stopped
/// being local edits — the data schema changed, or something started
/// rewriting documents wholesale.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PatchStats {
    /// `Copy` opcodes emitted.
    pub copies: u32,
    /// `Add` opcodes emitted.
    pub literals: u32,
    /// Divergence points that required a resync.
    pub resyncs: u32,
    /// Resyncs closed by the remembered-drift cache, without any search.
    pub shift_cache_hits: u32,
    /// Substring searches performed over a window of `old`.
    pub window_searches: u32,
    /// Bytes of `old` handed to the substring search.
    pub old_bytes_scanned: u64,
    /// Resyncs that fell through to the global index.
    pub escalations: u32,
    /// Whether the diff was restarted after the index was built. At most 1:
    /// the index is built once, and everything decided before it existed was
    /// decided without being able to see the whole base.
    pub restarts: u32,
}

/// A planned opcode, still referring to `old` / `new` by position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chunk {
    /// Copy `len` bytes of `old` starting at `offset`.
    Copy {
        /// Offset within `old`.
        offset: usize,
        /// Number of bytes to copy.
        len: usize,
    },
    /// Emit `len` bytes of `new` starting at `start` verbatim.
    Add {
        /// Offset within `new`.
        start: usize,
        /// Number of bytes to emit.
        len: usize,
    },
}

/// Where the matcher stood when an event was recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct At {
    /// Cursor in the new document.
    pub new: usize,
    /// Cursor in the old document.
    pub old: usize,
}

/// Which tier answered a resync.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Via {
    /// A remembered drift, verified without searching for anything.
    Cache,
    /// A windowed anchor search.
    Window,
}

/// One decision the matcher made, in the order it made it.
///
/// The payloads exist for the recorder in `demo`, which a build without that
/// feature deliberately does not compile — so in such a build every field is
/// genuinely unread, and saying so is more honest than inventing a consumer.
#[cfg_attr(not(feature = "demo"), allow(dead_code))]
#[derive(Debug, Clone, Copy)]
pub enum Event<'a> {
    /// Compared the two documents from the cursors onward.
    Compared {
        /// Bytes that matched.
        run: usize,
        /// Whether that was enough to spend an opcode on.
        accepted: bool,
    },
    /// Wrote a copy opcode.
    EmittedCopy {
        /// Offset in the old document.
        offset: usize,
        /// Bytes copied.
        len: usize,
    },
    /// A divergence: the tiers begin here.
    ResyncOpened,
    /// This gap has no anchor left to take.
    GapTooLate {
        /// The gap that did not fit.
        gap: usize,
    },
    /// Took an anchor at this gap.
    GapProbed {
        /// Offset of the anchor from the divergence point.
        gap: usize,
        /// Where the anchor was taken from.
        at_new: usize,
        /// Where it would sit in the old document under zero drift.
        at_old: usize,
        /// The anchor itself.
        anchor: &'a [u8],
        /// How far a candidate must extend to be believed.
        need: usize,
        /// The remembered drifts, most recent first.
        drifts: &'a [i64],
    },
    /// Tried one remembered drift against the anchor.
    DriftTried {
        /// The drift tried. Zero is always tried first.
        drift: i64,
        /// Where it pointed, or `None` if that was outside the document.
        at: Option<usize>,
        /// How far the match extended from there.
        extends: usize,
        /// Whether that cleared the bar.
        accepted: bool,
    },
    /// Opened a window of the old document for a substring search.
    WindowOpened {
        /// Start of the window.
        lo: usize,
        /// End of the window.
        hi: usize,
    },
    /// The search returned an occurrence and it was verified.
    CandidateTried {
        /// Where the anchor was found.
        at: usize,
        /// How far the match extended from there.
        extends: usize,
        /// Whether that cleared the bar.
        accepted: bool,
    },
    /// A tier answered. The walk-back has already been applied.
    Aligned {
        /// Which tier.
        via: Via,
        /// The gap whose anchor matched.
        gap: usize,
        /// Where the anchor was found in the old document.
        at: usize,
        /// Distance between where the anchor was expected and where it was.
        drift: i64,
        /// Literal length after walking back.
        literal: usize,
        /// Where matching resumes in the old document.
        resume: usize,
        /// Bytes handed back to the copy by the walk-back.
        walked_back: usize,
    },
    /// Wrote a literal opcode.
    EmittedLiteral {
        /// Offset in the new document.
        start: usize,
        /// Bytes carried.
        len: usize,
        /// Where matching resumes in the old document.
        resume: usize,
    },
    /// No tier answered.
    ResyncFailed,
    /// Weighed one more round of local searching against indexing the base.
    LadderWeighed {
        /// Bytes of the base read by window searches so far.
        scanned: u64,
        /// What indexing the base would cost.
        threshold: u64,
        /// Whether the index won.
        escalating: bool,
    },
    /// Built the global index.
    IndexBuilt {
        /// Entries it holds — one per key, first occurrence only.
        entries: usize,
        /// Bytes of `old` each key covers.
        key_bytes: usize,
        /// The positions in `old` it kept, ascending. Every other position is
        /// a later occurrence of a key already stored, and was dropped.
        positions: &'a [usize],
    },
    /// Looked one position up in the index.
    IndexProbed {
        /// How far forward in the new document this was.
        step: usize,
        /// Where the index pointed.
        at: usize,
        /// How far the match extended from there.
        extends: usize,
        /// Whether that cleared the minimum match.
        accepted: bool,
    },
    /// The index swept the rest of the new document and found nothing.
    IndexExhausted,
    /// Walked the match back into a literal that had already been emitted,
    /// handing those bytes to the copy instead.
    Reclaimed {
        /// Bytes taken back from the literal.
        bytes: usize,
    },
    /// Carried a bounded literal run and moved on.
    SkippedAhead {
        /// Bytes carried.
        len: usize,
    },
    /// Dropped everything decided before the index existed and started over.
    Restarted {
        /// Literal bytes that had been carried blind.
        carried: usize,
    },
    /// Carried the remainder of the new document verbatim.
    Finished,
}

/// A sink for [`Event`]s.
///
/// `ENABLED` is what keeps this free: every record site is behind
/// `if T::ENABLED`, so for [`NoTrace`] the compiler removes the branch, the
/// event construction and the call together.
pub trait Trace {
    /// Whether anything is listening.
    const ENABLED: bool;
    /// Record one event. `stats` is the running cost so far, which a
    /// recorder would otherwise have to reconstruct.
    fn record(&mut self, at: At, stats: &PatchStats, event: Event<'_>);
}

/// The sink used by every path in the crate itself.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoTrace;

impl Trace for NoTrace {
    const ENABLED: bool = false;
    #[inline(always)]
    fn record(&mut self, _at: At, _stats: &PatchStats, _event: Event<'_>) {}
}

/// Plan the opcodes reconstructing `new` from `old`, appending them to
/// `chunks`.
pub fn diff(old: &[u8], new: &[u8], chunks: &mut Vec<Chunk>) -> PatchStats {
    diff_with(old, new, Params::for_input(new.len()), chunks)
}

/// Plan the opcodes under explicit parameters.
pub fn diff_with(old: &[u8], new: &[u8], params: Params, chunks: &mut Vec<Chunk>) -> PatchStats {
    diff_traced(old, new, params, NoTrace, chunks).0
}

/// Plan the opcodes, recording every decision into `trace`, which is handed
/// back when the diff is done.
pub fn diff_traced<T: Trace>(
    old: &[u8],
    new: &[u8],
    params: Params,
    trace: T,
    chunks: &mut Vec<Chunk>,
) -> (PatchStats, T) {
    let mut differ = Differ {
        old,
        new,
        shifts: ShiftCache::new(),
        index: None,
        params,
        cursor_new: 0,
        cursor_old: 0,
        fruitless: 0,
        wide_exhausted: false,
        trace,
        stats: PatchStats::default(),
    };
    differ.run(chunks);
    (differ.stats, differ.trace)
}

// ---------------------------------------------------------------------------

struct Differ<'a, T: Trace> {
    old: &'a [u8],
    new: &'a [u8],
    shifts: ShiftCache,
    /// Built at most once per call, on the first escalation.
    index: Option<FxHashMap<u64, usize>>,
    params: Params,
    /// Kept on the differ rather than in `run` so that a tier several calls
    /// deep can report where it is without threading the cursors through.
    cursor_new: usize,
    cursor_old: usize,
    /// Bytes of `old` walked by window searches that found nothing, since the
    /// last one that did. What the escalation budget weighs: total scanning
    /// says how hard the diff is, this says whether searching still works.
    fruitless: u64,
    /// Whether the widest window has already failed at this stretch of `old`
    /// with nothing resyncing since.
    wide_exhausted: bool,
    trace: T,
    stats: PatchStats,
}

/// What to do at a divergence point.
enum Step {
    /// Emit `literal` bytes of `new`, then resume matching `old` at `aligned`.
    Realign { literal: usize, aligned: usize },
    /// Nothing matched nearby. Carry `literal` bytes and probe again further
    /// along.
    Skip { literal: usize },
    /// Nothing will match again. Carry the rest of `new` verbatim.
    Finish,
}

/// One anchor lookup: where it was taken from in `new`, where it would sit in
/// `old` under zero drift, and how far the alignment must extend to count.
struct Probe<'a> {
    gap: usize,
    at_new: usize,
    at_old: usize,
    anchor: &'a [u8],
    need: usize,
}

impl<'a, T: Trace> Differ<'a, T> {
    fn at(&self) -> At {
        At {
            new: self.cursor_new,
            old: self.cursor_old,
        }
    }

    fn run(&mut self, chunks: &mut Vec<Chunk>) {
        self.cursor_new = 0;
        self.cursor_old = 0;

        while self.cursor_new < self.new.len() {
            let run = self.extend(self.cursor_new, self.cursor_old);
            let long_enough = run >= self.params.min_match;
            if T::ENABLED {
                let at = self.at();
                self.trace.record(
                    at,
                    &self.stats,
                    Event::Compared {
                        run,
                        accepted: long_enough,
                    },
                );
            }
            if long_enough {
                chunks.push(Chunk::Copy {
                    offset: self.cursor_old,
                    len: run,
                });
                self.stats.copies = self.stats.copies.saturating_add(1);
                if T::ENABLED {
                    let at = self.at();
                    self.trace.record(
                        at,
                        &self.stats,
                        Event::EmittedCopy {
                            offset: self.cursor_old,
                            len: run,
                        },
                    );
                }
                self.cursor_new = self.cursor_new.saturating_add(run);
                self.cursor_old = self.cursor_old.saturating_add(run);
                if self.cursor_new >= self.new.len() {
                    break;
                }
            }

            self.stats.resyncs = self.stats.resyncs.saturating_add(1);
            if T::ENABLED {
                let at = self.at();
                self.trace.record(at, &self.stats, Event::ResyncOpened);
            }

            let blind = self.index.is_none();
            let outcome = self.step();

            // Everything decided before the index existed was decided without
            // being able to see the whole base — in particular every literal
            // carried by skip-ahead, which gives up on content the index can
            // find. Once it exists, redo the whole diff with it in hand.
            //
            // The index is built at most once, so this triggers at most once;
            // `restarts` makes that a fact of the loop rather than an
            // inference about `escalate`.
            if blind && self.index.is_some() && self.stats.restarts == 0 {
                let carried = if T::ENABLED { literal_bytes(chunks) } else { 0 };
                self.stats.restarts = 1;
                self.stats.copies = 0;
                self.stats.literals = 0;
                chunks.clear();
                self.cursor_new = 0;
                self.cursor_old = 0;
                if T::ENABLED {
                    let at = self.at();
                    self.trace
                        .record(at, &self.stats, Event::Restarted { carried });
                }
                continue;
            }

            // The guards reject a step that would leave both cursors where
            // they are. The tier rules make that unreachable — an accepted
            // alignment always extends at least `min_match` bytes, which the
            // run above did not — but the loop must terminate by
            // construction, not by argument.
            match outcome {
                Step::Realign { literal, aligned } if literal > 0 || aligned != self.cursor_old => {
                    // The walk-back inside `align` stops at the divergence
                    // point. If it got all the way there, the match may keep
                    // running into whatever was emitted before it.
                    let aligned = if literal == 0 {
                        let back = self.reclaim(chunks, aligned);
                        self.cursor_new = self.cursor_new.saturating_sub(back);
                        aligned.saturating_sub(back)
                    } else {
                        aligned
                    };
                    self.carry(chunks, literal, aligned);
                }
                Step::Skip { literal } if literal > 0 => {
                    let resume = self.cursor_old;
                    self.carry(chunks, literal, resume);
                }
                Step::Realign { .. } | Step::Skip { .. } | Step::Finish => {
                    // The closing literal goes through `carry` like every
                    // other one: emitting it silently would leave a recorder
                    // describing a patch that is missing its tail.
                    let rest = self.new.len().saturating_sub(self.cursor_new);
                    let resume = self.cursor_old;
                    self.carry(chunks, rest, resume);
                    if T::ENABLED {
                        let at = self.at();
                        self.trace.record(at, &self.stats, Event::Finished);
                    }
                    break;
                }
            }
        }
    }

    /// Emit `literal` bytes verbatim and resume matching `old` at `resume`.
    fn carry(&mut self, chunks: &mut Vec<Chunk>, literal: usize, resume: usize) {
        let start = self.cursor_new;
        self.push_literal(chunks, start, literal);
        if T::ENABLED {
            let at = self.at();
            self.trace.record(
                at,
                &self.stats,
                Event::EmittedLiteral {
                    start,
                    len: literal,
                    resume,
                },
            );
        }
        self.cursor_new = self.cursor_new.saturating_add(literal);
        self.cursor_old = resume;
    }

    /// Decide what to do at a divergence, cheapest tier first.
    fn step(&mut self) -> Step {
        if let Some((literal, aligned)) = self.resync() {
            return Step::Realign { literal, aligned };
        }
        if T::ENABLED {
            let at = self.at();
            self.trace.record(at, &self.stats, Event::ResyncFailed);
        }

        // The global index costs a pass over `old` to build. Build it as soon
        // as one more round of local searching would cost as much — which is
        // immediately for a small base, and only after many failed rounds for
        // a large one. Once it exists it is the cheapest tier available.
        let spent = self
            .fruitless
            .saturating_add(u64::try_from(self.params.search_round_cost()).unwrap_or(u64::MAX));
        let index_cost = u64::try_from(self.old.len()).unwrap_or(u64::MAX);
        let escalating = self.index.is_some() || spent >= index_cost;
        if T::ENABLED {
            let at = self.at();
            self.trace.record(
                at,
                &self.stats,
                Event::LadderWeighed {
                    scanned: self.stats.old_bytes_scanned,
                    threshold: index_cost,
                    escalating,
                },
            );
        }
        if escalating {
            return match self.escalate() {
                Some((literal, aligned)) => Step::Realign { literal, aligned },
                // The index scanned the rest of `new` and found nothing;
                // nothing further can match either.
                None => Step::Finish,
            };
        }

        let skip = self.params.skip_ahead();
        // Newly written data — an insertion longer than the widest gap, or an
        // append with no counterpart at all. Carry it and try again further
        // along; consecutive skips merge into one literal.
        if self.new.len().saturating_sub(self.cursor_new) <= skip {
            Step::Finish
        } else {
            if T::ENABLED {
                let at = self.at();
                self.trace
                    .record(at, &self.stats, Event::SkippedAhead { len: skip });
            }
            Step::Skip { literal: skip }
        }
    }

    /// Length of the common prefix of `new[at_new..]` and `old[at_old..]`.
    fn extend(&self, at_new: usize, at_old: usize) -> usize {
        match (self.new.get(at_new..), self.old.get(at_old..)) {
            (Some(head), Some(tail)) => simd_memcmp(head, tail),
            _ => 0,
        }
    }

    fn push_literal(&mut self, chunks: &mut Vec<Chunk>, start: usize, len: usize) {
        if len == 0 {
            return;
        }
        if let Some(Chunk::Add {
            start: prev_start,
            len: prev_len,
        }) = chunks.last_mut()
        {
            if prev_start.saturating_add(*prev_len) == start {
                *prev_len = prev_len.saturating_add(len);
                return;
            }
        }
        chunks.push(Chunk::Add { start, len });
        self.stats.literals = self.stats.literals.saturating_add(1);
    }

    /// Find where `old` continues after a divergence. Returns the literal
    /// length to emit and the position in `old` to resume from.
    fn resync(&mut self) -> Option<(usize, usize)> {
        for gap in self.params.ladder() {
            let Some(probe) = self.probe(gap) else {
                if T::ENABLED {
                    let at = self.at();
                    self.trace
                        .record(at, &self.stats, Event::GapTooLate { gap });
                }
                continue;
            };
            self.announce_gap(&probe);
            if let Some((pos, drift)) = self.probe_remembered(&probe) {
                self.stats.shift_cache_hits = self.stats.shift_cache_hits.saturating_add(1);
                self.wide_exhausted = false;
                return Some(self.accept(Via::Cache, &probe, pos, drift));
            }
        }

        // The window search exists to avoid paying for the index. Once the
        // index exists that reason is gone: it sees the whole base, in one
        // lookup, where a window sees a few kilobytes for the cost of scanning
        // them. Searching anyway means paying the window at every divergence
        // for the rest of the diff — on two documents that share structure but
        // no content that is thousands of divergences, and it dominates
        // everything else the differ does.
        if self.index.is_some() {
            return None;
        }

        // Narrow first. Most divergences displace the rest of the document by
        // a record or two, and widening the search for those would scan
        // kilobytes to find something that was always a few hundred bytes
        // away. The wide pass exists only for the displacements a narrow one
        // structurally cannot reach, and it runs only once the narrow one has
        // ruled them out.
        //
        // A failed wide pass is not repeated until something resyncs, because
        // a failure means the bytes here are new: nothing in `old` corresponds
        // to them, and the skip that follows moves `new` forward while leaving
        // the window over the same stretch of `old`. Searching it again each
        // skip is how an append ends up scanning the base once per two
        // kilobytes of appended data.
        for half_width in self.params.window_widths() {
            if half_width > WINDOW_MIN && self.wide_exhausted {
                break;
            }
            for gap in self.params.ladder() {
                let Some(probe) = self.probe(gap) else {
                    continue;
                };
                if let Some(pos) = self.search_window(&probe, half_width) {
                    let drift = shift_between(probe.at_old, pos).unwrap_or(0);
                    self.shifts.record(drift);
                    self.wide_exhausted = false;
                    return Some(self.accept(Via::Window, &probe, pos, drift));
                }
            }
        }
        self.wide_exhausted = true;
        None
    }

    fn announce_gap(&mut self, probe: &Probe<'a>) {
        if !T::ENABLED {
            return;
        }
        let mut drifts = [0_i64; SHIFT_SLOTS];
        drifts.copy_from_slice(&self.shifts.slots);
        let len = self.shifts.len.min(self.params.slots);
        let at = self.at();
        self.trace.record(
            at,
            &self.stats,
            Event::GapProbed {
                gap: probe.gap,
                at_new: probe.at_new,
                at_old: probe.at_old,
                anchor: probe.anchor,
                need: probe.need,
                drifts: drifts.get(..len).unwrap_or(&[]),
            },
        );
    }

    /// Hand the bytes that already matched between the divergence point and
    /// the anchor back to the copy, and report the alignment.
    fn accept(&mut self, via: Via, probe: &Probe<'_>, pos: usize, drift: i64) -> (usize, usize) {
        let (literal, resume) = self.align(probe.gap, pos);
        if T::ENABLED {
            let at = self.at();
            self.trace.record(
                at,
                &self.stats,
                Event::Aligned {
                    via,
                    gap: probe.gap,
                    at: pos,
                    drift,
                    literal,
                    resume,
                    walked_back: probe.gap.saturating_sub(literal),
                },
            );
        }
        (literal, resume)
    }

    /// `None` when this gap runs past the end of `new` and has no anchor.
    fn probe(&self, gap: usize) -> Option<Probe<'a>> {
        let new = self.new;
        let at_new = self.cursor_new.checked_add(gap)?;
        let anchor = new.get(at_new..at_new.checked_add(self.params.anchor)?)?;
        Some(Probe {
            gap,
            at_new,
            at_old: self.cursor_old.saturating_add(gap).min(self.old.len()),
            anchor,
            // Near the end of `new` there may be fewer than `anchor_verify`
            // bytes left; requiring the whole remainder is the strongest
            // check available.
            need: self
                .params
                .anchor_verify
                .min(new.len().saturating_sub(at_new)),
        })
    }

    fn probe_remembered(&mut self, probe: &Probe<'_>) -> Option<(usize, i64)> {
        // Zero drift is the most common outcome — an in-place edit that
        // changed no length — so it is tried before anything remembered.
        if let Some(pos) = self.try_drift(probe, 0) {
            return Some((pos, 0));
        }
        for slot in 0..self.shifts.len.min(self.params.slots) {
            let drift = self.shifts.slots.get(slot).copied()?;
            if let Some(pos) = self.try_drift(probe, drift) {
                self.shifts.record(drift);
                return Some((pos, drift));
            }
        }
        None
    }

    fn try_drift(&mut self, probe: &Probe<'_>, drift: i64) -> Option<usize> {
        let pos = shift_position(probe.at_old, drift);
        let extends = pos.map_or(0, |p| self.reach(probe, p));
        let accepted = pos.is_some() && extends >= probe.need;
        if T::ENABLED {
            let at = self.at();
            self.trace.record(
                at,
                &self.stats,
                Event::DriftTried {
                    drift,
                    at: pos,
                    extends,
                    accepted,
                },
            );
        }
        accepted.then_some(pos).flatten()
    }

    /// How far the new document matches the old one from `pos` onward.
    fn reach(&self, probe: &Probe<'_>, pos: usize) -> usize {
        match (self.new.get(probe.at_new..), self.old.get(pos..)) {
            (Some(head), Some(tail)) => simd_memcmp(head, tail),
            _ => 0,
        }
    }

    fn search_window(&mut self, probe: &Probe<'_>, half_width: usize) -> Option<usize> {
        let lo = probe.at_old.saturating_sub(half_width);
        let hi = probe
            .at_old
            .saturating_add(half_width)
            .saturating_add(self.params.anchor)
            .min(self.old.len());
        self.search_range(probe, lo, hi)
    }

    fn search_range(&mut self, probe: &Probe<'_>, lo: usize, hi: usize) -> Option<usize> {
        let haystack = self.old.get(lo..hi)?;
        self.stats.window_searches = self.stats.window_searches.saturating_add(1);
        if T::ENABLED {
            let at = self.at();
            self.trace
                .record(at, &self.stats, Event::WindowOpened { lo, hi });
        }

        let mut accepted = None;
        for offset in memmem::find_iter(haystack, probe.anchor) {
            let pos = lo.saturating_add(offset);
            let extends = self.reach(probe, pos);
            let ok = extends >= probe.need;
            if T::ENABLED {
                let at = self.at();
                self.trace.record(
                    at,
                    &self.stats,
                    Event::CandidateTried {
                        at: pos,
                        extends,
                        accepted: ok,
                    },
                );
            }
            if ok {
                accepted = Some(offset);
                break;
            }
        }
        // Charge only what the search actually walked: a hit stops there.
        let walked = accepted.map_or(haystack.len(), |offset| {
            offset.saturating_add(self.params.anchor)
        });
        let walked = u64::try_from(walked).unwrap_or(u64::MAX);
        self.stats.old_bytes_scanned = self.stats.old_bytes_scanned.saturating_add(walked);
        match accepted {
            // A hit means the local tier is still doing its job, whatever it
            // has cost so far. The tally only has to answer whether searching
            // has stopped working, so a hit clears it.
            Some(_) => self.fruitless = 0,
            None => self.fruitless = self.fruitless.saturating_add(walked),
        }

        accepted.map(|offset| lo.saturating_add(offset))
    }

    /// A skip commits bytes to a literal before knowing whether they would
    /// have matched — that is what makes it cheap. When an alignment is
    /// finally found, the match can run back past the start of this step and
    /// into that literal. Everything that does agree is handed back.
    ///
    /// Without this, every skip that is later realigned leaves a full
    /// `skip_ahead` of literal in the patch that the copy could have carried
    /// for nothing.
    fn reclaim(&mut self, chunks: &mut Vec<Chunk>, aligned: usize) -> usize {
        let Some(&Chunk::Add { start, len }) = chunks.last() else {
            return 0;
        };
        if start.saturating_add(len) != self.cursor_new {
            return 0;
        }

        let mut back: usize = 0;
        while back < len {
            let step = back.saturating_add(1);
            let (Some(from_new), Some(from_old)) = (
                self.cursor_new
                    .checked_sub(step)
                    .and_then(|i| self.new.get(i)),
                aligned.checked_sub(step).and_then(|i| self.old.get(i)),
            ) else {
                break;
            };
            if from_new != from_old {
                break;
            }
            back = step;
        }
        if back == 0 {
            return 0;
        }

        if back == len {
            chunks.pop();
            self.stats.literals = self.stats.literals.saturating_sub(1);
        } else if let Some(Chunk::Add { len: carried, .. }) = chunks.last_mut() {
            *carried = carried.saturating_sub(back);
        }
        if T::ENABLED {
            let at = self.at();
            self.trace
                .record(at, &self.stats, Event::Reclaimed { bytes: back });
        }
        back
    }

    /// Hand back the bytes that already matched between the divergence point
    /// and the anchor: they belong in the copy, not in the literal.
    fn align(&self, gap: usize, pos: usize) -> (usize, usize) {
        let mut literal = gap;
        let mut aligned = pos;
        while literal > 0 && aligned > 0 {
            let from_new = self
                .new
                .get(self.cursor_new.saturating_add(literal).saturating_sub(1));
            let from_old = self.old.get(aligned.saturating_sub(1));
            if from_new.is_none() || from_new != from_old {
                break;
            }
            literal = literal.saturating_sub(1);
            aligned = aligned.saturating_sub(1);
        }
        (literal, aligned)
    }

    /// Last resort: consult the global index. Scans `new` forward from the
    /// cursor until a verified match is found; a failed scan ends the diff, so
    /// the total cost across a call stays linear in `new`.
    fn escalate(&mut self) -> Option<(usize, usize)> {
        self.stats.escalations = self.stats.escalations.saturating_add(1);
        if self.index.is_none() {
            let built = build_hash_map(self.old);
            if T::ENABLED {
                let mut positions: Vec<usize> = built.values().copied().collect();
                positions.sort_unstable();
                let at = self.at();
                self.trace.record(
                    at,
                    &self.stats,
                    Event::IndexBuilt {
                        entries: built.len(),
                        key_bytes: WINDOW_SIZE,
                        positions: &positions,
                    },
                );
            }
            self.index = Some(built);
        }

        let cursor = self.cursor_new;
        let tail = self.new.get(cursor..)?;
        let hashes = RollingHash::new(tail, WINDOW_SIZE, HASH_BASE)?.enumerate();
        for (step, hash) in hashes {
            let Some(&pos) = self.index.as_ref()?.get(&hash) else {
                continue;
            };
            let extends = match (
                self.new.get(cursor.saturating_add(step)..),
                self.old.get(pos..),
            ) {
                (Some(head), Some(candidate)) => simd_memcmp(head, candidate),
                _ => 0,
            };
            let accepted = extends >= self.params.min_match;
            if T::ENABLED {
                let at = self.at();
                self.trace.record(
                    at,
                    &self.stats,
                    Event::IndexProbed {
                        step,
                        at: pos,
                        extends,
                        accepted,
                    },
                );
            }
            if accepted {
                return Some((step, pos));
            }
        }
        if T::ENABLED {
            let at = self.at();
            self.trace.record(at, &self.stats, Event::IndexExhausted);
        }
        None
    }
}

fn literal_bytes(chunks: &[Chunk]) -> usize {
    chunks
        .iter()
        .map(|chunk| match *chunk {
            Chunk::Add { len, .. } => len,
            Chunk::Copy { .. } => 0,
        })
        .sum()
}

// ---------------------------------------------------------------------------

/// Move-to-front cache of recent drift values, sized to stay on the stack.
struct ShiftCache {
    slots: [i64; SHIFT_SLOTS],
    len: usize,
}

impl ShiftCache {
    const fn new() -> Self {
        Self {
            slots: [0; SHIFT_SLOTS],
            len: 0,
        }
    }

    fn record(&mut self, shift: i64) {
        // Zero is probed unconditionally before the cache; storing it would
        // only displace a useful entry.
        if shift == 0 {
            return;
        }
        let known = self.slots.iter().take(self.len).position(|&s| s == shift);
        let mut at = known.unwrap_or_else(|| self.len.min(SHIFT_SLOTS.saturating_sub(1)));
        while at > 0 {
            let previous = self.slots.get(at.saturating_sub(1)).copied().unwrap_or(0);
            if let Some(slot) = self.slots.get_mut(at) {
                *slot = previous;
            }
            at = at.saturating_sub(1);
        }
        if let Some(slot) = self.slots.first_mut() {
            *slot = shift;
        }
        if known.is_none() {
            self.len = self.len.saturating_add(1).min(SHIFT_SLOTS);
        }
    }
}

fn shift_between(from: usize, to: usize) -> Option<i64> {
    i64::try_from(to)
        .ok()?
        .checked_sub(i64::try_from(from).ok()?)
}

fn shift_position(base: usize, shift: i64) -> Option<usize> {
    usize::try_from(i64::try_from(base).ok()?.checked_add(shift)?).ok()
}
