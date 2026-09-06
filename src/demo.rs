//! Instrumentation for the algorithm walkthrough.
//!
//! The walkthrough needs two things the library itself never wants: the
//! ability to shrink the whole search ladder until it is visible on a document
//! of a few dozen bytes, and a record of every decision the matcher made. Both
//! live behind the `demo` feature so a normal build carries neither.
//!
//! The record is emitted as JSON, one object per decision, in the order the
//! matcher made them. Nothing here reimplements any part of the matcher: the
//! events arrive from the same code that produces real patches, so the
//! walkthrough cannot drift away from it.

pub use crate::differ::{Params, ParamsError};

use crate::differ::{diff_traced, At, Chunk, Event, NoTrace, PatchStats, Trace, Via};
use std::fmt::Write as _;

/// Plan a diff under caller-supplied parameters.
///
/// # Errors
///
/// [`ParamsError`] if the ladder would only waste work. The matcher would
/// still produce a correct patch — the check exists so a caller tuning the
/// parameters is told when a combination is pointless, not to protect the
/// differ from it.
pub fn diff_tuned(
    old: &[u8],
    new: &[u8],
    params: Params,
) -> Result<(Vec<Chunk>, PatchStats), ParamsError> {
    params.check()?;
    let mut chunks = Vec::new();
    let stats = diff_traced(old, new, params, NoTrace, &mut chunks).0;
    Ok((chunks, stats))
}

/// Diff `old` into `new` and return the whole decision log as JSON.
///
/// # Errors
///
/// [`ParamsError`] if the parameters would only waste work.
pub fn trace_json(old: &[u8], new: &[u8], params: Params) -> Result<String, ParamsError> {
    params.check()?;
    let mut chunks = Vec::new();
    let recorder = diff_traced(old, new, params, Recorder::new(), &mut chunks).1;
    Ok(recorder.finish())
}

/// Writes each event as a JSON object into one growing string.
struct Recorder {
    out: String,
    frames: usize,
}

impl Recorder {
    fn new() -> Self {
        Self {
            out: String::from("["),
            frames: 0,
        }
    }

    fn finish(mut self) -> String {
        self.out.push(']');
        self.out
    }

    /// Open a frame: the separator, the event tag, and the state every frame
    /// carries.
    fn open(&mut self, tag: &str, at: At, stats: &PatchStats) {
        if self.frames > 0 {
            self.out.push(',');
        }
        self.frames = self.frames.saturating_add(1);
        // Writing into a String cannot fail; the result is discarded because
        // there is nothing a recorder could do about it if it could.
        let _ = write!(
            self.out,
            concat!(
                r#"{{"t":"{}","new":{},"old":{},"#,
                r#""s":[{},{},{},{},{},{},{},{}]"#
            ),
            tag,
            at.new,
            at.old,
            stats.copies,
            stats.literals,
            stats.resyncs,
            stats.shift_cache_hits,
            stats.window_searches,
            stats.old_bytes_scanned,
            stats.escalations,
            stats.restarts,
        );
    }

    fn field(&mut self, name: &str, value: usize) {
        let _ = write!(self.out, r#","{name}":{value}"#);
    }

    fn signed(&mut self, name: &str, value: i64) {
        let _ = write!(self.out, r#","{name}":{value}"#);
    }

    fn flag(&mut self, name: &str, value: bool) {
        let _ = write!(self.out, r#","{name}":{value}"#);
    }

    /// Bytes as a JSON string, with everything outside printable ASCII
    /// escaped. The walkthrough runs on text, but nothing guarantees it.
    fn bytes(&mut self, name: &str, value: &[u8]) {
        let _ = write!(self.out, r#","{name}":""#);
        for &byte in value {
            match byte {
                b'"' => self.out.push_str("\\\""),
                b'\\' => self.out.push_str("\\\\"),
                0x20..=0x21 | 0x23..=0x5B | 0x5D..=0x7E => self.out.push(char::from(byte)),
                other => {
                    let _ = write!(self.out, "\\u{other:04x}");
                }
            }
        }
        self.out.push('"');
    }

    fn drifts(&mut self, value: &[i64]) {
        let _ = write!(self.out, r#","drifts":["#);
        for (i, drift) in value.iter().enumerate() {
            if i > 0 {
                self.out.push(',');
            }
            let _ = write!(self.out, "{drift}");
        }
        self.out.push(']');
    }

    fn list(&mut self, name: &str, value: &[usize]) {
        let _ = write!(self.out, r#","{name}":["#);
        for (i, item) in value.iter().enumerate() {
            if i > 0 {
                self.out.push(',');
            }
            let _ = write!(self.out, "{item}");
        }
        self.out.push(']');
    }

    fn close(&mut self) {
        self.out.push('}');
    }
}

impl Recorder {
    /// The lockstep scan and the opcodes it produces.
    fn scan(&mut self, at: At, stats: &PatchStats, event: Event<'_>) -> bool {
        match event {
            Event::Compared { run, accepted } => {
                self.open("compared", at, stats);
                self.field("run", run);
                self.flag("ok", accepted);
            }
            Event::EmittedCopy { offset, len } => {
                self.open("copy", at, stats);
                self.field("offset", offset);
                self.field("len", len);
            }
            Event::ResyncOpened => self.open("resync", at, stats),
            Event::EmittedLiteral { start, len, resume } => {
                self.open("literal", at, stats);
                self.field("start", start);
                self.field("len", len);
                self.field("resume", resume);
            }
            Event::Restarted { carried } => {
                self.open("restart", at, stats);
                self.field("carried", carried);
            }
            Event::Finished => self.open("finished", at, stats),
            _ => return false,
        }
        true
    }

    /// The resync ladder.
    fn ladder(&mut self, at: At, stats: &PatchStats, event: Event<'_>) -> bool {
        match event {
            Event::GapTooLate { gap } => {
                self.open("gap-too-late", at, stats);
                self.field("gap", gap);
            }
            Event::GapProbed {
                gap,
                at_new,
                at_old,
                anchor,
                need,
                drifts,
            } => {
                self.open("gap", at, stats);
                self.field("gap", gap);
                self.field("atNew", at_new);
                self.field("atOld", at_old);
                self.field("need", need);
                self.bytes("anchor", anchor);
                self.drifts(drifts);
            }
            Event::DriftTried {
                drift,
                at: pos,
                extends,
                accepted,
            } => {
                self.open("drift", at, stats);
                self.signed("drift", drift);
                match pos {
                    Some(p) => self.field("at", p),
                    None => self.flag("oob", true),
                }
                self.field("extends", extends);
                self.flag("ok", accepted);
            }
            Event::WindowOpened { lo, hi } => {
                self.open("window", at, stats);
                self.field("lo", lo);
                self.field("hi", hi);
            }
            Event::CandidateTried {
                at: pos,
                extends,
                accepted,
            } => {
                self.open("candidate", at, stats);
                self.field("at", pos);
                self.field("extends", extends);
                self.flag("ok", accepted);
            }
            Event::Aligned {
                via,
                gap,
                at: pos,
                drift,
                literal,
                resume,
                walked_back,
            } => {
                self.open("aligned", at, stats);
                let _ = write!(
                    self.out,
                    r#","via":"{}""#,
                    match via {
                        Via::Cache => "cache",
                        Via::Window => "window",
                    }
                );
                self.field("gap", gap);
                self.field("at", pos);
                self.signed("drift", drift);
                self.field("literal", literal);
                self.field("resume", resume);
                self.field("walkedBack", walked_back);
            }
            Event::ResyncFailed => self.open("resync-failed", at, stats),
            _ => return false,
        }
        true
    }

    /// The escalation tier.
    fn escalation(&mut self, at: At, stats: &PatchStats, event: Event<'_>) -> bool {
        match event {
            Event::LadderWeighed {
                scanned,
                threshold,
                escalating,
            } => {
                self.open("ladder", at, stats);
                self.signed("scanned", i64::try_from(scanned).unwrap_or(i64::MAX));
                self.signed("threshold", i64::try_from(threshold).unwrap_or(i64::MAX));
                self.flag("escalating", escalating);
            }
            Event::IndexBuilt {
                entries,
                key_bytes,
                positions,
            } => {
                self.open("index-built", at, stats);
                self.field("entries", entries);
                self.field("keyBytes", key_bytes);
                self.list("positions", positions);
            }
            Event::IndexProbed {
                step,
                at: pos,
                extends,
                accepted,
            } => {
                self.open("index-probe", at, stats);
                self.field("step", step);
                self.field("at", pos);
                self.field("extends", extends);
                self.flag("ok", accepted);
            }
            Event::IndexExhausted => self.open("index-exhausted", at, stats),
            Event::Reclaimed { bytes } => {
                self.open("reclaim", at, stats);
                self.field("bytes", bytes);
            }
            Event::SkippedAhead { len } => {
                self.open("skip", at, stats);
                self.field("len", len);
            }
            _ => return false,
        }
        true
    }
}

impl Trace for Recorder {
    const ENABLED: bool = true;

    fn record(&mut self, at: At, stats: &PatchStats, event: Event<'_>) {
        if self.scan(at, stats, event)
            || self.ladder(at, stats, event)
            || self.escalation(at, stats, event)
        {
            self.close();
        }
    }
}
