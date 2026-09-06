//! WASM shim for the algorithm walkthrough.
//!
//! The page hands over two documents and a set of matcher parameters, and gets
//! back the decision log as JSON. There is no binding generator involved: the
//! ABI is four exported functions and a linear-memory buffer, which is less
//! machinery than a generated binding and one fewer dependency.
//!
//! Calling convention, all from a single thread:
//!
//! 1. `alloc(len)` twice, write the two documents into the returned offsets;
//! 2. `alloc(gap_count * 4)`, write the gap ladder as `u32`s;
//! 3. `trace(...)`, which returns the length of the result, or `usize::MAX` if
//!    the parameters were rejected;
//! 4. `result_ptr()` and read that many bytes of UTF-8 JSON;
//! 5. `dealloc(ptr, len)` for each buffer handed out by `alloc`.
//!
//! Every entry point is `unsafe` at the boundary because the ABI is raw
//! pointers. Nothing below the boundary is: the slices are rebuilt once, at
//! the top of each function, and everything after that is safe Rust.

#![allow(clippy::missing_safety_doc)]

use bitcut::demo::{trace_json, Params};
use std::alloc::{alloc as raw_alloc, dealloc as raw_dealloc, Layout};
use std::cell::RefCell;

thread_local! {
    /// The last log produced, kept alive for the page to read. Its heap
    /// buffer does not move until the next `trace` call replaces it, which is
    /// the contract the page follows.
    static RESULT: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Rejected parameters, so the page can tell "no log" from "empty log".
const REJECTED: usize = usize::MAX;

fn layout(len: usize) -> Option<Layout> {
    Layout::from_size_align(len.max(1), 1).ok()
}

/// Reserve `len` bytes in linear memory.
///
/// Returns null if the allocation fails. The caller owns the result and must
/// hand it back to [`dealloc`].
#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    match layout(len) {
        // SAFETY: the layout has a non-zero size and an alignment of 1.
        Some(l) => unsafe { raw_alloc(l) },
        None => std::ptr::null_mut(),
    }
}

/// Release a buffer previously returned by [`alloc`].
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    if let Some(l) = layout(len) {
        // SAFETY: the caller guarantees `ptr` came from `alloc(len)`, so the
        // layout matches the one it was allocated with.
        unsafe { raw_dealloc(ptr, l) };
    }
}

/// Run the matcher and record every decision.
///
/// Returns the byte length of the JSON log, or [`REJECTED`] if the parameters
/// would only waste work.
#[no_mangle]
pub unsafe extern "C" fn trace(
    old_ptr: *const u8,
    old_len: usize,
    new_ptr: *const u8,
    new_len: usize,
    min_match: usize,
    anchor: usize,
    anchor_verify: usize,
    gaps_ptr: *const u32,
    gap_count: usize,
    window_half_width: usize,
    slots: usize,
) -> usize {
    if old_ptr.is_null() || new_ptr.is_null() || gaps_ptr.is_null() {
        return REJECTED;
    }
    // SAFETY: the caller guarantees each pointer addresses at least the stated
    // number of initialised bytes, all of it inside linear memory, and that
    // nothing mutates them for the duration of this call.
    let (old, new, gap_values) = unsafe {
        (
            std::slice::from_raw_parts(old_ptr, old_len),
            std::slice::from_raw_parts(new_ptr, new_len),
            std::slice::from_raw_parts(gaps_ptr, gap_count),
        )
    };

    let mut params = Params::for_input(new.len());
    params.min_match = min_match;
    params.anchor = anchor;
    params.anchor_verify = anchor_verify;
    params.window_half_width = window_half_width;
    params.slots = slots;
    params.gaps = [0; 8];
    params.gap_count = gap_count.min(params.gaps.len());
    for (slot, value) in params.gaps.iter_mut().zip(gap_values) {
        *slot = usize::try_from(*value).unwrap_or(0);
    }

    let Ok(json) = trace_json(old, new, params) else {
        return REJECTED;
    };
    let len = json.len();
    RESULT.with_borrow_mut(|slot| *slot = json);
    len
}

/// Start of the JSON produced by the last [`trace`] call.
#[no_mangle]
pub extern "C" fn result_ptr() -> *const u8 {
    RESULT.with_borrow(|slot| slot.as_ptr())
}
