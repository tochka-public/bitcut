#![no_main]

//! Fuzz target: arbitrary `(old, patch)` bytes must never panic
//! `apply_patch` — only return `Err`.
//!
//! Run with: `cargo +nightly fuzz run apply_patch`

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    // Split data into (old, patch) at a pseudo-random position.
    let split = (data[0] as usize) % data.len();
    let (old, patch) = data.split_at(split);
    let _ = bitcut::apply_patch(old, patch);
});
