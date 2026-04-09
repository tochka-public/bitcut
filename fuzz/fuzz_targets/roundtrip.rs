#![no_main]

//! Fuzz target: round-trip property
//! `apply_patch(old, make_diff(old, new)) == new` for arbitrary inputs.
//!
//! Run with: `cargo +nightly fuzz run roundtrip`

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let split = (data[0] as usize) % data.len();
    let (old, new) = data.split_at(split);
    let patch = bitcut::make_diff(old, new);
    let restored = bitcut::apply_patch(old, &patch).expect("valid patch must apply");
    assert_eq!(restored.as_slice(), new);
});
