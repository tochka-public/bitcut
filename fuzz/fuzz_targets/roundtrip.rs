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
    let patch = bitcut::make_patch(old, new).expect("make_patch must succeed");
    let restored = bitcut::apply_patch(old, &patch).expect("valid patch must apply");
    assert_eq!(restored.as_slice(), new);

    // The patch must name the base it was built from, and refuse any other.
    let header = bitcut::inspect(&patch)
        .expect("own patch must parse")
        .expect("own patch is v2");
    assert_eq!(
        (header.base_len, header.base_hash),
        bitcut::base_fingerprint(old)
    );
});
