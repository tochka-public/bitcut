//! Self-checks for the benchmark corpus generator (`benches/corpus/mod.rs`).
//!
//! Lives under `tests/` so the plain `cargo test` gate covers it: a silently
//! broken corpus would invalidate every measurement taken from it.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::default_numeric_fallback,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::as_conversions,
    clippy::missing_panics_doc
)]

#[path = "../benches/corpus/mod.rs"]
mod corpus;

use corpus::{base_profile, Params, Scenario};

#[test]
fn generation_is_deterministic() {
    let a = base_profile(Params::default());
    let b = base_profile(Params::default());
    assert_eq!(a, b, "base profile generation must be deterministic");

    for scenario in Scenario::ALL {
        assert_eq!(
            scenario.apply(Params::default()),
            scenario.apply(Params::default()),
            "scenario {} must be deterministic",
            scenario.name()
        );
    }
}

#[test]
fn base_profile_has_production_scale() {
    let base = base_profile(Params::default());
    assert!(
        (1 << 20..1 << 22).contains(&base.len()),
        "expected a 1-4 MiB base profile, got {} bytes",
        base.len()
    );
}

#[test]
fn scenarios_differ_from_base_and_from_each_other() {
    let base = base_profile(Params::default());
    let variants: Vec<Vec<u8>> = Scenario::ALL
        .iter()
        .map(|s| s.apply(Params::default()))
        .collect();

    for (scenario, variant) in Scenario::ALL.iter().zip(&variants) {
        assert_ne!(
            variant,
            &base,
            "scenario {} changed nothing",
            scenario.name()
        );
    }
    for (i, a) in variants.iter().enumerate() {
        for b in variants.iter().skip(i + 1) {
            assert_ne!(a, b, "scenarios must be distinguishable");
        }
    }
}

#[test]
fn append_only_changes_the_operation_count_and_the_tail() {
    // Sanity check on the corpus, not on bitcut. Appending to a definite-length
    // CBOR array rewrites the array header in place (12000 -> 12040, same
    // two-byte width) and leaves everything after it untouched. This is exactly
    // the production shape: a tiny edit early, then a pure suffix append.
    let base = base_profile(Params::default());
    let appended = Scenario::Append.apply(Params::default());
    assert!(appended.len() > base.len(), "append must grow the document");

    let common = base
        .iter()
        .zip(appended.iter())
        .take_while(|(x, y)| x == y)
        .count();
    let after_header = common + 2;
    assert_eq!(
        &base[after_header..],
        &appended[after_header..base.len()],
        "append must not disturb the body after the operations-array header"
    );
}

#[test]
fn status_edits_change_length() {
    // Length-changing edits are the whole point: they create the piecewise
    // constant drift the tiered differ is built around.
    let base = base_profile(Params::default());
    let edited = Scenario::StatusEdits.apply(Params::default());
    assert!(
        edited.len() > base.len(),
        "status edits must lengthen the document ({} -> {})",
        base.len(),
        edited.len()
    );
}

#[test]
fn roundtrip_holds_on_every_scenario() {
    let base = base_profile(Params::default());
    for scenario in Scenario::ALL {
        let new = scenario.apply(Params::default());
        let patch = bitcut::make_patch(&base, &new).unwrap();
        let restored = bitcut::apply_patch(&base, &patch).unwrap();
        assert_eq!(restored, new, "roundtrip failed for {}", scenario.name());
    }
}
