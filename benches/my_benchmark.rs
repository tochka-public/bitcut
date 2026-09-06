#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::default_numeric_fallback,
    clippy::semicolon_if_nothing_returned
)]

//! Timing benchmarks over the production-shaped corpus.
//!
//! Run: `cargo bench --bench my_benchmark`

#[path = "corpus/mod.rs"]
mod corpus;

use bitcut::{apply_patch, make_patch};
use corpus::{as_u64, base_profile, Params, Scenario};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;

fn build(c: &mut Criterion) {
    let params = Params::default();
    let old = base_profile(params);
    let mut group = c.benchmark_group("make_patch");
    for scenario in Scenario::ALL {
        let new = scenario.apply(params);
        group.throughput(Throughput::Bytes(as_u64(new.len())));
        group.bench_function(scenario.name(), |b| {
            b.iter(|| make_patch(black_box(&old), black_box(&new)).unwrap());
        });
    }
    group.finish();
}

fn apply(c: &mut Criterion) {
    let params = Params::default();
    let old = base_profile(params);
    let mut group = c.benchmark_group("apply_patch");
    for scenario in Scenario::ALL {
        let new = scenario.apply(params);
        let patch = make_patch(&old, &new).unwrap();
        group.throughput(Throughput::Bytes(as_u64(new.len())));
        group.bench_function(scenario.name(), |b| {
            b.iter(|| apply_patch(black_box(&old), black_box(&patch)).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, build, apply);
criterion_main!(benches);
