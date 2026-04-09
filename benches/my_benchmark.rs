#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::semicolon_if_nothing_returned
)]

use bitcut::make_patch;
use criterion::{criterion_group, criterion_main, Criterion};
use std::{fs, hint::black_box};

fn make_patch_huge(c: &mut Criterion) {
    let old = fs::read("benches/fixtures/huge_old").unwrap();
    let new = fs::read("benches/fixtures/huge_new").unwrap();
    c.bench_function("make_patch_huge", |b| {
        b.iter(|| make_patch(black_box(&old), black_box(&new)).unwrap());
    });
}

fn make_patch_p90(c: &mut Criterion) {
    let old = fs::read("benches/fixtures/p90_old").unwrap();
    let new = fs::read("benches/fixtures/p90_new").unwrap();
    c.bench_function("make_patch_p90", |b| {
        b.iter(|| make_patch(black_box(&old), black_box(&new)).unwrap());
    });
}

criterion_group!(benches, make_patch_huge, make_patch_p90);
criterion_main!(benches);
