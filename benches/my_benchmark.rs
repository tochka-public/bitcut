use bitcut::make_diff;
use criterion::{criterion_group, criterion_main, Criterion};
use std::{fs, hint::black_box};

fn make_diff_huge(c: &mut Criterion) {
    let old = fs::read("benches/fixtures/huge_old").unwrap();
    let new = fs::read("benches/fixtures/huge_new").unwrap();
    c.bench_function("make_diff_huge", |b| {
        b.iter(|| make_diff(black_box(&old), black_box(&new)))
    });
}

fn make_diff_p90(c: &mut Criterion) {
    let old = fs::read("benches/fixtures/p90_old").unwrap();
    let new = fs::read("benches/fixtures/p90_new").unwrap();
    c.bench_function("make_diff_p90", |b| {
        b.iter(|| make_diff(black_box(&old), black_box(&new)))
    });
}

criterion_group!(benches, make_diff_huge, make_diff_p90);
criterion_main!(benches);
