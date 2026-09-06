# Step 1 — Baseline Measurement Harness

> **For the implementer:** Use `executing-plans` to execute this plan task-by-task.

**Goal:** Produce reproducible baseline numbers (build time, apply time, peak memory,
raw patch size, zstd-19 size, opcode count) for the *current* differ on four
production-shaped scenarios, before any algorithmic change.

**Architecture:** A deterministic corpus generator (seeded xorshift, no `rand`
dependency) emits CBOR-shaped profile documents: a counterparty dictionary followed by
an operation log. Four mutation scenarios derive `new` from `old`. Criterion benches
measure time; a separate `--bin` report target measures size/memory/opcode metrics that
Criterion cannot express. Both consume the same generator module.

**Tech stack:** Rust 2021, criterion 0.5, zstd (dev-only), no new runtime dependencies
in this step.

**Constraints:**
- Public API unchanged: `make_patch`, `make_patch_into`, `apply_patch`,
  `apply_patch_into`, `Op::iter`.
- Repository lint table stays as-is (`warnings = "deny"`, clippy `pedantic`/`cargo`,
  `arithmetic_side_effects`, `indexing_slicing`, `as_conversions` denied). Bench and
  report code may keep the existing `#![allow(...)]` bench header pattern.
- `justfile` gates (`fmt-check`, `clippy` with and without default features, `test`
  with and without default features) must stay green.

**Non-goals:** No differ changes, no format changes, no minmatch tuning. Those are
steps 2–4 and get their own plans.

---

## Findings that shape this step

| Observation | Evidence | Consequence |
|---|---|---|
| `benches/fixtures/` contains only `.gitkeep` | `ls benches/fixtures` | `cargo bench` currently panics on `fs::read`. Existing benches are dead code. |
| Fuzz target calls `bitcut::make_diff` | `fuzz/fuzz_targets/roundtrip.rs:15` | Symbol does not exist; fuzz workspace does not build. |
| Match threshold is `WINDOW_SIZE = 10` | `src/lib.rs:26`, `src/lib.rs:285` | Left of the 24–32 optimum the task statement measured; step 4 territory. |
| Index keeps the **first** occurrence in an unbounded `FxHashMap<u64, usize>` | `src/lib.rs:504-519` | Not an overwriting fixed-size table. Memory is O(old_len) entries (~50M entries at 50 MB base). The task statement's premise "narrow windows lose nothing to overwriting-table collisions" does not apply verbatim; the real baseline weakness is index build cost + memory, not collisions. |
| Hash lookup runs per byte of `new` | `src/lib.rs:280-281` | One `FxHashMap` probe per byte dominates build time. |
| `estimate_patch_capacity` clamps to 4096 | `src/lib.rs:325` | Patches above 4 KiB realloc; noise in the baseline, not a target. |

---

### Task 1: Deterministic corpus generator

**Files:**
- Create: `benches/corpus/mod.rs`

**Step 1: Write the failing test**

Add to `benches/corpus/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic() {
        assert_eq!(base_profile(Params::default()), base_profile(Params::default()));
    }

    #[test]
    fn scenarios_differ_from_base_and_from_each_other() {
        let base = base_profile(Params::default());
        let variants: Vec<Vec<u8>> =
            Scenario::ALL.iter().map(|s| s.apply(&base, Params::default())).collect();
        for v in &variants {
            assert_ne!(v, &base, "scenario must modify the base");
        }
        for (i, a) in variants.iter().enumerate() {
            for b in variants.iter().skip(i + 1) {
                assert_ne!(a, b, "scenarios must be distinguishable");
            }
        }
    }
}
```

**Step 2: Verify RED**

Run: `cargo test --bench corpus_selftest`
Expected: FAIL — module and symbols do not exist.

**Step 3: Implement the minimum change**

```rust
pub struct Params {
    pub counterparties: usize, // default 500
    pub operations: usize,     // default 12_000
    pub seed: u64,             // default 0x5EED_1234_5EED_1234
}

pub enum Scenario {
    DictionaryInsert,   // 3 counterparties inserted at arbitrary dictionary positions
    StatusEdits,        // 60 length-changing status edits scattered through the log
    Append,             // 40 operations appended at the end
    Combined,           // all three
}

impl Scenario {
    pub const ALL: [Scenario; 4] = [...];
    pub fn name(&self) -> &'static str;
    pub fn apply(&self, base: &[u8], params: Params) -> Vec<u8>;
}

pub fn base_profile(params: Params) -> Vec<u8>;
```

Encoding: real CBOR major-type bytes emitted by hand (definite-length map/array/text/
uint) — no `serde_cbor` dependency. Randomness: inline xorshift64* seeded from
`params.seed`, so the corpus is byte-identical across machines and runs.
Status edits must change length (`"done"` → `"reversed"`) so the drift is piecewise
constant with non-zero steps, matching the production shape.

**Step 4: Verify GREEN**

Run: `cargo test --bench corpus_selftest`
Expected: PASS, 2 tests.

**Step 5: Refactor and run gates**

Run: `cargo fmt --all && cargo clippy --all-targets --all-features -- -D warnings`
Expected: PASS with no new warnings.

---

### Task 2: Replace the dead fixture benches with corpus benches

**Files:**
- Modify: `benches/my_benchmark.rs` (full rewrite, 30 lines)
- Modify: `Cargo.toml` — add `[[bench]] name = "corpus_selftest"`, keep `harness = false`
  for the criterion bench

**Step 1: Write the failing test**

Not a unit test — the verification is that `cargo bench --bench my_benchmark -- --test`
runs to completion instead of panicking in `fs::read`.

**Step 2: Verify RED**

Run: `cargo bench --bench my_benchmark -- --test`
Expected: FAIL — `called \`Result::unwrap()\` on an \`Err\` value` from
`fs::read("benches/fixtures/huge_old")` (file is empty/absent).

**Step 3: Implement the minimum change**

Two criterion groups, four scenarios each:

- `make_patch/<scenario>` — `make_patch(&old, &new)`
- `apply_patch/<scenario>` — `apply_patch(&old, &patch)` on a pre-built patch

Corpus is generated once per group via `Criterion::benchmark_group` setup, not per
iteration. `black_box` both inputs.

**Step 4: Verify GREEN**

Run: `cargo bench --bench my_benchmark -- --test`
Expected: PASS, 8 benchmark functions execute once each.

**Step 5: Refactor and run gates**

Run: `cargo clippy --all-targets --all-features -- -D warnings`
Expected: PASS.

---

### Task 3: Size / memory / opcode report target

**Files:**
- Create: `benches/report.rs` (a second `[[bench]]` with `harness = false`, so it runs
  under `cargo bench --bench report` without adding a published binary)
- Modify: `Cargo.toml` — `zstd = "0.13"` under `[dev-dependencies]`, `[[bench]]` entry

**Step 3: Implement**

Emit a Markdown table to stdout with one row per scenario:

| column | source |
|---|---|
| `old_len`, `new_len` | corpus |
| `patch_raw` | `patch.len()` |
| `patch_zstd19` | `zstd::encode_all(&patch[..], 19)?.len()` |
| `ops_copy`, `ops_add` | `Op::iter(&patch)` tally |
| `literal_bytes` | sum of `Op::Add` payload lengths |
| `peak_alloc_bytes` | a `#[global_allocator]` counting shim local to this bench |
| `build_ms`, `apply_ms` | single-shot `Instant` timing, marked indicative — Criterion owns the authoritative timing |

Peak memory uses a counting allocator wrapper (`AtomicUsize` current + peak, `Relaxed`
loads, `AcqRel` on the peak CAS). It is bench-local; the library gains no allocator.

Columns for resync-cache hit rate and escalation rate are **added in step 2**, when the
statistics struct exists. In step 1 those cells print `n/a`.

**Step 4: Verify GREEN**

Run: `cargo bench --bench report`
Expected: a four-row Markdown table on stdout; exit code 0.

**Step 5: Gates**

Run: `just check && just test`
Expected: PASS.

---

### Task 4: Repair the fuzz round-trip target

**Files:**
- Modify: `fuzz/fuzz_targets/roundtrip.rs:15`

**Step 3: Implement**

Replace `bitcut::make_diff(old, new)` with
`bitcut::make_patch(old, new).expect("make_patch must succeed")`.

Scope note: the acceptance criteria also ask for empty inputs, fully-dissimilar inputs,
and the 4 GiB boundary. The 4 GiB case cannot run under libFuzzer (memory) and belongs
in `tests/property.rs` as an `#[ignore]`-gated test; it is deferred to step 2 together
with the `base_hash` mismatch test, because both target behavior that does not exist
yet.

**Step 4: Verify GREEN**

Run: `cargo +nightly fuzz build` (skip with a recorded note if the nightly toolchain is
absent on this machine)
Expected: both targets compile.

---

### Task 5: Record the baseline

**Files:**
- Create: `docs/plans/2026-09-04-baseline-numbers.md`

Paste the `cargo bench --bench report` table and the Criterion medians for all eight
timing benchmarks, with `rustc -vV` and CPU model. This file is the reference every
later step is compared against.

---

## Decisions, as resolved

| Decision | Outcome |
|---|---|
| `memchr` as a runtime dependency | Added. `xxhash-rust` (xxh3) and a dev-only `zstd` followed. |
| How statistics reach the caller without changing `make_patch` | New `make_patch_stats` / `make_patch_into_stats`; the existing four functions are untouched. |
| v2 encoder selection: Cargo feature vs. runtime | Neither. A Cargo feature that changes `make_patch` output is unsafe under feature unification, and a runtime switch would be a knob with no right answer. v2 is simply what gets written; v1 is what still gets read, detected from the first byte. |
| Header `base_len:u64` vs. `Op::Copy { offset: u32 }` | Kept as specified. The 4 GiB opcode limit stands; the u64 header field is forward room. |
| minmatch value | Swept 10..=32 against zstd-19. Settled on 16 — see `2026-09-04-baseline-numbers.md`, which also records why the sweep contradicted the brief's prediction. |
