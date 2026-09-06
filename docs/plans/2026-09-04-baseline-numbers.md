# Measurements

Apple M4 Pro, `rustc 1.98.0 (88d9e12ae 2026-08-18)`, `cargo bench` (release, no
`target-cpu=native`). Corpus: `benches/corpus/mod.rs` — 500 counterparties +
12 000 operations, CBOR, 1.6 MiB base.

| scenario | update |
|---|---|
| `dictionary_insert` | 3 counterparties inserted at arbitrary dictionary positions |
| `status_edits` | 60 length-changing status edits scattered through the log (`done` → `reversed`) |
| `append` | 40 operations appended (also rewrites the operations-array header) |
| `combined` | all three |

---

## Step 1 — baseline (global rolling-hash index, `WINDOW_SIZE = 10`)

`make_patch`, criterion median:

| scenario | time | throughput |
|---|---|---|
| dictionary_insert | 6.96 ms | 235 MiB/s |
| status_edits | 8.37 ms | 195 MiB/s |
| append | 8.16 ms | 201 MiB/s |
| combined | 7.00 ms | 234 MiB/s |

`apply_patch`, criterion median: 22.4 / 49.6 / 45.7 / 54.3 µs.

Size and memory:

| scenario | patch raw | zstd-19 | Copy | Add | literal bytes | peak heap (build) |
|---|---|---|---|---|---|---|
| dictionary_insert | 254 | 163 | 17 | 10 | 51 | 34.0 MiB |
| status_edits | 999 | 416 | 111 | 0 | 0 | 34.0 MiB |
| append | 3584 | 1546 | 263 | 155 | 442 | 34.0 MiB |
| combined | 4796 | 2064 | 386 | 166 | 492 | 34.0 MiB |

Peak heap is flat at 34 MiB regardless of the update: the index is built over
the whole base on every call, ~21× the size of the input. On a 50 MB base that
extrapolates to roughly 1 GB of transient heap per `make_patch`.

---

## Step 2 — tiered resync differ (v1 wire format unchanged)

`make_patch`, criterion median:

| scenario | before | after | speedup |
|---|---|---|---|
| dictionary_insert | 6.96 ms | **71.0 µs** | 98× |
| status_edits | 8.37 ms | **73.9 µs** | 113× |
| append | 8.16 ms | **71.4 µs** | 114× |
| combined | 7.00 ms | **80.4 µs** | 87× |

`apply_patch` is unchanged in kind (21.6 / 49.1 / 43.9 / 30.9 µs); the format
did not change, only the opcode mix.

Size and memory:

| scenario | patch raw | zstd-19 | Δ zstd | Copy | Add | peak heap (build) |
|---|---|---|---|---|---|---|
| dictionary_insert | 265 | 207 | +27% | 4 | 4 | 64.3 KiB |
| status_edits | 1274 | 411 | −1% | 56 | 55 | 68.5 KiB |
| append | 5596 | 1463 | **−5%** | 2 | 2 | 64.1 KiB |
| combined | 7117 | 2061 | −0.1% | 60 | 61 | 68.5 KiB |

Search behaviour:

| scenario | resyncs | closed by drift cache | window searches | base bytes scanned | escalations |
|---|---|---|---|---|---|
| dictionary_insert | 4 | 1 | 9 | 60.1 KiB | 0 |
| status_edits | 55 | 53 | 4 | 24.1 KiB | 0 |
| append | 4 | 1 | 14 | 56.0 KiB | 0 |
| combined | 63 | 55 | 27 | 140.2 KiB | 0 |

### Reading the numbers

- **Peak heap: 34 MiB → 64 KiB (500×).** The index is no longer built at all
  for a local update; nothing on the hot path scales with the base.
- **Base scanned: 100% → 1.5–8.6%** of a 1.6 MiB base. The quantity that used
  to be `O(old.len())` is now `O(edits × window)`, so the gap widens with base
  size rather than staying constant.
- **The drift cache carries the workload it was built for.** 53 of 55 resyncs
  in `status_edits` are answered by a remembered drift with no search at all —
  every `done` → `reversed` edit has the same length delta.
- **Zero escalations** on all four scenarios. A rise in this column in
  production is the signal that updates stopped being local edits.

### Where the expectation was not met

`dictionary_insert` compressed 27% *worse* (163 → 207 bytes) — on a patch of
200 bytes, so 44 bytes absolute, far below anything that matters against the
6 KiB threshold. Cause: the old differ found 17 short copies inside the three
inserted counterparty records, because the inserted names are drawn from the
same vocabulary as existing ones. The new differ carries those 209 bytes as
literals instead. That trade is deliberate — it is the same trade that makes
`append` 5% smaller — but on the smallest scenario the opcode saving does not
cover the literal cost.

Raw patch size grew on `append` (3584 → 5596) and `combined` (4796 → 7117)
while the *compressed* size did not. The appended tail is now one literal run
rather than 250 opcodes interleaved with fragments; zstd recovers the internal
redundancy itself. Since only the compressed size is stored, this is not a
regression — but it does mean raw patch size is no longer a useful proxy.

---

## Steps 3 and 4 — v2 format, and `min_match` swept against compressed size

`make_patch`, criterion median:

| scenario | baseline | step 2 | **final** | vs baseline |
|---|---|---|---|---|
| dictionary_insert | 6.96 ms | 71.0 µs | **111.1 µs** | **63×** |
| status_edits | 8.37 ms | 73.9 µs | **114.2 µs** | **73×** |
| append | 8.16 ms | 71.4 µs | **111.0 µs** | **73×** |
| combined | 7.00 ms | 80.4 µs | **121.0 µs** | **58×** |

`apply_patch`, criterion median:

| scenario | baseline | step 2 | final |
|---|---|---|---|
| dictionary_insert | 22.4 µs | 21.6 µs | 65.7 µs |
| status_edits | 49.6 µs | 49.1 µs | 90.3 µs |
| append | 45.7 µs | 43.9 µs | 83.9 µs |
| combined | 54.3 µs | 30.9 µs | 72.0 µs |

Size, memory, and search behaviour:

| scenario | raw | zstd-19 | vs baseline | Copy | Add | peak heap (build) | escalations |
|---|---|---|---|---|---|---|---|
| dictionary_insert | 261 | 213 | +31% | 5 | 4 | 64.6 KiB | 0 |
| status_edits | 916 | **261** | **−37%** | 56 | 55 | 68.5 KiB | 0 |
| append | 5622 | 1487 | −4% | 2 | 2 | 64.1 KiB | 0 |
| combined | 6711 | **1834** | **−11%** | 61 | 61 | 68.5 KiB | 0 |

### Both time columns move for exactly one reason

Every scenario got ~40 µs slower on **both** build and apply, and the number
is the same on all four because it is not a property of the update — it is one
xxh3 pass over the 1.6 MiB base, at ~40 GB/s. That pass is what `base_hash`
costs. It was called "копейки" in the brief; at these speeds it is not. It is
a third of the build time and it doubles apply time, because the differ no
longer reads the base at all and hashing is now the only thing that does.

The check is kept on by default: the failure it prevents — assembling a
different customer's financial history from a stale base, with no error — is
worse than 40 µs. A caller that already knows its base is current can skip the
re-hash by comparing `inspect(&patch)` against a cached `base_fingerprint`.

The `combined` apply number is *lower* than `status_edits` despite a larger
patch: it copies in fewer, longer runs.

### `min_match` did not behave as the brief predicted

Swept against zstd-19 on the corpus, `combined` scenario:

| min_match | 10 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|
| raw | 6711 | 6711 | 6725 | 6725 | 6725 | 6725 |
| zstd-19 | 1834 | 1834 | 1849 | 1849 | 1849 | 1849 |

The brief measured a 26% swing between `min_match` 16 and 32 and an optimum at
the high end. Here the curve is flat to within 0.8%, and it tilts the *other*
way. The cause is that the two differs emit different quantities of opcodes:
the brief's numbers came from a matcher producing 156–300 copies, where opcode
count really did dominate the compressed patch. The tiered matcher emits 61.
With two orders of magnitude less opcode pressure there is nothing left for a
higher threshold to buy, so the raw wire cost wins and the smaller value is
weakly better. Settled on 16.

Values above 32 are rejected at compile time: the differ's progress argument
requires `min_match <= ANCHOR`.

### The post-pass from task 3 was not implemented, and why

- *"Copies shorter than a threshold become literals."* Vacuous. `min_match`
  already gates every emission path, so no copy shorter than it can exist —
  the pass would never fire.
- *"Copies separated by a short literal merge into one copy when the deltas
  agree."* Not lossless as stated. Given `Copy(o, a) Add(k) Copy(o+a+k, b)`,
  merging into `Copy(o, a+k+b)` substitutes `old[o+a .. o+a+k]` for the literal
  bytes — and those bytes differ, which is why the literal is there. The only
  saving actually available is one tag byte, and after delta coding the second
  copy's delta is already the small number `k`.

The motivation behind the task — half the compressed patch being opcodes
rather than data — was addressed instead by the opcode count itself falling
from 386 to 122 on `combined`, and by the sectioned layout separating the tag
stream from the literals.

---

## The escalation path, measured separately

The four corpus scenarios all report `escalations = 0`, so none of the numbers
above say anything about what happens when an update is *not* a local edit.
Measured directly (`cargo test --release --test escalation -- --nocapture`), on
a 2 MiB base of incompressible noise, against the pre-change implementation
that always built the index:

| update | always-index | tiered, first attempt | tiered, after the fix |
|---|---|---|---|
| one insertion | 39.4 ms, patch 31 B | **0.8 ms**, patch 60 B | 0.8 ms, patch 60 B |
| halves swapped | 16.8 ms, patch 18 B | 28.8 ms, patch **180 282 B** | 30.4 ms, patch **54 B** |

The middle column is the defect. Before the index exists, a failed resync
carries `SKIP_AHEAD` bytes as a literal and probes again; the number of such
skips before the scan budget trips is roughly `old.len() / SEARCH_ROUND_COST`,
so the literal accumulated blind is about `old.len() × SKIP_AHEAD /
SEARCH_ROUND_COST` — 5% by that estimate, 8.6% measured. Every one of those
bytes existed verbatim in the base; the index could see them and the local
search could not. A document whose entire content is present in the base
produced a patch the size of the moved block.

The fix is to treat the index as a change of information, not just another
tier: when it is built, discard the opcodes decided without it and redo the
diff from zero with the index in hand. It is built at most once, so the diff
restarts at most once, and `PatchStats::restarts` records it. Cost on the
local-edit path: nothing, the branch is never taken. Cost on the escalation
path: one more pass, which measured as noise — the second pass finds long
copies immediately instead of grinding through skip-ahead rounds.

What remains true after the fix: on a relocated block the tiered differ is
**1.8× slower** than indexing unconditionally (30.4 ms against 16.8 ms),
because it pays for the local search that failed before paying for the index.
That is the price of the 49× win on the local-edit path, and
`PatchStats::escalations` is what makes the trade observable in production.

### Where the expectation was not met

`dictionary_insert` compressed 31% worse than the baseline (163 → 213 bytes).
Two independent causes, both understood: the literals discussed under step 2,
plus the 38-byte v2 header, 8 bytes of which are an incompressible hash. On a
200-byte patch a fixed header is simply large. It stays 28× below the 6 KiB
threshold, so this does not affect the rewrite rate it was meant to protect.
