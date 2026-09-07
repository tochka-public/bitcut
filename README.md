# bitcut

Binary diff and patch for Rust, as a library and a CLI.

Builds a patch between two byte strings, then reconstructs the second from the
first plus the patch.

[Watch the algorithm run](https://tochka-public.github.io/bitcut/), one decision at a
time. The page drives this crate compiled to wasm, so what it draws is what the
differ did.

## What it does differently

Most binary differs index the base before they start: every position hashed,
every hash in a table. That cost lands on every call, whatever the update was.

This one walks both documents in lockstep and, at each divergence, spends a
bounded search to find where the base continues. An edit shifts everything
after it by the same amount, so a handful of remembered shifts closes most
divergences with a single comparison, and a windowed substring search closes
the rest. Search cost then tracks the number of edits rather than the size of
the base.

An update that is not a local edit (a relocated block, a reordered document, a
change of format) still falls through to a global index. `PatchStats::escalations`
counts how often that happened.

## Measurements

Against the crate's own 0.1.6, which is the index-first implementation. Median
of 25 runs on an M-series laptop, patch sizes uncompressed.

| shape | base | 0.1.6 | this | peak heap, 0.1.6 | peak heap, this | patch, 0.1.6 | patch, this |
|---|---|---|---|---|---|---|---|
| one insertion | 2.0 MiB | 20.3 ms | 0.11 ms | 51 MiB | 64 KiB | 31 B | 60 B |
| 40 scattered edits | 1.5 MiB | 17.3 ms | 0.10 ms | 51 MiB | 68 KiB | 855 B | 474 B |
| 5% appended | 1.5 MiB | 17.7 ms | 0.11 ms | 51 MiB | 192 KiB | 80 268 B | 80 062 B |
| halves swapped | 1.5 MiB | 17.4 ms | 5.50 ms | 51 MiB | 34 MiB | 27 B | 59 B |
| unrelated documents | 1.5 MiB | 29.0 ms | 15.8 ms | 51 MiB | 39 MiB | 1 573 392 B | 1 192 561 B |

The two escalating rows swing by tens of percent with how warm the machine is.
The first three do not.

The last two rows are the shapes local search cannot see. They build the index
after all, and the heap goes with it. Searching before giving up is not free,
but it is bounded: the matcher stops once fruitless scanning has cost about
what the index costs, so the worst case is the index plus that much again.

They still come out ahead of indexing first, for two reasons that have nothing
to do with the search. The table is reserved at its final size instead of
grown by doubling, which is where 0.1.6 spends 51 MiB to hold 32 MiB of
entries and re-hashes every key it has already inserted on each doubling. And
the index is consulted only at divergences, where 0.1.6 recomputes a ten-byte
hash from scratch at every byte it fails to match.

The smallest patches grew because v2 carries a 38-byte header. The opcode
encoding wins some of that back, so the net cost on a one-insertion patch is 29
bytes; above a few hundred bytes it stops being visible.

## Library usage

```toml
[dependencies]
bitcut = { version = "1", default-features = false }
```

`default-features = false` drops the CLI dependencies, `clap` and `anyhow`.

### Create and apply a patch

```rust
use bitcut::{make_patch, apply_patch};

let old = b"the quick brown fox jumps over the lazy dog";
let new = b"the quick brown cat jumps over the lazy dog";

let patch = make_patch(old, new)?;
assert_eq!(apply_patch(old, &patch)?, new);
```

### See what the matcher spent

```rust
use bitcut::make_patch_stats;

let (patch, stats) = make_patch_stats(old, new)?;
println!(
    "read {} bytes of a {}-byte base, escalated {} time(s)",
    stats.old_bytes_scanned,
    old.len(),
    stats.escalations,
);
```

`escalations` counts the divergences local search gave up on and handed to the
index, which is built once per call. Expect it above zero on a small base,
where indexing costs less than a single round of searching. On a large one it
is the number worth watching: a rising share means updates stopped being local
edits, and this crate stopped being the right tool for them.

### Inspect opcodes

```rust
use bitcut::Op;

for op in Op::iter(&patch) {
    match op? {
        Op::Copy { offset, len } => println!("copy {len} bytes from offset {offset}"),
        Op::Add(bytes) => println!("add {} new bytes", bytes.len()),
    }
}
```

### Check a patch against a base without applying it

```rust
use bitcut::{base_fingerprint, inspect};

if let Some(header) = inspect(&patch)? {
    assert_eq!((header.base_len, header.base_hash), base_fingerprint(old));
}
```

`inspect` returns `None` for a v1 patch, which identifies neither its base nor
its opcode count.

### Reuse buffers

```rust
use bitcut::{make_patch_into, apply_patch_into};

let mut patch_buf = Vec::new();
make_patch_into(old, new, &mut patch_buf)?;

let mut output = Vec::new();
apply_patch_into(old, &patch_buf, &mut output)?;
```

Neither call clears its buffer first.

## Patch format

Little-endian throughout. The format caps either input at 4 GiB (`u32::MAX`).

### v2, written and read

```text
magic:4 "BCUT"  version:u8=2  flags:u8
base_len:u64  base_hash:u64          xxh3 of the base
ops:u32  deltas_len:u32  lengths_len:u32  literals_len:u32
[tags]      one byte per opcode
[deltas]    StreamVByte, zigzag(offset - cursor), Copy only
[lengths]   StreamVByte, every opcode
[literals]  every Add payload, concatenated
```

The encoder writes each offset relative to a cursor that follows the previous
copy, which turns scattered 32-bit addresses into small numbers. Splitting
tags, numbers and literals into separate sections then lets a compressor model
each one on its own. Neither change is worth much without the other.

`flags` bit 0 marks a patch whose copies only ever read forward, so an applier
can stream the base rather than seek within it.

`base_hash` catches a wrong or stale base. It is not a signature: patches from
an untrusted source need authenticating at the transport or storage layer.

### v1, read only

```text
Copy: 0x00 offset:u32 len:u32
Add : 0x01 len:u32 bytes...
```

Patches written before v2 still apply. They carry no base identity, so nothing
about the base can be checked for them.

## CLI

```sh
cargo install bitcut

bitcut diff old.bin new.bin > patch.bin
bitcut patch old.bin patch.bin > new.bin
bitcut debug patch.bin            # the header, then every opcode
```

## Build from source

```sh
cargo build --release
./target/release/bitcut diff old.bin new.bin > patch.bin
```
