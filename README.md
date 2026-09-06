# bitcut

Compact binary diff & patch library and CLI for Rust.

Generates a patch between two binary files, then reconstructs the target by applying the patch to the original.

## Features

- **Search cost tracks edits, not file size** — the base is walked in lockstep with the new document instead of being indexed up front, so a local edit to a multi-megabyte base reads a few kilobytes of it. On a 2 MiB base a single insertion takes 0.8 ms and never builds an index at all, against 39 ms for indexing the base first. The benchmark corpus (1.6 MiB, scattered edits) shows the same shape: ~60× faster, 500× less peak memory.
- **…except when the update is not a local edit** — a relocated block, a reordered document or a format change falls through to a global index, and then the cost is the index plus the local search that failed first: ~30 ms against 17 ms for indexing unconditionally. The patch is the same size either way. `PatchStats::escalations` reports how often this happens; on the benchmark corpus it is zero.
- **Patches name their base** — `old` is fingerprinted into the patch header and re-checked on apply, so a patch built against a different version is refused instead of silently reconstructing the wrong document.
- **SIMD-accelerated comparison** — AVX2 on x86_64, NEON on AArch64, automatic scalar fallback
- **Streaming opcode iterator** — allocation-free `OpIter` for lazy patch traversal, over either format version
- **No unsafe in public API** — SIMD is internal; the library surface is fully safe
- **Minimal dependencies** — `memchr`, `rustc-hash` and `xxhash-rust` for the library

## Library Usage

Add to `Cargo.toml`:

```toml
[dependencies]
bitcut = { version = "2", default-features = false }
```

> `default-features = false` disables the CLI dependencies (`clap`, `anyhow`).

### Create and apply a patch

```rust
use bitcut::{make_patch, apply_patch};

let old = b"the quick brown fox jumps over the lazy dog";
let new = b"the quick brown cat jumps over the lazy dog";

let patch = make_patch(old, new)?;
let result = apply_patch(old, &patch)?;
assert_eq!(result, new);
```

### Inspect opcodes

```rust
use bitcut::Op;

let ops: Vec<Op> = Op::iter(&patch).collect::<Result<_, _>>()?;
for op in &ops {
    match op {
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

### Reuse buffers

```rust
use bitcut::{make_patch_into, apply_patch_into};

let mut patch_buf = Vec::new();
make_patch_into(old, new, &mut patch_buf)?;

let mut output = Vec::new();
apply_patch_into(old, &patch_buf, &mut output)?;
```

## Patch Format

All integers are little-endian. Maximum input size: 4 GiB (`u32::MAX`).

### v2 — written and read

```text
magic:4 "BCUT"  version:u8=2  flags:u8
base_len:u64  base_hash:u64          xxh3 of the base
ops:u32  deltas_len:u32  lengths_len:u32  literals_len:u32
[tags]      one byte per opcode
[deltas]    StreamVByte, zigzag(offset - cursor), Copy only
[lengths]   StreamVByte, every opcode
[literals]  every Add payload, concatenated
```

Offsets are stored relative to a cursor that follows the previous copy, which
turns scattered 32-bit addresses into small numbers; splitting tags, numbers
and literals into separate sections then lets a compressor model each one on
its own. Neither change is worth much without the other.

`flags` bit 0 marks a patch whose copies only ever read forward, which allows
applying it while streaming the base.

`base_hash` detects a wrong or stale base. It is not a signature — patches
from an untrusted source need authenticating at the transport or storage layer.

### v1 — read only

```text
Copy: 0x00 offset:u32 len:u32
Add : 0x01 len:u32 bytes...
```

Patches written before v2 still apply. They carry no base identity, so nothing
about the base can be checked for them.

## CLI

### Install

```sh
cargo install bitcut
```

### Commands

```sh
# create a patch
bitcut diff old.bin new.bin > patch.bin

# apply a patch
bitcut patch old.bin patch.bin > new.bin
```

## Build from Source

```sh
cargo build --release
./target/release/bitcut diff old.bin new.bin > patch.bin
```
