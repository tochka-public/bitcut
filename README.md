# bitcut

Compact binary diff & patch library and CLI for Rust.

Generates minimal patches between two binary files using rolling-hash matching, then reconstructs the target by applying the patch to the original.

## Features

- **SIMD-accelerated comparison** — AVX2 on x86_64, NEON on AArch64, automatic scalar fallback
- **Streaming opcode iterator** — zero-allocation `OpIter` for lazy patch traversal
- **Compact binary format** — only two opcodes: `Copy` (reference old) and `Add` (literal bytes)
- **No unsafe in public API** — SIMD is internal; the library surface is fully safe
- **Minimal dependencies** — only `rustc-hash` for the library

## Library Usage

Add to `Cargo.toml`:

```toml
[dependencies]
bitcut = { version = "1", default-features = false }
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

### Reuse buffers

```rust
use bitcut::{make_patch_into, apply_patch_into};

let mut patch_buf = Vec::new();
make_patch_into(old, new, &mut patch_buf)?;

let mut output = Vec::new();
apply_patch_into(old, &patch_buf, &mut output)?;
```

## Patch Format

A patch is a sequence of two opcodes (little-endian integers):

| Opcode | Tag | Layout |
|--------|-----|--------|
| Copy | `0x00` | `offset:u32` `len:u32` |
| Add | `0x01` | `len:u32` `bytes...` |

Maximum input size: 4 GiB (`u32::MAX`).

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
