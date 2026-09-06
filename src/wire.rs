//! Patch container formats.
//!
//! Two layouts are readable; only v2 is written.
//!
//! **v1** — a single stream of self-describing opcodes:
//!
//! ```text
//! Copy: 0x00 offset:u32_le len:u32_le
//! Add : 0x01 len:u32_le bytes...
//! ```
//!
//! **v2** — a header identifying the base, then four separate sections:
//!
//! ```text
//! magic:4 "BCUT"  version:u8=2  flags:u8
//! base_len:u64_le  base_hash:u64_le
//! ops:u32_le  deltas_len:u32_le  lengths_len:u32_le  literals_len:u32_le
//! [tags]      one byte per opcode
//! [deltas]    `StreamVByte`, zigzag(offset - cursor), Copy only
//! [lengths]   `StreamVByte`, every opcode
//! [literals]  every Add payload, concatenated
//! ```
//!
//! Two changes carry the size win, and neither works without the other.
//! Offsets become deltas against a cursor that follows the previous copy, so
//! a value that was a uniformly-distributed 32-bit address becomes a small
//! number clustered at zero. Splitting the streams then lets the compressor
//! model each one separately — the tags, the small deltas and the literals
//! have nothing in common, and interleaving them hides all three.
//!
//! The v1 tags `0x00` and `0x01` cannot begin a v2 patch, so the two layouts
//! are told apart by their first byte.

use crate::error::PatchError;

pub const MAGIC: [u8; 4] = *b"BCUT";
pub const VERSION: u8 = 2;

/// Set when every copy reads forward of the previous one, which lets an
/// applier stream the base instead of seeking within it.
pub const FLAG_MONOTONE: u8 = 0b0000_0001;

const HEADER_LEN: usize = 38;

/// The parts of a v2 patch, borrowed from the patch buffer.
#[derive(Debug, Clone, Copy)]
pub struct PatchV2<'a> {
    pub flags: u8,
    pub base_len: u64,
    pub base_hash: u64,
    pub ops: usize,
    pub tags: &'a [u8],
    pub deltas: &'a [u8],
    pub lengths: &'a [u8],
    pub literals: &'a [u8],
}

/// Which layout a patch buffer holds.
#[derive(Debug, Clone, Copy)]
pub enum Layout<'a> {
    /// Self-describing opcode stream, no base identity.
    V1(&'a [u8]),
    /// Sectioned layout with a base fingerprint.
    V2(PatchV2<'a>),
}

/// Classify `patch` and, for v2, split it into sections.
///
/// # Errors
///
/// [`PatchError::UnexpectedEof`] on a truncated header or sections,
/// [`PatchError::UnsupportedVersion`] on a future format.
pub fn parse(patch: &[u8]) -> Result<Layout<'_>, PatchError> {
    let Some((magic, rest)) = patch.split_first_chunk::<4>() else {
        return Ok(Layout::V1(patch));
    };
    if *magic != MAGIC {
        return Ok(Layout::V1(patch));
    }

    let (&version, rest) = rest.split_first().ok_or(PatchError::UnexpectedEof)?;
    if version != VERSION {
        return Err(PatchError::UnsupportedVersion(version));
    }
    let (&flags, rest) = rest.split_first().ok_or(PatchError::UnexpectedEof)?;
    let (base_len, rest) = take_u64(rest)?;
    let (base_hash, rest) = take_u64(rest)?;
    let (ops, rest) = take_u32(rest)?;
    let (deltas_len, rest) = take_u32(rest)?;
    let (lengths_len, rest) = take_u32(rest)?;
    let (literals_len, rest) = take_u32(rest)?;

    let ops = usize::try_from(ops).map_err(|_| PatchError::Overflow)?;
    let (tags, rest) = rest
        .split_at_checked(ops)
        .ok_or(PatchError::UnexpectedEof)?;
    let (deltas, rest) = split_section(rest, deltas_len)?;
    let (lengths, rest) = split_section(rest, lengths_len)?;
    let (literals, _rest) = split_section(rest, literals_len)?;

    Ok(Layout::V2(PatchV2 {
        flags,
        base_len,
        base_hash,
        ops,
        tags,
        deltas,
        lengths,
        literals,
    }))
}

/// Everything a v2 header states, before it is written.
pub struct Header {
    pub flags: u8,
    pub base_len: u64,
    pub base_hash: u64,
    pub ops: usize,
    pub deltas_len: usize,
    pub lengths_len: usize,
    pub literals_len: usize,
}

impl Header {
    /// Append this header to `out`. The four sections follow it, in the order
    /// their lengths are declared.
    ///
    /// # Errors
    ///
    /// [`PatchError::InputTooLarge`] if a section does not fit its 32-bit
    /// length field.
    pub fn write(&self, out: &mut Vec<u8>) -> Result<(), PatchError> {
        out.reserve(HEADER_LEN);
        out.extend_from_slice(&MAGIC);
        out.push(VERSION);
        out.push(self.flags);
        out.extend_from_slice(&self.base_len.to_le_bytes());
        out.extend_from_slice(&self.base_hash.to_le_bytes());
        for field in [
            self.ops,
            self.deltas_len,
            self.lengths_len,
            self.literals_len,
        ] {
            let value =
                u32::try_from(field).map_err(|_| PatchError::InputTooLarge { len: field })?;
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }
}

/// Fingerprint of a base document, as recorded in a v2 header.
#[must_use]
pub fn fingerprint(base: &[u8]) -> (u64, u64) {
    (
        u64::try_from(base.len()).unwrap_or(u64::MAX),
        xxhash_rust::xxh3::xxh3_64(base),
    )
}

impl PatchV2<'_> {
    /// Reject a patch built against a different base.
    ///
    /// # Errors
    ///
    /// [`PatchError::WrongBase`] when the length or fingerprint disagrees.
    pub fn check_base(&self, base: &[u8]) -> Result<(), PatchError> {
        let (actual_len, actual_hash) = fingerprint(base);
        if actual_len == self.base_len && actual_hash == self.base_hash {
            return Ok(());
        }
        Err(PatchError::WrongBase {
            expected_len: self.base_len,
            expected_hash: self.base_hash,
            actual_len,
            actual_hash,
        })
    }

    /// Number of `Copy` opcodes, which is how many deltas the section holds.
    #[must_use]
    pub fn copies(&self) -> usize {
        // `bytecount` would vectorise this, but the tag section is one byte
        // per opcode — hundreds, not megabytes — and it is not worth a
        // dependency.
        #[allow(clippy::naive_bytecount)]
        self.tags.iter().filter(|&&tag| tag == TAG_COPY).count()
    }
}

pub const TAG_COPY: u8 = 0x00;
pub const TAG_ADD: u8 = 0x01;

fn take_u64(input: &[u8]) -> Result<(u64, &[u8]), PatchError> {
    let (bytes, rest) = input
        .split_first_chunk::<8>()
        .ok_or(PatchError::UnexpectedEof)?;
    Ok((u64::from_le_bytes(*bytes), rest))
}

fn take_u32(input: &[u8]) -> Result<(u32, &[u8]), PatchError> {
    let (bytes, rest) = input
        .split_first_chunk::<4>()
        .ok_or(PatchError::UnexpectedEof)?;
    Ok((u32::from_le_bytes(*bytes), rest))
}

fn split_section(input: &[u8], len: u32) -> Result<(&[u8], &[u8]), PatchError> {
    let len = usize::try_from(len).map_err(|_| PatchError::Overflow)?;
    input.split_at_checked(len).ok_or(PatchError::UnexpectedEof)
}
