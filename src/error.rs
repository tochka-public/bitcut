//! The single error type of the crate.

use std::fmt;

/// Errors produced by the patch deserializer / applier.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PatchError {
    /// Patch ended in the middle of an opcode.
    UnexpectedEof,
    /// Opcode tag is neither `Copy` nor `Add`.
    InvalidOpcode(u8),
    /// `Copy` references a range outside `old`.
    CopyOutOfBounds {
        /// Requested offset.
        offset: u32,
        /// Requested length.
        len: u32,
        /// Actual length of `old`.
        old_len: usize,
    },
    /// Arithmetic overflow while computing a copy/add range. Indicates a
    /// crafted or corrupted patch.
    Overflow,
    /// `make_patch` was called with an input larger than `u32::MAX` bytes.
    InputTooLarge {
        /// The offending length.
        len: usize,
    },
    /// Patch header names a format version this build cannot read.
    UnsupportedVersion(u8),
    /// The patch was built against a different base document. Applying it
    /// anyway would silently produce a plausible but wrong result.
    WrongBase {
        /// Base length the patch was built against.
        expected_len: u64,
        /// Base fingerprint the patch was built against.
        expected_hash: u64,
        /// Length of the base it was applied to.
        actual_len: u64,
        /// Fingerprint of the base it was applied to.
        actual_hash: u64,
    },
}

impl fmt::Display for PatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            PatchError::UnexpectedEof => f.write_str("unexpected end of patch"),
            PatchError::InvalidOpcode(tag) => write!(f, "invalid opcode tag: 0x{tag:02X}"),
            PatchError::CopyOutOfBounds {
                offset,
                len,
                old_len,
            } => write!(
                f,
                "copy out of bounds: offset={offset} len={len} old_len={old_len}"
            ),
            PatchError::Overflow => f.write_str("arithmetic overflow in patch range"),
            PatchError::InputTooLarge { len } => {
                write!(f, "input too large for patch format: {len} bytes")
            }
            PatchError::UnsupportedVersion(version) => {
                write!(f, "unsupported patch format version: {version}")
            }
            PatchError::WrongBase {
                expected_len,
                expected_hash,
                actual_len,
                actual_hash,
            } => write!(
                f,
                "patch was built against a different base: \
                 expected len={expected_len} hash={expected_hash:016x}, \
                 got len={actual_len} hash={actual_hash:016x}"
            ),
        }
    }
}

impl std::error::Error for PatchError {}
