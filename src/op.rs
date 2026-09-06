//! Patch opcodes and their v1 wire format.
//!
//! ```text
//! Copy: 0x00 offset:u32_le len:u32_le
//! Add : 0x01 len:u32_le bytes...
//! ```

use crate::error::PatchError;
use crate::{vbyte, wire};
use std::fmt;

use wire::{TAG_ADD, TAG_COPY};

pub const ADD_HEADER_LEN: usize = 5;

/// A single patch opcode.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op<'a> {
    /// Copy `len` bytes from `old` starting at `offset`.
    Copy {
        /// Offset within `old`.
        offset: u32,
        /// Number of bytes to copy.
        len: u32,
    },
    /// Append a literal byte run.
    Add(&'a [u8]),
}

impl fmt::Debug for Op<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        enum Content<'a> {
            Text(&'a str),
            Bytes(&'a [u8]),
        }

        impl<'a> From<&'a [u8]> for Content<'a> {
            fn from(value: &'a [u8]) -> Self {
                match std::str::from_utf8(value) {
                    Ok(s) => Content::Text(s),
                    Err(_) => Content::Bytes(value),
                }
            }
        }

        impl fmt::Debug for Content<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    Content::Text(s) => write!(f, "Text({s:?})"),
                    Content::Bytes(b) => {
                        write!(f, "Bytes(")?;
                        for (i, byte) in b.iter().enumerate() {
                            if i > 0 {
                                write!(f, " ")?;
                            }
                            write!(f, "{byte:02X}")?;
                        }
                        write!(f, ")")
                    }
                }
            }
        }

        match *self {
            Self::Copy { offset, len } => f.debug_tuple("Copy").field(&offset).field(&len).finish(),
            Self::Add(content) => f.debug_tuple("Add").field(&Content::from(content)).finish(),
        }
    }
}

impl<'a> Op<'a> {
    /// Serialize this op into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`PatchError::InputTooLarge`] if an `Add` payload exceeds
    /// `u32::MAX` bytes.
    pub fn serialize_to(&self, out: &mut Vec<u8>) -> Result<(), PatchError> {
        match *self {
            Op::Copy { offset, len } => {
                out.push(TAG_COPY);
                out.extend_from_slice(&offset.to_le_bytes());
                out.extend_from_slice(&len.to_le_bytes());
                Ok(())
            }
            Op::Add(bytes) => {
                let len = u32::try_from(bytes.len())
                    .map_err(|_| PatchError::InputTooLarge { len: bytes.len() })?;
                out.push(TAG_ADD);
                out.extend_from_slice(&len.to_le_bytes());
                out.extend_from_slice(bytes);
                Ok(())
            }
        }
    }

    /// Parse the next opcode from `input`, returning the op and the remaining
    /// bytes.
    ///
    /// # Errors
    ///
    /// See [`PatchError`] variants.
    pub fn deserialize(input: &'a [u8]) -> Result<(Self, &'a [u8]), PatchError> {
        let (&tag, rest) = input.split_first().ok_or(PatchError::UnexpectedEof)?;
        match tag {
            TAG_COPY => {
                let (offset_bytes, rest) = rest
                    .split_first_chunk::<4>()
                    .ok_or(PatchError::UnexpectedEof)?;
                let (len_bytes, rest) = rest
                    .split_first_chunk::<4>()
                    .ok_or(PatchError::UnexpectedEof)?;
                Ok((
                    Op::Copy {
                        offset: u32::from_le_bytes(*offset_bytes),
                        len: u32::from_le_bytes(*len_bytes),
                    },
                    rest,
                ))
            }
            TAG_ADD => {
                let (len_bytes, rest) = rest
                    .split_first_chunk::<4>()
                    .ok_or(PatchError::UnexpectedEof)?;
                let declared = u32::from_le_bytes(*len_bytes);
                let len = usize::try_from(declared).map_err(|_| PatchError::Overflow)?;
                let (payload, rest) = rest
                    .split_at_checked(len)
                    .ok_or(PatchError::UnexpectedEof)?;
                Ok((Op::Add(payload), rest))
            }
            other => Err(PatchError::InvalidOpcode(other)),
        }
    }

    /// Iterate the opcodes in `patch` lazily, without allocating a `Vec`.
    ///
    /// Both container formats are accepted; the sectioned v2 layout is
    /// reassembled into the same opcode sequence a v1 patch would have
    /// carried. The base fingerprint in a v2 header is *not* checked here —
    /// [`crate::apply_patch`] does that.
    #[must_use]
    pub fn iter(patch: &'a [u8]) -> OpIter<'a> {
        match wire::parse(patch) {
            Ok(wire::Layout::V1(input)) => OpIter {
                state: State::V1 { input },
            },
            Ok(wire::Layout::V2(patch)) => OpIter {
                state: Sections::open(&patch).map_or_else(
                    || State::Failed(PatchError::UnexpectedEof),
                    |sections| State::V2(Box::new(sections)),
                ),
            },
            Err(error) => OpIter {
                state: State::Failed(error),
            },
        }
    }
}

/// Iterator over the opcodes of a patch.
///
/// Yields `Err` once and then stops on the first malformed opcode.
#[derive(Debug, Clone)]
pub struct OpIter<'a> {
    state: State<'a>,
}

#[derive(Debug, Clone)]
enum State<'a> {
    V1 { input: &'a [u8] },
    V2(Box<Sections<'a>>),
    Failed(PatchError),
    Done,
}

/// Cursor over the four sections of a v2 patch, reading one opcode per step.
#[derive(Debug, Clone)]
struct Sections<'a> {
    tags: &'a [u8],
    deltas: vbyte::Reader<'a>,
    lengths: vbyte::Reader<'a>,
    literals: &'a [u8],
    index: usize,
    /// End of the previous copy, which the next delta is measured against.
    cursor: u32,
    consumed_literals: usize,
}

impl<'a> Sections<'a> {
    fn open(patch: &wire::PatchV2<'a>) -> Option<Self> {
        // Section readers borrow the patch buffer, not the header struct.
        Some(Self {
            tags: patch.tags,
            deltas: vbyte::Reader::new(patch.deltas, patch.copies())?,
            lengths: vbyte::Reader::new(patch.lengths, patch.ops)?,
            literals: patch.literals,
            index: 0,
            cursor: 0,
            consumed_literals: 0,
        })
    }

    fn next_op(&mut self) -> Option<Result<Op<'a>, PatchError>> {
        let &tag = self.tags.get(self.index)?;
        self.index = self.index.saturating_add(1);
        let Some(len) = self.lengths.next() else {
            return Some(Err(PatchError::UnexpectedEof));
        };
        match tag {
            wire::TAG_COPY => {
                let Some(delta) = self.deltas.next() else {
                    return Some(Err(PatchError::UnexpectedEof));
                };
                let step = vbyte::unzigzag(delta);
                let offset = self
                    .cursor
                    .wrapping_add(u32::from_ne_bytes(step.to_ne_bytes()));
                self.cursor = offset.wrapping_add(len);
                Some(Ok(Op::Copy { offset, len }))
            }
            wire::TAG_ADD => {
                let Ok(len) = usize::try_from(len) else {
                    return Some(Err(PatchError::Overflow));
                };
                let Some(end) = self.consumed_literals.checked_add(len) else {
                    return Some(Err(PatchError::Overflow));
                };
                let Some(payload) = self.literals.get(self.consumed_literals..end) else {
                    return Some(Err(PatchError::UnexpectedEof));
                };
                self.consumed_literals = end;
                Some(Ok(Op::Add(payload)))
            }
            other => Some(Err(PatchError::InvalidOpcode(other))),
        }
    }
}

impl<'a> Iterator for OpIter<'a> {
    type Item = Result<Op<'a>, PatchError>;

    fn next(&mut self) -> Option<Self::Item> {
        // Any error ends iteration, so a caller that stops at the first `Err`
        // and one that drains the iterator see the same opcode sequence.
        match &mut self.state {
            State::Done => None,
            State::Failed(error) => {
                let error = error.clone();
                self.state = State::Done;
                Some(Err(error))
            }
            State::V1 { input } => {
                if input.is_empty() {
                    self.state = State::Done;
                    return None;
                }
                match Op::deserialize(input) {
                    Ok((op, rest)) => {
                        *input = rest;
                        Some(Ok(op))
                    }
                    Err(error) => {
                        self.state = State::Done;
                        Some(Err(error))
                    }
                }
            }
            State::V2(sections) => match sections.next_op() {
                None => {
                    self.state = State::Done;
                    None
                }
                Some(Ok(op)) => Some(Ok(op)),
                Some(Err(error)) => {
                    self.state = State::Done;
                    Some(Err(error))
                }
            },
        }
    }
}
