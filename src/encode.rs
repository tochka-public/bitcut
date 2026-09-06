//! Turning planned chunks into a patch buffer.

use crate::differ::Chunk;
use crate::error::PatchError;
#[cfg(test)]
use crate::op::Op;
use crate::{vbyte, wire};

/// Encode chunks in the legacy v1 layout: one self-describing opcode after
/// another. Nothing writes v1 any more; this exists so the compatibility test
/// can build a real legacy patch instead of a hand-written fixture.
///
/// # Errors
///
/// [`PatchError::Overflow`] if a chunk does not fit the 32-bit fields of the
/// format, or refers outside `new`.
#[cfg(test)]
pub fn v1(chunks: &[Chunk], new: &[u8], out: &mut Vec<u8>) -> Result<(), PatchError> {
    for chunk in chunks {
        match *chunk {
            Chunk::Copy { offset, len } => Op::Copy {
                offset: u32::try_from(offset).map_err(|_| PatchError::Overflow)?,
                len: u32::try_from(len).map_err(|_| PatchError::Overflow)?,
            }
            .serialize_to(out)?,
            Chunk::Add { start, len } => Op::Add(literal(new, start, len)?).serialize_to(out)?,
        }
    }
    Ok(())
}

/// Encode chunks in the v2 layout: a header identifying the base, then tags,
/// cursor-relative offsets, lengths and literals as four separate sections.
///
/// # Errors
///
/// [`PatchError::Overflow`] if a chunk does not fit the 32-bit fields of the
/// format, [`PatchError::InputTooLarge`] if a section does not.
pub fn v2(chunks: &[Chunk], old: &[u8], new: &[u8], out: &mut Vec<u8>) -> Result<(), PatchError> {
    let mut tags: Vec<u8> = Vec::with_capacity(chunks.len());
    let mut deltas: Vec<u32> = Vec::with_capacity(chunks.len());
    let mut lengths: Vec<u32> = Vec::with_capacity(chunks.len());
    let mut literals_len: usize = 0;
    // Follows the end of the previous copy; every offset is written relative
    // to it, which is what turns scattered 32-bit addresses into small numbers.
    let mut cursor: u32 = 0;
    let mut monotone = true;

    for chunk in chunks {
        match *chunk {
            Chunk::Copy { offset, len } => {
                let offset = u32::try_from(offset).map_err(|_| PatchError::Overflow)?;
                let len = u32::try_from(len).map_err(|_| PatchError::Overflow)?;
                // Wrapping keeps the difference exact in 32-bit arithmetic and
                // the decoder undoes it the same way, so any pair of offsets
                // round-trips even when the signed difference would not fit.
                let step = i32::from_ne_bytes(offset.wrapping_sub(cursor).to_ne_bytes());
                monotone = monotone && step >= 0_i32;
                tags.push(wire::TAG_COPY);
                deltas.push(vbyte::zigzag(step));
                lengths.push(len);
                cursor = offset.wrapping_add(len);
            }
            Chunk::Add { start, len } => {
                tags.push(wire::TAG_ADD);
                lengths.push(u32::try_from(len).map_err(|_| PatchError::Overflow)?);
                literals_len = literals_len.checked_add(len).ok_or(PatchError::Overflow)?;
                // Validate the range now so the second pass cannot fail.
                literal(new, start, len)?;
            }
        }
    }

    let (base_len, base_hash) = wire::fingerprint(old);
    wire::Header {
        flags: if monotone { wire::FLAG_MONOTONE } else { 0 },
        base_len,
        base_hash,
        ops: tags.len(),
        deltas_len: vbyte::encoded_len(&deltas),
        lengths_len: vbyte::encoded_len(&lengths),
        literals_len,
    }
    .write(out)?;
    out.extend_from_slice(&tags);
    vbyte::encode(&deltas, out);
    vbyte::encode(&lengths, out);
    for chunk in chunks {
        if let Chunk::Add { start, len } = *chunk {
            out.extend_from_slice(literal(new, start, len)?);
        }
    }
    Ok(())
}

fn literal(new: &[u8], start: usize, len: usize) -> Result<&[u8], PatchError> {
    let end = start.checked_add(len).ok_or(PatchError::Overflow)?;
    new.get(start..end).ok_or(PatchError::Overflow)
}
