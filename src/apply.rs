//! Patch application.

use crate::error::PatchError;
use crate::op::Op;
use crate::wire::{self, Layout};

/// Apply `patch` to `old` and return the reconstructed bytes.
///
/// # Errors
///
/// See [`PatchError`] variants. The applier rejects out-of-bounds copies
/// and overflows; it never panics.
pub fn apply_patch(old: &[u8], patch: &[u8]) -> Result<Vec<u8>, PatchError> {
    let mut out = Vec::with_capacity(estimate_apply_capacity(patch.len()));
    apply_patch_into(old, patch, &mut out)?;
    Ok(out)
}

/// Apply `patch` into a caller-supplied buffer. The buffer is **not**
/// cleared first; reconstructed bytes are appended.
///
/// # Errors
///
/// See [`PatchError`] variants.
pub fn apply_patch_into(old: &[u8], patch: &[u8], out: &mut Vec<u8>) -> Result<(), PatchError> {
    // A v1 patch carries no identity, so nothing can be checked; a v2 patch
    // names the base it was built from and applying it to any other one is
    // refused rather than silently producing a plausible wrong document.
    match wire::parse(patch)? {
        Layout::V1(_) => {}
        Layout::V2(header) => header.check_base(old)?,
    }

    for op in Op::iter(patch) {
        match op? {
            Op::Copy { offset, len } => {
                let start = usize::try_from(offset).map_err(|_| PatchError::Overflow)?;
                let len_usize = usize::try_from(len).map_err(|_| PatchError::Overflow)?;
                let end = start.checked_add(len_usize).ok_or(PatchError::Overflow)?;
                let slice = old.get(start..end).ok_or(PatchError::CopyOutOfBounds {
                    offset,
                    len,
                    old_len: old.len(),
                })?;
                out.extend_from_slice(slice);
            }
            Op::Add(bytes) => out.extend_from_slice(bytes),
        }
    }
    Ok(())
}

#[inline]
fn estimate_apply_capacity(patch_len: usize) -> usize {
    patch_len.saturating_mul(2).clamp(64, 1 << 20)
}
