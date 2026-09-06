//! Rolling hash and the global content index.
//!
//! The index is the differ's last resort. It costs `O(old.len())` time and
//! memory to build, so it is constructed lazily — only when local resync has
//! failed and the update turns out to be a reorder, a bulk move, or a format
//! change rather than a local edit.

use rustc_hash::{FxBuildHasher, FxHashMap};

pub const WINDOW_SIZE: usize = 10;
pub const HASH_BASE: u64 = 1_934_123_457;

/// Maps a window hash to the first position in `old` where it occurs.
///
/// Keeping the *first* occurrence tends to yield longer matches and more
/// compact patches than keeping the last.
pub fn build_hash_map(data: &[u8]) -> FxHashMap<u64, usize> {
    let Some(rh) = RollingHash::new(data, WINDOW_SIZE, HASH_BASE) else {
        return FxHashMap::default();
    };
    let cap = data
        .len()
        .saturating_sub(WINDOW_SIZE)
        .saturating_add(1_usize);
    let mut map: FxHashMap<u64, usize> = FxHashMap::with_capacity_and_hasher(cap, FxBuildHasher);
    for (i, h) in rh.enumerate() {
        map.entry(h).or_insert(i);
    }
    map
}

#[inline]
fn window_hash(data: &[u8], base: u64) -> (u64, u64) {
    let mut hash: u64 = 0;
    let mut base_pow: u64 = 1;
    let last = data.len().saturating_sub(1);
    for (i, &byte) in data.iter().enumerate() {
        hash = hash.wrapping_mul(base).wrapping_add(u64::from(byte));
        if i < last {
            base_pow = base_pow.wrapping_mul(base);
        }
    }
    (hash, base_pow)
}

/// Iterator over rolling hashes of fixed-size windows.
pub struct RollingHash<'a> {
    data: &'a [u8],
    pos: usize,
    window_size: usize,
    base_pow: u64,
    hash: u64,
    base: u64,
}

impl<'a> RollingHash<'a> {
    /// Create a rolling hash iterator over `data`.
    ///
    /// Returns `None` if `data` is shorter than `window_size`.
    #[must_use]
    pub fn new(data: &'a [u8], window_size: usize, base: u64) -> Option<Self> {
        let initial = data.get(..window_size)?;
        let (hash, base_pow) = window_hash(initial, base);
        Some(Self {
            data,
            pos: 0,
            window_size,
            base_pow,
            hash,
            base,
        })
    }
}

impl Iterator for RollingHash<'_> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let end = self.pos.checked_add(self.window_size)?;
        if end > self.data.len() {
            return None;
        }

        let result = self.hash;

        // Roll the hash forward to the next window, if any.
        if let (Some(&out_byte), Some(&in_byte)) = (self.data.get(self.pos), self.data.get(end)) {
            self.hash = self
                .hash
                .wrapping_sub(u64::from(out_byte).wrapping_mul(self.base_pow))
                .wrapping_mul(self.base)
                .wrapping_add(u64::from(in_byte));
        }

        self.pos = self.pos.checked_add(1)?;
        Some(result)
    }
}
