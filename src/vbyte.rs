//! `StreamVByte`: a run of `u32` values encoded as a control stream of 2-bit
//! width codes followed by a data stream of the significant bytes.
//!
//! Splitting the two streams is what makes decoding branch-free-ish and lets
//! the compressor see the control bytes — which are highly repetitive — apart
//! from the payload. LEB128 would interleave them into one unpredictable
//! byte-at-a-time loop.

/// Bytes of control stream needed for `count` values.
pub fn control_len(count: usize) -> usize {
    count.div_ceil(4)
}

/// Append `values` to `out` as a control stream followed by a data stream.
pub fn encode(values: &[u32], out: &mut Vec<u8>) {
    let control_start = out.len();
    out.resize(control_start.saturating_add(control_len(values.len())), 0);

    for (index, &value) in values.iter().enumerate() {
        let width = byte_width(value);
        let code = u8::try_from(width.saturating_sub(1)).unwrap_or(0);
        let slot = control_start.saturating_add(index / 4);
        let shift = u32::try_from((index % 4).saturating_mul(2)).unwrap_or(0);
        if let Some(byte) = out.get_mut(slot) {
            *byte |= code.wrapping_shl(shift);
        }
        if let Some(bytes) = value.to_le_bytes().get(..width) {
            out.extend_from_slice(bytes);
        }
    }
}

/// Total encoded size of `values`, without encoding them.
pub fn encoded_len(values: &[u32]) -> usize {
    values
        .iter()
        .fold(control_len(values.len()), |total, &value| {
            total.saturating_add(byte_width(value))
        })
}

fn byte_width(value: u32) -> usize {
    match value {
        0..=0xFF => 1,
        0x100..=0xFFFF => 2,
        0x1_0000..=0x00FF_FFFF => 3,
        0x0100_0000..=0xFFFF_FFFF => 4,
    }
}

/// Streaming decoder over a section produced by [`encode`].
#[derive(Debug, Clone)]
pub struct Reader<'a> {
    control: &'a [u8],
    data: &'a [u8],
    index: usize,
    offset: usize,
}

impl<'a> Reader<'a> {
    /// Split `section` into its control and data streams.
    ///
    /// Returns `None` if the section is too short to hold `count` control
    /// codes.
    pub fn new(section: &'a [u8], count: usize) -> Option<Self> {
        let (control, data) = section.split_at_checked(control_len(count))?;
        Some(Self {
            control,
            data,
            index: 0,
            offset: 0,
        })
    }
}

impl Iterator for Reader<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        let slot = self.control.get(self.index / 4)?;
        let shift = u32::try_from((self.index % 4).saturating_mul(2)).unwrap_or(0);
        let code = usize::try_from(u32::from(*slot).wrapping_shr(shift) & 0b11).ok()?;
        let width = code.saturating_add(1);

        let end = self.offset.checked_add(width)?;
        let bytes = self.data.get(self.offset..end)?;
        let mut buffer = [0_u8; 4];
        buffer.get_mut(..width)?.copy_from_slice(bytes);

        self.offset = end;
        self.index = self.index.checked_add(1)?;
        Some(u32::from_le_bytes(buffer))
    }
}

/// Map a signed delta onto an unsigned value whose magnitude tracks the
/// delta's, so that small steps in either direction stay one byte wide.
pub fn zigzag(value: i32) -> u32 {
    let folded = value.wrapping_shl(1) ^ value.wrapping_shr(31);
    u32::from_ne_bytes(folded.to_ne_bytes())
}

/// Inverse of [`zigzag`].
pub fn unzigzag(value: u32) -> i32 {
    // `value >> 1` clears the sign bit, so the bit-cast cannot change it.
    let magnitude = i32::from_ne_bytes(value.wrapping_shr(1).to_ne_bytes());
    let sign = i32::from_ne_bytes((value & 1).to_ne_bytes()).wrapping_neg();
    magnitude ^ sign
}
