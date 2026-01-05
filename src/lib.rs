use rustc_hash::FxHashMap;
use std::fmt;
use std::fmt::Debug;
const WINDOW_SIZE: usize = 10;
const BASE: u64 = 1_934_123_457;

#[derive(PartialEq, Eq)]
pub enum Op<'a> {
    Copy(u32, u32),
    Add(&'a [u8]),
}

impl<'a> Op<'a> {
    #[inline]
    pub fn serialize_to(&self, out: &mut Vec<u8>) {
        match self {
            Op::Copy(offset, len) => {
                out.push(0x00);
                out.extend_from_slice(&offset.to_le_bytes());
                out.extend_from_slice(&len.to_le_bytes());
            }
            Op::Add(bytes) => {
                out.push(0x01);
                let len = bytes.len() as u32;
                out.extend_from_slice(&len.to_le_bytes());
                out.extend_from_slice(bytes);
            }
        }
    }

    #[inline]
    pub fn deserialize(input: &'a [u8]) -> Result<(Self, &'a [u8]), &'static str> {
        match input {
            [0x00, rest @ ..] => {
                if rest.len() < 8 {
                    return Err("unexpected EOF in Copy");
                }
                let offset = u32::from_le_bytes(rest[0..4].try_into().unwrap());
                let len = u32::from_le_bytes(rest[4..8].try_into().unwrap());
                Ok((Op::Copy(offset, len), &rest[8..]))
            }
            [0x01, rest @ ..] => {
                if rest.len() < 4 {
                    return Err("unexpected EOF in Add header");
                }
                let len = u32::from_le_bytes(rest[0..4].try_into().unwrap()) as usize;
                if rest.len() < len + 4 {
                    return Err("unexpected EOF in Add body");
                }
                Ok((Op::Add(&rest[4..4 + len]), &rest[4 + len..]))
            }
            _ => Err("invalid opcode"),
        }
    }

    pub fn deserialize_all(input: &'a [u8]) -> Result<Vec<Self>, &'static str> {
        let mut input = input;
        let mut ops = vec![];
        while !input.is_empty() {
            let (op, next) = Self::deserialize(input)?;
            input = next;
            ops.push(op);
        }
        Ok(ops)
    }
}

impl Debug for Op<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

        impl<'a> fmt::Debug for Content<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    Content::Text(s) => write!(f, "Text({:?})", s),
                    Content::Bytes(b) => {
                        write!(f, "Bytes(")?;
                        for (i, byte) in b.iter().enumerate() {
                            if i > 0 {
                                write!(f, " ")?;
                            }
                            write!(f, "{:02X}", byte)?;
                        }
                        write!(f, ")")
                    }
                }
            }
        }

        match self {
            Self::Copy(offset, len) => f.debug_tuple("Copy").field(offset).field(len).finish(),
            Self::Add(content) => f
                .debug_tuple("Add")
                .field(&Content::from(*content))
                .finish(),
        }
    }
}

fn build_hash_map(data: &[u8]) -> FxHashMap<u64, usize> {
    RollingHash::new(data, WINDOW_SIZE, BASE)
        .map(|rh| rh.into_iter().enumerate().map(|(i, h)| (h, i)).collect())
        .unwrap_or_default()
}

pub fn make_diff(old: &[u8], new: &[u8]) -> Vec<u8> {
    if new.len() < WINDOW_SIZE || old.len() < WINDOW_SIZE {
        let mut patch = Vec::with_capacity(new.len() + 5);
        Op::Add(new).serialize_to(&mut patch);
        return patch;
    }

    let map = build_hash_map(old);
    let mut patch = Vec::with_capacity(512);
    let mut last_emitted = 0;
    let mut idx = 0;
    while idx + WINDOW_SIZE <= new.len() {
        let window = &new[idx..idx + WINDOW_SIZE];
        let (hash, _) = window_hash(window, BASE);
        if let Some(&match_pos) = map.get(&hash) {
            let len = simd_memcmp(&new[idx..], &old[match_pos..]);
            if len >= WINDOW_SIZE {
                if idx > last_emitted {
                    Op::Add(&new[last_emitted..idx]).serialize_to(&mut patch);
                }
                Op::Copy(match_pos as u32, len as u32).serialize_to(&mut patch);
                idx += len;
                last_emitted = idx;
                continue;
            }
        }
        idx += 1;
    }
    if last_emitted < new.len() {
        Op::Add(&new[last_emitted..]).serialize_to(&mut patch);
    }
    patch
}

pub fn apply_patch(old: &[u8], patch: &[u8]) -> Result<Vec<u8>, &'static str> {
    let mut patch = patch;
    let mut out = Vec::new();

    while !patch.is_empty() {
        let (op, next_patch) = Op::deserialize(patch)?;
        patch = next_patch;
        match op {
            Op::Copy(offset, len) => {
                let start = offset as usize;
                let end = start + len as usize;
                if end > old.len() {
                    return Err("copy out of bounds");
                }
                out.extend_from_slice(&old[start..end]);
            }
            Op::Add(bytes) => {
                out.extend_from_slice(bytes);
            }
        }
    }

    Ok(out)
}

#[inline]
fn simd_memcmp(a: &[u8], b: &[u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { simd_memcmp_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { simd_memcmp_neon(a, b) };
        }
    }

    simd_memcmp_fallback(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn simd_memcmp_avx2(a: &[u8], b: &[u8]) -> usize {
    use std::arch::x86_64::*;
    let len = a.len().min(b.len());
    let mut i = 0;
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    while i + 32 <= len {
        let chunk_a = _mm256_loadu_si256(pa.add(i) as *const __m256i);
        let chunk_b = _mm256_loadu_si256(pb.add(i) as *const __m256i);
        let cmp = _mm256_cmpeq_epi8(chunk_a, chunk_b);
        let mask = _mm256_movemask_epi8(cmp) as u32;
        if mask != 0xFFFF_FFFF {
            let diff_index = _tzcnt_u32(!mask);
            return i + diff_index as usize;
        }
        i += 32;
    }

    while i < len && a[i] == b[i] {
        i += 1;
    }

    i
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn simd_memcmp_neon(a: &[u8], b: &[u8]) -> usize {
    use std::arch::aarch64::*;
    let len = a.len().min(b.len());
    let mut i = 0;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    while i + 16 <= len {
        let chunk_a = vld1q_u8(a_ptr.add(i));
        let chunk_b = vld1q_u8(b_ptr.add(i));
        let cmp = vceqq_u8(chunk_a, chunk_b);
        if vminvq_u8(cmp) != 0xFF {
            for j in 0..16 {
                if a[i + j] != b[i + j] {
                    return i + j;
                }
            }
        }
        i += 16;
    }

    while i < len && a[i] == b[i] {
        i += 1;
    }

    i
}

#[inline]
fn simd_memcmp_fallback(a: &[u8], b: &[u8]) -> usize {
    let len = a.len().min(b.len());
    let mut i = 0;
    while i < len && a[i] == b[i] {
        i += 1;
    }
    i
}

#[inline]
fn window_hash(data: &[u8], base: u64) -> (u64, u64) {
    let mut hash: u64 = 0;
    let mut base_pow: u64 = 1;
    for (i, &byte) in data.iter().enumerate() {
        hash = hash.wrapping_mul(base).wrapping_add(byte as u64);
        if i < data.len() - 1 {
            base_pow = base_pow.wrapping_mul(base);
        }
    }
    (hash, base_pow)
}

pub struct RollingHash<'a> {
    data: &'a [u8],
    pos: usize,
    window_size: usize,
    base_pow: u64,
    hash: u64,
    base: u64,
}

impl<'a> RollingHash<'a> {
    pub fn new(data: &'a [u8], window_size: usize, base: u64) -> Option<Self> {
        if data.len() < window_size {
            return None;
        }
        let (hash, base_pow) = window_hash(&data[..window_size], base);
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

impl<'a> Iterator for RollingHash<'a> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + self.window_size > self.data.len() {
            return None;
        }

        let out = self.data[self.pos] as u64;
        let next = self.pos + self.window_size;
        let result = self.hash;

        if next < self.data.len() {
            let r#in = self.data[next] as u64;
            self.hash = self
                .hash
                .wrapping_sub(out.wrapping_mul(self.base_pow))
                .wrapping_mul(self.base)
                .wrapping_add(r#in);
        }

        self.pos += 1;
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_patches() {
        let digits = (0..100).collect::<Vec<_>>();
        let tail = vec![0, 1];
        let shifted_digits = digits
            .clone()
            .into_iter()
            .skip(2)
            .chain(tail.iter().copied())
            .collect::<Vec<_>>();
        let baz = b"baz".to_vec();

        let cases: Vec<(Vec<u8>, Vec<u8>, _)> = vec![
            (digits.clone(), digits.clone(), Some(vec![Op::Copy(0, 100)])),
            (
                digits.clone(),
                shifted_digits.clone(),
                Some(vec![Op::Copy(2, 98), Op::Add(tail.as_slice())]),
            ),
            (vec![], vec![1, 2, 3, 4, 5, 6, 7, 8], None),
            (vec![], vec![1, 2, 3], None),
            (vec![], vec![], Some(vec![Op::Add(&[])])),
            (
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                vec![],
                Some(vec![Op::Add(&[])]),
            ),
            (
                b"-foo-baar-hello-world".into(),
                b"hello-world-foo-baar-baz".into(),
                Some(vec![Op::Copy(10, 11), Op::Copy(0, 10), Op::Add(&baz)]),
            ),
            (
                b"just-swaps-with-no-adds".into(),
                b"-with-no-addsjust-swaps".into(),
                Some(vec![Op::Copy(10, 13), Op::Copy(0, 10)]),
            ),
        ];

        for (old, new, expected_ops) in cases {
            let patch = make_diff(&old, &new);
            let ops = Op::deserialize_all(&patch).unwrap();
            if let Some(expected) = expected_ops {
                assert_eq!(ops, expected);
            }
            let patched = apply_patch(&old, &patch).unwrap();
            assert_eq!(&patched, &new);
        }
    }

    #[test]
    fn test_rolling_hash() {
        const WND: usize = 3;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 8];
        let rh = RollingHash::new(&data, WND, 1_934_123_457).unwrap();
        let hashes: Vec<_> = data
            .as_slice()
            .windows(WND)
            .enumerate()
            .zip(rh)
            .filter(|((_, wnd), _)| wnd == &[2, 3, 4])
            .map(|((_, _), hash)| hash)
            .collect();
        assert!(hashes.len() > 1);
        assert_eq!(hashes.into_iter().collect::<HashSet<_>>().len(), 1);
    }

    #[test]
    fn test_edge_cases() {
        #[derive(Debug)]
        struct Case<'a> {
            old: &'a [u8],
            new: Vec<u8>,
            desc: &'a str,
        }
        const W: usize = WINDOW_SIZE;
        let rep = b"abcdefghij";
        let mut repeated = Vec::new();
        for _ in 0..5 {
            repeated.extend_from_slice(rep);
        }
        let cases = vec![
            Case {
                old: b"abcdefghij12345abcdefghij",
                new: b"abcdefghij12345abcdefghij".to_vec(),
                desc: "long matching block",
            },
            {
                let mut v = b"abcdefghij12345abcdefghij".to_vec();
                v.insert(W + 2, b'X');
                Case {
                    old: b"abcdefghij12345abcdefghij",
                    new: v,
                    desc: "insertion inside long block",
                }
            },
            {
                let mut v = b"abcdefghij12345abcdefghij".to_vec();
                v.remove(W + 2);
                Case {
                    old: b"abcdefghij12345abcdefghij",
                    new: v,
                    desc: "deletion inside long block",
                }
            },
            {
                let mut v = b"abcdefghij12345abcdefghij".to_vec();
                v[W + 2] = b'Z';
                Case {
                    old: b"abcdefghij12345abcdefghij",
                    new: v,
                    desc: "replacement inside long block",
                }
            },
            Case {
                old: &repeated,
                new: repeated.clone(),
                desc: "repeated windows > WINDOW_SIZE",
            },
            Case {
                old: b"abcdefghij12345",
                new: b"ZZZabcdefghij12345".to_vec(),
                desc: "Add at start, Copy after window",
            },
            Case {
                old: b"abcdefghij12345",
                new: b"abcdefghij12345YYY".to_vec(),
                desc: "Add at end, Copy at start",
            },
            Case {
                old: b"abcdefghij12345abcdefghij",
                new: b"abcdefghij12345Xabcdefghij".to_vec(),
                desc: "Add at window boundary",
            },
        ];
        for case in cases {
            let patch = make_diff(case.old, &case.new);
            let patched = apply_patch(case.old, &patch).unwrap();
            assert_eq!(&patched, &case.new, "failed: {}", case.desc);
        }
    }

    #[rstest]
    #[case(&[] as &[u8], &[], 0)]
    #[case(b"Hello, world!", b"", 0)]
    #[case(b"", b"Hello, world!", 0)]
    #[case(b"Hello, world!", b"Hello, world!", 13)]
    #[case(b"abc", b"xbc", 0)]
    #[case(b"abc", b"axc", 1)]
    #[case(b"abc", b"abd", 2)]
    #[case(b"abcdef", b"abc", 3)]
    #[case(&vec![0u8; 1000][..], &vec![0u8; 1000][..], 1000)]
    fn test_simd_memcmp(#[case] a: &[u8], #[case] b: &[u8], #[case] expected: usize) {
        assert_eq!(simd_memcmp(a, b), expected);
    }

    #[test]
    fn test_simd_memcmp_large_diff() {
        let a = vec![0u8; 1000];
        let mut b = vec![0u8; 1000];
        b[500] = 1;
        assert_eq!(simd_memcmp(&a, &b), 500);
    }
}
