//! Bit codecs between finite symbol spaces and the binary agent alphabet
//! (JAIR §2: actions and percepts are
//! binary strings of fixed per-environment widths).
//!
//! Convention: values are encoded MSB-first, so the codec is the big-endian
//! binary expansion truncated to `width` bits. Rewards are shifted to
//! non-negative *codes* by the environment before encoding (JAIR §7 does the
//! same for its domains).

/// Number of bits needed to encode symbols `0..n_symbols` (minimum 1).
pub fn bits_required(n_symbols: u64) -> u32 {
    debug_assert!(n_symbols >= 1);
    let max = n_symbols.saturating_sub(1);
    if max == 0 {
        1
    } else {
        64 - max.leading_zeros()
    }
}

/// Append the `width`-bit MSB-first encoding of `value` to `out`.
///
/// Panics if `value` does not fit in `width` bits (contract violation).
pub fn encode_bits_msb(value: u64, width: u32, out: &mut Vec<u8>) {
    assert!((1..=64).contains(&width), "width out of range: {width}");
    assert!(
        width == 64 || value < (1u64 << width),
        "value {value} does not fit in {width} bits"
    );
    for i in (0..width).rev() {
        out.push(((value >> i) & 1) as u8);
    }
}

/// Decode an MSB-first bit slice back to the value.
pub fn decode_bits_msb(bits: &[u8]) -> u64 {
    assert!(bits.len() <= 64);
    let mut v = 0u64;
    for &b in bits {
        debug_assert!(b <= 1);
        v = (v << 1) | u64::from(b);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_required_matches_ceil_log2() {
        assert_eq!(bits_required(1), 1);
        assert_eq!(bits_required(2), 1);
        assert_eq!(bits_required(3), 2);
        assert_eq!(bits_required(4), 2);
        assert_eq!(bits_required(5), 3);
        assert_eq!(bits_required(111), 7); // Tiger reward codes 0..=110
    }

    #[test]
    fn roundtrip_exhaustive_small_widths() {
        for width in 1..=10u32 {
            for value in 0..(1u64 << width) {
                let mut bits = Vec::new();
                encode_bits_msb(value, width, &mut bits);
                assert_eq!(bits.len(), width as usize);
                assert_eq!(decode_bits_msb(&bits), value);
            }
        }
    }

    #[test]
    fn msb_first_order() {
        let mut bits = Vec::new();
        encode_bits_msb(0b110, 3, &mut bits);
        assert_eq!(bits, vec![1, 1, 0]);
    }

    #[test]
    #[should_panic]
    fn rejects_overflow() {
        let mut bits = Vec::new();
        encode_bits_msb(4, 2, &mut bits);
    }
}
