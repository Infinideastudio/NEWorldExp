use serde::{Deserialize, Serialize};

/// Index into some block registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockId(u16);

impl BlockId {
    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

impl Default for BlockId {
    /// The registry's reserved empty slot.
    fn default() -> Self {
        Self::new(0)
    }
}

/// Packed (sky, block) lighting byte. Each channel is in `0..=15`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockLight(u8);

impl BlockLight {
    pub const fn new(value: u8) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u8 {
        self.0
    }

    pub const fn sky_and_block(sky: u8, block: u8) -> Self {
        assert!(sky <= 15, "BlockLight::new: sky > 15");
        assert!(block <= 15, "BlockLight::new: block > 15");
        Self::new((sky << 4) | (block & 0x0F))
    }

    pub const fn sky(self) -> u8 {
        self.get() >> 4
    }

    pub const fn block(self) -> u8 {
        self.get() & 0x0F
    }
}

impl Default for BlockLight {
    /// Default to no light.
    fn default() -> Self {
        Self::sky_and_block(0, 0)
    }
}

/// 16-bit block state with an external-storage tag bit.
///
/// Layout: the high bit (`0x8000`) is the *external* flag; the low 15 bits
/// are the payload.
///
/// * `BlockState(0x0000..=0x7FFF)` — **inline**: the 15-bit value is the
///   block's full state (orientation, sub-variant, etc.). `BlockState(0)`
///   is the default and the natural "no-state" value.
/// * `BlockState(0x8000..=0xFFFF)` — **external**: the 15 low bits index
///   into per-cell state stored outside the chunk grid (signs, chests,
///   anything bigger than 15 bits). The chunk array still holds a fixed
///   `BlockState`; whatever side table the world keeps does the lookup
///   when the block actually needs the larger payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockState(u16);

impl BlockState {
    /// High bit marking external storage.
    const EXTERNAL_BIT: u16 = 0x8000;
    /// Low 15 bits — payload mask for both variants.
    const VALUE_MASK: u16 = 0x7FFF;

    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u16 {
        self.0
    }

    /// Inline state. `value` must fit in 15 bits.
    pub const fn inline(value: u16) -> Self {
        assert!(
            value <= Self::VALUE_MASK,
            "BlockState::inline: value > 0x7FFF"
        );
        Self::new(value)
    }

    /// External-storage tag with the given 15-bit index.
    pub const fn external(index: u16) -> Self {
        assert!(
            index <= Self::VALUE_MASK,
            "BlockState::external: index > 0x7FFF"
        );
        Self::new(Self::EXTERNAL_BIT | index)
    }

    /// True iff the high bit is set (external storage).
    pub const fn is_external(self) -> bool {
        self.get() & Self::EXTERNAL_BIT != 0
    }

    /// 15-bit external index when [`Self::is_external`], else `None`.
    pub const fn external_index(self) -> Option<u16> {
        if self.is_external() {
            Some(self.get() & Self::VALUE_MASK)
        } else {
            None
        }
    }

    /// 15-bit inline value when *not* external, else `None`. The inline
    /// payload is what `face_for` / `Orientation::for_block` consume.
    pub const fn inline_value(self) -> Option<u16> {
        if self.is_external() {
            None
        } else {
            Some(self.get() & Self::VALUE_MASK)
        }
    }

    /// Convenience: inline value or zero when external. The chunk mesher and
    /// orientation table both treat external states as "no inline payload",
    /// which falls back to the canonical placement.
    pub const fn inline_or_zero(self) -> u16 {
        if self.is_external() {
            0
        } else {
            self.get() & Self::VALUE_MASK
        }
    }
}

impl Default for BlockState {
    /// The default state for any block which doesn't need a state.
    fn default() -> Self {
        Self::inline(0)
    }
}

/// Per-cell payload stored in chunks. Three fields, encoded on disk as
/// `id (u16 LE) + state (u16 LE) + light (u8)` (5 bytes per cell).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default, Serialize, Deserialize)]
pub struct BlockData {
    pub id: BlockId,
    pub state: BlockState,
    pub light: BlockLight,
}

impl BlockData {
    /// Per-cell on-disk size (5 bytes: id u16 + state u16 + light u8).
    pub const ENCODED_LEN: usize = 5;

    /// Encode to little-endian bytes filling a [`Self::ENCODED_LEN`]-sized
    /// slot. The chunk save path calls this once per cell, writing into a
    /// preallocated body buffer — appending into a growable `Vec` instead
    /// cost ~7× as much across a chunk (a capacity check plus a memcpy
    /// call per field, 4096 times).
    pub fn encode_into(self, out: &mut [u8]) {
        debug_assert_eq!(out.len(), Self::ENCODED_LEN);
        out[0..2].copy_from_slice(&self.id.get().to_le_bytes());
        out[2..4].copy_from_slice(&self.state.get().to_le_bytes());
        out[4] = self.light.get();
    }

    /// Decode from a 5-byte slice. Caller is responsible for splitting the
    /// chunk body into [`Self::ENCODED_LEN`]-sized chunks.
    pub fn decode_from(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), Self::ENCODED_LEN);
        Self {
            id: BlockId::new(u16::from_le_bytes([bytes[0], bytes[1]])),
            state: BlockState::new(u16::from_le_bytes([bytes[2], bytes[3]])),
            light: BlockLight::new(bytes[4]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_all_zero() {
        let d = BlockData::default();
        assert_eq!(d.id, BlockId::new(0));
        assert_eq!(d.state, BlockState::new(0));
        assert_eq!(d.light, BlockLight::new(0));
    }

    #[test]
    fn encode_decode_round_trips() {
        let cases = [
            BlockData::default(),
            BlockData {
                id: BlockId::new(7),
                state: BlockState::inline(0x1234),
                light: BlockLight::sky_and_block(15, 7),
            },
            BlockData {
                id: BlockId::new(0xFFFF),
                state: BlockState::external(0x7FFF),
                light: BlockLight::sky_and_block(0, 0),
            },
        ];
        let mut buf = [0u8; BlockData::ENCODED_LEN];
        for &c in &cases {
            buf.fill(0);
            c.encode_into(&mut buf);
            assert_eq!(BlockData::decode_from(&buf), c);
        }
    }
}
