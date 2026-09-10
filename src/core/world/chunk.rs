//! `Chunk` — one entry in the world's chunk store, plus the
//! [`ChunkData`] block array and on-disk codec it wraps.
//!
//! `Chunk` is both the runtime cache-slot type (lock + atomics +
//! LSN pair + lease counter + eviction state) and the namespace
//! for the chunk-geometry constants ([`Chunk::SIZE`],
//! [`Chunk::SIZE_LOG`]). [`ChunkData`] is the bare
//! `SIZE × SIZE × SIZE` block array that lives under each
//! `Chunk`'s `RwLock`.
//!
//! ## Pin discipline
//!
//! A `Chunk` is reachable from the world's
//! `DashMap<Vec3i, Arc<Chunk>>` while it's resident. Outside of
//! the world module, **the only way to obtain a usable reference
//! is to acquire a [`Lease`]** — an RAII handle that increments
//! the chunk's lease counter on construction and decrements it on
//! drop. While any lease is live, eviction cannot finish.
//!
//! Eviction is a state machine:
//!
//! 1. The world CAS-flips the chunk's state from `RESIDENT` to
//!    `EVICTING`. If the CAS fails, the caller bails — somebody
//!    else is already evicting this chunk.
//! 2. After the CAS, new [`Chunk::try_acquire_lease`] calls observe
//!    `EVICTING` and refuse — no fresh leases can appear.
//! 3. The evictor calls [`Chunk::wait_drain`] until the lease
//!    counter hits zero (currently a `yield_now` spin — eviction
//!    is the cold path).
//! 4. The evictor flushes the chunk to disk and removes it from
//!    the world map.
//!
//! With this discipline, a transaction holding a lease and the
//! `Arc<Chunk>` it points at always agree on identity — the world
//! cannot install a fresh `Chunk` for the same coord while the
//! evicting one is still leased. Orphan-arcs are impossible.
//!
//! ## Owned guards
//!
//! Transactions need to hold the chunk's `RwLock` guard across
//! many operations, so we hand out *owned* guards: a
//! [`ChunkReadGuard`] / [`ChunkWriteGuard`] bundles the
//! parking_lot guard with an `Arc<Chunk>` keepalive. Field
//! declaration order in the guard struct ensures the lock is
//! released *before* the keepalive Arc decrements; see the SAFETY
//! comment on [`Chunk::read_owned`].
//!
//! ## LSN pair
//!
//! Each chunk carries `commit_lsn` (under the lock; mutated under
//! the write guard at commit time) and `persisted_lsn` (atomic;
//! advanced by writeback after a successful disk write). The pair
//! `persisted_lsn < commit_lsn` means dirty.
//!
//! Putting `commit_lsn` *under the lock* fixes the order-inversion
//! bug in the previous design: any reader who holds the read lock
//! sees a consistent (data, lsn) pair.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::core::blocks::{BlockData, BlockId};
use crate::core::math::Vec3u;

use super::errors::ChunkError;

// ----------------------------------------------------------------------
//   Eviction-state encoding
// ----------------------------------------------------------------------

pub(super) const RESIDENT: u8 = 0;
pub(super) const EVICTING: u8 = 1;

// ----------------------------------------------------------------------
//   ChunkData — the SIZE × SIZE × SIZE block array
// ----------------------------------------------------------------------

/// The `SIZE × SIZE × SIZE` block array for one chunk. Lives under
/// [`ChunkInner`] which itself lives under each [`Chunk`]'s
/// `RwLock`.
pub struct ChunkData([BlockData; Chunk::SIZE * Chunk::SIZE * Chunk::SIZE]);

impl ChunkData {
    /// Magic bytes (`"NEWC"`) identifying a packaged chunk on disk.
    pub const MAGIC: u32 = 0x4E45_5743;

    /// Current on-disk format version. Version 3 adds zstd compression of
    /// the body — the header stays plain so corrupt files can be
    /// diagnosed without invoking the decoder.
    pub const VERSION: u32 = 3;

    /// Header size in bytes.
    pub const HEADER_SIZE: usize = 8;

    /// Total per-block bytes when serialised to disk.
    pub const DATA_SIZE: usize = Chunk::SIZE * Chunk::SIZE * Chunk::SIZE * BlockData::ENCODED_LEN;

    /// zstd compression level used when packaging a chunk. Level 3 is
    /// zstd's "balanced" default — fast enough for save-on-evict on the
    /// main thread (~tens of µs per chunk on contemporary CPUs) and
    /// gives a strong ratio on homogeneous voxel data.
    pub const COMPRESSION_LEVEL: i32 = 3;

    /// Linear index for `(x, y, z)` block-local coords (X-major,
    /// matching the C++ port).
    pub fn index(bcoord: Vec3u) -> usize {
        let x = bcoord.x as usize;
        let y = bcoord.y as usize;
        let z = bcoord.z as usize;
        debug_assert!(
            x < Chunk::SIZE && y < Chunk::SIZE && z < Chunk::SIZE,
            "block coordinates out of bounds"
        );
        (x * Chunk::SIZE + y) * Chunk::SIZE + z
    }

    pub fn block(&self, bcoord: Vec3u) -> BlockData {
        self.0[Self::index(bcoord)]
    }

    pub fn block_mut(&mut self, bcoord: Vec3u) -> &mut BlockData {
        &mut self.0[Self::index(bcoord)]
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, BlockData> {
        self.0.iter_mut()
    }

    pub fn as_slice(&self) -> &[BlockData] {
        &self.0
    }

    /// Serialize to bytes: an 8-byte plain header followed by the
    /// zstd-compressed block array (`SIZE_CUBED` little-endian
    /// records before compression). `current_to_canonical` translates
    /// each cell's in-memory id to the canonical id stored on disk;
    /// pass an empty slice for an identity mapping.
    ///
    /// Compression typically shrinks chunks 5–200× — homogeneous
    /// chunks (pure air above terrain, pure rock below) collapse
    /// almost to nothing, mixed chunks settle around 4–10×.
    pub fn package_to(&self, current_to_canonical: &[BlockId]) -> Vec<u8> {
        let mut body = Vec::with_capacity(Self::DATA_SIZE);
        let identity = current_to_canonical.is_empty();
        for cell in self.0.iter() {
            let mut translated = *cell;
            if !identity {
                translated.id = current_to_canonical
                    .get(cell.id.get() as usize)
                    .copied()
                    .unwrap_or(BlockId::default());
            }
            translated.encode_to(&mut body);
        }
        let compressed = zstd::bulk::compress(&body, Self::COMPRESSION_LEVEL)
            .expect("zstd compression of a fixed-size buffer cannot fail");
        let mut out = Vec::with_capacity(Self::HEADER_SIZE + compressed.len());
        out.extend_from_slice(&Self::MAGIC.to_le_bytes());
        out.extend_from_slice(&Self::VERSION.to_le_bytes());
        out.extend_from_slice(&compressed);
        out
    }

    /// Deserialize bytes (produced by [`Self::package_to`]) into
    /// `self`. Verifies magic + version, then decompresses the body
    /// directly into a `SIZE_DATA`-bounded buffer (so a malicious
    /// header can't trigger an unbounded allocation).
    /// `canonical_to_current` translates each cell's canonical id
    /// (read from disk) to the in-memory `BlockId`; pass an empty
    /// slice for identity.
    pub fn unpackage_from(
        &mut self,
        bytes: &[u8],
        canonical_to_current: &[BlockId],
    ) -> Result<(), ChunkError> {
        if bytes.len() < Self::HEADER_SIZE {
            return Err(ChunkError::Size {
                expected: Self::HEADER_SIZE,
                got: bytes.len(),
            });
        }
        let (head, body) = bytes.split_at(Self::HEADER_SIZE);
        let magic = u32::from_le_bytes([head[0], head[1], head[2], head[3]]);
        let version = u32::from_le_bytes([head[4], head[5], head[6], head[7]]);
        if magic != Self::MAGIC {
            return Err(ChunkError::Magic);
        }
        if version != Self::VERSION {
            return Err(ChunkError::Version { got: version });
        }
        let mut decompressed = vec![0u8; Self::DATA_SIZE];
        let n = zstd::bulk::decompress_to_buffer(body, &mut decompressed)
            .map_err(|_| ChunkError::Compression)?;
        if n != Self::DATA_SIZE {
            return Err(ChunkError::Size {
                expected: Self::DATA_SIZE,
                got: n,
            });
        }
        let identity = canonical_to_current.is_empty();
        for (i, slot) in self.0.iter_mut().enumerate() {
            let off = i * BlockData::ENCODED_LEN;
            let mut cell = BlockData::decode_from(&decompressed[off..off + BlockData::ENCODED_LEN]);
            if !identity {
                cell.id = canonical_to_current
                    .get(cell.id.get() as usize)
                    .copied()
                    .unwrap_or(BlockId::default());
            }
            *slot = cell;
        }
        Ok(())
    }
}

impl Default for ChunkData {
    /// Empty chunk with all blocks set to the default (empty, no light) state.
    fn default() -> Self {
        Self([BlockData::default(); Chunk::SIZE * Chunk::SIZE * Chunk::SIZE])
    }
}

// ----------------------------------------------------------------------
//   ChunkInner — what lives under the lock
// ----------------------------------------------------------------------

/// What lives under each [`Chunk`]'s `RwLock`. Pairing
/// `commit_lsn` with the data inside the same lock guarantees that
/// any reader holding the lock sees a consistent (data, lsn) pair.
pub struct ChunkInner {
    pub data: ChunkData,
    pub commit_lsn: u64,
}

// ----------------------------------------------------------------------
//   Chunk — the runtime cache slot
// ----------------------------------------------------------------------

pub struct Chunk {
    blocks: RwLock<ChunkInner>,
    persisted_lsn: AtomicU64,
    leases: AtomicUsize,
    state: AtomicU8,
    updated: AtomicBool,
}

impl Chunk {
    /// `log2(SIZE)` — chunks are `SIZE × SIZE × SIZE` blocks.
    pub const SIZE_LOG: usize = 4;

    /// Edge length of a chunk (= 16).
    pub const SIZE: usize = 1 << Self::SIZE_LOG;

    /// Build a chunk just loaded from disk.
    /// `(commit_lsn, persisted_lsn) = (0, 0)` — clean.
    pub(super) fn from_disk(data: ChunkData) -> Arc<Self> {
        Arc::new(Self {
            blocks: RwLock::new(ChunkInner {
                data,
                commit_lsn: 0,
            }),
            persisted_lsn: AtomicU64::new(0),
            leases: AtomicUsize::new(0),
            state: AtomicU8::new(RESIDENT),
            updated: AtomicBool::new(false),
        })
    }

    /// Build a chunk for freshly-generated terrain. `lsn` must be a
    /// non-zero LSN allocated from the world's counter; storing it
    /// in `commit_lsn` while `persisted_lsn` stays zero marks the
    /// chunk dirty until writeback.
    pub(super) fn from_generated(data: ChunkData, lsn: u64) -> Arc<Self> {
        debug_assert!(lsn > 0, "lsn 0 is reserved for clean from_disk chunks");
        Arc::new(Self {
            blocks: RwLock::new(ChunkInner {
                data,
                commit_lsn: lsn,
            }),
            persisted_lsn: AtomicU64::new(0),
            leases: AtomicUsize::new(0),
            state: AtomicU8::new(RESIDENT),
            updated: AtomicBool::new(false),
        })
    }

    // ---- lease + eviction state ----------------------------------------

    /// Try to take a lease. Returns `None` if the chunk is in the
    /// `EVICTING` state — caller should treat that as "chunk not
    /// currently available" and retry.
    ///
    /// Pattern: optimistic increment, then check state. If we raced
    /// with [`Self::start_eviction`], we revert. The transient
    /// `leases >= 1` doesn't reflect a real user — the lease isn't
    /// returned to the caller.
    pub(super) fn try_acquire_lease(self: &Arc<Self>) -> Option<Lease> {
        self.leases.fetch_add(1, Ordering::AcqRel);
        if self.state.load(Ordering::Acquire) != RESIDENT {
            self.leases.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
        Some(Lease {
            chunk: Arc::clone(self),
        })
    }

    /// Cheap peek at the eviction state — true iff the chunk is
    /// `RESIDENT`. Used by [`super::World::is_loaded`] /
    /// [`super::World::load_chunk`] to decide whether the entry in
    /// the chunk map is usable as-is or whether to wait/retry.
    pub(super) fn is_resident(&self) -> bool {
        self.state.load(Ordering::Acquire) == RESIDENT
    }

    /// CAS-flip `RESIDENT → EVICTING`. Returns true on success.
    /// After this returns true, no fresh leases can be acquired.
    /// The caller must then [`Self::wait_drain`] before flushing.
    pub(super) fn start_eviction(&self) -> bool {
        self.state
            .compare_exchange(RESIDENT, EVICTING, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Spin (with `yield_now`) until `leases == 0`. Eviction is
    /// rare and lease holds are short; if this becomes a problem,
    /// switch to a Condvar.
    pub(super) fn wait_drain(&self) {
        while self.leases.load(Ordering::Acquire) > 0 {
            std::thread::yield_now();
        }
    }

    // ---- owned guards (the unsafe lives here) --------------------------

    pub(super) fn read_owned(self: &Arc<Self>) -> ChunkReadGuard {
        let guard = self.blocks.read();
        // SAFETY: the returned `ChunkReadGuard` stores
        // `Arc::clone(self)` alongside the guard. That Arc keeps
        // the chunk — and thus the RwLock allocation the guard
        // points at — alive for at least as long as the guard.
        // `ChunkReadGuard`'s field declaration order ensures the
        // guard is dropped before the Arc, so the lock is always
        // released before any potential deallocation.
        let guard: RwLockReadGuard<'static, ChunkInner> = unsafe { std::mem::transmute(guard) };
        ChunkReadGuard {
            guard,
            _chunk: Arc::clone(self),
        }
    }

    pub(super) fn write_owned(self: &Arc<Self>) -> ChunkWriteGuard {
        let guard = self.blocks.write();
        // SAFETY: same justification as `read_owned`.
        let guard: RwLockWriteGuard<'static, ChunkInner> = unsafe { std::mem::transmute(guard) };
        ChunkWriteGuard {
            guard,
            _chunk: Arc::clone(self),
        }
    }

    // ---- LSN pair ------------------------------------------------------

    pub(super) fn persisted_lsn(&self) -> u64 {
        self.persisted_lsn.load(Ordering::Acquire)
    }

    /// Advance `persisted_lsn` to `max(persisted_lsn, captured)`.
    /// Called by writeback after a successful disk write.
    /// Monotonic and idempotent.
    pub(super) fn advance_persisted_lsn(&self, captured: u64) {
        self.persisted_lsn.fetch_max(captured, Ordering::AcqRel);
    }

    /// True iff `persisted_lsn < commit_lsn`. Reads `commit_lsn`
    /// under the read lock — `dirty()` is intended for cold-path
    /// callers (sweep, save) where the lock cost is acceptable.
    pub(super) fn dirty(&self) -> bool {
        let persisted = self.persisted_lsn.load(Ordering::Acquire);
        persisted < self.blocks.read().commit_lsn
    }

    // ---- updated (renderer hook) ---------------------------------------

    pub(super) fn mark_updated(&self) {
        self.updated.store(true, Ordering::Release);
    }
    pub(super) fn clear_updated(&self) {
        self.updated.store(false, Ordering::Release);
    }
    pub(super) fn updated(&self) -> bool {
        self.updated.load(Ordering::Acquire)
    }
}

// ----------------------------------------------------------------------
//   Lease
// ----------------------------------------------------------------------

/// RAII pin on a chunk. Holding a `Lease` blocks eviction
/// completion for the pinned coord; dropping it allows eviction
/// to proceed (if [`Chunk::start_eviction`] already fired).
pub struct Lease {
    chunk: Arc<Chunk>,
}

impl Lease {
    pub(super) fn chunk(&self) -> &Arc<Chunk> {
        &self.chunk
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.chunk.leases.fetch_sub(1, Ordering::AcqRel);
    }
}

// ----------------------------------------------------------------------
//   Owned guards
// ----------------------------------------------------------------------

/// Owned read guard. Field order is load-bearing: `guard` must
/// drop first (releasing the lock) before `_chunk` (which may free
/// the allocation the lock lives in).
pub struct ChunkReadGuard {
    guard: RwLockReadGuard<'static, ChunkInner>,
    _chunk: Arc<Chunk>,
}

impl Deref for ChunkReadGuard {
    type Target = ChunkInner;
    fn deref(&self) -> &ChunkInner {
        &self.guard
    }
}

pub struct ChunkWriteGuard {
    guard: RwLockWriteGuard<'static, ChunkInner>,
    _chunk: Arc<Chunk>,
}

impl Deref for ChunkWriteGuard {
    type Target = ChunkInner;
    fn deref(&self) -> &ChunkInner {
        &self.guard
    }
}
impl DerefMut for ChunkWriteGuard {
    fn deref_mut(&mut self) -> &mut ChunkInner {
        &mut self.guard
    }
}

// ----------------------------------------------------------------------
//   Tests
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::blocks::BlockLight;

    // ---------- ChunkData codec ----------

    #[test]
    fn block_mut_writes_value() {
        let mut blocks = ChunkData::default();
        blocks.block_mut(Vec3u::new(2, 3, 4)).id = BlockId::new(1);
        assert_eq!(blocks.block(Vec3u::new(2, 3, 4)).id, BlockId::new(1));
        assert_eq!(blocks.block(Vec3u::new(0, 0, 0)).id, BlockId::default());
    }

    #[test]
    fn package_round_trips_through_unpackage() {
        let mut original = ChunkData::default();
        original.block_mut(Vec3u::new(0, 0, 0)).id = BlockId::new(1);
        original.block_mut(Vec3u::new(15, 15, 15)).id = BlockId::new(2);
        original.block_mut(Vec3u::new(8, 4, 2)).id = BlockId::new(3);
        original.block_mut(Vec3u::new(1, 2, 3)).light = BlockLight::sky_and_block(7, 11);

        let bytes = original.package_to(&[]);
        // After zstd compression a near-uniform chunk is much
        // smaller than the raw body — strict upper bound only.
        assert!(bytes.len() >= ChunkData::HEADER_SIZE);
        assert!(bytes.len() < ChunkData::HEADER_SIZE + ChunkData::DATA_SIZE);

        let mut loaded = ChunkData::default();
        loaded.unpackage_from(&bytes, &[]).expect("unpackage");
        assert_eq!(loaded.block(Vec3u::new(0, 0, 0)).id, BlockId::new(1));
        assert_eq!(loaded.block(Vec3u::new(15, 15, 15)).id, BlockId::new(2));
        assert_eq!(loaded.block(Vec3u::new(8, 4, 2)).id, BlockId::new(3));
        assert_eq!(
            loaded.block(Vec3u::new(1, 2, 3)).light,
            BlockLight::sky_and_block(7, 11)
        );
    }

    #[test]
    fn air_only_chunk_compresses_far_below_raw() {
        // Pure-air chunk: zstd should knock the body down to a
        // handful of bytes. Lock in the win so a future codec
        // regression doesn't silently bloat the database.
        let original = ChunkData::default();
        let bytes = original.package_to(&[]);
        assert!(
            bytes.len() < ChunkData::HEADER_SIZE + 64,
            "expected pure-air chunk to compress to < 64 bytes after header, got {}",
            bytes.len() - ChunkData::HEADER_SIZE,
        );
    }

    #[test]
    fn unpackage_rejects_bad_magic() {
        let mut blocks = ChunkData::default();
        let bytes = vec![0_u8; ChunkData::HEADER_SIZE + ChunkData::DATA_SIZE];
        assert_eq!(blocks.unpackage_from(&bytes, &[]), Err(ChunkError::Magic));
    }

    #[test]
    fn unpackage_rejects_bad_version() {
        let mut blocks = ChunkData::default();
        let mut bytes = vec![0_u8; ChunkData::HEADER_SIZE + ChunkData::DATA_SIZE];
        bytes[0..4].copy_from_slice(&ChunkData::MAGIC.to_le_bytes());
        bytes[4..8].copy_from_slice(&999_u32.to_le_bytes());
        assert_eq!(
            blocks.unpackage_from(&bytes, &[]),
            Err(ChunkError::Version { got: 999 })
        );
    }

    // ---------- Chunk runtime (lease, LSN, eviction state) ----------

    #[test]
    fn from_disk_starts_clean() {
        let c = Chunk::from_disk(ChunkData::default());
        assert_eq!(c.persisted_lsn(), 0);
        assert!(!c.dirty());
    }

    #[test]
    fn from_gen_starts_dirty() {
        let c = Chunk::from_generated(ChunkData::default(), 1);
        assert_eq!(c.persisted_lsn(), 0);
        assert!(c.dirty());
    }

    #[test]
    fn writeback_clears_dirty() {
        let c = Chunk::from_generated(ChunkData::default(), 3);
        assert!(c.dirty());
        c.advance_persisted_lsn(3);
        assert!(!c.dirty());
    }

    #[test]
    fn second_commit_during_writeback_keeps_dirty() {
        // Writeback captured LSN=3, but a later commit bumped
        // commit_lsn to 4 before the disk write completed; persisted
        // advances to 3 only, so still dirty.
        let c = Chunk::from_disk(ChunkData::default());
        c.write_owned().commit_lsn = 3;
        let captured = 3;
        c.write_owned().commit_lsn = 4;
        c.advance_persisted_lsn(captured);
        assert!(c.dirty());
    }

    #[test]
    fn lease_blocks_then_drains() {
        let c = Chunk::from_disk(ChunkData::default());
        let lease = c.try_acquire_lease().expect("acquired");
        assert!(c.start_eviction());
        // start_eviction succeeded; new leases now blocked.
        assert!(c.try_acquire_lease().is_none());
        drop(lease);
        c.wait_drain(); // returns immediately
    }

    #[test]
    fn cant_acquire_lease_in_evicting_state() {
        let c = Chunk::from_disk(ChunkData::default());
        assert!(c.start_eviction());
        assert!(c.try_acquire_lease().is_none());
    }

    #[test]
    fn read_guard_outlives_only_other_arc() {
        // Drop-order check: hold the only outstanding Arc<Chunk> via
        // the guard's keepalive, then access through the guard.
        let c = Chunk::from_disk(ChunkData::default());
        let g = c.read_owned();
        drop(c);
        let _ = g.commit_lsn;
    }

    #[test]
    fn write_excludes_concurrent_readers() {
        let c = Chunk::from_disk(ChunkData::default());
        let mut w = c.write_owned();
        w.commit_lsn = 7;
        // Direct try_read on the lock returns None while writer holds it.
        assert!(c.blocks.try_read().is_none());
        drop(w);
        let r = c.read_owned();
        assert_eq!(r.commit_lsn, 7);
    }

    #[test]
    fn updated_default_false_set_and_clear() {
        let c = Chunk::from_disk(ChunkData::default());
        assert!(!c.updated());
        c.mark_updated();
        assert!(c.updated());
        c.clear_updated();
        assert!(!c.updated());
    }
}
