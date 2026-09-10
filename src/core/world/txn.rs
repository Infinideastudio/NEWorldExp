//! `WorkingSet`, `ReadTxn`, `WriteTxn` — transactional access to
//! the chunk store.
//!
//! All transactions declare a *working set* (one chunk, an
//! explicit list, or an axis-aligned chunk-coord box). At begin
//! time the txn:
//!
//! 1. Sorts and dedups the working-set coords (lex order — gives
//!    deadlock-free per-chunk lock acquisition).
//! 2. Acquires a [`Lease`] on every coord. If any chunk is missing
//!    from the world map or is in the `EVICTING` state, the begin
//!    fails with [`TxnError::NotLoaded`] and any leases collected
//!    so far drop (RAII).
//! 3. Acquires the per-chunk read or write guard via the chunk's
//!    owned-guard API; the guard pairs the parking_lot guard with
//!    the chunk Arc, so the guard outlives any subsequent eviction.
//!
//! A `WriteTxn` buffers writes locally and applies them at commit
//! time under the held write guards. The commit point is a single
//! LSN allocated from `World`'s monotonic counter, written into
//! each touched chunk's `commit_lsn` under its write guard — but
//! only chunks whose *content* (`(id, state)`, light aside) the
//! commit actually changed are stamped; pure light writes leave
//! the chunk clean. Commit point = LSN allocation; visibility
//! point = guard release. The linearised order over all commits is
//! exactly the LSN order.
//!
//! Reads and writes outside the declared set return
//! [`TxnError::OutOfRange`].

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::blocks::BlockData;
use crate::core::math::{Vec3i, Vec3u};

use super::chunk::{ChunkData, ChunkReadGuard, ChunkWriteGuard, Lease};
use super::{World, block_coord, chunk_coord};

// ----------------------------------------------------------------------
//   WorkingSet
// ----------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum WorkingSet {
    Single(Vec3i),
    List(Vec<Vec3i>),
    /// Inclusive lower, exclusive upper bounds in chunk-coord space.
    Range(Vec3i, Vec3i),
}

impl WorkingSet {
    fn collect_sorted(&self) -> Vec<Vec3i> {
        let mut out: Vec<Vec3i> = match self {
            Self::Single(c) => vec![*c],
            Self::List(v) => v.clone(),
            Self::Range(lo, hi) => {
                let mut v = Vec::new();
                for x in lo.x..hi.x {
                    for y in lo.y..hi.y {
                        for z in lo.z..hi.z {
                            v.push(Vec3i::new(x, y, z));
                        }
                    }
                }
                v
            }
        };
        out.sort_by_key(|c| (c.x, c.y, c.z));
        out.dedup();
        out
    }
}

impl From<Vec3i> for WorkingSet {
    fn from(c: Vec3i) -> Self {
        Self::Single(c)
    }
}
impl From<Vec<Vec3i>> for WorkingSet {
    fn from(v: Vec<Vec3i>) -> Self {
        Self::List(v)
    }
}
impl From<&[Vec3i]> for WorkingSet {
    fn from(s: &[Vec3i]) -> Self {
        Self::List(s.to_vec())
    }
}

// ----------------------------------------------------------------------
//   Errors
// ----------------------------------------------------------------------

/// Single error type covering both txn-begin and txn-access failures.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TxnError {
    /// Begin: one or more working-set coords aren't currently in
    /// memory (chunk vacant, or chunk in the `EVICTING` state). Caller
    /// should warm the cache (sync prefetch or async load) and retry.
    #[error("chunks not loaded: {0:?}")]
    NotLoaded(Vec<Vec3i>),
    /// Access: caller asked for a block whose chunk isn't in the txn's
    /// declared working set.
    #[error("coord {coord:?} is out of the txn's working set")]
    OutOfRange { coord: Vec3i },
}

// ----------------------------------------------------------------------
//   ReadTxn
// ----------------------------------------------------------------------

/// One pinned chunk held by a `ReadTxn`. The `_lease` blocks
/// eviction completion; the owned `guard` keeps the chunk Arc
/// alive independently. Field order means `guard` drops first,
/// then `_lease` — both before the WriteTxn is dropped.
struct ReadEntry {
    coord: Vec3i,
    guard: ChunkReadGuard,
    _lease: Lease,
}

pub struct ReadTxn {
    entries: Vec<ReadEntry>,
}

impl ReadTxn {
    fn entry(&self, cc: Vec3i) -> Option<&ReadEntry> {
        self.entries.iter().find(|e| e.coord == cc)
    }

    /// Read one block. Errors `OutOfRange` if the containing chunk
    /// isn't in this txn's working set.
    pub fn read(&self, coord: Vec3i) -> Result<BlockData, TxnError> {
        let cc = chunk_coord(coord);
        let entry = self.entry(cc).ok_or(TxnError::OutOfRange { coord })?;
        Ok(entry.guard.data.block(block_coord(coord)))
    }

    /// Borrow the held cell array at `ccoord`. Returns `None` if the
    /// coord isn't in this txn's working set. Borrowed for the
    /// lifetime of `&self`.
    pub fn chunk_at(&self, ccoord: Vec3i) -> Option<&ChunkData> {
        self.entry(ccoord).map(|e| &e.guard.data)
    }
}

// ----------------------------------------------------------------------
//   WriteTxn
// ----------------------------------------------------------------------

struct WriteEntry {
    coord: Vec3i,
    guard: ChunkWriteGuard,
    _lease: Lease,
}

pub struct WriteTxn {
    entries: Vec<WriteEntry>,
    buffered: HashMap<Vec3i, Vec<(Vec3u, BlockData)>>,
    /// Cloned from `World::next_lsn` at begin time. Decouples the
    /// txn type from a `&World` lifetime; `commit` allocates one
    /// LSN here, regardless of whether the originating world is
    /// still around (it always is, in practice).
    lsn_counter: Arc<AtomicU64>,
}

impl WriteTxn {
    fn entry(&self, cc: Vec3i) -> Option<&WriteEntry> {
        self.entries.iter().find(|e| e.coord == cc)
    }

    fn entry_mut(&mut self, cc: Vec3i) -> Option<&mut WriteEntry> {
        self.entries.iter_mut().find(|e| e.coord == cc)
    }

    /// Read one block. Sees committed state plus this txn's buffered
    /// writes (last-write-wins).
    pub fn read(&self, coord: Vec3i) -> Result<BlockData, TxnError> {
        let cc = chunk_coord(coord);
        let entry = self.entry(cc).ok_or(TxnError::OutOfRange { coord })?;
        let bcoord = block_coord(coord);
        if let Some(buf) = self.buffered.get(&cc)
            && let Some((_, data)) = buf.iter().rev().find(|(b, _)| *b == bcoord)
        {
            return Ok(*data);
        }
        Ok(entry.guard.data.block(bcoord))
    }

    /// Buffer one write.
    pub fn write(&mut self, coord: Vec3i, data: BlockData) -> Result<(), TxnError> {
        let cc = chunk_coord(coord);
        if self.entry(cc).is_none() {
            return Err(TxnError::OutOfRange { coord });
        }
        self.buffered
            .entry(cc)
            .or_default()
            .push((block_coord(coord), data));
        Ok(())
    }

    /// Apply all buffered writes in place under the held write
    /// guards. Allocates a single LSN per non-empty commit and
    /// stamps it as `commit_lsn` on every touched chunk **whose
    /// content the commit actually changed** — content is a cell's
    /// `(id, state)`, compared last-write-wins against the
    /// pre-commit value. Pure light writes still land (the renderer
    /// samples in-memory light) but leave the chunk clean, so light
    /// relaxation alone never sends a chunk to disk.
    ///
    /// Stamping happens under each held write guard, so any reader
    /// who later acquires the chunk's lock sees a consistent
    /// (data, commit_lsn) pair.
    ///
    /// On return, the entries' write guards drop first (commit
    /// point — visible to subsequent txns), then the leases drop
    /// (chunks become eligible for eviction).
    pub fn commit(mut self) {
        let drained: Vec<(Vec3i, Vec<(Vec3u, BlockData)>)> = self.buffered.drain().collect();
        if drained.is_empty() {
            return;
        }
        let lsn = self.lsn_counter.fetch_add(1, Ordering::AcqRel);
        for (cc, writes) in drained {
            let Some(entry) = self.entry_mut(cc) else {
                debug_assert!(false, "buffered write for coord without entry");
                continue;
            };
            // Last-write-wins final value per buffered cell, gathered
            // BEFORE anything is applied: `guard.data` still holds the
            // pre-commit state here.
            let mut finals: HashMap<Vec3u, BlockData> = HashMap::with_capacity(writes.len());
            for (bcoord, data) in &writes {
                finals.insert(*bcoord, *data);
            }
            // Content = (id, state); light is runtime-only. Stamp the
            // chunk iff its content actually changed — a light-only
            // commit updates memory but leaves the chunk clean, and a
            // content write reverted within the txn is a no-op.
            let content_changed = finals.iter().any(|(bcoord, data)| {
                let orig = entry.guard.data.block(*bcoord);
                orig.id != data.id || orig.state != data.state
            });
            for (bcoord, data) in writes {
                *entry.guard.data.block_mut(bcoord) = data;
            }
            if content_changed {
                entry.guard.commit_lsn = lsn;
            }
        }
    }
}

// ----------------------------------------------------------------------
//   Begin transactions (sync)
// ----------------------------------------------------------------------

pub(super) fn begin_read_sync(world: &World, set: WorkingSet) -> Result<ReadTxn, TxnError> {
    let coords = set.collect_sorted();
    let leases = acquire_leases(world, &coords)?;
    // Acquire read guards in lex order — `coords` is sorted, and
    // `leases` zips with it.
    let entries: Vec<ReadEntry> = coords
        .into_iter()
        .zip(leases)
        .map(|(coord, lease)| {
            let guard = lease.chunk().read_owned();
            ReadEntry {
                coord,
                guard,
                _lease: lease,
            }
        })
        .collect();
    Ok(ReadTxn { entries })
}

pub(super) fn begin_write_sync(world: &World, set: WorkingSet) -> Result<WriteTxn, TxnError> {
    let coords = set.collect_sorted();
    let leases = acquire_leases(world, &coords)?;
    let entries: Vec<WriteEntry> = coords
        .into_iter()
        .zip(leases)
        .map(|(coord, lease)| {
            let guard = lease.chunk().write_owned();
            WriteEntry {
                coord,
                guard,
                _lease: lease,
            }
        })
        .collect();
    Ok(WriteTxn {
        entries,
        buffered: HashMap::new(),
        lsn_counter: world.lsn_counter(),
    })
}

/// Try to take a lease on every coord in `coords`. On any failure,
/// return `NotLoaded(missing)` listing every coord that wasn't
/// available — successfully-acquired leases drop as the local
/// `Vec<Lease>` goes out of scope.
fn acquire_leases(world: &World, coords: &[Vec3i]) -> Result<Vec<Lease>, TxnError> {
    let mut leases = Vec::with_capacity(coords.len());
    let mut missing = Vec::new();
    for &c in coords {
        match world.try_acquire_lease(c) {
            Some(lease) => leases.push(lease),
            None => missing.push(c),
        }
    }
    if !missing.is_empty() {
        // Drop any leases acquired so far via Vec drop. The remaining
        // RAII collapse is automatic.
        return Err(TxnError::NotLoaded(missing));
    }
    Ok(leases)
}

// ----------------------------------------------------------------------
//   Tests
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::blocks::{BlockId, BlockLight, BlockState};
    use crate::core::world::{Metadata, WorldTables};

    /// Tiny RAII scratch dir — `tempfile` isn't a dev-dep on this
    /// crate, so we roll our own minimal version (matches the
    /// pattern used in `globalization.rs` tests).
    struct ScratchDir {
        path: std::path::PathBuf,
    }

    impl ScratchDir {
        fn new() -> Self {
            // Unique enough for the test suite: nanos + addr-of-stack.
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let salt = &nanos as *const _ as usize;
            let path = std::env::temp_dir().join(format!("neworld-txn-{nanos:x}-{salt:x}"));
            std::fs::create_dir_all(&path).expect("create scratch");
            Self { path }
        }
        fn path(&self) -> &std::path::Path {
            &self.path
        }
    }
    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn temp_world() -> (ScratchDir, World) {
        let dir = ScratchDir::new();
        let world = World::new_at(
            dir.path(),
            "test".to_string(),
            WorldTables {
                metadata: Metadata {
                    block_mapping: Vec::new(),
                    seed: 0,
                    seed_needs_migration: false,
                },
                load_table: Vec::new(),
                save_table: Vec::new(),
            },
        )
        .expect("new_at");
        (dir, world)
    }

    fn install_chunks(world: &World, coords: &[Vec3i]) {
        for &c in coords {
            world.load_chunk(c, ChunkData::default);
        }
    }

    fn chunk_dirty(world: &World, cc: Vec3i) -> bool {
        world.chunks.get(&cc).expect("chunk loaded").value().dirty()
    }

    fn chunk_commit_lsn(world: &World, cc: Vec3i) -> u64 {
        world
            .chunks
            .get(&cc)
            .expect("chunk loaded")
            .value()
            .read_owned()
            .commit_lsn
    }

    #[test]
    fn working_set_collect_sorted_dedups_and_orders() {
        let s = WorkingSet::List(vec![
            Vec3i::new(2, 0, 0),
            Vec3i::new(0, 0, 0),
            Vec3i::new(0, 0, 0),
            Vec3i::new(1, 0, 0),
        ]);
        assert_eq!(
            s.collect_sorted(),
            vec![
                Vec3i::new(0, 0, 0),
                Vec3i::new(1, 0, 0),
                Vec3i::new(2, 0, 0),
            ]
        );
    }

    #[test]
    fn working_set_aabb_iterates_inclusive_lo_exclusive_hi() {
        let s = WorkingSet::Range(Vec3i::new(0, 0, 0), Vec3i::new(2, 2, 1));
        let coords = s.collect_sorted();
        assert_eq!(coords.len(), 4);
        assert!(coords.contains(&Vec3i::new(0, 0, 0)));
        assert!(!coords.contains(&Vec3i::new(2, 0, 0)));
    }

    #[test]
    fn begin_read_sync_errors_when_coord_not_loaded() {
        let (_dir, world) = temp_world();
        match world.begin_read_txn_sync(WorkingSet::Single(Vec3i::new(0, 0, 0))) {
            Err(TxnError::NotLoaded(v)) => assert_eq!(v, vec![Vec3i::new(0, 0, 0)]),
            Err(other) => panic!("expected NotLoaded; got {other:?}"),
            Ok(_) => panic!("expected NotLoaded; got Ok"),
        }
    }

    #[test]
    fn read_txn_returns_air_for_empty_chunk_in_set() {
        let (_dir, world) = temp_world();
        let cc = Vec3i::new(0, 1, 0);
        install_chunks(&world, &[cc]);
        let txn = world
            .begin_read_txn_sync(WorkingSet::Single(cc))
            .expect("loaded");
        let b = txn.read(Vec3i::new(0, 16, 0)).expect("in range");
        assert_eq!(b.id, BlockId::default());
    }

    #[test]
    fn read_txn_out_of_range_returns_error() {
        let (_dir, world) = temp_world();
        let cc = Vec3i::new(0, 0, 0);
        install_chunks(&world, &[cc]);
        let txn = world
            .begin_read_txn_sync(WorkingSet::Single(cc))
            .expect("loaded");
        let err = txn.read(Vec3i::new(16, 0, 0)).expect_err("out of range");
        assert_eq!(
            err,
            TxnError::OutOfRange {
                coord: Vec3i::new(16, 0, 0)
            }
        );
    }

    #[test]
    fn write_txn_commit_makes_value_visible_to_subsequent_read_txn() {
        let (_dir, world) = temp_world();
        let cc = Vec3i::new(0, 0, 0);
        install_chunks(&world, &[cc]);

        let mut wtxn = world
            .begin_write_txn_sync(WorkingSet::Single(cc))
            .expect("loaded");
        let target = BlockData {
            id: BlockId::new(1),
            ..BlockData::default()
        };
        wtxn.write(Vec3i::new(3, 4, 5), target).expect("in range");
        wtxn.commit();

        let rtxn = world
            .begin_read_txn_sync(WorkingSet::Single(cc))
            .expect("loaded");
        let b = rtxn.read(Vec3i::new(3, 4, 5)).expect("in range");
        assert_eq!(b.id, BlockId::new(1));
    }

    #[test]
    fn write_txn_drop_without_commit_rolls_back() {
        let (_dir, world) = temp_world();
        let cc = Vec3i::new(0, 0, 0);
        install_chunks(&world, &[cc]);
        {
            let mut wtxn = world
                .begin_write_txn_sync(WorkingSet::Single(cc))
                .expect("loaded");
            let target = BlockData {
                id: BlockId::new(1),
                ..BlockData::default()
            };
            wtxn.write(Vec3i::new(3, 4, 5), target).expect("in range");
        }
        let rtxn = world
            .begin_read_txn_sync(WorkingSet::Single(cc))
            .expect("loaded");
        let b = rtxn.read(Vec3i::new(3, 4, 5)).expect("in range");
        assert_eq!(b.id, BlockId::default());
    }

    #[test]
    fn write_txn_self_read_sees_buffered_writes_before_commit() {
        let (_dir, world) = temp_world();
        let cc = Vec3i::new(0, 0, 0);
        install_chunks(&world, &[cc]);
        let mut wtxn = world
            .begin_write_txn_sync(WorkingSet::Single(cc))
            .expect("loaded");
        let target = BlockData {
            id: BlockId::new(1),
            ..BlockData::default()
        };
        let coord = Vec3i::new(1, 1, 1);
        wtxn.write(coord, target).expect("in range");
        let b = wtxn.read(coord).expect("in range");
        assert_eq!(b.id, BlockId::new(1));
    }

    #[test]
    fn light_only_commit_updates_memory_but_keeps_chunk_clean() {
        let (_dir, world) = temp_world();
        let cc = Vec3i::new(0, 0, 0);
        install_chunks(&world, &[cc]);

        let mut wtxn = world
            .begin_write_txn_sync(WorkingSet::Single(cc))
            .expect("loaded");
        wtxn.write(
            Vec3i::new(1, 1, 1),
            BlockData {
                light: BlockLight::sky_and_block(15, 0),
                ..BlockData::default()
            },
        )
        .expect("in range");
        wtxn.commit();

        assert!(!chunk_dirty(&world, cc));
        assert_eq!(chunk_commit_lsn(&world, cc), 0);
        // The light write itself still landed.
        let rtxn = world
            .begin_read_txn_sync(WorkingSet::Single(cc))
            .expect("loaded");
        assert_eq!(
            rtxn.read(Vec3i::new(1, 1, 1)).expect("in range").light,
            BlockLight::sky_and_block(15, 0)
        );
    }

    #[test]
    fn content_commit_marks_chunk_dirty() {
        let (_dir, world) = temp_world();
        let cc = Vec3i::new(0, 0, 0);
        install_chunks(&world, &[cc]);

        let mut wtxn = world
            .begin_write_txn_sync(WorkingSet::Single(cc))
            .expect("loaded");
        wtxn.write(
            Vec3i::new(2, 2, 2),
            BlockData {
                id: BlockId::new(5),
                ..BlockData::default()
            },
        )
        .expect("in range");
        wtxn.commit();

        assert!(chunk_dirty(&world, cc));
        assert!(chunk_commit_lsn(&world, cc) > 0);
    }

    #[test]
    fn state_only_commit_marks_chunk_dirty() {
        let (_dir, world) = temp_world();
        let cc = Vec3i::new(0, 0, 0);
        install_chunks(&world, &[cc]);

        let mut wtxn = world
            .begin_write_txn_sync(WorkingSet::Single(cc))
            .expect("loaded");
        wtxn.write(
            Vec3i::new(2, 2, 2),
            BlockData {
                state: BlockState::inline(3),
                ..BlockData::default()
            },
        )
        .expect("in range");
        wtxn.commit();

        assert!(chunk_dirty(&world, cc));
    }

    #[test]
    fn content_reverted_within_txn_keeps_chunk_clean() {
        let (_dir, world) = temp_world();
        let cc = Vec3i::new(0, 0, 0);
        install_chunks(&world, &[cc]);

        // Same cell: change content, then restore the original
        // BlockData — last-write-wins equals the pre-commit value.
        let mut wtxn = world
            .begin_write_txn_sync(WorkingSet::Single(cc))
            .expect("loaded");
        wtxn.write(
            Vec3i::new(3, 3, 3),
            BlockData {
                id: BlockId::new(9),
                ..BlockData::default()
            },
        )
        .expect("in range");
        wtxn.write(Vec3i::new(3, 3, 3), BlockData::default())
            .expect("in range");
        wtxn.commit();

        assert!(!chunk_dirty(&world, cc));
    }

    #[test]
    fn mixed_light_and_content_commit_is_dirty_and_applies_light() {
        let (_dir, world) = temp_world();
        let cc = Vec3i::new(0, 0, 0);
        install_chunks(&world, &[cc]);

        let mut wtxn = world
            .begin_write_txn_sync(WorkingSet::Single(cc))
            .expect("loaded");
        wtxn.write(
            Vec3i::new(4, 4, 4),
            BlockData {
                light: BlockLight::sky_and_block(9, 3),
                ..BlockData::default()
            },
        )
        .expect("in range");
        wtxn.write(
            Vec3i::new(5, 5, 5),
            BlockData {
                id: BlockId::new(2),
                ..BlockData::default()
            },
        )
        .expect("in range");
        wtxn.commit();

        assert!(chunk_dirty(&world, cc));
        let rtxn = world
            .begin_read_txn_sync(WorkingSet::Single(cc))
            .expect("loaded");
        assert_eq!(
            rtxn.read(Vec3i::new(4, 4, 4)).expect("in range").light,
            BlockLight::sky_and_block(9, 3)
        );
    }

    #[test]
    fn multi_chunk_txn_stamps_only_content_changed_chunks() {
        // One txn over two chunks: content change in `a`, light-only
        // in `b`. `a` stamps, `b` stays clean.
        let (_dir, world) = temp_world();
        let a = Vec3i::new(0, 0, 0);
        let b = Vec3i::new(1, 0, 0);
        install_chunks(&world, &[a, b]);

        let mut wtxn = world
            .begin_write_txn_sync(WorkingSet::List(vec![a, b]))
            .expect("loaded");
        wtxn.write(
            Vec3i::new(0, 0, 0),
            BlockData {
                id: BlockId::new(4),
                ..BlockData::default()
            },
        )
        .expect("in range");
        wtxn.write(
            Vec3i::new(16, 0, 0), // block-local (0,0,0) of chunk b
            BlockData {
                light: BlockLight::sky_and_block(3, 5),
                ..BlockData::default()
            },
        )
        .expect("in range");
        wtxn.commit();

        assert!(chunk_dirty(&world, a));
        assert!(!chunk_dirty(&world, b));
    }

    #[test]
    fn lease_held_by_txn_blocks_eviction_until_drop() {
        // The orphan-arc bug from the old design: `unload_chunk`
        // could drop a chunk while a txn was still using it. With
        // the new lease discipline, eviction must wait.
        let (_dir, world) = temp_world();
        let cc = Vec3i::new(0, 0, 0);
        install_chunks(&world, &[cc]);

        let mut wtxn = world
            .begin_write_txn_sync(WorkingSet::Single(cc))
            .expect("loaded");
        wtxn.write(
            Vec3i::new(1, 1, 1),
            BlockData {
                id: BlockId::new(7),
                ..BlockData::default()
            },
        )
        .expect("in range");

        // Try to evict on a background thread. The eviction CAS
        // succeeds and bounces new leases, but `wait_drain` blocks
        // on the lease this txn holds.
        let world2 = std::sync::Arc::new(world);
        let w2 = std::sync::Arc::clone(&world2);
        let evictor = std::thread::spawn(move || {
            w2.unload_chunk(cc);
        });

        // Evictor is blocked draining; meanwhile we commit safely.
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(
            !evictor.is_finished(),
            "evictor must not finish while WriteTxn lease is alive"
        );
        wtxn.commit();
        // After commit, the WriteTxn (and its lease) is dropped;
        // evictor can now drain and proceed.
        evictor.join().expect("evictor join");
        // Chunk has been evicted and removed from the map.
        assert!(!world2.is_loaded(cc));
    }
}
