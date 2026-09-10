//! `World` — the chunk-store database, registry-agnostic.
//!
//! Owns a `DashMap<Vec3i, Arc<Chunk>>` plus a monotonic LSN counter,
//! the per-world canonical id translation tables, and the world's
//! directory + sled store. Knows nothing about the block registry,
//! base blocks, terrain generation, the player, or the game clock —
//! those all live in `core::game::*` modules and consume `World`
//! via its small, number-only API.
//!
//! ## Concurrency model — strict 2PL with lease pinning
//!
//! Each map entry is an [`Arc<Chunk>`](Chunk) carrying the block
//! array under a `RwLock`, an LSN pair, and a lease counter.
//! The **only** way to gain access to a chunk's data is to
//! acquire a [`Lease`] (RAII; pins the chunk against eviction)
//! and then take an owned `RwLock` guard from it. Transactions
//! hold one lease per chunk in their working set.
//!
//! Eviction is a state machine: the world CAS-flips the chunk
//! from `RESIDENT` to `EVICTING` (refusing new leases), waits for
//! outstanding leases to drop, flushes, then removes the entry.
//! See [`chunk`] for the full discipline.
//!
//! Each `WriteTxn::commit` allocates one LSN from
//! [`World::next_lsn`]; that LSN becomes the chunks' `commit_lsn`
//! under their write guards. The pair `persisted_lsn < commit_lsn`
//! drives writeback's "dirty" flag; the LSN ordering itself
//! linearises all committed write txns into a single sequence.
//!
//! **Storage encapsulation.** The sled-backed [`Store`] is a
//! private implementation detail; nothing outside this module can
//! reach it. Disk-load on chunk install, dirty-flush on eviction,
//! the periodic [`World::sweep_dirty`] sweep, and the `save_to_disk`
//! fence all run through `World`.
//!
//! **Save policy.**
//! - [`World::load_chunk`] tries to load the chunk from disk first;
//!   only on miss does it call the caller-supplied generator closure.
//! - [`World::unload_chunk`] starts the eviction state machine:
//!   refuse new leases, drain, flush, remove. If a chunk has
//!   uncommitted changes they are persisted synchronously first; on
//!   I/O failure the chunk is still removed and the error is logged.
//! - [`World::sweep_dirty`] flushes up to a budget of dirty chunks
//!   each call; intended to be called periodically so on-quit save
//!   has little remaining work.
//! - [`World::save_to_disk`] is the durability fence used by autosave
//!   and on-quit.
//!
//! Renderer-side mesh-dirty hooks (`mark_neighbour_chunks_updated`,
//! `drain_updated_chunks`, `clear_updated_chunks`,
//! `mark_all_loaded_for_remesh`) currently live on `World` but are
//! not part of the database interpretation; a future round will move
//! them to a renderer-side dirty-set.

mod chunk;
mod errors;
mod metadata;
mod store;
mod txn;

pub use chunk::{Chunk, ChunkData, Lease};
pub use errors::WorldError;
pub use metadata::Metadata;
pub use store::Store;
pub use txn::{ReadTxn, TxnError, WorkingSet, WriteTxn};

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;

use crate::core::blocks::{BlockData, BlockId};
use crate::core::math::{Vec3i, Vec3u};

// ----------------------------------------------------------------------
//   Coord helpers
// ----------------------------------------------------------------------

/// Floor-divide a block coord by chunk size to get the containing
/// chunk coord.
pub fn chunk_coord(coord: Vec3i) -> Vec3i {
    Vec3i::new(
        coord.x >> Chunk::SIZE_LOG,
        coord.y >> Chunk::SIZE_LOG,
        coord.z >> Chunk::SIZE_LOG,
    )
}

/// Block-local coord (within its chunk) for a global block coord.
pub fn block_coord(coord: Vec3i) -> Vec3u {
    let mask = (Chunk::SIZE - 1) as u32;
    Vec3u::new(
        coord.x as u32 & mask,
        coord.y as u32 & mask,
        coord.z as u32 & mask,
    )
}

// ----------------------------------------------------------------------
//   Construction tables
// ----------------------------------------------------------------------

/// Canonical id translation tables for a world. Built by the
/// registry-aware caller from the world's `world.dat` (or freshly
/// snapshotted from the registry on first run), then handed to
/// [`World::new_at`].
pub struct WorldTables {
    pub metadata: Metadata,
    /// Translation applied when loading a chunk: indexed by canonical
    /// id, gives the in-memory `BlockId`. Empty `Vec` means identity.
    pub load_table: Vec<BlockId>,
    /// Translation applied when saving a chunk: indexed by in-memory
    /// id, gives the canonical `BlockId`. Empty `Vec` means identity.
    pub save_table: Vec<BlockId>,
}

// ----------------------------------------------------------------------
//   World
// ----------------------------------------------------------------------

pub struct World {
    name: String,
    dir: PathBuf,
    store: Store,
    metadata: Metadata,
    chunk_load_table: Vec<BlockId>,
    chunk_save_table: Vec<BlockId>,
    /// Sharded `Vec3i → Arc<Chunk>` map. Lookups clone the
    /// `Arc<Chunk>` and drop the shard guard immediately. The Arc
    /// itself does not pin the chunk against eviction — only a
    /// [`Lease`] does.
    chunks: DashMap<Vec3i, Arc<Chunk>>,
    /// Monotone source of commit ordering. Each `WriteTxn::commit`
    /// (and each [`Chunk::from_gen`]) allocates one. Starts at 1;
    /// LSN 0 is reserved as the "clean from_disk" sentinel in
    /// [`Chunk`].
    next_lsn: Arc<AtomicU64>,
}

impl World {
    /// Create or open the world named `name` rooted at
    /// `<root>/worlds/<name>/`. The caller (registry-aware) provides
    /// pre-built [`WorldTables`] — see
    /// [`crate::core::game::worldgen::world_tables_for`].
    pub fn new_at(root: &Path, name: String, tables: WorldTables) -> Result<Self, WorldError> {
        let dir = root.join("worlds").join(&name);
        std::fs::create_dir_all(&dir)?;
        let store = Store::open_at(&dir.join("chunks.db"))?;
        Ok(Self {
            name,
            dir,
            store,
            metadata: tables.metadata,
            chunk_load_table: tables.load_table,
            chunk_save_table: tables.save_table,
            chunks: DashMap::new(),
            next_lsn: Arc::new(AtomicU64::new(1)),
        })
    }

    /// List the names of every directory under `<root>/worlds/`.
    pub fn list_at(root: &Path) -> Vec<String> {
        let worlds_dir = root.join("worlds");
        let Ok(entries) = std::fs::read_dir(&worlds_dir) else {
            return Vec::new();
        };
        let mut names: Vec<String> = entries
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .filter_map(|e| e.file_name().into_string().ok())
            .collect();
        names.sort();
        names
    }

    /// Recursively delete `<root>/worlds/<name>/`. No-op if the directory
    /// doesn't exist.
    pub fn delete_at(root: &Path, name: &str) -> Result<(), std::io::Error> {
        let dir = root.join("worlds").join(name);
        if !dir.exists() {
            return Ok(());
        }
        std::fs::remove_dir_all(&dir)
    }

    // ---- identity -------------------------------------------------------

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    // ---- chunk presence / iteration -------------------------------------

    pub fn is_loaded(&self, ccoord: Vec3i) -> bool {
        self.chunks
            .get(&ccoord)
            .is_some_and(|r| r.value().is_resident())
    }

    pub fn loaded_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn loaded_coords(&self) -> Vec<Vec3i> {
        self.chunks.iter().map(|r| *r.key()).collect()
    }

    /// Try to acquire a lease on the chunk at `ccoord`. Returns
    /// `None` if no chunk is mapped at that coord, or if the chunk
    /// is in the `EVICTING` state.
    ///
    /// A `Lease` is the canonical pin token in the new design:
    /// while it's alive, [`Self::unload_chunk`] for that coord
    /// will block in its drain step until the lease is dropped.
    /// Long-lived pin holders (e.g. the streaming range loader)
    /// store leases instead of `Arc<Chunk>` clones.
    pub fn try_acquire_lease(&self, ccoord: Vec3i) -> Option<Lease> {
        let chunk = self.chunks.get(&ccoord)?;
        chunk.value().try_acquire_lease()
    }

    /// Allocate the next commit LSN. Called by [`WriteTxn::commit`]
    /// and by [`Self::load_chunk`] for freshly-generated terrain.
    pub(super) fn next_lsn(&self) -> u64 {
        self.next_lsn.fetch_add(1, Ordering::AcqRel)
    }

    /// Hand out a clone of the LSN counter Arc — `WriteTxn` carries
    /// one to allocate an LSN at commit time without holding a
    /// reference back to the world.
    pub(super) fn lsn_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.next_lsn)
    }

    // ---- transactional read/write entry points --------------------------

    pub fn begin_read_txn_sync(
        &self,
        working_set: impl Into<WorkingSet>,
    ) -> Result<txn::ReadTxn, TxnError> {
        txn::begin_read_sync(self, working_set.into())
    }

    pub fn begin_write_txn_sync(
        &self,
        working_set: impl Into<WorkingSet>,
    ) -> Result<txn::WriteTxn, TxnError> {
        txn::begin_write_sync(self, working_set.into())
    }

    // ---- single-coord block lookup --------------------------------------

    /// Single-coord block lookup. Routes through a 1-coord `ReadTxn`.
    /// **Inefficient on hot paths** — one DashMap lookup, lease
    /// acquire, and lock acquire per call. For multi-cell reads
    /// against the same chunk(s), open one `ReadTxn` and reuse it.
    pub fn block(&self, coord: Vec3i) -> Option<BlockData> {
        let cc = chunk_coord(coord);
        let txn = self.begin_read_txn_sync(WorkingSet::Single(cc)).ok()?;
        txn.read(coord).ok()
    }

    // ---- chunk load / unload --------------------------------------------

    /// Loads the chunk at `ccoord`. World tries its on-disk store
    /// first; on miss, calls `init` to produce a fresh chunk.
    /// The closure is consumed only on disk-miss, so callers can
    /// hand in pre-generated blocks (e.g. from an async worker)
    /// without wasting them on hits.
    ///
    /// No-op if a resident chunk already exists at `ccoord`. If the
    /// chunk is mid-eviction, spins until the evictor finishes and
    /// then re-tries the install.
    pub fn load_chunk<F>(&self, ccoord: Vec3i, init: F)
    where
        F: FnOnce() -> ChunkData,
    {
        // The init closure can only be moved into one branch, so we
        // wrap it in Option to satisfy the borrow checker across loop
        // iterations.
        let mut init = Some(init);
        loop {
            match self.chunks.entry(ccoord) {
                Entry::Occupied(o) => {
                    let s = Arc::clone(o.get());
                    drop(o);
                    if s.is_resident() {
                        return;
                    }
                    // EVICTING — wait for the evictor to finish, then retry.
                    std::thread::yield_now();
                    continue;
                }
                Entry::Vacant(vacant) => {
                    let new_chunk = match self.try_load_from_disk(ccoord) {
                        Ok(Some(data)) => Chunk::from_disk(data),
                        Ok(None) => {
                            let f = init.take().expect("init only consumed once");
                            Chunk::from_generated(f(), self.next_lsn())
                        }
                        Err(err) => {
                            tracing::warn!(?ccoord, error = %err, "chunk disk load failed; regenerating");
                            let f = init.take().expect("init only consumed once");
                            Chunk::from_generated(f(), self.next_lsn())
                        }
                    };
                    vacant.insert(new_chunk);
                    return;
                }
            }
        }
    }

    /// Evict the chunk at `ccoord`. Runs the full eviction
    /// state-machine: CAS to `EVICTING`, drain outstanding leases,
    /// flush if dirty, remove from the map. No-op if no chunk is
    /// mapped or if another thread is already evicting this chunk.
    pub fn unload_chunk(&self, ccoord: Vec3i) {
        let Some(chunk) = self.chunks.get(&ccoord).map(|r| Arc::clone(r.value())) else {
            return;
        };
        if !chunk.start_eviction() {
            return;
        }
        chunk.wait_drain();
        if let Err(err) = self.flush_chunk(ccoord, &chunk) {
            tracing::warn!(?ccoord, error = %err, "chunk save on unloading failed");
        }
        self.chunks.remove(&ccoord);
    }

    // ---- save / housekeeping --------------------------------------------

    /// Flush at most `budget` dirty chunks to disk. Returns the
    /// number actually written. Intended for periodic background
    /// calls so [`Self::save_to_disk`] on quit has little to do.
    /// Errors during individual flushes are logged and swallowed —
    /// the sweep keeps going and surfaces nothing to the caller.
    pub fn sweep_dirty(&self, budget: usize) -> Result<usize, WorldError> {
        if budget == 0 {
            return Ok(0);
        }
        // Collect Arc clones (so we don't hold shard guards across
        // the per-chunk flush, which itself takes the chunk's lock).
        let mut dirty: Vec<(Vec3i, Arc<Chunk>)> = Vec::with_capacity(budget);
        for r in self.chunks.iter() {
            if dirty.len() >= budget {
                break;
            }
            if r.value().dirty() {
                dirty.push((*r.key(), Arc::clone(r.value())));
            }
        }
        let mut written = 0;
        for (cc, chunk) in dirty {
            match self.flush_chunk(cc, &chunk) {
                Ok(()) => written += 1,
                Err(err) => tracing::warn!(?cc, error = %err, "sweep flush failed"),
            }
        }
        self.store.flush()?;
        Ok(written)
    }

    /// Persist every dirty chunk to disk and write `world.dat`. Does
    /// not save the player — that's the caller's responsibility (the
    /// player lives outside `World`).
    pub fn save_to_disk(&self) -> Result<(), WorldError> {
        let snapshot: Vec<(Vec3i, Arc<Chunk>)> = self
            .chunks
            .iter()
            .map(|r| (*r.key(), Arc::clone(r.value())))
            .collect();
        for (cc, chunk) in snapshot {
            self.flush_chunk(cc, &chunk)?;
        }
        self.store.flush()?;
        self.metadata.save_to(&self.dir.join("world.dat"))?;
        Ok(())
    }

    // ---- private helpers ------------------------------------------------

    /// Try to deserialize the chunk at `coord` from sled. `Ok(None)`
    /// means there is no on-disk copy.
    fn try_load_from_disk(&self, coord: Vec3i) -> Result<Option<ChunkData>, WorldError> {
        let Some(bytes) = self.store.load(coord)? else {
            return Ok(None);
        };
        let mut blocks = ChunkData::default();
        match blocks.unpackage_from(&bytes, &self.chunk_load_table) {
            Ok(()) => Ok(Some(blocks)),
            Err(err) => {
                tracing::warn!(?coord, error = %err, "chunk unpackage failed; regenerating");
                Ok(None)
            }
        }
    }

    /// Snapshot the chunk at `ccoord`, write it to sled if dirty,
    /// and advance its `persisted_lsn`. Captures the LSN under the
    /// read lock — fixes the order-inversion in the old
    /// gen-captured-before-lock flush.
    fn flush_chunk(&self, ccoord: Vec3i, chunk: &Arc<Chunk>) -> Result<(), WorldError> {
        let bytes;
        let captured_lsn;
        {
            let guard = chunk.read_owned();
            if guard.commit_lsn <= chunk.persisted_lsn() {
                return Ok(());
            }
            captured_lsn = guard.commit_lsn;
            tracing::info!(?ccoord, lsn = captured_lsn, "flushing dirty chunk");
            bytes = guard.data.package_to(&self.chunk_save_table);
        }
        self.store.save(ccoord, &bytes)?;
        chunk.advance_persisted_lsn(captured_lsn);
        Ok(())
    }

    // ---- mesh-dirty tracking (TODO: factor out to render) ---------------

    /// Renderer hook: snapshot of coords whose mesh-dirty atomic is set.
    pub fn drain_updated_chunks(&self) -> Vec<Vec3i> {
        self.chunks
            .iter()
            .filter(|r| r.value().updated())
            .map(|r| *r.key())
            .collect()
    }

    /// Renderer hook: clear the mesh-dirty atomic on `coords` (called
    /// by the renderer after it dispatches a remesh for each).
    pub fn clear_updated_chunks(&self, coords: &[Vec3i]) {
        for &cc in coords {
            if let Some(s) = self.chunks.get(&cc) {
                s.value().clear_updated();
            }
        }
    }

    /// Renderer hook: mark every loaded chunk's mesh-dirty atomic.
    /// Used to force a full re-mesh after meshing rules flip.
    pub fn mark_all_loaded_for_remesh(&self) {
        for r in self.chunks.iter() {
            r.value().mark_updated();
        }
    }

    /// Renderer hook: mark every chunk in the 3×3×3 cube around
    /// `ccoord` as needing a re-mesh. Called after a chunk lands so
    /// neighbouring chunks re-mesh against the real blocks.
    pub fn mark_neighbour_chunks_updated(&self, ccoord: Vec3i) {
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let target = ccoord + Vec3i::new(dx, dy, dz);
                    if let Some(s) = self.chunks.get(&target) {
                        s.value().mark_updated();
                    }
                }
            }
        }
    }
}
