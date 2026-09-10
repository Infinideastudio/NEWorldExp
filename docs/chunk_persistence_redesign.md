# 区块持久化重设计：仅内容变更触发落盘

@JelawatIHPC

解决"区块刚生成就被标记为 Dirty"问题的实施方案。核心思路：把区块数据拆成**内容**（id + state，决定 dirty）与**光照**（运行时字段，其变更不构成区块变更）两个层面——`WriteTxn` 提交时只在实际改变 id/state 时才盖 LSN，生成区块出生即为 clean。由此磁盘存档只包含玩家真正编辑过的区块，其余区块依据存档元数据中的种子重新生成。**光照数据本期仍随区块持久化，磁盘格式不变。**

本文档为方案规约与实施计划。

---

## 0. 分支与工作流

1. 拉取远程仓库同步（`git pull`）。
2. 从 `main` 创建新分支 `refactor/dirty-mechanism`。
3. 在新分支上完成开发工作。

---

## 1. 背景与问题

当前实现中有两个行为叠加，导致纯生成区块也会进入存档：

### 1.1 生成即 Dirty

`Chunk::from_generated`（`src/core/world/chunk.rs:276`）以非零 LSN 构造：`commit_lsn = next_lsn()`、`persisted_lsn = 0`，于是刚生成的区块立即满足 `persisted_lsn < commit_lsn` 即 dirty。`World::load_chunk`（`src/core/world/mod.rs:280`）在磁盘未命中时走这条路（`mod.rs:305`、`mod.rs:310`）。结果是每个新探索的区块都会在首次卸载时写盘。

### 1.2 光照松弛写入也会 bump commit_lsn

`WriteTxn::commit`（`src/core/world/txn.rs:211`）对所有触及的区块无条件盖 LSN，不区分写入究竟改变了什么。光照松弛路径 `process_one_in_txn`（`src/core/game/block_update.rs:239`）虽然只在 `new_data != curr` 时写入，但其差异**只来自 light 字段**（id/state 原样保留，见 `block_update.rs:261-268`）。所以纯光照更新同样把区块打成 dirty——这在修复 1.1 之后仍会让"被玩家编辑波及的邻区块"不必要地落盘。

### 1.3 合成后果

存档 `chunks.db` 被从未被玩家触碰的区块填满，存档体积近似等于探索面积而非编辑面积。磁盘格式本身（每格 5 字节 `id + state + light`，`ChunkData::VERSION = 3`）没有问题，本期不动。

---

## 2. 目标与非目标

**目标**

1. 光照更新不视为区块变更：只有当某次提交真正改变了区块内方块的 **id 或 state** 时，该区块才变 dirty。
2. 磁盘存档只包含玩家放置/破坏过方块的区块；不在存档中的区块依据 `world.dat` 元数据里的种子重新生成。
3. 磁盘格式零变化：`WORLD_META_VERSION = 2`、`ChunkData::VERSION = 3`、5 字节 cell 编码（含 light）全部维持原样。

**非目标**

- 光照去持久化与加载后重建——本期不做（留作后续方案）。
- 不改动 mesh-dirty（`Chunk::updated` 标记与 `drain_updated_chunks`）机制。
- 不做"编辑后还原到原始状态即从存档剔除"的优化（见 §8）。

---

## 3. 核心不变式

重设计后整个系统建立在三条不变式上：

| # | 不变式 | 含义 |
|---|--------|------|
| I1 | **dirty 由内容决定** | 内容 = (id, state)。`commit_lsn` 只在内容实际变化时盖章。光照是运行时字段：它的变更会被应用、会驱动渲染，但不构成区块变更。 |
| I2 | **再生不变式** | 未 dirty 的生成区块内容恒等于 `f(seed, chunk_coord, 世界生成规则, block_mapping)`。丢弃它们是安全的——重载时从种子重建出逐位相同的内容（含 `init_generate` 烘焙的光照）。 |
| I3 | **存档集合 = 有过内容提交的区块** | 一个区块进入 `chunks.db` 当且仅当它发生过至少一次内容（id/state）变更。光照数据作为落盘时的内存快照随内容一并写入。 |

I2 要求世界生成对 `(seed, coord)` 完全确定（当前 WorldGen 移植已满足）。它同时意味着一个**有意接受的行为**：若游戏版本改变了世界生成规则，未编辑区域的地形会在升级后改变——这正是"按种子重新生成"的题中之义。

---

## 4. 修改点：commit 按内容判别 dirty

### 4.1 方案选型

| 方案 | 描述 | 结论 |
|------|------|------|
| A. 双写 API | `WriteTxn` 增加 `write_light_only`，由调用方声明写入性质 | 否决。调用方误用方向危险（把内容写当光照写 → 区块永不落盘 → 数据丢失），且新增 API 面积。 |
| B. commit 时自动判别 | 提交时对每个被写 cell 比较"内容是否相对提交前发生变化"，据此决定是否盖 LSN | **采用**。调用方零改动、零误用风险，未来新增写路径（爆炸、重力方块等）自动获得正确语义。 |

### 4.2 新的 `WriteTxn::commit` 算法

```text
commit(self):
    drained = buffered.drain()          // Vec<(chunk_coord, Vec<(block_coord, BlockData)>)>
    if drained.is_empty(): return
    lsn = lsn_counter.fetch_add(1)      // 照常单调递增
    for (cc, writes) in drained:
        entry = entry_mut(cc)
        // 1) 每个 cell 取 last-write-wins 的最终值
        final = HashMap<block_coord, BlockData>  // 按 writes 顺序后写覆盖先写
        // 2) 在应用任何写入前，从 guard.data 读出各 cell 的原始值
        content_changed = any { final[c].id != orig[c].id || final[c].state != orig[c].state }
        // 3) 按原顺序应用全部写入（含纯光照写入——内存光照必须更新）
        for (b, data) in writes: guard.data.block_mut(b) = data
        // 4) 仅当内容变化才盖 LSN
        if content_changed: entry.guard.commit_lsn = lsn
```

要点：

- **纯光照写入仍然生效**——渲染端依赖内存光照（`src/client/render/mesh.rs:447-455` 的 smooth-lighting 采样），松弛算法照常运行；变化的只是不再盖 `commit_lsn`。
- 原始值在应用前读取：写入此刻仍是缓冲态，`guard.data` 即提交前状态。同 cell 多次写入按最终值 vs 原始值比较，天然把"先改后还原"判为无内容变化。
- 判别只比较 `(id, state)`，忽略 light。
- LSN 计数器照常每 commit 递增一次，保持全序语义不变（只是部分区块不再被盖章）。

### 4.3 现有写路径在新语义下的归类

| 写路径 | 位置 | 写入内容 | 新语义下 |
|--------|------|----------|----------|
| `set_id_and_state`（玩家放置/破坏） | `block_update.rs:144` | id/state 变化（已有 `new_data == curr` 早退保护） | dirty ✓ |
| `process_one_in_txn`（光照松弛，单格与批量同体） | `block_update.rs:239` | 仅 light 变化 | 不 dirty ✓ |
| `set_block` / `set_block_with_state` | `block_update.rs:104/122` | 转发到上两者 | 随之正确 |
| 未来写路径（explode、随机 tick 等） | — | 任意 | 自动判别，无需各自处理 |

游戏代码中 `begin_write_txn_sync` 的调用点只有 `block_update.rs` 三处（`L154`、`L210`、`L373`）与 `txn.rs` 测试，收敛性已核实。

### 4.4 `from_generated` 转 clean

`Chunk::from_generated` 去掉 `lsn` 参数，`(commit_lsn, persisted_lsn) = (0, 0)`，与 `from_disk` 一致：

- `chunk.rs:272-288`：删除 `lsn` 参数与 `debug_assert!(lsn > 0)`。
- `mod.rs:305/310`：`Chunk::from_generated(f(), self.next_lsn())` → `Chunk::from_generated(f())`。
- `World::next_lsn` / `lsn_counter` 保留，仅供 `WriteTxn` 提交使用；`mod.rs:229` 的注释同步更新。

由此**生成区块从出生就是 clean**：卸载时 `flush_chunk`（`mod.rs:410`）因 `commit_lsn(0) <= persisted_lsn(0)` 直接跳过，纯生成区块永不写盘。玩家在生成区块上做第一次内容编辑时，commit 盖章 → dirty → 卸载落盘。

---

## 5. 光照与持久化的交互（本期语义）

光照数据本期仍持久化，但其持久化时机完全由内容决定：

| 场景 | 行为 |
|------|------|
| 区块因内容编辑 flush | `package_to` 打包当前内存态——**光照松弛的最新结果顺带保存** |
| 区块只有光照变化（如邻区块编辑的光照溢出） | 不 dirty → 不 flush → 光照更新不落盘，盘上保留旧快照 |
| 生成区块（含 `init_generate` 烘焙的天空光） | 出生 clean → 从不落盘；重载时重新生成并重新烘焙，光照自洽 |
| 磁盘区块重载 | 光照从磁盘恢复（现状行为不变） |

即：光照落盘 = 内容落盘的搭车快照（ride-along）。纯光照变化丢失的问题与处理见 §8 / §9。

---

## 6. 存档形态与迁移

### 6.1 修改后的存档布局

```
<world_dir>/
├── world.dat      # metadata，magic "NEWD"，WORLD_META_VERSION = 2（不变）
│                  #   body: block_mapping + seed —— 再生所需信息已齐备，无新增字段
└── chunks.db      # sled K/V：key = 12B LE chunk coord
                   # value = "NEWC" + version=3 + zstd(id+state+light 5B/格)   （格式不变）
                   # 集合 = 有过内容提交的区块（≈ 玩家编辑过的区块）
```

各场景落点：

| 场景 | 结果 |
|------|------|
| 生成区块，玩家从未编辑 | clean → 卸载不写盘 → 不在存档；重载时按 seed 再生 |
| 生成区块，玩家编辑过 | 编辑时 dirty → 卸载落盘 |
| 磁盘区块，只有光照变化（编辑波及） | 不 dirty → 不重写（盘上旧值与旧光照快照保留） |
| 磁盘区块，内容编辑 | dirty → 落盘（含最新光照快照） |

### 6.2 迁移

磁盘格式与 metadata 均无任何变化，无需迁移。

---

## 7. 测试计划

| 层 | 用例 | 断言 |
|----|------|------|
| `txn.rs` | commit 判别 | 仅 light 不同的 write → commit 后 `!dirty()`；id/state 变化 → `dirty()`；同 txn 先改内容后还原 → `!dirty()`；light+content 混合 → `dirty()` 且 light 生效 |
| `chunk.rs` | 构造语义 | `from_generated` 无 lsn 参数且 clean；`from_disk` clean |
| 集成（World 级） | 存档形态 | 生成→卸载→sled 无 key；编辑→卸载→有 key；重载后内容一致、光照从磁盘恢复 |
| 集成（World 级） | 搭车快照 | 磁盘区块经历光照松弛但无内容变化 → 卸载不重写（盘上字节不变） |

现有需同步修订的用例：`txn.rs::from_gen_starts_dirty`（语义反转）、`chunk.rs::writeback_clears_dirty` / `second_commit_during_writeback_keeps_dirty` 等以 `from_generated` 构造 dirty 的用例改为经内容提交构造。

---

## 8. 实施清单

| 文件 | 修改 |
|------|------|
| `src/core/world/txn.rs` | `commit` 内容判别算法（§4.2）；新增判别测试 |
| `src/core/world/chunk.rs` | `from_generated` 去掉 `lsn` 参数；相关测试修订 |
| `src/core/world/mod.rs` | 生成路径不再分配 LSN；`next_lsn` 随之成为死代码，实施时删除（计数器仅经 `lsn_counter` Arc 由 `WriteTxn` 持有） |

**明确不动**：`data.rs`（编码含 light 不变）、`store.rs`、`block_update.rs`、`range_loader.rs`、`terrain_generator.rs`（`init_generate` 烘焙光照保留）、`metadata.rs`（v2 不变）、`chunk.rs` 的 `VERSION`（3 不变）。

---

## 9. 风险与开放问题

1. **纯光照变化不落盘**：邻区块被编辑波及、仅光照变化时，重载后该区块回到盘上旧光照快照（或生成区块重新烘焙的值），可能与编辑前的稳态不一致——属视觉层面的局部偏差，且玩家在该区块内的任何后续内容编辑都会把最新光照搭车保存。彻底解法是"光照去持久化 + 加载重建"，本期明确不做，留作后续方案。
2. **编辑还原不弃存**：区块一旦发生内容提交就永久 dirty（LSN 单调），即使玩家把方块改回生成原貌也会落盘（内容与再生结果等价，仅多占空间）。
3. **世界生成规则演进**：未编辑区域随版本升级而变化（I2 的直接推论，属"按种子再生"的定义）。
