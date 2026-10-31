//! Host-side bookkeeping for the paged KV cache.
//!
//! The backend owns the actual K/V memory (a pool of fixed-size blocks per
//! layer). This crate decides which blocks each sequence uses. Blocks are
//! reference-counted so prefix caching can later share them between sequences.

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("KV cache out of blocks (need {need}, have {free})")]
    OutOfBlocks { need: usize, free: usize },
    #[error("KV block copy: {0}")]
    Copy(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub struct BlockAllocator {
    block_size: usize,
    free: Vec<u32>,
    refcount: Vec<u32>,
}

impl BlockAllocator {
    pub fn new(n_blocks: usize, block_size: usize) -> Self {
        // Hand out low block ids first.
        let free = (0..n_blocks as u32).rev().collect();
        Self { block_size, free, refcount: vec![0; n_blocks] }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn n_free(&self) -> usize {
        self.free.len()
    }

    pub fn n_blocks(&self) -> usize {
        self.refcount.len()
    }

    fn alloc(&mut self) -> Option<u32> {
        let b = self.free.pop()?;
        self.refcount[b as usize] = 1;
        Some(b)
    }

    /// Allocate `n` blocks (one reference each), or none if fewer are free.
    pub fn alloc_n(&mut self, n: usize) -> Option<Vec<u32>> {
        (n <= self.free.len()).then(|| (0..n).map(|_| self.alloc().expect("checked above")).collect())
    }

    /// References held to `block` (0: free).
    pub fn refcount(&self, block: u32) -> u32 {
        self.refcount[block as usize]
    }

    /// Add a reference to an allocated block (prefix sharing).
    pub fn share(&mut self, block: u32) {
        let rc = &mut self.refcount[block as usize];
        debug_assert!(*rc > 0, "sharing free block {block}");
        *rc += 1;
    }

    /// Drop one reference; the block is free once none are left.
    pub fn release(&mut self, block: u32) {
        let rc = &mut self.refcount[block as usize];
        debug_assert!(*rc > 0, "double free of block {block}");
        *rc -= 1;
        if *rc == 0 {
            self.free.push(block);
        }
    }
}

/// One sequence's view of the cache: its block table and token count.
#[derive(Debug, Default, Clone)]
pub struct SeqKv {
    pub block_table: Vec<u32>,
    pub len: usize,
}

impl SeqKv {
    /// A sequence starting with `len` tokens already in `blocks` (references
    /// the caller has taken, e.g. from [`PrefixCache::lookup`]).
    pub fn with_prefix(blocks: Vec<u32>, len: usize) -> Self {
        Self { block_table: blocks, len }
    }

    /// Blocks needed to hold `n` more tokens.
    pub fn blocks_needed(&self, block_size: usize, n: usize) -> usize {
        (self.len + n).div_ceil(block_size).saturating_sub(self.block_table.len())
    }

    /// Reserve room for `n` more tokens, allocating blocks as needed.
    /// Returns the slot index for each new token. All-or-nothing.
    pub fn append(&mut self, alloc: &mut BlockAllocator, n: usize) -> Result<Vec<u32>> {
        let bs = alloc.block_size();
        let need = (self.len + n).div_ceil(bs).saturating_sub(self.block_table.len());
        if need > alloc.n_free() {
            return Err(Error::OutOfBlocks { need, free: alloc.n_free() });
        }
        for _ in 0..need {
            self.block_table.push(alloc.alloc().expect("checked above"));
        }
        let slots = (self.len..self.len + n)
            .map(|p| self.block_table[p / bs] * bs as u32 + (p % bs) as u32)
            .collect();
        self.len += n;
        Ok(slots)
    }

    /// Keep only the first `len` tokens (rejected speculative tokens), and
    /// release blocks no longer needed. Their slots are simply overwritten
    /// by later tokens.
    pub fn truncate(&mut self, alloc: &mut BlockAllocator, len: usize) {
        if len >= self.len {
            return;
        }
        for b in self.block_table.drain(len.div_ceil(alloc.block_size())..) {
            alloc.release(b);
        }
        self.len = len;
    }

    /// Return all blocks to the allocator.
    pub fn free(&mut self, alloc: &mut BlockAllocator) {
        for b in self.block_table.drain(..) {
            alloc.release(b);
        }
        self.len = 0;
    }
}

/// Where cached blocks live, for [`PrefixCache`]: the device pool, and
/// optionally a host tier (pinned system memory) that evicted entries move to
/// instead of being dropped.
pub trait BlockStore {
    fn device(&mut self) -> &mut BlockAllocator;

    /// The host tier's allocator, if there is one.
    fn host(&mut self) -> Option<&mut BlockAllocator> {
        None
    }

    /// Copy device blocks `dev[i]` to host blocks `host[i]`.
    fn to_host(&mut self, dev: &[u32], host: &[u32]) -> Result<()> {
        let _ = (dev, host);
        Err(Error::Copy("no host tier".into()))
    }

    /// Copy host blocks `host[i]` to device blocks `dev[i]`.
    fn to_device(&mut self, host: &[u32], dev: &[u32]) -> Result<()> {
        let _ = (host, dev);
        Err(Error::Copy("no host tier".into()))
    }
}

/// A plain device pool, without a host tier.
impl BlockStore for BlockAllocator {
    fn device(&mut self) -> &mut BlockAllocator {
        self
    }
}

/// Finished sequences kept for reuse: a later request whose tokens start
/// with the same prefix shares their full blocks instead of recomputing them.
///
/// Only whole blocks are shared, so a sharing sequence never writes into a
/// block someone else reads. Entries hold one reference to each of their
/// blocks; they are evicted least-recently-used first when blocks run out.
///
/// With a host tier ([`BlockStore::host`]), an evicted entry is copied to
/// host blocks instead of dropped, and a later lookup that matches it
/// further than anything on the device copies it back: a copy over PCIe
/// instead of recomputing the prefix. Host entries are evicted LRU in turn.
#[derive(Default)]
pub struct PrefixCache {
    entries: Vec<Entry>,
    clock: u64,
    stats: TierStats,
}

struct Entry {
    tokens: Vec<u32>,
    /// Device block ids, or host block ids if `on_host`.
    blocks: Vec<u32>,
    used: u64,
    on_host: bool,
}

/// Host tier activity since start.
#[derive(Debug, Default, Clone, Copy)]
pub struct TierStats {
    /// Blocks copied to the host.
    pub offloaded: usize,
    /// Blocks copied back to the device.
    pub restored: usize,
}

impl PrefixCache {
    /// Upper bound on device entries; beyond it the oldest is evicted on insert.
    const MAX_ENTRIES: usize = 256;
    /// Upper bound on host entries (lookups scan every entry).
    const MAX_HOST_ENTRIES: usize = 2048;

    /// Entries on the device.
    pub fn len(&self) -> usize {
        self.entries.iter().filter(|e| !e.on_host).count()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Device blocks held by the cache (some may also be in use by sequences).
    pub fn n_blocks(&self) -> usize {
        self.entries.iter().filter(|e| !e.on_host).map(|e| e.blocks.len()).sum()
    }

    /// Entries in the host tier.
    pub fn host_len(&self) -> usize {
        self.entries.iter().filter(|e| e.on_host).count()
    }

    /// Host blocks held by the cache.
    pub fn host_blocks(&self) -> usize {
        self.entries.iter().filter(|e| e.on_host).map(|e| e.blocks.len()).sum()
    }

    pub fn stats(&self) -> TierStats {
        self.stats
    }

    fn touch(&mut self, i: usize) {
        self.clock += 1;
        self.entries[i].used = self.clock;
    }

    /// The longest cached block-aligned prefix of `tokens`, leaving at least
    /// one token to compute (the model needs it for the next logits).
    /// Takes a reference to each returned block. Returns (tokens covered, blocks).
    pub fn lookup(&mut self, store: &mut impl BlockStore, tokens: &[u32]) -> (usize, Vec<u32>) {
        let bs = store.device().block_size();
        let limit = tokens.len().saturating_sub(1) / bs;
        let mut best = self.best_matches(tokens, bs, limit);
        let on_device = best[0].map_or(0, |b| b.1);
        if let Some((i, n)) = best[1]
            && n > on_device
        {
            if let Some(hit) = self.restore(store, i, n) {
                return hit;
            }
            // A failed restore reorders and may evict entries: the device
            // match found above can point at another entry now.
            best = self.best_matches(tokens, bs, limit);
        }
        let Some((i, n)) = best[0] else {
            return (0, Vec::new());
        };
        self.touch(i);
        let blocks = self.entries[i].blocks[..n].to_vec();
        for &b in &blocks {
            store.device().share(b);
        }
        (n * bs, blocks)
    }

    /// Best match on the device and on the host: (entry index, blocks).
    fn best_matches(&self, tokens: &[u32], bs: usize, limit: usize) -> [Option<(usize, usize)>; 2] {
        let mut best: [Option<(usize, usize)>; 2] = [None, None];
        for (i, e) in self.entries.iter().enumerate() {
            let common = e.tokens.iter().zip(tokens).take_while(|(a, b)| a == b).count();
            let n = (common / bs).min(limit).min(e.blocks.len());
            let b = &mut best[e.on_host as usize];
            if n > 0 && b.is_none_or(|(_, m)| n > m) {
                *b = Some((i, n));
            }
        }
        best
    }

    /// Copy the first `n` blocks of host entry `i` back to the device as a
    /// device entry, and share them with the caller. None if there isn't
    /// room on the device or the copy fails.
    fn restore(&mut self, store: &mut impl BlockStore, i: usize, n: usize) -> Option<(usize, Vec<u32>)> {
        let bs = store.device().block_size();
        // Out of the list while making room, so evictions can't touch it.
        let mut e = self.entries.swap_remove(i);
        self.clock += 1;
        e.used = self.clock;
        let dev = if self.make_room(store, n) { store.device().alloc_n(n) } else { None };
        let Some(dev) = dev else {
            self.entries.push(e);
            return None;
        };
        if store.to_device(&e.blocks[..n], &dev).is_err() {
            dev.into_iter().for_each(|b| store.device().release(b));
            self.entries.push(e);
            return None;
        }
        self.stats.restored += n;
        let tokens = e.tokens[..n * bs].to_vec();
        if n == e.blocks.len() {
            // The device now has all of it.
            let host = store.host().expect("host entries need a host tier");
            e.blocks.into_iter().for_each(|b| host.release(b));
        } else {
            self.entries.push(e);
        }
        self.add_device(store, tokens, dev.clone());
        for &b in &dev {
            store.device().share(b);
        }
        Some((n * bs, dev))
    }

    /// Retire a sequence whose cache holds `tokens` (`seq.len == tokens.len()`).
    /// Its full blocks join the cache; the partial last block is released.
    pub fn insert(&mut self, store: &mut impl BlockStore, tokens: &[u32], mut seq: SeqKv) {
        let bs = store.device().block_size();
        let full = seq.len.min(tokens.len()) / bs;
        for b in seq.block_table.drain(full..) {
            store.device().release(b);
        }
        if full == 0 {
            return;
        }
        let tokens = &tokens[..full * bs];
        // An existing device entry that already covers this one: keep it instead.
        if let Some(i) = self.entries.iter().position(|e| !e.on_host && e.tokens.starts_with(tokens)) {
            self.touch(i);
            for b in seq.block_table.drain(..) {
                store.device().release(b);
            }
            return;
        }
        if self.len() >= Self::MAX_ENTRIES {
            self.evict(store);
        }
        self.add_device(store, tokens.to_vec(), std::mem::take(&mut seq.block_table));
    }

    /// Add a device entry, dropping the entries (either tier) it extends.
    fn add_device(&mut self, store: &mut impl BlockStore, tokens: Vec<u32>, blocks: Vec<u32>) {
        self.drop_extended(store, &tokens);
        self.clock += 1;
        self.entries.push(Entry { tokens, blocks, used: self.clock, on_host: false });
    }

    /// Drop entries whose tokens are a prefix of `tokens`: redundant once
    /// `tokens` is cached.
    fn drop_extended(&mut self, store: &mut impl BlockStore, tokens: &[u32]) {
        let mut i = 0;
        while i < self.entries.len() {
            if tokens.starts_with(&self.entries[i].tokens) {
                let e = self.entries.swap_remove(i);
                Self::release(store, e);
            } else {
                i += 1;
            }
        }
    }

    fn release(store: &mut impl BlockStore, e: Entry) {
        if e.on_host {
            let host = store.host().expect("host entries need a host tier");
            e.blocks.into_iter().for_each(|b| host.release(b));
        } else {
            e.blocks.into_iter().for_each(|b| store.device().release(b));
        }
    }

    /// Free the least recently used device entry's blocks, moving it to the
    /// host tier if there is one. False if no device entry is left.
    pub fn evict(&mut self, store: &mut impl BlockStore) -> bool {
        self.evict_where(store, |_, _| true)
    }

    /// Evict the least recently used device entry that `pick` accepts.
    fn evict_where(&mut self, store: &mut impl BlockStore, pick: impl Fn(&Entry, &BlockAllocator) -> bool) -> bool {
        let dev = store.device();
        let lru = self.entries.iter().enumerate().filter(|(_, e)| !e.on_host && pick(e, dev)).min_by_key(|(_, e)| e.used).map(|(i, _)| i);
        let Some(i) = lru else {
            return false;
        };
        let e = self.entries.swap_remove(i);
        self.offload(store, &e);
        Self::release(store, e);
        true
    }

    /// Copy a device entry into the host tier (if there is one, it fits,
    /// and the host doesn't already hold it).
    fn offload(&mut self, store: &mut impl BlockStore, e: &Entry) {
        let n = e.blocks.len();
        let Some(capacity) = store.host().map(|h| h.n_blocks()) else { return };
        if n > capacity || self.entries.iter().any(|h| h.on_host && h.tokens.starts_with(&e.tokens)) {
            return;
        }
        while store.host().unwrap().n_free() < n || self.host_len() >= Self::MAX_HOST_ENTRIES {
            let lru = self.entries.iter().enumerate().filter(|(_, h)| h.on_host).min_by_key(|(_, h)| h.used).map(|(i, _)| i);
            let Some(j) = lru else { return };
            let h = self.entries.swap_remove(j);
            Self::release(store, h);
        }
        let host = store.host().unwrap().alloc_n(n).expect("checked above");
        if store.to_host(&e.blocks, &host).is_err() {
            let h = store.host().unwrap();
            host.into_iter().for_each(|b| h.release(b));
            return;
        }
        self.stats.offloaded += n;
        // Host entries this one extends are redundant now.
        let mut j = 0;
        while j < self.entries.len() {
            if self.entries[j].on_host && e.tokens.starts_with(&self.entries[j].tokens) {
                let h = self.entries.swap_remove(j);
                Self::release(store, h);
            } else {
                j += 1;
            }
        }
        self.entries.push(Entry { tokens: e.tokens.clone(), blocks: host, used: e.used, on_host: true });
    }

    /// Evict until `need` device blocks are free (or nothing is left to evict).
    pub fn make_room(&mut self, store: &mut impl BlockStore, need: usize) -> bool {
        // Blocks only the cache holds: evicting is pointless (and costs a
        // copy to the host tier) when even all of them are too few.
        let sole = |e: &Entry, dev: &BlockAllocator| e.blocks.iter().filter(|&&b| dev.refcount(b) == 1).count();
        let dev = store.device();
        let freeable: usize = self.entries.iter().filter(|e| !e.on_host).map(|e| sole(e, dev)).sum();
        if dev.n_free() + freeable < need {
            return false;
        }
        while store.device().n_free() < need {
            // Skip entries whose blocks running sequences still hold.
            if !self.evict_where(store, |e, dev| sole(e, dev) > 0) {
                return false;
            }
        }
        true
    }

    /// Drop every entry, in both tiers.
    pub fn clear(&mut self, store: &mut impl BlockStore) {
        for e in std::mem::take(&mut self.entries) {
            Self::release(store, e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_follow_block_table() {
        let mut a = BlockAllocator::new(4, 2);
        let mut s = SeqKv::default();
        assert_eq!(s.append(&mut a, 3).unwrap(), vec![0, 1, 2]);
        assert_eq!(s.block_table, vec![0, 1]);
        assert_eq!(s.append(&mut a, 1).unwrap(), vec![3]);
        assert_eq!(a.n_free(), 2);

        let mut t = SeqKv::default();
        assert_eq!(t.append(&mut a, 1).unwrap(), vec![4]);
        s.free(&mut a);
        assert_eq!(a.n_free(), 3);
        // t's next block comes from s's freed blocks.
        assert_eq!(t.append(&mut a, 2).unwrap(), vec![5, 2]);
    }

    #[test]
    fn truncate_releases_whole_blocks() {
        let mut a = BlockAllocator::new(4, 4);
        let mut s = SeqKv::default();
        s.append(&mut a, 10).unwrap();
        s.truncate(&mut a, 5);
        assert_eq!((s.len, s.block_table.len(), a.n_free()), (5, 2, 2));
        // The next token goes right after the kept ones.
        assert_eq!(s.append(&mut a, 1).unwrap(), vec![s.block_table[1] * 4 + 1]);
        s.truncate(&mut a, 8);
        assert_eq!(s.len, 6);
    }

    #[test]
    fn out_of_blocks_is_atomic() {
        let mut a = BlockAllocator::new(1, 4);
        let mut s = SeqKv::default();
        assert!(s.append(&mut a, 5).is_err());
        assert_eq!(s.len, 0);
        assert_eq!(a.n_free(), 1);
    }

    #[test]
    fn prefix_cache_shares_full_blocks() {
        let mut a = BlockAllocator::new(8, 4);
        let mut pc = PrefixCache::default();
        let toks: Vec<u32> = (0..10).collect();
        let mut s = SeqKv::default();
        s.append(&mut a, 10).unwrap();
        pc.insert(&mut a, &toks, s);
        // Two full blocks kept, the partial third released.
        assert_eq!(pc.n_blocks(), 2);
        assert_eq!(a.n_free(), 6);

        // Same first 9 tokens, then different: 2 blocks (8 tokens) reused.
        let mut q: Vec<u32> = (0..9).collect();
        q.push(99);
        let (n, blocks) = pc.lookup(&mut a, &q);
        assert_eq!((n, blocks.len()), (8, 2));
        let mut t = SeqKv::with_prefix(blocks, n);
        assert_eq!(t.append(&mut a, 2).unwrap(), vec![8, 9]);

        // An exact match still leaves the last token to compute.
        let (n, blocks) = pc.lookup(&mut a, &toks[..8]);
        assert_eq!(n, 4);
        blocks.into_iter().for_each(|b| a.release(b));

        // Evicting the entry keeps blocks t still uses.
        assert!(pc.evict(&mut a));
        assert_eq!(a.n_free(), 5);
        t.free(&mut a);
        assert_eq!(a.n_free(), 8);
    }

    #[test]
    fn prefix_cache_replaces_extended_entries() {
        let mut a = BlockAllocator::new(8, 2);
        let mut pc = PrefixCache::default();
        let mut s = SeqKv::default();
        s.append(&mut a, 4).unwrap();
        pc.insert(&mut a, &[1, 2, 3, 4], s);
        let (n, blocks) = pc.lookup(&mut a, &[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(n, 4);
        let mut s = SeqKv::with_prefix(blocks, n);
        s.append(&mut a, 2).unwrap();
        pc.insert(&mut a, &[1, 2, 3, 4, 5, 6], s);
        assert_eq!(pc.len(), 1);
        assert_eq!(pc.n_blocks(), 3);
        pc.clear(&mut a);
        assert_eq!(a.n_free(), 8);
    }

    /// A device pool and host tier whose blocks hold a payload, so tests can
    /// check that the right data moves.
    struct TwoTier {
        dev: BlockAllocator,
        host: BlockAllocator,
        dev_data: Vec<u32>,
        host_data: Vec<u32>,
        fail_restore: bool,
    }

    impl TwoTier {
        fn new(dev: usize, host: usize, bs: usize) -> Self {
            Self { dev: BlockAllocator::new(dev, bs), host: BlockAllocator::new(host, bs), dev_data: vec![0; dev], host_data: vec![0; host], fail_restore: false }
        }

        /// A sequence holding `tokens`; each block's payload is its first token.
        fn seq(&mut self, tokens: &[u32]) -> SeqKv {
            let mut s = SeqKv::default();
            s.append(&mut self.dev, tokens.len()).unwrap();
            let bs = self.dev.block_size();
            for (j, &b) in s.block_table.iter().enumerate() {
                self.dev_data[b as usize] = tokens[j * bs];
            }
            s
        }
    }

    impl BlockStore for TwoTier {
        fn device(&mut self) -> &mut BlockAllocator {
            &mut self.dev
        }
        fn host(&mut self) -> Option<&mut BlockAllocator> {
            Some(&mut self.host)
        }
        fn to_host(&mut self, dev: &[u32], host: &[u32]) -> Result<()> {
            for (&d, &h) in dev.iter().zip(host) {
                self.host_data[h as usize] = self.dev_data[d as usize];
            }
            Ok(())
        }
        fn to_device(&mut self, host: &[u32], dev: &[u32]) -> Result<()> {
            if self.fail_restore {
                return Err(Error::Copy("injected".into()));
            }
            for (&h, &d) in host.iter().zip(dev) {
                self.dev_data[d as usize] = self.host_data[h as usize];
            }
            Ok(())
        }
    }

    #[test]
    fn host_tier_keeps_evicted_prefixes() {
        let mut st = TwoTier::new(4, 8, 2);
        let mut pc = PrefixCache::default();
        let a: Vec<u32> = (10..16).collect(); // 3 blocks
        let s = st.seq(&a);
        pc.insert(&mut st, &a, s);
        // Room for 3 more blocks: `a` moves to the host.
        assert!(pc.make_room(&mut st, 3));
        assert_eq!((pc.len(), pc.host_len(), pc.host_blocks()), (0, 1, 3));
        assert_eq!(st.dev.n_free(), 4);

        // A prompt continuing `a` gets its blocks back, with their data.
        let mut q = a.clone();
        q.extend([99, 98]);
        let (n, blocks) = pc.lookup(&mut st, &q);
        assert_eq!(n, 6);
        let payload: Vec<u32> = blocks.iter().map(|&b| st.dev_data[b as usize]).collect();
        assert_eq!(payload, vec![10, 12, 14]);
        // Restored whole: back on the device, gone from the host.
        assert_eq!((pc.len(), pc.host_len(), st.host.n_free()), (1, 0, 8));
        assert_eq!(pc.stats().offloaded, 3);
        assert_eq!(pc.stats().restored, 3);
        blocks.into_iter().for_each(|b| st.dev.release(b));
        pc.clear(&mut st);
        assert_eq!((st.dev.n_free(), st.host.n_free()), (4, 8));
    }

    #[test]
    fn host_tier_restores_partial_matches_and_evicts_lru() {
        let mut st = TwoTier::new(4, 4, 2);
        let mut pc = PrefixCache::default();
        let a: Vec<u32> = (0..6).collect();
        let b: Vec<u32> = (100..104).collect();
        for t in [&a, &b] {
            let s = st.seq(t);
            pc.insert(&mut st, t, s);
            assert!(pc.make_room(&mut st, 4));
        }
        // The host holds 4 blocks: `b` (2) arrived last and pushed out `a` (3).
        assert_eq!((pc.host_len(), pc.host_blocks()), (1, 2));
        let s = st.seq(&a);
        pc.insert(&mut st, &a, s);
        assert!(pc.make_room(&mut st, 4));
        // Now `a` is in and `b` (LRU) is out.
        assert_eq!((pc.host_len(), pc.host_blocks()), (1, 3));

        // Matching only `a`'s first block restores that block alone; the
        // host keeps the whole entry.
        let (n, blocks) = pc.lookup(&mut st, &[0, 1, 7, 7]);
        assert_eq!(n, 2);
        assert_eq!(st.dev_data[blocks[0] as usize], 0);
        assert_eq!((pc.len(), pc.host_len()), (1, 1));
        blocks.into_iter().for_each(|b| st.dev.release(b));
        // Evicting the restored block again: the host already covers it.
        assert!(pc.make_room(&mut st, 4));
        assert_eq!((pc.len(), pc.host_len(), pc.stats().offloaded), (0, 1, 8));
    }

    #[test]
    fn device_hits_win_over_equal_host_hits() {
        let mut st = TwoTier::new(8, 8, 2);
        let mut pc = PrefixCache::default();
        let a: Vec<u32> = (0..4).collect();
        let s = st.seq(&a);
        pc.insert(&mut st, &a, s);
        assert!(pc.make_room(&mut st, 8));
        let s = st.seq(&a);
        pc.insert(&mut st, &a, s);
        // Same prefix in both tiers: no copy.
        let (n, blocks) = pc.lookup(&mut st, &[0, 1, 2, 3, 4]);
        assert_eq!(n, 4);
        assert_eq!(pc.stats().restored, 0);
        blocks.into_iter().for_each(|b| st.dev.release(b));
    }

    #[test]
    fn failed_restore_falls_back_to_the_device_match() {
        let mut st = TwoTier::new(8, 8, 2);
        let mut pc = PrefixCache::default();
        // Keep host block ids apart from device ones.
        let pad = st.host.alloc_n(4).unwrap();
        let a: Vec<u32> = (0..6).collect();
        let s = st.seq(&a);
        pc.insert(&mut st, &a, s);
        assert!(pc.make_room(&mut st, 8));
        pad.into_iter().for_each(|b| st.host.release(b));
        // Entries: [a (host), x, d]; the failed restore reorders them.
        let mut d_blocks = Vec::new();
        for t in [vec![50, 51, 52, 53], vec![0, 1, 9, 9]] {
            let s = st.seq(&t);
            d_blocks = s.block_table.clone();
            pc.insert(&mut st, &t, s);
        }
        st.fail_restore = true;
        let (n, blocks) = pc.lookup(&mut st, &[0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(n, 2);
        assert_eq!(blocks, d_blocks[..1]);
        assert_eq!((pc.len(), pc.host_len()), (2, 1));
        blocks.into_iter().for_each(|b| st.dev.release(b));
        pc.clear(&mut st);
        assert_eq!((st.dev.n_free(), st.host.n_free()), (8, 8));
    }

    #[test]
    fn make_room_keeps_entries_that_free_nothing() {
        let mut st = TwoTier::new(6, 8, 2);
        let mut pc = PrefixCache::default();
        let a: Vec<u32> = (0..4).collect();
        let s = st.seq(&a);
        pc.insert(&mut st, &a, s);
        // A running sequence holds `a`'s blocks.
        let (n, held) = pc.lookup(&mut st, &[0, 1, 2, 3, 4]);
        assert_eq!(n, 4);
        // Evicting `a` would free nothing: no eviction, no copy to the host.
        assert!(!pc.make_room(&mut st, 5));
        assert_eq!((pc.len(), pc.host_len(), pc.stats().offloaded), (1, 0, 0));
        held.into_iter().for_each(|b| st.dev.release(b));
        assert!(pc.make_room(&mut st, 6));
        assert_eq!((pc.len(), pc.host_len()), (0, 1));
    }
}
