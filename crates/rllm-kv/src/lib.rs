//! Host-side bookkeeping for the paged KV cache.
//!
//! The backend owns the actual K/V memory (a pool of fixed-size blocks per
//! layer). This crate decides which blocks each sequence uses. Blocks are
//! reference-counted so prefix caching can later share them between sequences.

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("KV cache out of blocks (need {need}, have {free})")]
    OutOfBlocks { need: usize, free: usize },
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

    /// Return all blocks to the allocator.
    pub fn free(&mut self, alloc: &mut BlockAllocator) {
        for b in self.block_table.drain(..) {
            alloc.release(b);
        }
        self.len = 0;
    }
}

/// Finished sequences kept for reuse: a later request whose tokens start
/// with the same prefix shares their full blocks instead of recomputing them.
///
/// Only whole blocks are shared, so a sharing sequence never writes into a
/// block someone else reads. Entries hold one reference to each of their
/// blocks; they are evicted least-recently-used first when blocks run out.
#[derive(Default)]
pub struct PrefixCache {
    entries: Vec<Entry>,
    clock: u64,
}

struct Entry {
    tokens: Vec<u32>,
    blocks: Vec<u32>,
    used: u64,
}

impl PrefixCache {
    /// Upper bound on entries; beyond it the oldest is evicted on insert.
    const MAX_ENTRIES: usize = 256;

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Blocks held by the cache (some may also be in use by sequences).
    pub fn n_blocks(&self) -> usize {
        self.entries.iter().map(|e| e.blocks.len()).sum()
    }

    /// The longest cached block-aligned prefix of `tokens`, leaving at least
    /// one token to compute (the model needs it for the next logits).
    /// Takes a reference to each returned block. Returns (tokens covered, blocks).
    pub fn lookup(&mut self, alloc: &mut BlockAllocator, tokens: &[u32]) -> (usize, Vec<u32>) {
        let bs = alloc.block_size();
        let limit = tokens.len().saturating_sub(1) / bs;
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            let common = e.tokens.iter().zip(tokens).take_while(|(a, b)| a == b).count();
            let n = (common / bs).min(limit).min(e.blocks.len());
            if n > 0 && best.is_none_or(|(_, m)| n > m) {
                best = Some((i, n));
            }
        }
        let Some((i, n)) = best else {
            return (0, Vec::new());
        };
        self.clock += 1;
        self.entries[i].used = self.clock;
        let blocks = self.entries[i].blocks[..n].to_vec();
        for &b in &blocks {
            alloc.share(b);
        }
        (n * bs, blocks)
    }

    /// Retire a sequence whose cache holds `tokens` (`seq.len == tokens.len()`).
    /// Its full blocks join the cache; the partial last block is released.
    pub fn insert(&mut self, alloc: &mut BlockAllocator, tokens: &[u32], mut seq: SeqKv) {
        let bs = alloc.block_size();
        let full = seq.len.min(tokens.len()) / bs;
        for b in seq.block_table.drain(full..) {
            alloc.release(b);
        }
        if full == 0 {
            return;
        }
        let tokens = &tokens[..full * bs];
        // An existing entry that already covers this one: keep it instead.
        if let Some(e) = self.entries.iter_mut().find(|e| e.tokens.starts_with(tokens)) {
            self.clock += 1;
            e.used = self.clock;
            for b in seq.block_table.drain(..) {
                alloc.release(b);
            }
            return;
        }
        // Entries this one extends are now redundant.
        let mut i = 0;
        while i < self.entries.len() {
            if tokens.starts_with(&self.entries[i].tokens) {
                let e = self.entries.swap_remove(i);
                e.blocks.into_iter().for_each(|b| alloc.release(b));
            } else {
                i += 1;
            }
        }
        if self.entries.len() >= Self::MAX_ENTRIES {
            self.evict(alloc);
        }
        self.clock += 1;
        self.entries.push(Entry { tokens: tokens.to_vec(), blocks: std::mem::take(&mut seq.block_table), used: self.clock });
    }

    /// Drop the least recently used entry. False if the cache is empty.
    pub fn evict(&mut self, alloc: &mut BlockAllocator) -> bool {
        let Some(i) = self.entries.iter().enumerate().min_by_key(|(_, e)| e.used).map(|(i, _)| i) else {
            return false;
        };
        let e = self.entries.swap_remove(i);
        e.blocks.into_iter().for_each(|b| alloc.release(b));
        true
    }

    /// Evict until `need` blocks are free (or the cache is empty).
    pub fn make_room(&mut self, alloc: &mut BlockAllocator, need: usize) -> bool {
        while alloc.n_free() < need {
            if !self.evict(alloc) {
                return false;
            }
        }
        true
    }

    pub fn clear(&mut self, alloc: &mut BlockAllocator) {
        while self.evict(alloc) {}
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
}
