"""
src/retrieval/hybrid.py
=========================
ELI5: Sparse search (BM25) is great at exact terms but misses paraphrases.
Dense search (LSA) is great at "vibe" matches but can be fooled by short,
number-heavy queries with little vocabulary to work with. Hybrid search
asks BOTH, then combines their opinions into one final ranking — like
asking two different experts for their top picks and merging the lists,
rather than trusting only one.

The specific merging technique used here — Reciprocal Rank Fusion (RRF)
— is a real, widely-used, and pleasantly simple method: for each chunk,
take 1/(rank + k) from EACH retriever's ranking (not its raw score) and
add them together, where `rank` is that chunk's position in that
retriever's results (1st place, 2nd place, etc.) and `k` is a small
constant (60 is the standard default from the original research) that
softens the impact of small rank differences. Using RANK instead of raw
score is the key trick — it means you never have to worry about BM25 and
LSA producing scores on totally different numeric scales, which would
make a naive "just add the scores together" approach unreliable.
"""

from collections import defaultdict

from src.chunking.simple import Chunk
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import ScoredChunk, SparseRetriever


class HybridRetriever:
    def __init__(self, rrf_k: int = 60, candidate_pool: int = 20):
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool  # how many results each sub-retriever contributes before fusion
        self._sparse = SparseRetriever()
        self._dense = DenseRetriever()

    def build(self, chunks: list[Chunk]) -> None:
        self._sparse.build(chunks)
        self._dense.build(chunks)

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        sparse_results = self._sparse.search(query, top_k=self.candidate_pool)
        dense_results = self._dense.search(query, top_k=self.candidate_pool)

        fused_scores: dict[str, float] = defaultdict(float)
        chunk_lookup: dict[str, Chunk] = {}

        for rank, result in enumerate(sparse_results, start=1):
            fused_scores[result.chunk.chunk_id] += 1.0 / (self.rrf_k + rank)
            chunk_lookup[result.chunk.chunk_id] = result.chunk

        for rank, result in enumerate(dense_results, start=1):
            fused_scores[result.chunk.chunk_id] += 1.0 / (self.rrf_k + rank)
            chunk_lookup[result.chunk.chunk_id] = result.chunk

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [ScoredChunk(chunk=chunk_lookup[cid], score=score) for cid, score in ranked[:top_k]]
