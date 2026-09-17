"""
src/retrieval/sparse.py
=========================
ELI5: "Sparse" search is old-school keyword search, done well. Think of
the index at the back of a textbook — it tells you exactly which page a
word appears on. BM25 is a well-known formula (used inside real search
engines like Elasticsearch) that scores each chunk by how well its exact
words match the query's exact words, with two smart adjustments: rare
words count for more than common words (matching "geofence" matters more
than matching "the"), and a chunk repeating a word 20 times isn't 20x
more relevant than one that has it once — the formula intentionally
diminishes the returns of repetition.

BM25 is genuinely still used in production search systems today, often
paired with a semantic/dense method (see hybrid.py) — it's not a toy.
"""

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.chunking.simple import Chunk


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class SparseRetriever:
    def __init__(self):
        self._bm25 = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._chunks, scores), key=lambda x: x[1], reverse=True)
        return [ScoredChunk(chunk=c, score=float(s)) for c, s in ranked[:top_k]]
