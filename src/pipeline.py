"""
src/pipeline.py
=================
ELI5: This file is the "recipe book" — it defines the 4 complete methods
being compared, each one a specific combination of a chunking strategy
and a retrieval strategy (plus, for one method, a reranking step). Every
method is built from the same small set of LEGO-brick pieces defined in
chunking/ and retrieval/ — the comparison is really about which
COMBINATION of bricks works best, not about four totally separate systems.
"""

from src.chunking.semantic import chunk_semantic
from src.chunking.simple import Chunk, chunk_simple
from src.corpus import Document
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import rerank
from src.retrieval.sparse import ScoredChunk, SparseRetriever


class RetrievalMethod:
    name: str
    description: str
    chunks: list[Chunk] = None

    def build(self, documents: list[Document]):
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        raise NotImplementedError


class SimpleChunkingMethod(RetrievalMethod):
    """Method 1: fixed-size chunking + plain BM25 keyword search. The
    baseline every other method is measured against."""

    def __init__(self):
        self.name = "simple_chunking"
        self.description = "Fixed-size chunking + BM25 keyword search (baseline)"
        self._retriever = SparseRetriever()

    def build(self, documents: list[Document]):
        self.chunks = chunk_simple(documents)
        self._retriever.build(self.chunks)

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        return self._retriever.search(query, top_k)


class SemanticChunkingMethod(RetrievalMethod):
    """Method 2: semantic (topic-boundary-aware) chunking + BM25. Isolates
    the effect of chunking quality alone, keeping the retrieval algorithm
    identical to the baseline."""

    def __init__(self):
        self.name = "semantic_chunking"
        self.description = "Semantic (topic-boundary) chunking + BM25 keyword search"
        self._retriever = SparseRetriever()

    def build(self, documents: list[Document]):
        self.chunks = chunk_semantic(documents)
        self._retriever.build(self.chunks)

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        return self._retriever.search(query, top_k)


class HybridSearchMethod(RetrievalMethod):
    """Method 3: semantic chunking + hybrid (BM25 + LSA fused via RRF)
    search. Adds a dense/semantic retrieval signal on top of Method 2."""

    def __init__(self):
        self.name = "hybrid_search"
        self.description = "Semantic chunking + hybrid search (BM25 + LSA, fused via RRF)"
        self._retriever = HybridRetriever()

    def build(self, documents: list[Document]):
        self.chunks = chunk_semantic(documents)
        self._retriever.build(self.chunks)

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        return self._retriever.search(query, top_k)


class RerankingMethod(RetrievalMethod):
    """Method 4: semantic chunking + hybrid search retrieves a wider
    shortlist, then a reranking pass reorders that shortlist using more
    careful signals before returning the final top-k."""

    def __init__(self, candidate_pool: int = 15):
        self.name = "reranking"
        self.description = "Semantic chunking + hybrid search + lexical reranking of top candidates"
        self._retriever = HybridRetriever()
        self.candidate_pool = candidate_pool

    def build(self, documents: list[Document]):
        self.chunks = chunk_semantic(documents)
        self._retriever.build(self.chunks)

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        candidates = self._retriever.search(query, top_k=self.candidate_pool)
        return rerank(query, candidates, top_k=top_k)


def build_all_methods(documents: list[Document]) -> list[RetrievalMethod]:
    """Constructs and builds (indexes) all 4 methods, ready to be searched."""
    methods = [
        SimpleChunkingMethod(),
        SemanticChunkingMethod(),
        HybridSearchMethod(),
        RerankingMethod(),
    ]
    for method in methods:
        method.build(documents)
    return methods
