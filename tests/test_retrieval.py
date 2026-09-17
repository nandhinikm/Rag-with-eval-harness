"""
tests/test_retrieval.py
=========================
ELI5: These tests check each retriever actually finds the right chunk for
an obvious, unambiguous query — using a small hand-built set of chunks
where we already know the right answer, rather than the full corpus.
"""

from src.chunking.simple import Chunk
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import rerank
from src.retrieval.sparse import SparseRetriever

SAMPLE_CHUNKS = [
    Chunk("c1", "doc1.md", "Doc 1", "Employees receive 22 days of paid annual leave per year."),
    Chunk("c2", "doc2.md", "Doc 2", "The Falcon-5 drone has a battery capacity of 3.4 kWh."),
    Chunk("c3", "doc3.md", "Doc 3", "Pull requests require approval from 2 reviewers before merging."),
    Chunk("c4", "doc4.md", "Doc 4", "The office kitchen is restocked with snacks every Monday morning."),
]


def test_sparse_retriever_finds_exact_keyword_match():
    retriever = SparseRetriever()
    retriever.build(SAMPLE_CHUNKS)
    results = retriever.search("how many days of annual leave", top_k=1)
    assert results[0].chunk.chunk_id == "c1"


def test_dense_retriever_returns_results_for_every_query():
    retriever = DenseRetriever()
    retriever.build(SAMPLE_CHUNKS)
    results = retriever.search("battery capacity of the drone", top_k=2)
    assert len(results) == 2


def test_hybrid_retriever_finds_exact_match_too():
    retriever = HybridRetriever()
    retriever.build(SAMPLE_CHUNKS)
    results = retriever.search("reviewer approval pull request", top_k=1)
    assert results[0].chunk.chunk_id == "c3"


def test_reranker_reorders_by_phrase_match():
    retriever = SparseRetriever()
    retriever.build(SAMPLE_CHUNKS)
    initial = retriever.search("annual leave days", top_k=4)
    reranked = rerank("annual leave days", initial, top_k=4)
    # The chunk actually about annual leave should end up in first place
    # after reranking, regardless of where BM25 alone put it.
    assert reranked[0].chunk.chunk_id == "c1"
