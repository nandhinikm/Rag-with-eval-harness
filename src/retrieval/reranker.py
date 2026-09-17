"""
src/retrieval/reranker.py
============================
ELI5: Imagine a first-round of resume screening that quickly narrows
10,000 applicants down to 20 plausible ones (that's what hybrid search
just did). Reranking is the SECOND, more careful pass — someone sits down
and reads those 20 much more closely before deciding the final order. It
only has to look at a small number of candidates, so it can afford to be
slower and more thorough per-item than the first-round search.

In real production RAG systems, this second pass is usually a "cross-
encoder" — a small neural model that reads the query AND a candidate
chunk TOGETHER (rather than as two separate vectors, which is what BM25
and LSA both do) and outputs a single relevance score. That joint
reading is more accurate but needs a downloaded model. This project's
offline reranker is an honest, explainable stand-in built from several
lexical signals combined — good enough to demonstrate genuinely what
reranking DOES (reorder a shortlist using more careful signals than the
first pass used), even though a real cross-encoder would likely do
better still. Swapping this function for a real cross-encoder later
requires changing only this one file — everything upstream is unaffected.
"""

import re
from dataclasses import dataclass

from src.retrieval.sparse import ScoredChunk


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _rerank_score(query: str, chunk_text: str) -> float:
    query_tokens = _tokenize(query)
    chunk_tokens = _tokenize(chunk_text)

    if not query_tokens:
        return 0.0

    # Signal 1: what fraction of the query's own words appear in this
    # chunk at all? (plain overlap, similar spirit to keyword coverage
    # scoring used elsewhere in these projects)
    overlap_fraction = len(query_tokens & chunk_tokens) / len(query_tokens)

    # Signal 2: does the chunk contain the query as a near-exact PHRASE,
    # not just the same words scattered anywhere? Phrase matches are a
    # much stronger relevance signal than scattered word overlap.
    query_lower = query.lower().strip("?.! ")
    phrase_bonus = 0.3 if query_lower in chunk_text.lower() else 0.0

    # Signal 3: mild preference for shorter, more focused chunks over
    # very long ones when overlap is similar — a long chunk "matching"
    # partly by sheer volume of words is a weaker signal than a short,
    # tightly focused one.
    length_penalty = min(len(chunk_text) / 2000, 0.2)

    return overlap_fraction + phrase_bonus - length_penalty


def rerank(query: str, candidates: list[ScoredChunk], top_k: int = 5) -> list[ScoredChunk]:
    """Takes an already-retrieved shortlist (from hybrid search, typically)
    and reorders it using the more careful lexical signals above."""
    rescored = [
        ScoredChunk(chunk=c.chunk, score=_rerank_score(query, c.chunk.text))
        for c in candidates
    ]
    rescored.sort(key=lambda x: x.score, reverse=True)
    return rescored[:top_k]
