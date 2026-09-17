"""
src/retrieval/dense.py
========================
ELI5: "Dense" search is the "vibe match" approach — instead of requiring
the exact same words, it tries to capture MEANING, so a search for "time
off" can still find a chunk about "annual leave" even though they don't
share a single word. Real production systems do this with neural
embedding models. This project can't download one (no internet to Hugging
Face here), so it uses a genuine, decades-old alternative that achieves a
similar effect purely mathematically: LSA (Latent Semantic Analysis).

How LSA works, in plain English: first build a big table of "which words
appear in which chunks" (that's TF-IDF, the same building block as BM25's
cousin). Then run a mathematical technique called SVD (Singular Value
Decomposition) on that table, which compresses it down to a much smaller
number of dimensions that tend to capture broad THEMES rather than exact
words — e.g. one compressed dimension might end up loosely representing
"HR/leave-related concepts" even though no single word was told to mean
that. This is the actual technique search engines used for "semantic
search" for years before neural embeddings existed, and it's a completely
legitimate, real, explainable stand-in here.
"""

from dataclasses import dataclass

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunking.simple import Chunk
from src.retrieval.sparse import ScoredChunk


class DenseRetriever:
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self._vectorizer = None
        self._svd = None
        self._chunk_vectors = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        texts = [c.text for c in chunks]

        self._vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = self._vectorizer.fit_transform(texts)

        # SVD needs fewer components than there are documents/features —
        # cap it sensibly for a small corpus so this doesn't error out.
        n_components = min(self.n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        n_components = max(n_components, 2)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._chunk_vectors = self._svd.fit_transform(tfidf_matrix)

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        query_tfidf = self._vectorizer.transform([query])
        query_vector = self._svd.transform(query_tfidf)
        similarities = cosine_similarity(query_vector, self._chunk_vectors)[0]
        ranked = sorted(zip(self._chunks, similarities), key=lambda x: x[1], reverse=True)
        return [ScoredChunk(chunk=c, score=float(s)) for c, s in ranked[:top_k]]
