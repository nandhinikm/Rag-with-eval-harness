"""
src/chunking/semantic.py
==========================
ELI5: Instead of cutting a document with a ruler at a fixed measurement,
semantic chunking reads sentence by sentence and asks "does this next
sentence still feel like it's talking about the same thing as what came
before, or has the topic shifted?" — and only cuts a new chunk at the
points where the topic genuinely changes. Like a good editor breaking a
document into sections by MEANING, not by a fixed word count.

How do we measure "does this still feel like the same topic" without
downloading a neural embedding model (this project runs fully offline
with no model downloads)? We use TF-IDF vectors — a decades-old, purely
statistical way of turning a sentence into a vector of numbers based on
which words it uses and how rare/common those words are across the
document. Two sentences using similar vocabulary get vectors that point
in a similar direction (measured by "cosine similarity"). It's a real,
legitimate technique — simpler than a neural embedding model, but built
on the same core idea: turn text into vectors, and use vector distance as
a stand-in for meaning-similarity.
"""

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunking.simple import Chunk
from src.corpus import Document

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    # Split on blank lines first (paragraph breaks are a strong signal),
    # then split each paragraph into sentences.
    sentences = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        for sent in SENTENCE_SPLIT_RE.split(para):
            sent = sent.strip()
            if sent:
                sentences.append(sent)
    return sentences


def chunk_semantic(documents: list[Document], similarity_threshold: float = 0.15,
                    max_chunk_sentences: int = 6) -> list[Chunk]:
    """
    For each document: split into sentences, compute a TF-IDF vector per
    sentence, then walk through the sentences in order, growing a "current
    chunk". Whenever the next sentence's similarity to the current chunk's
    average vector drops below `similarity_threshold`, that's treated as a
    topic boundary — close off the current chunk and start a new one.
    `max_chunk_sentences` is a safety cap so one very homogeneous document
    doesn't become one giant chunk.
    """
    chunks = []
    for doc in documents:
        sentences = _split_sentences(doc.text)
        if len(sentences) <= 1:
            if sentences:
                chunks.append(Chunk(f"{doc.doc_id}::semantic::0", doc.doc_id, doc.title, sentences[0]))
            continue

        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            vectors = vectorizer.fit_transform(sentences).toarray()
        except ValueError:
            # A document with only stopwords/too little vocabulary — fall
            # back to treating it as a single chunk rather than crashing.
            chunks.append(Chunk(f"{doc.doc_id}::semantic::0", doc.doc_id, doc.title, doc.text))
            continue

        current_group = [0]
        idx = 0
        for i in range(1, len(sentences)):
            current_vector = vectors[current_group].mean(axis=0).reshape(1, -1)
            next_vector = vectors[i].reshape(1, -1)
            similarity = cosine_similarity(current_vector, next_vector)[0][0]

            if similarity >= similarity_threshold and len(current_group) < max_chunk_sentences:
                current_group.append(i)
            else:
                chunk_text = " ".join(sentences[j] for j in current_group)
                chunks.append(Chunk(f"{doc.doc_id}::semantic::{idx}", doc.doc_id, doc.title, chunk_text))
                idx += 1
                current_group = [i]

        if current_group:
            chunk_text = " ".join(sentences[j] for j in current_group)
            chunks.append(Chunk(f"{doc.doc_id}::semantic::{idx}", doc.doc_id, doc.title, chunk_text))

    return chunks
