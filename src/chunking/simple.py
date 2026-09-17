"""
src/chunking/simple.py
========================
ELI5: Imagine cutting a long book into pieces by measuring exactly 500
characters at a time with a ruler, no matter where a sentence happens to
be — sometimes you'll cut right through the middle of an important
sentence. That's "simple chunking": fast, easy to implement, and the
most common starting point in real RAG systems — but exactly because it
ignores meaning, it's also the technique most likely to accidentally
split a fact in half, leaving neither half fully answering a question.
"""

from dataclasses import dataclass

from src.corpus import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str


def chunk_simple(documents: list[Document], chunk_size: int = 400, overlap: int = 80) -> list[Chunk]:
    """
    Splits each document into fixed-size character windows, with a bit of
    OVERLAP between consecutive chunks. The overlap exists specifically to
    reduce (not eliminate) the "cut through the middle of a fact" problem:
    if a fact sentence falls right on a boundary, there's a decent chance
    it still appears whole in the overlapping region of the next chunk.
    """
    chunks = []
    for doc in documents:
        text = doc.text
        start = 0
        idx = 0
        step = max(chunk_size - overlap, 1)
        while start < len(text):
            piece = text[start:start + chunk_size]
            if piece.strip():
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}::simple::{idx}",
                    doc_id=doc.doc_id, doc_title=doc.title, text=piece,
                ))
                idx += 1
            start += step
    return chunks
