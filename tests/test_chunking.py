"""
tests/test_chunking.py
========================
ELI5: These tests check both chunkers actually produce sensible pieces
from a document — no piece completely empty, every original document
still represented, using a small hand-made test document instead of the
full 59-document corpus (faster, and the failure is easier to reason
about with known input).
"""

from src.chunking.semantic import chunk_semantic
from src.chunking.simple import chunk_simple
from src.corpus import Document

SAMPLE_DOC = Document(
    doc_id="test-doc.md",
    title="Test Document",
    text=(
        "# Test Document\n\n"
        "This is the first paragraph about vacation policy. Employees get 20 days off per year. "
        "This is still about vacation.\n\n"
        "This second paragraph is about something completely different: server maintenance. "
        "Servers are rebooted every Sunday at 2am."
    ),
)


def test_simple_chunking_covers_the_whole_document():
    chunks = chunk_simple([SAMPLE_DOC], chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.doc_id == "test-doc.md" for c in chunks)
    assert all(len(c.text.strip()) > 0 for c in chunks)


def test_simple_chunking_respects_chunk_size():
    chunks = chunk_simple([SAMPLE_DOC], chunk_size=100, overlap=20)
    assert all(len(c.text) <= 100 for c in chunks)


def test_semantic_chunking_produces_nonempty_chunks():
    chunks = chunk_semantic([SAMPLE_DOC])
    assert len(chunks) >= 1
    assert all(len(c.text.strip()) > 0 for c in chunks)
    assert all(c.doc_id == "test-doc.md" for c in chunks)


def test_semantic_chunking_handles_single_sentence_document():
    tiny_doc = Document(doc_id="tiny.md", title="Tiny", text="Just one sentence here.")
    chunks = chunk_semantic([tiny_doc])
    assert len(chunks) == 1
    assert "one sentence" in chunks[0].text
