"""
src/corpus.py
===============
ELI5: This file's only job is loading the fake company's documents off
disk into memory as simple Python objects, so everything downstream
(chunkers, retrievers) has a consistent starting point to work from.
"""

import os
from dataclasses import dataclass


@dataclass
class Document:
    doc_id: str    # filename, e.g. "product-rover-x1.md"
    title: str
    text: str


def load_corpus(corpus_dir: str) -> list[Document]:
    docs = []
    for filename in sorted(os.listdir(corpus_dir)):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(corpus_dir, filename)
        with open(path, "r") as f:
            text = f.read()
        title_line = next((l for l in text.splitlines() if l.startswith("# ")), filename)
        title = title_line.lstrip("# ").strip()
        docs.append(Document(doc_id=filename, title=title, text=text))
    return docs
