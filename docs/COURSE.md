# The Complete Course: Building a RAG Retrieval Evaluation Harness

Every concept in this project, from the ground up — a plain analogy first, then the real technical
explanation, then exactly which file implements it.

---

## Table of Contents

1. [What is RAG, and why does retrieval matter so much?](#1-what-is-rag)
2. [Building an honest evaluation set](#2-building-an-honest-evaluation-set)
3. [Chunking: why you can't just feed whole documents in](#3-chunking)
4. [Simple chunking vs. semantic chunking](#4-simple-vs-semantic-chunking)
5. [Sparse search (BM25)](#5-sparse-search-bm25)
6. [Dense/semantic search, and why this project uses LSA instead of neural embeddings](#6-dense-search-and-lsa)
7. [Hybrid search and Reciprocal Rank Fusion](#7-hybrid-search-and-rrf)
8. [Reranking — the second, more careful pass](#8-reranking)
9. [Hit Rate and MRR — grading retrieval, not generation](#9-hit-rate-and-mrr)
10. [Why the corpus has "filler" text around every fact](#10-why-filler-text)
11. [CI, exit codes, and evaluation gates](#11-ci-exit-codes-and-evaluation-gates)
12. [Reading the actual results honestly](#12-reading-the-results-honestly)
13. [Glossary](#13-glossary)

---

## 1. What is RAG?

**ELI5:** Imagine an extremely well-read friend who's still only ever read books up to a certain
date, and has never seen YOUR company's specific internal documents at all. RAG (Retrieval-
Augmented Generation) is like handing that friend the exact right page from your company handbook
right before asking them a question about it — so instead of guessing from general knowledge, they
answer from the actual, current, correct source.

**The real explanation:** RAG systems have two stages. **Retrieval**: given a question, search a
knowledge base and pull out the most relevant chunks of text. **Generation**: hand those chunks to
an AI model along with the question, so it answers grounded in real, specific source material
instead of only its own training data. This project focuses entirely on the RETRIEVAL half — if
retrieval finds the wrong (or no) source material, no amount of generation skill can produce a
correct answer, since the model never saw the right information in the first place. That's exactly
why evaluating retrieval on its own, separately from generation, is worth doing.

---

## 2. Building an honest evaluation set

**ELI5:** If you wrote an exam and also wrote the answer key completely separately, from memory,
you might accidentally write an answer that's wrong or that the exam doesn't actually support.
This project avoids that by building the exam and the documents from the exact SAME source of
truth, so every "correct answer" is guaranteed to genuinely exist in the material being searched.

**The real explanation:** `scripts/facts.py` defines 75 `Fact` records — each one is a single
sentence that becomes BOTH a fact embedded in a generated document AND a question/answer pair in
the eval set. `scripts/build_dataset.py` reads these once and writes out both the corpus and
`data/eval/questions.json` together. This "single source of truth" design is a real, important
practice in evaluation engineering: an eval set built by hand, separately from the source data, is
a common and easy way to accidentally introduce answers that aren't actually retrievable, which
silently makes the whole benchmark meaningless.

---

## 3. Chunking

**ELI5:** You can't hand someone an entire 500-page book and ask them to instantly point to the one
sentence that answers a specific question — you'd first break the book into smaller sections so
searching becomes practical. "Chunking" is that step: breaking documents into smaller pieces BEFORE
indexing them, so retrieval can return a focused, relevant piece rather than an entire long
document (most of which wouldn't be relevant to a given question).

**The real explanation:** Every retrieval method in this project starts with chunking. The choice
of HOW you chunk turns out to matter enormously for retrieval quality — cut in the wrong place and
you can literally split a fact's sentence across two separate chunks, so that neither chunk alone
fully contains the answer.

---

## 4. Simple vs. semantic chunking

**ELI5:** Simple chunking cuts a document with a ruler, every N characters, no matter what's there.
Semantic chunking reads along and cuts where the TOPIC actually changes, like a good editor
splitting a document into sections by meaning.

**The real explanation:** `src/chunking/simple.py`'s `chunk_simple()` slices each document into
fixed-size, overlapping windows — fast, simple, and the default in many real systems, but blind to
sentence or topic boundaries. `src/chunking/semantic.py`'s `chunk_semantic()` instead splits each
document into sentences, computes a TF-IDF vector for each sentence, and walks through them
greedily: keep adding sentences to the current chunk while the next sentence stays similar (cosine
similarity above a threshold) to what's already in the chunk; start a new chunk the moment
similarity drops — that drop is treated as a topic boundary. This is a simplified, fully offline
version of the real technique used in production RAG chunking libraries, which normally do the same
similarity check with neural sentence embeddings instead of TF-IDF.

**Worth noticing from the actual results:** semantic chunking alone (paired with plain BM25) scored
*slightly worse* than the simple baseline in this project's real run (0.720 vs 0.733). That's a
genuinely useful, honest finding, not a bug — a good chunking strategy isn't automatically better in
isolation; it depends on what retrieval algorithm sits on top of it. It's semantic chunking
COMBINED with hybrid search where the real gain shows up (0.787). This exact kind of nuance — "this
component alone didn't help, but paired with that other component it did" — is precisely why you
benchmark combinations empirically instead of assuming each individual technique is a strict
improvement.

---

## 5. Sparse search (BM25)

**ELI5:** The index at the back of a textbook, done really well. Look up a word, get sent straight
to the page. "Sparse" refers to how the underlying math represents text: most possible words in the
language DON'T appear in a given chunk, so the representation is mostly zeros ("sparse") with a few
meaningful nonzero entries for the words that actually do appear.

**The real explanation:** `src/retrieval/sparse.py` uses the `rank_bm25` library's `BM25Okapi`
implementation — a real, still widely-used-in-production scoring formula. It rewards a chunk for
containing the query's exact words, weighting rarer words more heavily than common ones (matching
"geofence" counts for far more than matching "the"), with diminishing returns for word repetition
inside one chunk. Sparse search is excellent at exact terminology and numbers, and struggles when a
question and its answer use different words for the same idea (a paraphrase).

---

## 6. Dense search, and LSA

**ELI5:** Sparse search needs the exact same words to match. Dense/"semantic" search instead tries
to capture MEANING, so a search for "time off" can still find a chunk about "annual leave" — no
shared words needed. Real production systems achieve this with neural embedding models (small AI
models trained specifically to turn text into meaning-capturing vectors). This project can't
download one (no internet access to model hubs in this build environment), so it uses a genuine,
much older alternative that gets a similar effect through pure linear algebra.

**The real explanation — LSA (Latent Semantic Analysis):** `src/retrieval/dense.py` first builds a
TF-IDF matrix (chunks x words), then runs `TruncatedSVD` (Singular Value Decomposition) on it,
compressing that matrix down to a much smaller number of dimensions. The mathematical effect is
that words which tend to co-occur across many chunks get pulled toward similar compressed
directions — so the compressed space ends up loosely capturing broad THEMES rather than exact
wording, purely as a side effect of the matrix math, with no meaning ever explicitly programmed in.
This was literally how "semantic search" worked in production information-retrieval systems for
years before neural embeddings existed, and it remains a completely legitimate, explainable
technique — just less powerful than a modern trained embedding model on genuinely subtle paraphrase
cases. **To swap in a real neural embedding model later:** you'd replace `DenseRetriever`'s
`TfidfVectorizer + TruncatedSVD` with a call to an embedding model (e.g.
`sentence-transformers`) that turns each chunk and each query into a vector directly — everything
downstream (hybrid fusion, reranking, evaluation) works identically either way, since they only
ever see "a list of scored chunks," not how the scoring was computed.

---

## 7. Hybrid search and RRF

**ELI5:** Ask two different experts — one who's great at exact-word matching, one who's great at
"vibe"/meaning matching — for their independent top picks, then merge both lists into one fair
final ranking, rather than trusting only one expert's opinion.

**The real explanation:** `src/retrieval/hybrid.py`'s `HybridRetriever` runs BOTH `SparseRetriever`
(BM25) and `DenseRetriever` (LSA) independently, then combines their two rankings using
**Reciprocal Rank Fusion (RRF)** — a real, widely-used, deliberately simple fusion technique. For
each chunk, take `1 / (rank + k)` from EACH retriever it appeared in (using the chunk's POSITION in
that retriever's ranking, not its raw numeric score) and sum those values together; `k` is a small
constant (60, the standard default from the original research) that softens the impact of small
rank differences near the top. Using rank instead of raw score is the key trick: it sidesteps the
real problem that BM25 scores and LSA cosine-similarity scores live on completely different numeric
scales, which would make a naive "just add the two raw scores" approach unreliable.

---

## 8. Reranking

**ELI5:** A first-round resume screen quickly narrows 10,000 applicants down to 20 plausible ones —
that's hybrid search. Reranking is the SECOND, slower, more careful read: someone sits down and
reads those 20 much more closely before finalizing the order. It only has to examine a small
shortlist, so it can afford to look more carefully at each one than the fast first pass did.

**The real explanation:** Real production rerankers are usually "cross-encoders" — small neural
models that read the query and a candidate chunk TOGETHER, letting them interact directly, rather
than as two separately-computed vectors compared only by distance afterward (which is what BM25 and
LSA both do). That joint reading tends to catch subtler relevance signals, but needs a downloaded
model. This project's `src/retrieval/reranker.py` is an honest, fully offline stand-in built from
several lexical signals combined: what fraction of the query's words appear in the chunk, whether
the query appears as a near-exact PHRASE (not just scattered words), and a mild penalty for very
long chunks. It genuinely demonstrates what reranking DOES structurally — narrow first, then
re-examine more carefully — even though a real cross-encoder would likely outperform it on subtler
paraphrase cases. Swapping it for a real cross-encoder later means changing only this one file.

---

## 9. Hit Rate and MRR

**ELI5:** Hit Rate is a simple yes/no per question: did the right answer show up ANYWHERE in the
top results? MRR additionally rewards finding it EARLIER, not just finding it somewhere at all.

**The real explanation:** `src/eval/metrics.py`'s `is_hit()` checks whether ALL of a question's
expected keywords appear in a given retrieved chunk's text — a deliberately strict, defensible
pass/fail bar, the same design principle used across every project in this series. **Hit Rate @ K**
(averaged across all 75 questions) answers "what fraction of the time did the correct chunk appear
somewhere in the top K results?" **MRR (Mean Reciprocal Rank)** additionally captures WHERE in the
ranking it appeared: 1st place scores a full 1.0, 2nd place scores 0.5, 5th place scores 0.2, not
found at all scores 0 — averaged across all questions. Two methods can have identical Hit Rates
while having very different MRRs, if one method usually finds the answer in 1st place and the other
usually buries it in 5th — and that difference matters a great deal in a real RAG system, since
whichever chunk lands in position 1 has the most influence on the AI's final generated answer.

---

## 10. Why filler text

**ELI5:** If every document were just one bare fact sitting completely alone, ANY retrieval method
— even a genuinely bad one — would trivially find it every time, and the whole comparison would be
meaningless. Real company documents are never one isolated sentence; they're full of realistic
surrounding text. Building that same realism into the fake corpus is what makes the benchmark a
genuine test rather than a rigged demonstration.

**The real explanation:** `scripts/build_dataset.py`'s `build_filler_paragraph()` surrounds every
real fact sentence with plausible-sounding but factually empty corporate boilerplate ("this
document is reviewed quarterly," "please direct questions to HR," etc.) drawn from a small pool of
template sentences. This forces every retrieval method to actually distinguish signal from noise,
the same challenge real-world retrieval faces against genuinely long, noisy company wikis.

---

## 11. CI, exit codes, and evaluation gates

**ELI5:** A program's "exit code" is a single number it hands back to whatever ran it when it
finishes — by strong convention, 0 means "everything's fine," any other number means "something
went wrong." GitHub Actions doesn't read any of the printed text your program produced; it watches
ONLY that one number to decide whether to mark a build green (passed) or red (failed).

**The real explanation:** `run_eval.py` ends with `sys.exit(1)` if the best method's hit rate falls
below `--threshold`, and `sys.exit(0)` otherwise (Python programs exit 0 automatically if they
finish without calling `sys.exit()` explicitly, which is why the passing path doesn't need to say
so out loud). `.github/workflows/eval.yml` runs this exact script on every push and pull request;
the moment any step in a GitHub Actions job exits non-zero, the whole job is marked failed
automatically — there is no more exotic mechanism than that single number underneath the "red X" or
"green check" you see on a pull request. This pattern — an automated quality check wired directly
into the merge process — is called a **quality gate**, and wiring an AI evaluation into one this way
is exactly how production AI teams catch a retrieval regression (someone's "small refactor" that
accidentally breaks chunking, say) automatically, before it ships, rather than discovering it only
after a customer complains.

---

## 12. Reading the results honestly

Worth saying plainly, since it's a stronger and more credible story than pretending every technique
is a strict win: in this project's actual run, `reranking` (0.800) beat `hybrid_search` (0.787) beat
`simple_chunking` (0.733) beat `semantic_chunking` alone (0.720). The weakest category throughout
was `product` (0.67 even for the best method) — genuinely explainable, since the 8 fictional
products deliberately have very similar names and numbers (Rover X1 vs X2, Falcon-3 vs Falcon-5),
which is exactly the kind of near-duplicate content real retrieval systems struggle with too. Being
able to explain not just the winning number but WHY a specific category is hardest, in an interview,
is worth more than a single clean headline score.

---

## 13. Glossary

| Term | Plain-English meaning |
|---|---|
| RAG | Retrieval-Augmented Generation — searching real documents before an AI answers, instead of relying only on its training data |
| Chunking | Splitting documents into smaller pieces before indexing them for search |
| Sparse search | Keyword-based search (e.g. BM25) — needs matching words |
| Dense / semantic search | Meaning-based search using vector similarity — can match paraphrases |
| TF-IDF | A statistical way to turn text into a vector based on word frequency and rarity |
| LSA (Latent Semantic Analysis) | Compressing a TF-IDF matrix via SVD to capture broad themes, without neural embeddings |
| BM25 | A well-established keyword-scoring formula used in real production search engines |
| Hybrid search | Combining sparse and dense search results into one ranking |
| RRF (Reciprocal Rank Fusion) | A method for combining multiple rankings using each item's rank position, not raw score |
| Reranking | A second, more careful pass that reorders an already-narrowed shortlist |
| Cross-encoder | A neural model that reads a query and a document together, rather than as separate vectors |
| Hit Rate @ K | The fraction of questions where the correct answer appeared anywhere in the top K results |
| MRR (Mean Reciprocal Rank) | A metric rewarding finding the correct answer EARLIER in the ranking, not just anywhere |
| Evaluation gate | An automated check wired into CI that blocks a merge if quality falls below a threshold |
| Exit code | The number a program returns on finishing; 0 means success, anything else signals failure |
