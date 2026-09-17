"""
src/eval/metrics.py
=====================
ELI5: Once we've retrieved, say, the top 5 chunks for a question, how do
we grade whether retrieval actually "worked"? Two standard, real metrics,
used industry-wide for evaluating search and retrieval systems:

HIT RATE @ K: a simple yes/no per question — did the correct answer show
up ANYWHERE in the top K results? Averaged across all questions, this
becomes "what fraction of the time did we find the answer at all?" —
easy to understand, but doesn't care whether the right answer was
result #1 or result #5.

MRR (Mean Reciprocal Rank): rewards finding the answer EARLIER, not just
finding it somewhere. If the correct chunk is the very first result, that
question scores a perfect 1.0. Second place scores 1/2 = 0.5. Fifth place
scores 1/5 = 0.2. Not found in the top K at all scores 0. Averaged across
all questions, this captures something Hit Rate misses entirely: two
methods might both "find" the answer 90% of the time, but if one method
usually finds it in 1st place and the other usually buries it in 5th
place, MRR tells them apart — which matters a lot in a real system, since
whatever comes back in position 1 is what most influences the final
generated answer.
"""

from dataclasses import dataclass

from src.retrieval.sparse import ScoredChunk


def is_hit(answer_keywords: list[str], chunk_text: str) -> bool:
    """A retrieved chunk counts as a 'hit' if ALL of the question's
    expected keywords appear in it — a deliberately strict bar (partial
    keyword matches don't count), the same design principle used in the
    other two projects: a clear, defensible pass/fail line rather than a
    fuzzy in-between."""
    chunk_lower = chunk_text.lower()
    return all(kw.lower() in chunk_lower for kw in answer_keywords)


@dataclass
class QuestionResult:
    question_id: str
    hit: bool
    reciprocal_rank: float  # 0.0 if no hit in the retrieved results


def evaluate_question(answer_keywords: list[str], retrieved: list[ScoredChunk]) -> QuestionResult:
    for rank, result in enumerate(retrieved, start=1):
        if is_hit(answer_keywords, result.chunk.text):
            return QuestionResult(question_id="", hit=True, reciprocal_rank=1.0 / rank)
    return QuestionResult(question_id="", hit=False, reciprocal_rank=0.0)
