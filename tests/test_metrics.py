"""
tests/test_metrics.py
=======================
ELI5: These tests check Hit Rate and MRR are computed correctly against
known, hand-constructed retrieval results — no real corpus needed.
"""

from src.chunking.simple import Chunk
from src.eval.metrics import evaluate_question, is_hit
from src.retrieval.sparse import ScoredChunk


def _sc(text):
    return ScoredChunk(chunk=Chunk("id", "doc.md", "Doc", text), score=1.0)


def test_is_hit_requires_all_keywords_present():
    assert is_hit(["22", "days"], "Employees get 22 days of leave.") is True
    assert is_hit(["22", "days"], "Employees get some leave.") is False


def test_is_hit_case_insensitive():
    assert is_hit(["PARIS"], "The capital is paris.") is True


def test_evaluate_question_first_place_hit_gets_full_score():
    retrieved = [_sc("22 days of annual leave"), _sc("unrelated chunk")]
    result = evaluate_question(["22", "days"], retrieved)
    assert result.hit is True
    assert result.reciprocal_rank == 1.0


def test_evaluate_question_second_place_hit_gets_half_score():
    retrieved = [_sc("unrelated chunk"), _sc("22 days of annual leave")]
    result = evaluate_question(["22", "days"], retrieved)
    assert result.hit is True
    assert result.reciprocal_rank == 0.5


def test_evaluate_question_no_hit_scores_zero():
    retrieved = [_sc("unrelated chunk one"), _sc("unrelated chunk two")]
    result = evaluate_question(["22", "days"], retrieved)
    assert result.hit is False
    assert result.reciprocal_rank == 0.0
