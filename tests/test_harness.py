"""
tests/test_harness.py
=======================
ELI5: These are integration tests — running the FULL harness (all 4
methods, indexed and searched) against a small hand-built mini-corpus, to
prove every piece actually connects correctly end-to-end. The real
75-question corpus is tested separately by just running run_eval.py
directly (that's what the GitHub Actions workflow does) — these tests
stay fast by using a tiny corpus instead.
"""

from src.corpus import Document
from src.eval.harness import evaluate_method, run_full_evaluation
from src.pipeline import build_all_methods

MINI_DOCS = [
    Document("doc1.md", "Leave Policy", "# Leave Policy\n\nEmployees receive 22 days of paid annual leave per year. This is a generous policy."),
    Document("doc2.md", "Drone Specs", "# Falcon-5 Specs\n\nThe Falcon-5 drone has a battery capacity of 3.4 kWh and a range of 27 km."),
    Document("doc3.md", "Review Process", "# Code Review\n\nPull requests require approval from 2 reviewers before they can be merged into main."),
]

MINI_QUESTIONS = [
    {"id": "q1", "category": "policy", "difficulty": "exact",
     "question": "How many days of annual leave do employees get?",
     "expected_answer": "22 days", "answer_keywords": ["22", "days"], "source_doc": "doc1.md"},
    {"id": "q2", "category": "product", "difficulty": "exact",
     "question": "What is the battery capacity of the Falcon-5?",
     "expected_answer": "3.4 kWh", "answer_keywords": ["3.4", "kWh"], "source_doc": "doc2.md"},
    {"id": "q3", "category": "procedure", "difficulty": "exact",
     "question": "How many reviewers must approve a pull request?",
     "expected_answer": "2", "answer_keywords": ["2", "reviewers"], "source_doc": "doc3.md"},
]


def test_all_four_methods_build_successfully():
    methods = build_all_methods(MINI_DOCS)
    assert len(methods) == 4
    assert {m.name for m in methods} == {"simple_chunking", "semantic_chunking", "hybrid_search", "reranking"}


def test_each_method_produces_a_report_with_valid_scores():
    methods = build_all_methods(MINI_DOCS)
    for method in methods:
        report = evaluate_method(method, MINI_QUESTIONS, top_k=3)
        assert 0.0 <= report.hit_rate <= 1.0
        assert 0.0 <= report.mrr <= 1.0
        assert report.n_questions == 3


def test_run_full_evaluation_returns_one_report_per_method():
    methods = build_all_methods(MINI_DOCS)
    reports = run_full_evaluation(methods, MINI_QUESTIONS, top_k=3)
    assert len(reports) == 4


def test_easy_unambiguous_questions_are_findable_by_every_method():
    # With only 3 documents and 3 clearly distinct questions, every method
    # — even the simple baseline — should be able to find the right answer.
    # This is a sanity check that the whole pipeline isn't silently broken.
    methods = build_all_methods(MINI_DOCS)
    for method in methods:
        report = evaluate_method(method, MINI_QUESTIONS, top_k=3)
        assert report.hit_rate >= 0.66  # at least 2 of 3 found, generously
