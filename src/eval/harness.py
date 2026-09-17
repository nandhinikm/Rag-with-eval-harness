"""
src/eval/harness.py
=====================
ELI5: This is the "exam invigilator" — it takes the 75-question eval set,
runs every single question through every one of the 4 methods, grades
each answer using metrics.py, and produces one summary report per method
(overall, and broken down by question category and difficulty, so you can
see not just WHICH method wins, but WHERE and WHY).
"""

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from src.eval.metrics import evaluate_question
from src.pipeline import RetrievalMethod


@dataclass
class MethodReport:
    method_name: str
    description: str
    hit_rate: float
    mrr: float
    n_questions: int
    by_category: dict = field(default_factory=dict)
    by_difficulty: dict = field(default_factory=dict)


def load_questions(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def evaluate_method(method: RetrievalMethod, questions: list[dict], top_k: int = 5) -> MethodReport:
    results = []
    for q in questions:
        retrieved = method.search(q["question"], top_k=top_k)
        result = evaluate_question(q["answer_keywords"], retrieved)
        results.append((q, result))

    hit_rate = statistics.mean(1.0 if r.hit else 0.0 for _, r in results)
    mrr = statistics.mean(r.reciprocal_rank for _, r in results)

    by_category = _breakdown(results, key=lambda q: q["category"])
    by_difficulty = _breakdown(results, key=lambda q: q["difficulty"])

    return MethodReport(
        method_name=method.name, description=method.description,
        hit_rate=hit_rate, mrr=mrr, n_questions=len(questions),
        by_category=by_category, by_difficulty=by_difficulty,
    )


def _breakdown(results, key) -> dict:
    groups: dict[str, list] = {}
    for q, r in results:
        groups.setdefault(key(q), []).append(r)
    return {
        group: {
            "hit_rate": statistics.mean(1.0 if r.hit else 0.0 for r in rs),
            "mrr": statistics.mean(r.reciprocal_rank for r in rs),
            "n": len(rs),
        }
        for group, rs in groups.items()
    }


def run_full_evaluation(methods: list[RetrievalMethod], questions: list[dict], top_k: int = 5) -> list[MethodReport]:
    return [evaluate_method(m, questions, top_k=top_k) for m in methods]


def save_report(reports: list[MethodReport], results_dir: str) -> str:
    import os
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {"timestamp": timestamp, "reports": [asdict(r) for r in reports]}

    filepath = os.path.join(results_dir, f"eval_{timestamp}.json")
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(results_dir, "latest.json"), "w") as f:
        json.dump(payload, f, indent=2)
    return filepath
