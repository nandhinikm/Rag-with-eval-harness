"""
run_eval.py
=============
ELI5: This is the script that actually runs the whole evaluation, prints
a results table, and — this is the important bit for the GitHub Actions
part of this project — EXITS WITH AN ERROR CODE if the best method's
score falls below a threshold. A program's "exit code" is a number it
hands back to whatever ran it: 0 conventionally means "everything's
fine", anything else means "something's wrong." GitHub Actions (and any
CI system) watches this exit code to decide whether to mark a build as
passed or failed — it's not reading the printed text at all, just that
one number.

Run with:  python3 run_eval.py
Or with a custom threshold:  python3 run_eval.py --threshold 0.85
"""

import argparse
import sys

from src.corpus import load_corpus
from src.eval.harness import load_questions, run_full_evaluation, save_report
from src.pipeline import build_all_methods


def print_report(reports, threshold: float):
    print(f"\n{'Method':<20} {'Hit Rate':<12} {'MRR':<10} {'Questions'}")
    print("-" * 55)
    for r in reports:
        print(f"{r.method_name:<20} {r.hit_rate:<12.3f} {r.mrr:<10.3f} {r.n_questions}")

    best = max(reports, key=lambda r: r.hit_rate)
    print(f"\nBest method: {best.method_name} (hit rate {best.hit_rate:.3f})")

    print(f"\n{best.method_name} — breakdown by category:")
    for cat, stats in sorted(best.by_category.items()):
        print(f"  {cat:<12} hit_rate={stats['hit_rate']:.2f}  mrr={stats['mrr']:.2f}  (n={stats['n']})")

    print(f"\n{best.method_name} — breakdown by difficulty:")
    for diff, stats in sorted(best.by_difficulty.items()):
        print(f"  {diff:<12} hit_rate={stats['hit_rate']:.2f}  mrr={stats['mrr']:.2f}  (n={stats['n']})")

    print(f"\nThreshold check: best hit rate {best.hit_rate:.3f} vs required {threshold:.3f}")
    return best


def main():
    parser = argparse.ArgumentParser(description="Run the RAG retrieval evaluation harness.")
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--questions-path", default="data/eval/questions.json")
    parser.add_argument("--results-dir", default="data/eval/results")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.75,
                         help="Minimum hit rate the BEST method must achieve, or this script exits non-zero.")
    args = parser.parse_args()

    print(f"Loading corpus from {args.corpus_dir}...")
    documents = load_corpus(args.corpus_dir)
    print(f"  {len(documents)} documents loaded")

    print(f"Loading questions from {args.questions_path}...")
    questions = load_questions(args.questions_path)
    print(f"  {len(questions)} questions loaded")

    print("Building all 4 retrieval methods (this indexes the corpus once per method)...")
    methods = build_all_methods(documents)

    print(f"Running evaluation (top_k={args.top_k})...")
    reports = run_full_evaluation(methods, questions, top_k=args.top_k)

    filepath = save_report(reports, args.results_dir)
    best = print_report(reports, args.threshold)
    print(f"\nFull report saved to {filepath}")

    if best.hit_rate < args.threshold:
        print(f"\nFAILED: best hit rate {best.hit_rate:.3f} is below the required threshold {args.threshold:.3f}")
        sys.exit(1)

    print("\nPASSED: retrieval quality meets the required threshold.")
    sys.exit(0)


if __name__ == "__main__":
    main()
