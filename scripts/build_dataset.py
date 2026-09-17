"""
scripts/build_dataset.py
==========================
ELI5: This script does two things in one pass, from the single "answer
key" in facts.py: (1) writes a folder of realistic-looking fake company
documents, each with the real fact sentence buried among plausible
surrounding text (not just a bare fact sitting alone — that would make
retrieval trivially easy and defeat the whole point of a benchmark), and
(2) writes out the 75-question evaluation set as JSON.

Run with:  python3 scripts/build_dataset.py
"""

import json
import os
import random

from facts import COMPANY_NAME, DECISIONS, GLOSSARY, POLICIES, PROCEDURES, PRODUCTS, Fact

random.seed(42)

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
EVAL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval", "questions.json")

# ELI5: filler sentences that sound like real corporate writing but carry
# no factual payload. Mixing these in around each real fact is what makes
# this a genuine retrieval CHALLENGE rather than a trivial lookup — a
# retrieval method has to find the one true sentence inside a paragraph
# of plausible-sounding noise, the same as a real company wiki.
FILLER_SENTENCES = [
    "This document is maintained by the {dept} team and reviewed quarterly.",
    "Please direct any questions about this topic to your manager or HR business partner.",
    "This policy applies to all Solara Robotics employees, contractors, and interns unless otherwise noted.",
    "For the most current version of this document, always refer to the internal wiki.",
    "This procedure was last reviewed and updated following the Q2 engineering retrospective.",
    "Exceptions to this guidance require written approval from a department head.",
    "Solara Robotics is committed to maintaining industry-leading standards in this area.",
    "This information is considered internal and should not be shared outside the company.",
    "A summary of related changes is available in the engineering changelog.",
    "Employees are encouraged to raise questions about this topic during team stand-ups.",
]

DEPTS = ["Engineering", "Operations", "People", "Product", "Safety"]


def build_filler_paragraph(n_sentences=3) -> str:
    chosen = random.sample(FILLER_SENTENCES, n_sentences)
    return " ".join(s.format(dept=random.choice(DEPTS)) for s in chosen)


def write_doc(filename: str, title: str, category: str, fact_sentences: list[str]) -> None:
    """Writes one markdown document: a title, an intro filler paragraph,
    then each real fact sentence embedded inside its own filler paragraph
    — so the fact is present but surrounded by realistic noise."""
    lines = [f"# {title}", "", f"**Category:** {category}  ", f"**Company:** {COMPANY_NAME}", ""]
    lines.append(build_filler_paragraph(2))
    lines.append("")
    for fact_sentence in fact_sentences:
        para = f"{build_filler_paragraph(1)} {fact_sentence} {build_filler_paragraph(1)}"
        lines.append(para)
        lines.append("")
    with open(os.path.join(CORPUS_DIR, filename), "w") as f:
        f.write("\n".join(lines))


def build_products():
    facts, docs = [], []
    for p in PRODUCTS:
        filename = f"product-{p['id']}.md"
        title = f"{p['name']} — Product Specification"

        range_sentence = (
            f"The {p['name']} has a {p['battery_kwh']} kWh battery providing an operating "
            f"range of up to {p['range_km']} km per full charge."
        )
        price_sentence = (
            f"The {p['name']} has a list price of ${p['price_usd']:,} for standard configuration."
        )
        warranty_sentence = (
            f"The {p['name']} ships with a {p['warranty_years']}-year manufacturer warranty, "
            f"first available starting {p['release_year']}."
        )

        write_doc(filename, title, "product", [range_sentence, price_sentence, warranty_sentence])

        facts.append(Fact(
            fact_id=f"{p['id']}-range", category="product", doc_id=filename, doc_title=title,
            fact_sentence=range_sentence,
            question=f"What is the operating range of the {p['name']}?",
            expected_answer=f"{p['range_km']} km",
            answer_keywords=[str(p["range_km"]), "km", "range"], difficulty="exact",
        ))
        facts.append(Fact(
            fact_id=f"{p['id']}-price", category="product", doc_id=filename, doc_title=title,
            fact_sentence=price_sentence,
            question=f"What is the list price of the {p['name']}?",
            expected_answer=f"${p['price_usd']:,}",
            answer_keywords=[str(p["price_usd"]), "price"], difficulty="exact",
        ))
        facts.append(Fact(
            fact_id=f"{p['id']}-warranty", category="product", doc_id=filename, doc_title=title,
            fact_sentence=warranty_sentence,
            question=f"How many years is the manufacturer warranty for the {p['name']}?",
            expected_answer=f"{p['warranty_years']} years",
            answer_keywords=[str(p["warranty_years"]), "year", "warranty"], difficulty="paraphrase",
        ))
    return facts


def build_simple_category(entries, category, id_field="id", title_field="title", sentence_field="sentence"):
    facts = []
    for e in entries:
        filename = f"{category}-{e[id_field]}.md"
        title = e.get(title_field, e.get("term", e[id_field]))
        write_doc(filename, title, category, [e[sentence_field]])
        facts.append(Fact(
            fact_id=e[id_field], category=category, doc_id=filename, doc_title=title,
            fact_sentence=e[sentence_field], question=e["question"],
            expected_answer=e["answer"], answer_keywords=e["keywords"], difficulty=e["difficulty"],
        ))
    return facts


def main():
    os.makedirs(CORPUS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(EVAL_PATH), exist_ok=True)
    # clear out any previously generated docs so re-runs don't leave stale files
    for f in os.listdir(CORPUS_DIR):
        if f.endswith(".md"):
            os.remove(os.path.join(CORPUS_DIR, f))

    all_facts: list[Fact] = []
    all_facts += build_products()
    all_facts += build_simple_category(POLICIES, "policy")
    all_facts += build_simple_category(PROCEDURES, "procedure")
    all_facts += build_simple_category(DECISIONS, "decision")
    all_facts += build_simple_category(GLOSSARY, "glossary", sentence_field="sentence")

    assert len(all_facts) == 75, f"Expected exactly 75 facts, got {len(all_facts)}"

    questions_payload = [
        {
            "id": f.fact_id,
            "category": f.category,
            "difficulty": f.difficulty,
            "question": f.question,
            "expected_answer": f.expected_answer,
            "answer_keywords": f.answer_keywords,
            "source_doc": f.doc_id,
        }
        for f in all_facts
    ]
    with open(EVAL_PATH, "w") as f:
        json.dump(questions_payload, f, indent=2)

    n_docs = len([f for f in os.listdir(CORPUS_DIR) if f.endswith(".md")])
    print(f"Wrote {n_docs} documents to {CORPUS_DIR}")
    print(f"Wrote {len(questions_payload)} questions to {EVAL_PATH}")
    by_category = {}
    for f in all_facts:
        by_category[f.category] = by_category.get(f.category, 0) + 1
    for cat, count in sorted(by_category.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
