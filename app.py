"""
app.py
======
ELI5: A visual way to browse the evaluation results — run the eval, then
see the 4 methods compared side by side with bar charts, without reading
raw JSON. Almost no logic lives here; it just re-uses run_eval.py's own
building blocks and draws the results.
"""

import pandas as pd
import streamlit as st

from src.corpus import load_corpus
from src.eval.harness import load_questions, run_full_evaluation, save_report
from src.pipeline import build_all_methods

st.set_page_config(page_title="RAG Retrieval Evaluation", page_icon="🔍", layout="centered")

st.title("🔍 RAG Retrieval Evaluation Dashboard")
st.caption("Comparing 4 retrieval methods against a 75-question benchmark — "
           "fully offline, no external AI calls.")


@st.cache_data
def load_data():
    documents = load_corpus("data/corpus")
    questions = load_questions("data/eval/questions.json")
    return documents, questions


documents, questions = load_data()
st.write(f"**{len(documents)} documents** in the corpus, **{len(questions)} questions** in the eval set.")

top_k = st.slider("top_k (how many chunks each method retrieves per question)", 1, 10, 5)
threshold = st.slider("Pass/fail threshold (hit rate)", 0.0, 1.0, 0.75, 0.05)

if st.button("Run evaluation"):
    with st.spinner("Indexing corpus and running all 4 methods..."):
        methods = build_all_methods(documents)
        reports = run_full_evaluation(methods, questions, top_k=top_k)
        filepath = save_report(reports, "data/eval/results")

    summary_df = pd.DataFrame([{
        "Method": r.method_name,
        "Hit Rate": round(r.hit_rate, 3),
        "MRR": round(r.mrr, 3),
        "Questions": r.n_questions,
    } for r in reports])

    best = max(reports, key=lambda r: r.hit_rate)
    if best.hit_rate >= threshold:
        st.success(f"PASSED — best method '{best.method_name}' scored {best.hit_rate:.3f}, "
                   f"above the {threshold:.2f} threshold.")
    else:
        st.error(f"FAILED — best method '{best.method_name}' scored {best.hit_rate:.3f}, "
                 f"below the {threshold:.2f} threshold.")

    st.subheader("Summary")
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Hit Rate by method")
    st.bar_chart(summary_df.set_index("Method")["Hit Rate"])

    st.subheader("MRR by method")
    st.bar_chart(summary_df.set_index("Method")["MRR"])

    st.subheader(f"'{best.method_name}' — breakdown by question category")
    cat_df = pd.DataFrame([
        {"Category": cat, "Hit Rate": round(stats["hit_rate"], 2), "MRR": round(stats["mrr"], 2), "n": stats["n"]}
        for cat, stats in best.by_category.items()
    ])
    st.dataframe(cat_df, use_container_width=True)

    st.subheader(f"'{best.method_name}' — breakdown by difficulty")
    diff_df = pd.DataFrame([
        {"Difficulty": diff, "Hit Rate": round(stats["hit_rate"], 2), "MRR": round(stats["mrr"], 2), "n": stats["n"]}
        for diff, stats in best.by_difficulty.items()
    ])
    st.dataframe(diff_df, use_container_width=True)

    st.caption(f"Full report saved to {filepath}")
