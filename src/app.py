from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

try:
    from src.pipeline import RagPipeline
    from src.evaluation.evaluator import RAGEvaluator
except ModuleNotFoundError:
    # Allow running app.py when Streamlit's module path excludes project root.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.pipeline import RagPipeline
    from src.evaluation.evaluator import RAGEvaluator


st.set_page_config(page_title="Local Multimodal RAG", layout="wide")

# Sidebar for Settings and Evaluation
with st.sidebar:
    st.title("Settings & Tools")
    st.markdown("---")
    top_k = st.slider("Retrieval Depth (Top-K)", min_value=3, max_value=25, value=15, step=1)
    
    st.markdown("### System Evaluation")
    eval_btn = st.button("Run RAGAS Benchmark 📊", use_container_width=True)
    
    if eval_btn:
        with st.spinner("Evaluating..."):
            eval_obj = RAGEvaluator()
            summary = eval_obj.run(dataset_path=Path("eval_dataset_tough5.jsonl"), max_cases=5)
            st.success("Benchmark Complete")
            
            st.metric("Faithfulness", f"{summary['avg_faithfulness']:.2f}")
            st.metric("Answer Relevancy", f"{summary['avg_answer_relevancy']:.2f}")
            st.metric("Context Precision", f"{summary['avg_context_precision']:.2f}")
            st.metric("Context Recall", f"{summary['avg_context_recall']:.2f}")
            
            with st.expander("View Full Report"):
                import pandas as pd
                df = pd.read_csv(summary["csv_path"])
                st.dataframe(df)
    st.markdown("---")

# Main Interface
st.title("Local Multimodal RAG")
st.caption("Powered by OpenAI + Gemini | Optimized for Synthesis")

@st.cache_resource
def get_pipeline() -> RagPipeline:
    return RagPipeline()

pipeline = get_pipeline()

# Session State for Query Management
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "trigger_query" not in st.session_state:
    st.session_state.trigger_query = False

def handle_submit():
    query_text = st.session_state.query_input.strip()
    if query_text:
        st.session_state.last_query = query_text
        st.session_state.trigger_query = True
        # Clear the input field
        st.session_state.query_input = ""

# Query input with Enter to submit
st.text_input(
    "How can I help you today?", 
    placeholder="Ask a technical question about your PDF documents... (Press Enter to send)", 
    key="query_input",
    on_change=handle_submit
)

if st.session_state.trigger_query:
    st.session_state.trigger_query = False
    query_to_run = st.session_state.last_query
    
    with st.spinner("Thinking..."):
        result = pipeline.ask(query_to_run, top_k=top_k)
    
    # Display the Question that was asked
    st.markdown(f"### ❓ Question\n**{query_to_run}**")
    st.divider()
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.subheader("Answer")
        st.markdown(result["answer_text"])

        st.subheader("Citations")
        if result["citations"]:
            for c in result["citations"]:
                st.info(f"**[{c['id']}]** {c['source_file']} (Page {c['page']})")
        else:
            st.info("No citations available.")

        if result["tables"]:
            with st.expander("Extracted Tables", expanded=True):
                for i, tbl in enumerate(result["tables"], start=1):
                    st.markdown(f"**Table {i}** - Source: {tbl['source_file']}")
                    st.markdown(tbl["markdown"])
                    st.divider()

    with col_b:
        if result["images"]:
            st.subheader("Visual Evidence")
            for img in result["images"]:
                st.image(img["path"], caption=f"{img['source_file']} (p.{img['page']})")
        
        with st.expander("System Diagnostics"):
            st.write("**Retrieval Metadata:**")
            st.json(result["retrieval"])
            st.write("**Generation Metadata:**")
            st.json(result["generation"])
            st.write("**Full Context Chunks:**")
            st.json(result["retrieved"])
